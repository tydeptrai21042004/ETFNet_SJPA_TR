from __future__ import annotations
import json, statistics, torch
import torch.nn.functional as F
from pathlib import Path
from ultra_modeling.nn.modules.cerc import CERCModel
from ultra_modeling.nn.modules.evidence_baselines import EvidenceFusionModel

torch.set_num_threads(2)

def kw(ch):
    return dict(input_channels=ch,widths=(8,12,16,20),group_width=4,stat_grid=4,trust_radius=.05,relation_kernel=3)

def multimed(n,size,seed):
    import math
    g=torch.Generator().manual_seed(seed)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij')
    ev={k:[] for k in ('t1','t2','flair','adc')}; ms=[]
    for i in range(n):
        cx=float(torch.empty(()).uniform_(-.3,.3,generator=g)); cy=float(torch.empty(()).uniform_(-.3,.3,generator=g)); rad=float(torch.empty(()).uniform_(.13,.23,generator=g))
        rr=(xx-cx)**2+(yy-cy)**2; m=(rr<rad*rad).float(); common=torch.exp(-rr/(2*(rad*.7)**2)); noise=lambda scale: scale*torch.randn((size,size),generator=g)
        ev['t1'].append((.32+.28*common+noise(.06)).unsqueeze(0)); ev['t2'].append((.40+.20*common+noise(.06)).unsqueeze(0)); ev['flair'].append((.22+.55*common+noise(.06)).unsqueeze(0)); ev['adc'].append((.58-.28*common+noise(.06)).unsqueeze(0)); ms.append(m.long())
    return {k:torch.stack(v) for k,v in ev.items()},torch.stack(ms)

def batches(x,y,batch,steps,seed):
    n=y.shape[0]; g=torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx=torch.randint(0,n,(batch,),generator=g)
        xb={k:v[idx] for k,v in x.items()} if isinstance(x,dict) else x[idx]
        yield xb,y[idx]

def dice_iou(logits,target):
    pred=logits.argmax(1); inter=((pred==1)&(target==1)).sum().item(); union=((pred==1)|(target==1)).sum().item()
    return inter/(union+1e-9)

def eval_task(model,task,x,y):
    model.eval()
    with torch.no_grad():
        o=model(x)
        if task=='classify': return {'score':float((o.argmax(1)==y).float().mean()),'loss':float(F.cross_entropy(o,y))}
        return {'score':dice_iou(o,y),'loss':float(F.cross_entropy(o,y))}

SEEDS=(0,1,2,3,4)

def train(model,x,y,seed,steps,lr,proposal_only=False):
    if proposal_only:
        for name,p in model.named_parameters():
            p.requires_grad_("backbone.relations." in name)
        model.train()
        # Frozen BatchNorm buffers must remain frozen too.
        for name,module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    else:
        model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0)
    for xb,yb in batches(x,y,8,steps,seed+991):
        if proposal_only:
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        opt.zero_grad(set_to_none=True); out=model(xb); loss=F.cross_entropy(out,yb); loss.backward(); opt.step()

def state_copy_common(source,target):
    ss=source.state_dict(); ts=target.state_dict()
    with torch.no_grad():
        for k,v in ts.items():
            if k in ss and ss[k].shape==v.shape: v.copy_(ss[k])
    target.load_state_dict(ts,strict=True)

def maxdiff_common(a,b):
    sa=a.state_dict(); sb=b.state_dict(); m=0.0
    for k,v in sa.items():
        if "backbone.relations." in k: continue
        if k in sb and sb[k].shape==v.shape: m=max(m,float((v-sb[k]).abs().max()))
    return m

raw={}
for seed in SEEDS:
    print('seed',seed,flush=True)
    tr=multimed(32,36,7000+seed); va=multimed(32,36,8000+seed); te=multimed(48,36,9000+seed); ch={'t1':1,'t2':1,'flair':1,'adc':1}
    torch.manual_seed(12345+seed)
    mean=EvidenceFusionModel('mean','segment',num_classes=2,backbone_kwargs=kw(ch),head_channels=8)
    train(mean,*tr,seed,16,.011,False)
    mean_full=eval_task(mean,'segment',*va)
    xv,yv=va; subset={'t1':xv['t1'],'flair':xv['flair']}; mean_sub=eval_task(mean,'segment',subset,yv)
    mean_score=(mean_full['score']+mean_sub['score'])/2
    xt,yt=te; test_subset={'t1':xt['t1'],'flair':xt['flair']}
    mean_test_full=eval_task(mean,'segment',*te); mean_test_sub=eval_task(mean,'segment',test_subset,yt)
    mean_test_score=(mean_test_full['score']+mean_test_sub['score'])/2

    torch.manual_seed(99999+seed)
    cerc=CERCModel('segment',num_classes=2,backbone_kwargs=kw(ch),head_channels=8)
    state_copy_common(mean,cerc)
    # Exact prediction parity at adaptation step 0.
    cerc_full0=eval_task(cerc,'segment',*va); cerc_sub0=eval_task(cerc,'segment',subset,yv)
    parity=max(abs(cerc_full0['score']-mean_full['score']),abs(cerc_sub0['score']-mean_sub['score']))
    # Tensor-level parity on a small batch.
    mean.eval(); cerc.eval()
    with torch.no_grad():
        mo=mean({k:v[:2] for k,v in xv.items()}); co=cerc({k:v[:2] for k,v in xv.items()})
    pred_diff=float((mo-co).abs().max())
    if pred_diff != 0.0: raise RuntimeError(f'initial parity failed: {pred_diff}')
    common_before={k:v.detach().clone() for k,v in cerc.state_dict().items() if 'backbone.relations.' not in k}

    best={'score':mean_score,'step':0,'full':mean_full,'subset':mean_sub,'state':{k:v.detach().clone() for k,v in cerc.state_dict().items()}}
    # Relation-only adaptation; keep exact baseline-equivalent step 0 as candidate.
    for chunk in range(1,5):
        train(cerc,*tr,seed+100*chunk,2,.012,True)
        full=eval_task(cerc,'segment',*va); sub=eval_task(cerc,'segment',subset,yv); score=(full['score']+sub['score'])/2
        if score>best['score']:
            best={'score':score,'step':2*chunk,'full':full,'subset':sub,'state':{k:v.detach().clone() for k,v in cerc.state_dict().items()}}
    cerc.load_state_dict(best['state'],strict=True)
    cerc_test_full=eval_task(cerc,'segment',*te); cerc_test_sub=eval_task(cerc,'segment',test_subset,yt); cerc_test_score=(cerc_test_full['score']+cerc_test_sub['score'])/2
    common_after={k:v.detach() for k,v in cerc.state_dict().items() if 'backbone.relations.' not in k}
    common_diff=max(float((common_before[k]-common_after[k]).abs().max()) for k in common_before)
    raw[str(seed)]={'mean_val_score':mean_score,'cerc_best_val_score':best['score'],'val_gain':best['score']-mean_score,'mean_test_score':mean_test_score,'cerc_test_score':cerc_test_score,'test_gain':cerc_test_score-mean_test_score,'best_relation_steps':best['step'],'prediction_parity_maxdiff':pred_diff,'common_state_maxdiff':common_diff}

val_gains=[v['val_gain'] for v in raw.values()]; test_gains=[v['test_gain'] for v in raw.values()]
report={'protocol':{'mean_training_steps':16,'mean_lr':.011,'cerc_relation_only_max_steps':8,'cerc_relation_lr':.012,'selection':'best validation among step 0,2,4,6,8; step 0 is exact trained mean baseline','final_report':'fresh held-out test fixture not used for checkpoint selection'},'mean_val_average':statistics.mean(v['mean_val_score'] for v in raw.values()),'cerc_val_average':statistics.mean(v['cerc_best_val_score'] for v in raw.values()),'val_average_gain':statistics.mean(val_gains),'validation_nondegradation_seeds':sum(g>=0 for g in val_gains),'validation_strict_improvement_seeds':sum(g>0 for g in val_gains),'mean_test_average':statistics.mean(v['mean_test_score'] for v in raw.values()),'cerc_test_average':statistics.mean(v['cerc_test_score'] for v in raw.values()),'test_average_gain':statistics.mean(test_gains),'test_nondegradation_seeds':sum(g>=0 for g in test_gains),'test_strict_improvement_seeds':sum(g>0 for g in test_gains),'raw':raw}
Path('VALIDATION/cerc_residual_from_mean_5seed.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
