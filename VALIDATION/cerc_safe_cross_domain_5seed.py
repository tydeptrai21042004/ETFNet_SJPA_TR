from __future__ import annotations
import json, math, statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from ultra_modeling.nn.modules.cerc import CERCModel
from ultra_modeling.nn.modules.evidence_baselines import EvidenceFusionModel

torch.set_num_threads(2)
SEEDS=(0,1,2,3,4)
ACCEPT_MARGIN=0.015

def kw(ch): return dict(input_channels=ch,widths=(8,12,16,20),group_width=4,stat_grid=4,trust_radius=.05,relation_kernel=3)

def fabric(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-math.pi,math.pi,size),torch.linspace(-math.pi,math.pi,size),indexing='ij'); ims=[]; ms=[]
    for i in range(n):
        phase=float(torch.rand((),generator=g))*math.pi; fx=8+int(torch.randint(0,5,(),generator=g)); fy=9+int(torch.randint(0,5,(),generator=g)); base=.48+.13*torch.sin(fx*xx+phase)+.11*torch.cos(fy*yy-.4*phase); m=torch.zeros(size,size)
        h=int(torch.randint(4,9,(),generator=g)); w=int(torch.randint(5,11,(),generator=g)); y0=int(torch.randint(3,size-h-3,(),generator=g)); x0=int(torch.randint(3,size-w-3,(),generator=g)); mode=i%3
        if mode==0: base[y0:y0+h,x0:x0+w]+=.45
        elif mode==1: base[y0:y0+h,x0:x0+w]-=.35
        else: base[y0:y0+h,x0:x0+w]=torch.roll(base[y0:y0+h,x0:x0+w],2,1)
        m[y0:y0+h,x0:x0+w]=1; rgb=torch.stack((base,base*.94,base*1.04)).clamp(0,1)+.025*torch.randn((3,size,size),generator=g); ims.append(rgb); ms.append(m.long())
    return torch.stack(ims),torch.stack(ms)

def medcls(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij'); ims=[]; ys=[]
    for i in range(n):
        label=i%2; cx=float(torch.empty(()).uniform_(-.18,.18,generator=g)); cy=float(torch.empty(()).uniform_(-.18,.18,generator=g)); r=(.22 if label==0 else .43)+float(torch.empty(()).uniform_(-.04,.04,generator=g)); rr=(xx-cx)**2+(yy-cy)**2; blob=torch.exp(-rr/(2*r*r));
        if label: blob=blob-.35*torch.exp(-rr/(2*.13**2))
        blob += .06*torch.randn((size,size),generator=g); ims.append(blob.unsqueeze(0)); ys.append(label)
    return torch.stack(ims),torch.tensor(ys)

def ultrasound(n,h,w,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij'); ims=[]; ms=[]
    for i in range(n):
        cx=float(torch.empty(()).uniform_(-.35,.35,generator=g)); cy=float(torch.empty(()).uniform_(-.3,.3,generator=g)); rx=float(torch.empty(()).uniform_(.18,.35,generator=g)); ry=float(torch.empty(()).uniform_(.15,.28,generator=g)); m=((((xx-cx)/rx)**2+((yy-cy)/ry)**2)<1).float(); bg=.30+.11*torch.randn((h,w),generator=g); im=bg+.16*bg*torch.randn((h,w),generator=g)+(.28+float(torch.rand((),generator=g))*.12)*m; ims.append(im.unsqueeze(0)); ms.append(m.long())
    return torch.stack(ims),torch.stack(ms)

def multimed(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij'); ev={k:[] for k in ('t1','t2','flair','adc')}; ms=[]
    for i in range(n):
        cx=float(torch.empty(()).uniform_(-.3,.3,generator=g)); cy=float(torch.empty(()).uniform_(-.3,.3,generator=g)); rad=float(torch.empty(()).uniform_(.13,.23,generator=g)); rr=(xx-cx)**2+(yy-cy)**2; m=(rr<rad*rad).float(); common=torch.exp(-rr/(2*(rad*.7)**2)); noise=lambda s:s*torch.randn((size,size),generator=g); ev['t1'].append((.32+.28*common+noise(.06)).unsqueeze(0)); ev['t2'].append((.40+.20*common+noise(.06)).unsqueeze(0)); ev['flair'].append((.22+.55*common+noise(.06)).unsqueeze(0)); ev['adc'].append((.58-.28*common+noise(.06)).unsqueeze(0)); ms.append(m.long())
    return {k:torch.stack(v) for k,v in ev.items()},torch.stack(ms)

def batches(x,y,batch,steps,seed):
    g=torch.Generator().manual_seed(seed); n=y.shape[0]
    for _ in range(steps):
        idx=torch.randint(0,n,(batch,),generator=g); yield ({k:v[idx] for k,v in x.items()} if isinstance(x,dict) else x[idx]),y[idx]

def train(model,x,y,seed,steps,lr,relation_only=False):
    if relation_only:
        for name,p in model.named_parameters(): p.requires_grad_("backbone.relations." in name)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0)
    for xb,yb in batches(x,y,8,steps,seed+700):
        model.train()
        if relation_only:
            for module in model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
        opt.zero_grad(set_to_none=True); out=model(xb); loss=F.cross_entropy(out,yb); loss.backward(); opt.step()

def metric(model,task,x,y):
    model.eval();
    with torch.no_grad(): out=model(x)
    if task=='classify': return float((out.argmax(1)==y).float().mean())
    pred=out.argmax(1); inter=((pred==1)&(y==1)).sum().item(); union=((pred==1)|(y==1)).sum().item(); return inter/(union+1e-9)

def score(model,task,data,subset=False):
    x,y=data
    if subset and isinstance(x,dict): x={'t1':x['t1'],'flair':x['flair']}
    return metric(model,task,x,y)

def copy_common(src,dst):
    ss=src.state_dict(); ds=dst.state_dict()
    with torch.no_grad():
        for k,v in ds.items():
            if k in ss and ss[k].shape==v.shape: v.copy_(ss[k])
    dst.load_state_dict(ds,strict=True)

def case_cfg(case,seed):
    if case=='fabric': return 'segment',3,fabric(32,40,1000+seed),fabric(32,40,2000+seed),fabric(48,40,3000+seed),20,.004,.008
    if case=='medcls': return 'classify',1,medcls(80,28,4000+seed),medcls(80,28,5000+seed),medcls(120,28,6000+seed),20,.004,.008
    if case=='ultrasound': return 'segment',1,ultrasound(32,40,36,7000+seed),ultrasound(32,40,36,8000+seed),ultrasound(48,40,36,9000+seed),20,.004,.008
    if case=='multimed':
        ch={'t1':1,'t2':1,'flair':1,'adc':1}; return 'segment',ch,multimed(32,36,10000+seed),multimed(32,36,11000+seed),multimed(48,36,12000+seed),16,.011,.012
    raise KeyError(case)

report={'accept_margin':ACCEPT_MARGIN,'cases':{}}
for case in ('fabric','medcls','ultrasound','multimed'):
    rows=[]
    for seed in SEEDS:
        print(case,seed,flush=True); task,ch,tr,va,te,base_steps,base_lr,rel_lr=case_cfg(case,seed)
        torch.manual_seed(12345+seed); base=EvidenceFusionModel('mean',task,num_classes=2,backbone_kwargs=kw(ch),head_channels=8); train(base,*tr,seed,base_steps,base_lr,False)
        use_subset=isinstance(ch,dict)
        base_val=(score(base,task,va,False)+score(base,task,va,True))/2 if use_subset else score(base,task,va)
        base_test=(score(base,task,te,False)+score(base,task,te,True))/2 if use_subset else score(base,task,te)
        torch.manual_seed(99999+seed); cerc=CERCModel(task,num_classes=2,backbone_kwargs=kw(ch),head_channels=8); copy_common(base,cerc)
        # exact tensor parity at step 0
        x0,y0=va; probe={k:v[:2] for k,v in x0.items()} if isinstance(x0,dict) else x0[:2]
        base.eval(); cerc.eval();
        with torch.no_grad(): parity=float((base(probe)-cerc(probe)).abs().max())
        if parity!=0.0: raise RuntimeError((case,seed,parity))
        common0={k:v.detach().clone() for k,v in cerc.state_dict().items() if 'backbone.relations.' not in k}
        best_val=base_val; best_step=0; best_state={k:v.detach().clone() for k,v in cerc.state_dict().items()}
        for chunk in range(1,5):
            train(cerc,*tr,seed+100*chunk,2,rel_lr,True)
            val=(score(cerc,task,va,False)+score(cerc,task,va,True))/2 if use_subset else score(cerc,task,va)
            if val >= base_val + ACCEPT_MARGIN and val>best_val:
                best_val=val; best_step=2*chunk; best_state={k:v.detach().clone() for k,v in cerc.state_dict().items()}
        cerc.load_state_dict(best_state,strict=True)
        test=(score(cerc,task,te,False)+score(cerc,task,te,True))/2 if use_subset else score(cerc,task,te)
        common1={k:v.detach() for k,v in cerc.state_dict().items() if 'backbone.relations.' not in k}; common_diff=max(float((common0[k]-common1[k]).abs().max()) for k in common0)
        rows.append({'seed':seed,'baseline_val':base_val,'selected_val':best_val,'baseline_test':base_test,'selected_test':test,'test_gain':test-base_test,'selected_relation_steps':best_step,'initial_prediction_maxdiff':parity,'common_state_maxdiff':common_diff})
    gains=[r['test_gain'] for r in rows]
    report['cases'][case]={'baseline_test_mean':statistics.mean(r['baseline_test'] for r in rows),'cerc_test_mean':statistics.mean(r['selected_test'] for r in rows),'test_gain_mean':statistics.mean(gains),'test_nondegradation':sum(g>=0 for g in gains),'test_strict_improvement':sum(g>0 for g in gains),'selected_adaptation_seeds':sum(r['selected_relation_steps']>0 for r in rows),'rows':rows}
Path('VALIDATION/cerc_safe_cross_domain_5seed.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
