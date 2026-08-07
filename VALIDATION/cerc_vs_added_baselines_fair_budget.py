from __future__ import annotations
import json, statistics, torch
import torch.nn.functional as F
from pathlib import Path
from ultra_modeling.nn.modules.cerc import CERCModel
from ultra_modeling.nn.modules.evidence_baselines import EvidenceFusionModel

torch.set_num_threads(2)
SEEDS=(0,1,2,3,4)
LRS={'mean':.011,'max':.011,'energy':.011,'gate':.012,'deepset':.011}

def kw(ch): return dict(input_channels=ch,widths=(8,12,16,20),group_width=4,stat_grid=4,trust_radius=.05,relation_kernel=3)

def multimed(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij'); ev={k:[] for k in ('t1','t2','flair','adc')}; ms=[]
    for i in range(n):
        cx=float(torch.empty(()).uniform_(-.3,.3,generator=g)); cy=float(torch.empty(()).uniform_(-.3,.3,generator=g)); rad=float(torch.empty(()).uniform_(.13,.23,generator=g)); rr=(xx-cx)**2+(yy-cy)**2; m=(rr<rad*rad).float(); common=torch.exp(-rr/(2*(rad*.7)**2)); noise=lambda s:s*torch.randn((size,size),generator=g); ev['t1'].append((.32+.28*common+noise(.06)).unsqueeze(0)); ev['t2'].append((.40+.20*common+noise(.06)).unsqueeze(0)); ev['flair'].append((.22+.55*common+noise(.06)).unsqueeze(0)); ev['adc'].append((.58-.28*common+noise(.06)).unsqueeze(0)); ms.append(m.long())
    return {k:torch.stack(v) for k,v in ev.items()},torch.stack(ms)

def batches(x,y,batch,steps,seed):
    g=torch.Generator().manual_seed(seed); n=y.shape[0]
    for _ in range(steps):
        idx=torch.randint(0,n,(batch,),generator=g); yield {k:v[idx] for k,v in x.items()},y[idx]

def train(model,x,y,seed,steps,lr):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0); model.train()
    for xb,yb in batches(x,y,8,steps,seed+404):
        opt.zero_grad(set_to_none=True); out=model(xb); loss=F.cross_entropy(out,yb); loss.backward(); opt.step()

def iou(model,data,subset=False):
    x,y=data
    if subset: x={'t1':x['t1'],'flair':x['flair']}
    model.eval()
    with torch.no_grad(): pred=model(x).argmax(1)
    inter=((pred==1)&(y==1)).sum().item(); union=((pred==1)|(y==1)).sum().item(); return inter/(union+1e-9)

def score(model,data): return (iou(model,data,False)+iou(model,data,True))/2

def copy_common(src,dst):
    ss=src.state_dict(); ds=dst.state_dict()
    with torch.no_grad():
        for k,v in ds.items():
            if k in ss and ss[k].shape==v.shape: v.copy_(ss[k])
    dst.load_state_dict(ds,strict=True)

# Safe-CERC held-out scores are read from the independently generated report.
safe=json.load(open('VALIDATION/cerc_safe_cross_domain_5seed.json'))['cases']['multimed']['rows']
safe_scores={int(r['seed']):r['selected_test'] for r in safe}
raw={name:{} for name in LRS}
ch={'t1':1,'t2':1,'flair':1,'adc':1}
for seed in SEEDS:
    print('seed',seed,flush=True); tr=multimed(32,36,10000+seed); te=multimed(48,36,12000+seed)
    torch.manual_seed(12345+seed); reference=EvidenceFusionModel('mean','segment',num_classes=2,backbone_kwargs=kw(ch),head_channels=8)
    for index,(name,lr) in enumerate(LRS.items()):
        torch.manual_seed(70000+seed*10+index); model=EvidenceFusionModel(name,'segment',num_classes=2,backbone_kwargs=kw(ch),head_channels=8); copy_common(reference,model); train(model,*tr,seed,24,lr); raw[name][str(seed)]=score(model,te)
summary={name:{'mean_test_score':statistics.mean(values.values()),'scores':[values[str(s)] for s in SEEDS]} for name,values in raw.items()}
summary['cerc_safe']={'mean_test_score':statistics.mean(safe_scores.values()),'scores':[safe_scores[s] for s in SEEDS]}
best_baseline=max((v['mean_test_score'],k) for k,v in summary.items() if k!='cerc_safe')
report={'protocol':{'baseline_full_training_steps':24,'baseline_lrs':LRS,'cerc_safe_budget':'16 common mean steps + at most 8 relation-only steps; validation margin 0.015','test':'fresh held-out full-four-evidence + missing-modality subset average'},'summary':summary,'best_added_baseline':{'name':best_baseline[1],'mean_test_score':best_baseline[0]},'cerc_minus_best_added_baseline':summary['cerc_safe']['mean_test_score']-best_baseline[0]}
Path('VALIDATION/cerc_vs_added_baselines_fair_budget.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
