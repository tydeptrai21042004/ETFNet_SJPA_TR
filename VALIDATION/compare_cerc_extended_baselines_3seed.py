from __future__ import annotations
import copy, json, math, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F
from ultra_modeling.nn.modules.cerc import CERCModel
from ultra_modeling.nn.modules.evidence_baselines import EvidenceFusionModel, FUSION_BASELINES

torch.set_num_threads(2)
METHODS=("cerc","mean","max","energy","gate","deepset","median","smoothmax","set_attention")
SEEDS=(0,1,2)

def kw(ch): return dict(input_channels=ch,widths=(8,12,16,20),group_width=4,stat_grid=4,trust_radius=.05,relation_kernel=3)

def copy_common(reference, target):
    rs=reference.state_dict(); ts=target.state_dict(); copied=0; maxdiff=0.0
    with torch.no_grad():
        for k,v in ts.items():
            if k in rs and rs[k].shape==v.shape:
                v.copy_(rs[k]); copied+=1; maxdiff=max(maxdiff,float((v-rs[k]).abs().max()))
    target.load_state_dict(ts,strict=True)
    return copied,maxdiff

def make_model(method,task,ch,num_classes=2):
    if method=="cerc": return CERCModel(task,num_classes=num_classes,backbone_kwargs=kw(ch),head_channels=8)
    return EvidenceFusionModel(method,task,num_classes=num_classes,backbone_kwargs=kw(ch),head_channels=8)

def fabric(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-math.pi,math.pi,size),torch.linspace(-math.pi,math.pi,size),indexing='ij')
    ims=[]; masks=[]
    for i in range(n):
        phase=float(torch.rand((),generator=g))*math.pi; freqx=8+int(torch.randint(0,5,(),generator=g)); freqy=9+int(torch.randint(0,5,(),generator=g))
        base=.48+.13*torch.sin(freqx*xx+phase)+.11*torch.cos(freqy*yy-.4*phase); m=torch.zeros(size,size)
        h=int(torch.randint(4,9,(),generator=g)); w=int(torch.randint(5,11,(),generator=g)); y0=int(torch.randint(3,size-h-3,(),generator=g)); x0=int(torch.randint(3,size-w-3,(),generator=g))
        mode=i%3
        if mode==0: base[y0:y0+h,x0:x0+w]+=.45
        elif mode==1: base[y0:y0+h,x0:x0+w]-=.35
        else: base[y0:y0+h,x0:x0+w]=torch.roll(base[y0:y0+h,x0:x0+w],shifts=2,dims=1)
        m[y0:y0+h,x0:x0+w]=1
        rgb=torch.stack((base,base*.94,base*1.04)).clamp(0,1); rgb += .025*torch.randn(rgb.shape,generator=g)
        ims.append(rgb); masks.append(m.long())
    return torch.stack(ims),torch.stack(masks)

def medcls(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij'); ims=[]; y=[]
    for i in range(n):
        label=i%2; cx=float(torch.empty(()).uniform_(-.18,.18,generator=g)); cy=float(torch.empty(()).uniform_(-.18,.18,generator=g)); r=(.22 if label==0 else .43)+float(torch.empty(()).uniform_(-.04,.04,generator=g))
        rr=(xx-cx)**2+(yy-cy)**2; blob=torch.exp(-rr/(2*r*r))
        if label: blob=blob-.35*torch.exp(-rr/(2*(.13**2)))
        blob += .06*torch.randn(blob.shape,generator=g); ims.append(blob.unsqueeze(0)); y.append(label)
    return torch.stack(ims),torch.tensor(y)

def ultrasound(n,h,w,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij'); ims=[]; ms=[]
    for i in range(n):
        cx=float(torch.empty(()).uniform_(-.35,.35,generator=g)); cy=float(torch.empty(()).uniform_(-.3,.3,generator=g)); rx=float(torch.empty(()).uniform_(.18,.35,generator=g)); ry=float(torch.empty(()).uniform_(.15,.28,generator=g))
        m=((((xx-cx)/rx)**2+((yy-cy)/ry)**2)<1).float(); bg=.30+.11*torch.randn((h,w),generator=g); speckle=.16*bg*torch.randn((h,w),generator=g); im=bg+speckle+(.28+float(torch.rand((),generator=g))*.12)*m
        ims.append(im.unsqueeze(0)); ms.append(m.long())
    return torch.stack(ims),torch.stack(ms)

def multimed(n,size,seed):
    g=torch.Generator().manual_seed(seed); yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij'); ev={k:[] for k in ('t1','t2','flair','adc')}; ms=[]
    for i in range(n):
        cx=float(torch.empty(()).uniform_(-.3,.3,generator=g)); cy=float(torch.empty(()).uniform_(-.3,.3,generator=g)); rad=float(torch.empty(()).uniform_(.13,.23,generator=g)); rr=(xx-cx)**2+(yy-cy)**2; m=(rr<rad*rad).float(); common=torch.exp(-rr/(2*(rad*.7)**2)); noise=lambda s: s*torch.randn((size,size),generator=g)
        ev['t1'].append((.32+.28*common+noise(.06)).unsqueeze(0)); ev['t2'].append((.40+.20*common+noise(.06)).unsqueeze(0)); ev['flair'].append((.22+.55*common+noise(.06)).unsqueeze(0)); ev['adc'].append((.58-.28*common+noise(.06)).unsqueeze(0)); ms.append(m.long())
    return {k:torch.stack(v) for k,v in ev.items()},torch.stack(ms)

def batches(x,y,batch,steps,seed):
    n=y.shape[0]; g=torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx=torch.randint(0,n,(batch,),generator=g)
        if isinstance(x,dict): xb={k:v[idx] for k,v in x.items()}
        else: xb=x[idx]
        yield xb,y[idx]

def dice_iou(logits,target):
    pred=logits.argmax(1); inter=((pred==1)&(target==1)).sum().item(); union=((pred==1)|(target==1)).sum().item(); ps=(pred==1).sum().item(); ts=(target==1).sum().item()
    return (2*inter/(ps+ts+1e-9), inter/(union+1e-9))

def eval_task(model,task,x,y):
    model.eval();
    with torch.no_grad():
        o=model(x)
        if task=='classify': return {'score':float((o.argmax(1)==y).float().mean()),'loss':float(F.cross_entropy(o,y))}
        d,i=dice_iou(o,y); return {'score':i,'dice':d,'loss':float(F.cross_entropy(o,y))}

def train_task(model,task,x,y,seed,steps=24,lr=.004):
    model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0)
    for xb,yb in batches(x,y,8,steps,seed+999):
        opt.zero_grad(set_to_none=True); out=model(xb); loss=F.cross_entropy(out,yb); loss.backward(); opt.step()


def param_count(m): return sum(p.numel() for p in m.parameters())

def run_case(case, seed):
    if case=='fabric': tr=fabric(32,40,1000+seed); va=fabric(32,40,2000+seed); task='segment'; ch=3; steps=16; lr=.008
    elif case=='medcls': tr=medcls(80,28,3000+seed); va=medcls(80,28,4000+seed); task='classify'; ch=1; steps=16; lr=.008
    elif case=='ultrasound': tr=ultrasound(32,40,36,5000+seed); va=ultrasound(32,40,36,6000+seed); task='segment'; ch=1; steps=16; lr=.008
    elif case=='multimed': tr=multimed(32,36,7000+seed); va=multimed(32,36,8000+seed); task='segment'; ch={'t1':1,'t2':1,'flair':1,'adc':1}; steps=16; lr=.008
    else: raise KeyError(case)
    torch.manual_seed(12345+seed); reference=make_model('mean',task,ch)
    common_ref={k:v.detach().clone() for k,v in reference.state_dict().items()}
    results={}
    for method in METHODS:
        torch.manual_seed(90000+seed)
        model=make_model(method,task,ch)
        copied,diff=copy_common(reference,model)
        # prove common tensors really equal reference now
        assert diff==0.0
        before=eval_task(model,task,*va)
        train_task(model,task,*tr,seed,steps=steps,lr=lr)
        after=eval_task(model,task,*va)
        results[method]={'before':before,'after':after,'params':param_count(model),'common_tensors_copied':copied}
        if case=='multimed':
            xv,yv=va; subset={'t1':xv['t1'],'flair':xv['flair']}; results[method]['subset_after']=eval_task(model,task,subset,yv)
    return results

start=time.time(); raw={}
for case in ('multimed',):
    raw[case]={}
    for seed in SEEDS:
        print('RUN',case,seed,flush=True); raw[case][str(seed)]=run_case(case,seed)
summary={}
for case,seeds in raw.items():
    summary[case]={}
    for method in METHODS:
        vals=[seeds[str(s)][method]['after']['score'] for s in SEEDS]
        if case=='multimed':
            subset=[seeds[str(s)][method]['subset_after']['score'] for s in SEEDS]
            vals_comb=[(a+b)/2 for a,b in zip(vals,subset)]
        else: vals_comb=vals
        summary[case][method]={'mean_score':statistics.mean(vals_comb),'scores':vals_comb,'wins_vs_cerc':0}
    c=summary[case]['cerc']['scores']
    for method in METHODS[1:]: summary[case][method]['wins_vs_cerc']=sum(b>a for a,b in zip(c,summary[case][method]['scores']))
report={'protocol':{'methods':METHODS,'seeds':SEEDS,'common_initialization':'all matching tensors copied from one CERC reference per seed','score':'accuracy for classification; IoU for segmentation; multimed averages full and missing-modality subset IoU'},'summary':summary,'raw':raw,'elapsed_seconds':time.time()-start}
out=Path('VALIDATION/cerc_vs_evidence_baselines_5seed.json'); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(summary,indent=2)); print('elapsed',report['elapsed_seconds'])
