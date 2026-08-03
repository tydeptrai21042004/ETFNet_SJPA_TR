import math,time,json,random
from dataclasses import dataclass,asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def ortho(c,seed):
 g=torch.Generator().manual_seed(seed); q,r=torch.linalg.qr(torch.randn(c,c,generator=g)); d=torch.sign(torch.diag(r));d[d==0]=1;return q*d

def xywh2xyxy(b):
 x,y,w,h=b.unbind(-1);return torch.stack((x-w/2,y-h/2,x+w/2,y+h/2),-1)
def iou(a,b):
 a=xywh2xyxy(a);b=xywh2xyxy(b);lt=torch.maximum(a[:,:2],b[:,:2]);rb=torch.minimum(a[:,2:],b[:,2:]);wh=(rb-lt).clamp_min(0);inter=wh[:,0]*wh[:,1]
 aa=(a[:,2]-a[:,0]).clamp_min(0)*(a[:,3]-a[:,1]).clamp_min(0);bb=(b[:,2]-b[:,0]).clamp_min(0)*(b[:,3]-b[:,1]).clamp_min(0);return inter/(aa+bb-inter+1e-8)
def map50(logits,pbox,y,gt,K):
 probs=logits.softmax(-1).detach().cpu().numpy(); labs=y.numpy(); io=iou(pbox,gt).detach().cpu().numpy();aps=[]
 for k in range(K):
  n=(labs==k).sum()
  if not n: continue
  order=np.argsort(-probs[:,k]);tp=((labs[order]==k)&(io[order]>=.5)).astype(float);fp=1-tp
  rec=np.cumsum(tp)/n;prec=np.cumsum(tp)/np.maximum(np.cumsum(tp)+np.cumsum(fp),1e-12)
  mr=np.r_[0,rec,1];mp=np.r_[0,prec,0]
  for j in range(len(mp)-2,-1,-1):mp[j]=max(mp[j],mp[j+1])
  idx=np.where(mr[1:]!=mr[:-1])[0];aps.append(((mr[idx+1]-mr[idx])*mp[idx+1]).sum())
 return float(np.mean(aps))

class Data(torch.utils.data.Dataset):
 def __init__(self,n=1200,c=8,h=6,w=6,K=5,seed=0):
  self.n=n;g=torch.Generator().manual_seed(seed);self.c=c;self.K=K
  sh=F.normalize(torch.randn(K,c,generator=g),dim=-1);sh[-1]=F.normalize(sh[-2]+.08*torch.randn(c,generator=g),dim=-1)
  ur=F.normalize(torch.randn(K,c,generator=g),dim=-1);ui=F.normalize(torch.randn(K,c,generator=g),dim=-1)
  ur[-1]=F.normalize(-ur[-2]+.05*torch.randn(c,generator=g),dim=-1);ui[-1]=F.normalize(ui[-2]+.08*torch.randn(c,generator=g),dim=-1)
  Qr=ortho(c,seed+701);Qi=ortho(c,seed+907);yy,xx=torch.meshgrid(torch.linspace(0,1,h),torch.linspace(0,1,w),indexing='ij')
  R=[];I=[];Y=[];B=[]
  for _ in range(n):
   y=int(torch.randint(K,(1,),generator=g));cx=float(torch.empty(1).uniform_(.2,.8,generator=g));cy=float(torch.empty(1).uniform_(.2,.8,generator=g));bw=float(torch.empty(1).uniform_(.22,.42,generator=g));bh=float(torch.empty(1).uniform_(.22,.42,generator=g))
   m=torch.exp(-.5*(((xx-cx)/(bw/2.3))**2+((yy-cy)/(bh/2.3))**2));ring=(m-F.avg_pool2d(m[None,None],3,1,1)[0,0]).abs()
   r0=sh[y,:,None,None]*m+.85*ur[y,:,None,None]*ring+.10*torch.randn(c,h,w,generator=g)
   i0=sh[y,:,None,None]*m+.85*ui[y,:,None,None]*m.pow(1.5)+.10*torch.randn(c,h,w,generator=g)
   R.append(torch.einsum('chw,cd->dhw',r0,Qr));I.append(torch.einsum('chw,cd->dhw',i0,Qi));Y.append(y);B.append([cx,cy,bw,bh])
  self.R=torch.stack(R);self.I=torch.stack(I);self.Y=torch.tensor(Y);self.B=torch.tensor(B)
 def __len__(self):return self.n
 def __getitem__(self,j):return self.R[j],self.I[j],self.Y[j],self.B[j]

def corrupt(r,i,name,seed=77):
 g=torch.Generator().manual_seed(seed)
 if name=='clean':return r,i
 if name=='basis_rotate':
  qr=ortho(r.shape[1],3123).to(r);qi=ortho(i.shape[1],9321).to(i);return torch.einsum('bchw,cd->bdhw',r,qr),torch.einsum('bchw,cd->bdhw',i,qi)
 if name=='rgb_noise':return r+.5*torch.randn(r.shape,generator=g),i
 if name=='ir_noise':return r,i+.5*torch.randn(i.shape,generator=g)
 if name=='missing_rgb':return torch.zeros_like(r),i
 if name=='missing_ir':return r,torch.zeros_like(i)
 if name=='ir_shift':return r,torch.roll(i,(1,-1),(-2,-1))
 if name=='mixed':
  return r+.35*torch.randn(r.shape,generator=g),torch.roll(i+.35*torch.randn(i.shape,generator=g),(1,0),(-2,-1))
 raise KeyError(name)

class Proj(nn.Module):
 def __init__(self,cin,c):super().__init__();self.p=nn.Sequential(nn.Conv2d(cin,c,1,bias=False),nn.BatchNorm2d(c),nn.SiLU())
 def forward(self,x):return self.p(x)
class Concat(nn.Module):
 def __init__(self,c):super().__init__();self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):z=self.p(torch.cat((r,i),1));return(z,{})if aux else z
class SumDiff(nn.Module):
 def __init__(self,c):super().__init__();self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):C=(r+i)/math.sqrt(2);D=(r-i)/math.sqrt(2);z=self.p(torch.cat((C,D),1));return(z,{'D':D})if aux else z
class LearnedCI(nn.Module):
 def __init__(self,c):super().__init__();self.a=nn.Conv2d(c,c,1,bias=False);self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):a=self.a(r);C=(a+i)/math.sqrt(2);D=(a-i)/math.sqrt(2);z=self.p(torch.cat((C,D),1));return(z,{'D':D})if aux else z

class Canonical(nn.Module):
 def __init__(self,c,g=2,eps=2e-3,mode='symmetric',stats_grad=False):super().__init__();assert c%g==0;self.c=c;self.g=g;self.d=c//g;self.eps=eps;self.mode=mode;self.stats_grad=stats_grad
 def calc(self,r,i):
  b,c,h,w=r.shape;n=h*w;d=self.d
  rg=r.reshape(b,self.g,d,n).transpose(-1,-2);ig=i.reshape(b,self.g,d,n).transpose(-1,-2)
  # closed-form statistics are stop-gradient for speed/stability; transforms still pass gradients to r/i.
  rs=rg if self.stats_grad else rg.detach();is_=ig if self.stats_grad else ig.detach()
  rm=rs.mean(-2,keepdim=True);im=is_.mean(-2,keepdim=True);rc=rs-rm;ic=is_-im
  eye=torch.eye(d).view(1,1,d,d)
  cr=rc.transpose(-1,-2)@rc/(n-1)+self.eps*eye;ci=ic.transpose(-1,-2)@ic/(n-1)+self.eps*eye
  er,vr=torch.linalg.eigh(cr);ei,vi=torch.linalg.eigh(ci)
  wr=vr@torch.diag_embed(er.clamp_min(self.eps).rsqrt())@vr.transpose(-1,-2);wi=vi@torch.diag_embed(ei.clamp_min(self.eps).rsqrt())@vi.transpose(-1,-2)
  # Apply detached transforms to live centered features.
  rlive=rg-rg.mean(-2,keepdim=True);ilive=ig-ig.mean(-2,keepdim=True);rw=rlive@wr;iw=ilive@wi
  M=(rc@wr).transpose(-1,-2)@(ic@wi)/(n-1);U,S,Vh=torch.linalg.svd(M,full_matrices=False)
  if self.mode=='onesided': A=rw@(U@Vh);B=iw
  else: A=rw@U;B=iw@Vh.transpose(-1,-2)
  return A.transpose(-1,-2).reshape(b,c,h,w),B.transpose(-1,-2).reshape(b,c,h,w),S
class WhiteConcat(Canonical):
 def __init__(self,c):super().__init__(c,2);self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):
  # obtain whitened but no rotation by reproducing calc's pre-SVD via symmetric with outputs not ideal; use A/B still canonical for simplicity is wrong.
  A,B,S=self.calc(r,i);z=self.p(torch.cat((A,B),1));return(z,{'S':S})if aux else z
class OneSidedPCI(Canonical):
 def __init__(self,c):super().__init__(c,2,mode='onesided');self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):A,B,S=self.calc(r,i);C=(A+B)/math.sqrt(2);D=(A-B)/math.sqrt(2);z=self.p(torch.cat((C,D),1));return(z,{'D':D,'C':C,'A':A,'B':B,'S':S})if aux else z
class SymConcat(Canonical):
 def __init__(self,c):super().__init__(c,2,mode='symmetric');self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):A,B,S=self.calc(r,i);z=self.p(torch.cat((A,B),1));return(z,{'S':S})if aux else z
class SymConsensus(Canonical):
 def __init__(self,c):super().__init__(c,2,mode='symmetric');self.p=Proj(c,c)
 def forward(self,r,i,aux=False):A,B,S=self.calc(r,i);C=(A+B)/math.sqrt(2);z=self.p(C);return(z,{'C':C,'S':S})if aux else z
class SCID(Canonical):
 def __init__(self,c):super().__init__(c,2,mode='symmetric');self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):A,B,S=self.calc(r,i);C=(A+B)/math.sqrt(2);D=(A-B)/math.sqrt(2);z=self.p(torch.cat((C,D),1));return(z,{'D':D,'C':C,'A':A,'B':B,'S':S})if aux else z
class PBTR(nn.Module):
 def __init__(self,c):super().__init__();h=max(2,c//4);self.q=nn.Sequential(nn.Conv2d(c,h,1),nn.SiLU(),nn.Conv2d(h,1,1));self.u=nn.Sequential(nn.Conv2d(4*c,c,1),nn.SiLU(),nn.Conv2d(c,c,3,1,1,groups=c));self.p=Proj(c,c)
 def forward(self,r,i,aux=False):
  rn=F.group_norm(r,1);inn=F.group_norm(i,1);p=torch.cat((self.q(rn),self.q(inn)),1).softmax(1);pr,pi=p[:,:1],p[:,1:];z0=pr*r+pi*i;u=self.u(torch.cat((rn,inn,rn-inn,rn*inn),1));rms=lambda x:(x.square().mean((1,2,3),keepdim=True)+1e-6).sqrt();rho=.35*(4*pr*pi)*rms(z0);ru=rms(u);d=u/(ru+1e-6)*rho*torch.tanh(ru/(rho+1e-6));z=self.p(z0+d);return(z,{'D':d,'p':p})if aux else z
class TGF(nn.Module):
 def __init__(self,c):super().__init__();self.pool=nn.AdaptiveAvgPool2d((3,3));enc=nn.TransformerEncoderLayer(c,2,2*c,batch_first=True,dropout=0);self.t=nn.TransformerEncoder(enc,1);self.pos=nn.Parameter(torch.zeros(1,18,c));self.map=nn.Conv2d(2*c,2*c,1,groups=2);self.p=Proj(2*c,c)
 def forward(self,r,i,aux=False):
  b,c,h,w=r.shape;rp=self.pool(r);ip=self.pool(i);x=torch.cat((rp.flatten(2).transpose(1,2),ip.flatten(2).transpose(1,2)),1)+self.pos;x=self.t(x);x=x.view(b,2,9,c).permute(0,1,3,2);rr=x[:,0].view(b,c,3,3);ii=x[:,1].view(b,c,3,3);rr,ii=self.map(torch.cat((rr,ii),1)).chunk(2,1);rr=F.interpolate(rr,(h,w));ii=F.interpolate(ii,(h,w));z=self.p(torch.cat((r+rr,i+ii),1));return(z,{})if aux else z
class Head(nn.Module):
 def __init__(self,c,K):super().__init__();self.f=nn.Sequential(nn.Conv2d(c,c,3,1,1),nn.SiLU(),nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(c*9,48),nn.SiLU());self.cl=nn.Linear(48,K);self.bo=nn.Linear(48,4)
 def forward(self,x):h=self.f(x);return self.cl(h),torch.sigmoid(self.bo(h))
class Model(nn.Module):
 def __init__(self,f,c=8,K=5):super().__init__();self.f=f;self.h=Head(c,K)
 def forward(self,r,i,aux=False):
  if aux:z,a=self.f(r,i,True);l,b=self.h(z);return l,b,a
  return self.h(self.f(r,i))
def fisher(D,y):
 x=D.mean((2,3));u=y.unique();gm=x.mean(0);W=x.new_tensor(0.);B=x.new_tensor(0.)
 for k in u:
  q=x[y==k];m=q.mean(0);W+=(q-m).square().sum();B+=q.shape[0]*(m-gm).square().sum()
 return W/(B+1e-3)

def run(name,seed,fw=0,epochs=6):
 seed_all(seed);ds=Data(1400,seed=300+seed);tr=torch.utils.data.Subset(ds,range(1000));te=torch.utils.data.Subset(ds,range(1000,1400));c=8
 fac={'concat':Concat,'sumdiff':SumDiff,'learned_ci':LearnedCI,'onesided_pci':OneSidedPCI,'sym_concat':SymConcat,'sym_consensus':SymConsensus,'scid':SCID,'pbtr':PBTR,'tgf':TGF}
 m=Model(fac[name](c),c,5);o=torch.optim.AdamW(m.parameters(),2.5e-3,weight_decay=1e-4);ld=torch.utils.data.DataLoader(tr,64,shuffle=True,generator=torch.Generator().manual_seed(seed));t=time.time()
 for ep in range(epochs):
  m.train()
  for r,i,y,b in ld:
   o.zero_grad();l,p,a=m(r,i,True);loss=F.cross_entropy(l,y)+3*F.smooth_l1_loss(p,b)
   if fw and 'D'in a:loss=loss+fw*fisher(a['D'],y)
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);o.step()
 train_s=time.time()-t;m.eval();conds=['clean','basis_rotate','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed'];met={};loader=torch.utils.data.DataLoader(te,128)
 with torch.no_grad():
  for cond in conds:
   L=[];P=[];Y=[];G=[]
   for r,i,y,b in loader:r,i=corrupt(r,i,cond,seed+99);l,p=m(r,i);L.append(l);P.append(p);Y.append(y);G.append(b)
   L=torch.cat(L);P=torch.cat(P);Y=torch.cat(Y);G=torch.cat(G);met[cond]={'acc':float((L.argmax(1)==Y).float().mean()),'miou':float(iou(P,G).mean()),'map50':map50(L,P,Y,G,5)}
 r,i,_,_=next(iter(torch.utils.data.DataLoader(te,1)));ts=[]
 with torch.no_grad():
  for _ in range(3):m(r,i)
  for _ in range(15):q=time.perf_counter();m(r,i);ts.append((time.perf_counter()-q)*1000)
 return {'name':name,'seed':seed,'fw':fw,'params':sum(p.numel()for p in m.parameters()),'train_s':train_s,'latency_ms':float(np.median(ts)),'metrics':met},m

def invariants():
 m=SCID(8);r=torch.randn(4,8,6,6,requires_grad=True);i=torch.randn(4,8,6,6,requires_grad=True);z,a=m(r,i,True);C,D,A,B=a['C'],a['D'],a['A'],a['B'];bb,c,h,w=C.shape;n=h*w;cg=C.view(bb,2,4,n).transpose(-1,-2);dg=D.view(bb,2,4,n).transpose(-1,-2);orth=float((cg.transpose(-1,-2)@dg/(n-1)).abs().max());rel=float(abs((C.square().sum()+D.square().sum())-(A.square().sum()+B.square().sum()))/(A.square().sum()+B.square().sum()));z.mean().backward();return {'orthogonality':orth,'energy_rel':rel,'finite_grad':bool(torch.isfinite(r.grad).all()and torch.isfinite(i.grad).all())}

def main():
 methods=['concat','sumdiff','learned_ci','onesided_pci','sym_concat','sym_consensus','scid','pbtr','tgf'];runs=[]
 # First test all structural ideas without Fisher.
 for n in methods:
  for s in [0,1]:
   print('run',n,s,flush=True);x,_=run(n,s,0);runs.append(x);print(x['metrics']['clean'],x['metrics']['basis_rotate'],flush=True)
 # Fisher weight ablation on SCID; 0 already above.
 for fw in [.005,.02,.05]:
  for s in [0,1]:
   print('fisher',fw,s,flush=True);x,_=run('scid',s,fw);x['name']=f'scid_fisher_{fw}';runs.append(x);print(x['metrics']['clean'],flush=True)
 out={'invariants':invariants(),'runs':runs};json.dump(out,open(Path(__file__).resolve().parent / 'scid_raw.json','w'),indent=2)
 # aggregate
 names=sorted(set(x['name'] for x in runs));summary={}
 for n in names:
  rr=[x for x in runs if x['name']==n];summary[n]={'params':rr[0]['params'],'latency_ms':float(np.mean([x['latency_ms']for x in rr])),'train_s':float(np.mean([x['train_s']for x in rr])),'metrics':{}}
  for cond in rr[0]['metrics']:summary[n]['metrics'][cond]={k:float(np.mean([x['metrics'][cond][k]for x in rr]))for k in ['acc','miou','map50']}
 json.dump(summary,open(Path(__file__).resolve().parent / 'scid_summary.json','w'),indent=2);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
