import sys,math,json,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scid_fast import *

class GlobalStats(nn.Module):
    def __init__(self,c,g=2,eps=2e-3,kind='pci',reliable=False):
        super().__init__();assert c%g==0;self.c=c;self.g=g;self.d=c//g;self.eps=eps;self.kind=kind;self.reliable=reliable
        self.register_buffer('mr',torch.zeros(g,1,self.d));self.register_buffer('mi',torch.zeros(g,1,self.d))
        eye=torch.eye(self.d).repeat(g,1,1)
        self.register_buffer('wr',eye.clone());self.register_buffer('wi',eye.clone());self.register_buffer('q',eye.clone())
        cin=c if kind=='consensus' else 2*c
        self.p=Proj(cin,c)
    @torch.no_grad()
    def fit(self,dataset,max_items=None):
        R=[];I=[];n=0
        for r,i,_,_ in torch.utils.data.DataLoader(dataset,128,shuffle=False):
            b,c,h,w=r.shape;R.append(r.reshape(b,self.g,self.d,h*w).permute(1,0,3,2).reshape(self.g,-1,self.d));I.append(i.reshape(b,self.g,self.d,h*w).permute(1,0,3,2).reshape(self.g,-1,self.d));n+=b
            if max_items and n>=max_items:break
        R=torch.cat(R,1);I=torch.cat(I,1);mr=R.mean(1,keepdim=True);mi=I.mean(1,keepdim=True);Rc=R-mr;Ic=I-mi;N=R.shape[1]
        eye=torch.eye(self.d).expand(self.g,-1,-1)
        cr=Rc.transpose(-1,-2)@Rc/(N-1)+self.eps*eye;ci=Ic.transpose(-1,-2)@Ic/(N-1)+self.eps*eye
        er,vr=torch.linalg.eigh(cr);ei,vi=torch.linalg.eigh(ci);wr=vr@torch.diag_embed(er.clamp_min(self.eps).rsqrt())@vr.transpose(-1,-2);wi=vi@torch.diag_embed(ei.clamp_min(self.eps).rsqrt())@vi.transpose(-1,-2)
        Rw=Rc@wr;Iw=Ic@wi;M=Rw.transpose(-1,-2)@Iw/(N-1);u,s,vh=torch.linalg.svd(M);q=u@vh
        self.mr.copy_(mr);self.mi.copy_(mi);self.wr.copy_(wr);self.wi.copy_(wi);self.q.copy_(q);self.svals=s
    def transform(self,x,m,w):
        b,c,h,ww=x.shape;n=h*ww;xg=x.reshape(b,self.g,self.d,n).transpose(-1,-2);z=(xg-m)@w;return z.transpose(-1,-2).reshape(b,c,h,ww)
    def forward(self,r,i,aux=False):
        A=self.transform(r,self.mr,self.wr)
        B=self.transform(i,self.mi,self.wi)
        # Map RGB into IR's global whitened basis.
        b,c,h,w=A.shape;n=h*w;Ag=A.reshape(b,self.g,self.d,n).transpose(-1,-2)@self.q;A=Ag.transpose(-1,-2).reshape(b,c,h,w)
        if self.reliable:
            # Global Gaussian typicality: clean whitened modality energy should be near 1.
            er=A.square().mean((1,2,3),keepdim=True);ei=B.square().mean((1,2,3),keepdim=True)
            # symmetric log-energy deviation handles missing (0) and excessive-noise (>1)
            sr=(torch.log(er+1e-4).abs());si=(torch.log(ei+1e-4).abs())
            logits=torch.cat((-1.5*sr,-1.5*si),1);p=logits.softmax(1);pr,pi=p[:,:1],p[:,1:]
            C=(pr*A+pi*B)/(pr.square()+pi.square()+1e-6).sqrt()
            D=(4*pr*pi).sqrt()*(A-B)/math.sqrt(2)
        else:
            pr=pi=None;C=(A+B)/math.sqrt(2);D=(A-B)/math.sqrt(2)
        if self.kind=='consensus':z=self.p(C)
        elif self.kind=='concat':z=self.p(torch.cat((A,B),1))
        else:z=self.p(torch.cat((C,D),1))
        d={'C':C,'D':D,'A':A,'B':B}
        if pr is not None:d['p']=torch.cat((pr,pi),1)
        return(z,d)if aux else z

class GlobalWhiteConcat(GlobalStats):
    def __init__(self,c):super().__init__(c,kind='concat')
    @torch.no_grad()
    def fit(self,dataset,max_items=None):
        super().fit(dataset,max_items);self.q.copy_(torch.eye(self.d).expand(self.g,-1,-1))


def run_global(name,seed,fw=0,epochs=8):
 seed_all(seed);ds=Data(1600,seed=500+seed);tr=torch.utils.data.Subset(ds,range(1200));te=torch.utils.data.Subset(ds,range(1200,1600));c=8
 if name=='global_white':f=GlobalWhiteConcat(c)
 elif name=='global_proc_concat':f=GlobalStats(c,kind='concat')
 elif name=='global_consensus':f=GlobalStats(c,kind='consensus')
 elif name=='global_pci':f=GlobalStats(c,kind='pci')
 elif name=='global_rpci':f=GlobalStats(c,kind='pci',reliable=True)
 else:raise KeyError(name)
 f.fit(tr);m=Model(f,c,5);o=torch.optim.AdamW(m.parameters(),2.5e-3,weight_decay=1e-4);ld=torch.utils.data.DataLoader(tr,64,shuffle=True,generator=torch.Generator().manual_seed(seed));t=time.time()
 for ep in range(epochs):
  m.train()
  for r,i,y,b in ld:
   o.zero_grad();l,p,a=m(r,i,True);loss=F.cross_entropy(l,y)+3*F.smooth_l1_loss(p,b)
   if fw:loss+=fw*fisher(a['D'],y)
   loss.backward();o.step()
 train_s=time.time()-t;m.eval();conds=['clean','basis_rotate','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed'];met={};loader=torch.utils.data.DataLoader(te,128)
 with torch.no_grad():
  for cond in conds:
   L=[];P=[];Y=[];G=[]
   for r,i,y,b in loader:r,i=corrupt(r,i,cond,seed+44);l,p=m(r,i);L.append(l);P.append(p);Y.append(y);G.append(b)
   L=torch.cat(L);P=torch.cat(P);Y=torch.cat(Y);G=torch.cat(G);met[cond]={'acc':float((L.argmax(1)==Y).float().mean()),'miou':float(iou(P,G).mean()),'map50':map50(L,P,Y,G,5)}
 r,i,_,_=next(iter(torch.utils.data.DataLoader(te,1)));ts=[]
 with torch.no_grad():
  for _ in range(3):m(r,i)
  for _ in range(20):q=time.perf_counter();m(r,i);ts.append((time.perf_counter()-q)*1000)
 return {'name':name if not fw else f'{name}_fisher_{fw}','seed':seed,'params':sum(p.numel()for p in m.parameters()),'latency_ms':float(np.median(ts)),'train_s':train_s,'metrics':met}

def main():
 runs=[];names=['global_white','global_proc_concat','global_consensus','global_pci','global_rpci']
 for n in names:
  for s in [0,1,2]:
   print(n,s,flush=True);x=run_global(n,s);runs.append(x);print(x['metrics']['clean'],x['metrics']['mixed'],flush=True)
 for n in ['global_pci','global_rpci']:
  for fw in [.002,.01,.05]:
   for s in [0,1,2]:
    print(n,fw,s,flush=True);runs.append(run_global(n,s,fw))
 json.dump(runs,open(Path(__file__).resolve().parent / 'global_pci_raw.json','w'),indent=2)
 summary={}
 for n in sorted(set(x['name']for x in runs)):
  rr=[x for x in runs if x['name']==n];summary[n]={'params':rr[0]['params'],'latency_ms':float(np.mean([x['latency_ms']for x in rr])),'metrics':{}}
  for cond in rr[0]['metrics']:summary[n]['metrics'][cond]={k:float(np.mean([x['metrics'][cond][k]for x in rr]))for k in ['acc','miou','map50']}
 json.dump(summary,open(Path(__file__).resolve().parent / 'global_pci_summary.json','w'),indent=2);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
