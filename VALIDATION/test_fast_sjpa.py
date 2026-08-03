import sys,json,math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scid_fast import *
from test_global_pci import GlobalStats

class FastSJPA(GlobalStats):
 def __init__(self,c,tau=.6,k=12,gamma=1.5,max_shift=1,penalty=.1,score_threshold=.3):super().__init__(c,kind='concat',reliable=False);self.tau=tau;self.k=k;self.gamma=gamma;self.max_shift=max_shift;self.penalty=penalty;self.score_threshold=score_threshold
 def _search(self,A,B,eligible):
  bs,c,h,w=A.shape;n=h*w;Ag=A.reshape(bs,self.g,self.d,n).transpose(-1,-2);Ac=Ag-Ag.mean(-2,keepdim=True);Bg=B.reshape(bs,self.g,self.d,n).transpose(-1,-2);Bc=Bg-Bg.mean(-2,keepdim=True);M0=Ac.transpose(-1,-2)@Bc/(n-1);s0=torch.linalg.svdvals(M0).sum((-1,-2))/(self.g*self.d);eligible=eligible&(s0<self.score_threshold)
  cands=[];scores=[]
  for dy in range(-self.max_shift,self.max_shift+1):
   for dx in range(-self.max_shift,self.max_shift+1):
    Bs=torch.roll(B,(dy,dx),(-2,-1));Bt=Bs.reshape(bs,self.g,self.d,n).transpose(-1,-2);Bt=Bt-Bt.mean(-2,keepdim=True);M=Ac.transpose(-1,-2)@Bt/(n-1);sc=torch.linalg.svdvals(M).sum((-1,-2))-self.penalty*(dy*dy+dx*dx);cands.append(Bs);scores.append(sc)
  S=torch.stack(scores,1);idx=S.argmax(1);idx=torch.where(eligible,idx,torch.full_like(idx,4));out=torch.stack(cands,1)[torch.arange(bs),idx];return out,idx,eligible
 def forward(self,r,i,aux=False):
  A=self.transform(r,self.mr,self.wr);B=self.transform(i,self.mi,self.wi);bs,c,h,w=A.shape;n=h*w;A=(A.reshape(bs,self.g,self.d,n).transpose(-1,-2)@self.q).transpose(-1,-2).reshape(bs,c,h,w);er=A.square().mean((1,2,3),keepdim=True);ei=B.square().mean((1,2,3),keepdim=True);dr=torch.log(er+1e-4).abs();di=torch.log(ei+1e-4).abs();anom=torch.maximum(dr,di);B,idx,elig=self._search(A,B,anom.flatten()<self.tau);p=torch.cat((-self.gamma*dr,-self.gamma*di),1).softmax(1);pr,pi=p[:,:1],p[:,1:];tr=torch.sigmoid(self.k*(anom-self.tau));Aw=(2*pr).sqrt()*A;Bw=(2*pi).sqrt()*B;Fr=(1-tr)*A+tr*Aw;Fi=(1-tr)*B+tr*Bw;z=self.p(torch.cat((Fr,Fi),1));a={'D':Fr-Fi,'shift_idx':idx,'searched':elig};return(z,a)if aux else z

def train_corrupt(r,i,step,seed):
 choices=['clean','clean','clean','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed'];cond=choices[(step+seed*7)%len(choices)];return corrupt(r,i,cond,seed*10000+step),cond

def run(s,thr):
 seed_all(s);ds=Data(1400,seed=1800+s);tr=torch.utils.data.Subset(ds,range(1050));te=torch.utils.data.Subset(ds,range(1050,1400));f=FastSJPA(8,score_threshold=thr);f.fit(tr);m=Model(f,8,5);o=torch.optim.AdamW(m.parameters(),2.5e-3,weight_decay=1e-4);ld=torch.utils.data.DataLoader(tr,64,shuffle=True,generator=torch.Generator().manual_seed(s));step=0
 for ep in range(8):
  m.train()
  for r,i,y,b in ld:(r,i),_=train_corrupt(r,i,step,s);step+=1;o.zero_grad();l,p,a=m(r,i,True);loss=F.cross_entropy(l,y)+3*F.smooth_l1_loss(p,b);loss.backward();o.step()
 m.eval();tl=torch.utils.data.DataLoader(te,100);met={}
 with torch.no_grad():
  for cnd in ['clean','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed']:
   L=[];P=[];Y=[];G=[];S=[]
   for r,i,y,b in tl:r,i=corrupt(r,i,cnd,s+666);l,p,a=m(r,i,True);L.append(l);P.append(p);Y.append(y);G.append(b);S.append(a['searched'])
   L=torch.cat(L);P=torch.cat(P);Y=torch.cat(Y);G=torch.cat(G);met[cnd]={'acc':float((L.argmax(1)==Y).float().mean()),'miou':float(iou(P,G).mean()),'map50':map50(L,P,Y,G,5),'search_rate':float(torch.cat(S).float().mean())}
 return {'seed':s,'thr':thr,'metrics':met}

def main():
 rr=[]
 for thr in [.25,.3,.35]:
  for s in [0,1,2]:print(thr,s,flush=True);rr.append(run(s,thr))
 su={}
 for thr in [.25,.3,.35]:
  x=[z for z in rr if z['thr']==thr];su[str(thr)]={c:{k:float(np.mean([q['metrics'][c][k]for q in x]))for k in ['acc','miou','map50','search_rate']}for c in x[0]['metrics']};su[str(thr)]['robust_mean']=float(np.mean([su[str(thr)][c]['map50']for c in ['rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed']]))
 json.dump({'runs':rr,'summary':su},open(Path(__file__).resolve().parent / 'fast_sjpa_results.json','w'),indent=2);print(json.dumps(su,indent=2))
if __name__=='__main__':main()
