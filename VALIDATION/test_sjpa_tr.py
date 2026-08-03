import sys,json,math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scid_fast import *
from test_global_pci import GlobalStats

class SJPATR(GlobalStats):
 def __init__(self,c,tau=.6,k=12,gamma=1.5,max_shift=1,penalty=.1):super().__init__(c,kind='concat',reliable=False);self.tau=tau;self.k=k;self.gamma=gamma;self.max_shift=max_shift;self.penalty=penalty
 def _search(self,A,B,eligible):
  bs,c,h,w=A.shape;n=h*w;Ag=A.reshape(bs,self.g,self.d,n).transpose(-1,-2);Ac=Ag-Ag.mean(-2,keepdim=True);cands=[];scores=[]
  for dy in range(-self.max_shift,self.max_shift+1):
   for dx in range(-self.max_shift,self.max_shift+1):
    Bs=torch.roll(B,(dy,dx),(-2,-1));Bg=Bs.reshape(bs,self.g,self.d,n).transpose(-1,-2);Bc=Bg-Bg.mean(-2,keepdim=True);M=Ac.transpose(-1,-2)@Bc/(n-1);sc=torch.linalg.svdvals(M).sum((-1,-2))-self.penalty*(dy*dy+dx*dx);cands.append(Bs);scores.append(sc)
  S=torch.stack(scores,1);idx=S.argmax(1);zero=(2*self.max_shift+1)**2//2;idx=torch.where(eligible,idx,torch.full_like(idx,zero));out=torch.stack(cands,1)[torch.arange(bs),idx];return out,idx
 def forward(self,r,i,aux=False):
  A=self.transform(r,self.mr,self.wr);B=self.transform(i,self.mi,self.wi);bs,c,h,w=A.shape;n=h*w;A=(A.reshape(bs,self.g,self.d,n).transpose(-1,-2)@self.q).transpose(-1,-2).reshape(bs,c,h,w)
  er0=A.square().mean((1,2,3),keepdim=True);ei0=B.square().mean((1,2,3),keepdim=True);dr0=torch.log(er0+1e-4).abs();di0=torch.log(ei0+1e-4).abs();anomaly=torch.maximum(dr0,di0);eligible=(anomaly.flatten()<self.tau)
  B,idx=self._search(A,B,eligible)
  er=A.square().mean((1,2,3),keepdim=True);ei=B.square().mean((1,2,3),keepdim=True);dr=torch.log(er+1e-4).abs();di=torch.log(ei+1e-4).abs();p=torch.cat((-self.gamma*dr,-self.gamma*di),1).softmax(1);pr,pi=p[:,:1],p[:,1:];trigger=torch.sigmoid(self.k*(torch.maximum(dr,di)-self.tau));Aw=(2*pr).sqrt()*A;Bw=(2*pi).sqrt()*B;Fr=(1-trigger)*A+trigger*Aw;Fi=(1-trigger)*B+trigger*Bw;z=self.p(torch.cat((Fr,Fi),1));a={'D':Fr-Fi,'shift_idx':idx,'trigger':trigger};return(z,a)if aux else z

def train_corrupt(r,i,step,seed):
 choices=['clean','clean','clean','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed'];cond=choices[(step+seed*7)%len(choices)];return corrupt(r,i,cond,seed*10000+step),cond

def run(s,pen=.1):
 seed_all(s);ds=Data(1400,seed=1800+s);tr=torch.utils.data.Subset(ds,range(1050));te=torch.utils.data.Subset(ds,range(1050,1400));f=SJPATR(8,penalty=pen);f.fit(tr);m=Model(f,8,5);o=torch.optim.AdamW(m.parameters(),2.5e-3,weight_decay=1e-4);ld=torch.utils.data.DataLoader(tr,64,shuffle=True,generator=torch.Generator().manual_seed(s));step=0
 for ep in range(8):
  m.train()
  for r,i,y,b in ld:
   (r,i),_=train_corrupt(r,i,step,s);step+=1;o.zero_grad();l,p,a=m(r,i,True);loss=F.cross_entropy(l,y)+3*F.smooth_l1_loss(p,b);loss.backward();o.step()
 m.eval();tl=torch.utils.data.DataLoader(te,100);met={}
 with torch.no_grad():
  for cnd in ['clean','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed']:
   L=[];P=[];Y=[];G=[];IDX=[]
   for r,i,y,b in tl:r,i=corrupt(r,i,cnd,s+666);l,p,a=m(r,i,True);L.append(l);P.append(p);Y.append(y);G.append(b);IDX.append(a['shift_idx'])
   L=torch.cat(L);P=torch.cat(P);Y=torch.cat(Y);G=torch.cat(G);idx=torch.cat(IDX);met[cnd]={'acc':float((L.argmax(1)==Y).float().mean()),'miou':float(iou(P,G).mean()),'map50':map50(L,P,Y,G,5),'zero_shift_rate':float((idx==4).float().mean())}
 return {'seed':s,'penalty':pen,'metrics':met}

def main():
 rr=[]
 for pen in [.03,.1,.3]:
  for s in [0,1,2]:print(pen,s,flush=True);rr.append(run(s,pen))
 su={}
 for pen in [.03,.1,.3]:
  x=[z for z in rr if z['penalty']==pen];su[str(pen)]={c:{k:float(np.mean([q['metrics'][c][k]for q in x]))for k in ['acc','miou','map50','zero_shift_rate']}for c in x[0]['metrics']};su[str(pen)]['robust_mean']=float(np.mean([su[str(pen)][c]['map50']for c in ['rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed']]))
 json.dump({'runs':rr,'summary':su},open(Path(__file__).resolve().parent / 'sjpa_tr_results.json','w'),indent=2);print(json.dumps(su,indent=2))
if __name__=='__main__':main()
