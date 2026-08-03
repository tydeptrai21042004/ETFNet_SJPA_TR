import sys,json,random,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scid_fast import *
from test_global_pci import GlobalStats
from test_goci_tr import GOCITR
from test_goci_safe import GOCISafe

METHODS=['concat','pbtr','tgf','global_proc','goci_tr','goci_safe']
def build(n,c,tr):
 if n=='concat':return Concat(c)
 if n=='pbtr':return PBTR(c)
 if n=='tgf':return TGF(c)
 if n=='global_proc':m=GlobalStats(c,kind='concat')
 if n=='goci_tr':m=GOCITR(c,hadamard=False,tau=.6)
 if n=='goci_safe':m=GOCISafe(c,tau=.6)
 if n in ['global_proc','goci_tr','goci_safe']:m.fit(tr);return m

def train_corrupt(r,i,step,seed):
 # Same deterministic corruption schedule for every method.
 choices=['clean','clean','clean','rgb_noise','ir_noise','missing_rgb','missing_ir','mixed']
 cond=choices[(step+seed*7)%len(choices)]
 return corrupt(r,i,cond,seed*10000+step),cond

def run(n,s,epochs=8):
 seed_all(s);ds=Data(1400,seed=1800+s);tr=torch.utils.data.Subset(ds,range(1050));te=torch.utils.data.Subset(ds,range(1050,1400));m=Model(build(n,8,tr),8,5);o=torch.optim.AdamW(m.parameters(),2.5e-3,weight_decay=1e-4);ld=torch.utils.data.DataLoader(tr,64,shuffle=True,generator=torch.Generator().manual_seed(s));step=0
 for ep in range(epochs):
  m.train()
  for r,i,y,b in ld:
   (r,i),_=train_corrupt(r,i,step,s);step+=1;o.zero_grad();l,p,a=m(r,i,True);loss=F.cross_entropy(l,y)+3*F.smooth_l1_loss(p,b);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);o.step()
 m.eval();tl=torch.utils.data.DataLoader(te,100);met={}
 with torch.no_grad():
  for cnd in ['clean','rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed']:
   L=[];P=[];Y=[];G=[]
   for r,i,y,b in tl:r,i=corrupt(r,i,cnd,s+666);l,p=m(r,i);L.append(l);P.append(p);Y.append(y);G.append(b)
   L=torch.cat(L);P=torch.cat(P);Y=torch.cat(Y);G=torch.cat(G);met[cnd]={'acc':float((L.argmax(1)==Y).float().mean()),'miou':float(iou(P,G).mean()),'map50':map50(L,P,Y,G,5)}
 return {'method':n,'seed':s,'metrics':met}

def main():
 rr=[]
 for n in METHODS:
  for s in [0,1,2]:print(n,s,flush=True);rr.append(run(n,s))
 su={}
 for n in METHODS:
  x=[z for z in rr if z['method']==n];su[n]={c:{k:float(np.mean([q['metrics'][c][k]for q in x]))for k in ['acc','miou','map50']}for c in x[0]['metrics']};su[n]['robust_mean']=float(np.mean([su[n][c]['map50']for c in ['rgb_noise','ir_noise','missing_rgb','missing_ir','ir_shift','mixed']]))
 json.dump({'runs':rr,'summary':su},open(Path(__file__).resolve().parent / 'augmented_fusion_results.json','w'),indent=2);print(json.dumps(su,indent=2))
if __name__=='__main__':main()
