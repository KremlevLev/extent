import argparse,os,sys,json,time,math,types,traceback
from pathlib import Path
sys.path.insert(0,os.getcwd());sys.path.insert(0,'/tmp/extent176-reference')
os.environ.update(USE_TORCH='1',USE_TF='0',USE_FLAX='0')
import torch,numpy as np
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from extent.torch_continue_176 import FullStudent,TrainingMamba,load_config
from sequential_oracle import sequential_oracle

p=argparse.ArgumentParser();p.add_argument('--length',type=int,default=128);p.add_argument('--double-only',action='store_true');p.add_argument('--cursor',type=int,default=235061248);a=p.parse_args()
work=Path('../extent176/source175-diagnostic').resolve();root=Path(json.loads((work/'ram-workspace.json').read_text())['path'])
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor');os.environ['TRITON_CACHE_DIR']=str(root/'triton')
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
saved=torch.load(root/'CURRENT.pt',map_location='cpu',mmap=True,weights_only=True)
model=FullStudent(load_config(root/'config'),backend='chunked');model.load_state_dict(saved['model'],strict=True,assign=True);model.cuda();step=saved['step'];cursor=a.cursor;del saved
corpus=Path('../engine-core/data-pack-edu-600m')
arr=np.memmap(corpus/'train.bin',dtype='<i4',mode='r');tokens=torch.from_numpy(np.array(arr[cursor:cursor+a.length],dtype=np.int64)[None]).cuda()
original=TrainingMamba.forward

def forward(self,u,**kw):
 if not self.backend.startswith('oracle'):return original(self,u,**kw)
 values=self.components(u)
 y=sequential_oracle(*values,self,dtype=torch.float64 if self.backend.endswith('64') else torch.float32)
 return F.linear(y.flatten(-2),self.out_weight)
TrainingMamba.forward=forward

bf16_hidden=FullStudent.hidden

def fp32_hidden(self,tokens):
 hidden=self.qwen.embed_tokens(tokens)
 positions=torch.arange(tokens.shape[1],device=tokens.device)[None]
 rope=self.qwen.rotary_emb(hidden,positions)
 for index,layer in enumerate(self.qwen.layers):
  mixer=None if index in self.retained else self.mixers[str(index)]
  args=(hidden,layer,mixer,positions,rope)
  hidden=checkpoint(self.block,*args,use_reentrant=False) if self.checkpoint_blocks and torch.is_grad_enabled() else self.block(*args)
 return self.qwen.norm(hidden)

results=dict(checkpoint_step=step,cursor=cursor,length=a.length,batch=1,optimizer_updates=0,tf32=False,rows=[])
base={}
trials=[('fp32','oracle64'),('fp32','chunked'),('bf16','chunked-compiled')]
if a.double_only:
 import extent.torch_continue_176 as tc
 def precise_rms(x,w,eps):return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w).to(x.dtype)
 tc.rms=precise_rms
 tc.ce_tile=lambda features,head,targets:F.cross_entropy(F.linear(features,head),targets,reduction='sum')
 def components64(self,u):
  z,x,b,c,dd,raw,trap,angles=F.linear(u,self.in_weight).split(self.widths,dim=-1)
  batch,length=u.shape[:2];shape=(batch,length,self.h,self.n)
  b=precise_rms(b.reshape(batch,length,self.r,self.g,self.n),self.b_norm,self.eps)
  c=precise_rms(c.reshape(batch,length,self.r,self.g,self.n),self.c_norm,self.eps)
  dt=F.softplus(dd+self.dt_bias);decay=-(raw.clamp_min(0)+1/(1-raw.clamp_max(0)))
  return x.reshape(shape),z.reshape(shape),b,c,dt,decay.clamp_max(-1e-4),trap,angles
 TrainingMamba.components=components64
 model.double();trials=[('fp64','oracle64')]

for precision,backend in trials:
 torch.manual_seed(176);torch.cuda.manual_seed_all(176)
 model.hidden=types.MethodType(fp32_hidden if precision!='bf16' else bf16_hidden,model)
 for m in model.mixers.values():m.backend=backend
 model.zero_grad(set_to_none=True);model.train();t=time.monotonic();print('START',precision,backend,'length',a.length,flush=True)
 try:
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):loss=model(tokens)
  loss.backward();torch.cuda.synchronize();squares={};dot=aa=bb=delta=0.
  for name,v in model.named_parameters():
   if v.grad is None:continue
   g=v.grad.detach();group='.'.join(name.split('.')[:2] if name.startswith('mixers.') else name.split('.')[:3])
   squares[group]=squares.get(group,0.)+float(g.double().square().sum())
   if not results['rows']:base[name]=g.cpu()
   else:
    ref=base[name].cuda().double();cur=g.double();dot+=float((ref*cur).sum());aa+=float(ref.square().sum());bb+=float(cur.square().sum());delta+=float((ref-cur).square().sum())
  row=dict(precision=precision,backend=backend,ce=float(loss.detach()),seconds=time.monotonic()-t,grad_norm=math.sqrt(sum(squares.values())),top_groups=sorted([(k,math.sqrt(v)) for k,v in squares.items()],key=lambda x:-x[1])[:8])
  if results['rows']:row.update(cosine_vs_fp32_oracle64=dot/math.sqrt(aa*bb),relative_l2_vs_fp32_oracle64=math.sqrt(delta/aa))
 except Exception as exc:
  row=dict(precision=precision,backend=backend,error=type(exc).__name__,traceback=traceback.format_exc());print(row,flush=True);results['rows'].append(row);(work/f"reference-{a.length}{'-double' if a.double_only else ''}.json").write_text(json.dumps(results,indent=2));raise
 results['rows'].append(row);(work/f'reference-{a.length}{'-double' if a.double_only else ''}.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 if a.double_only:
  selected=max(((float(v.grad.detach().abs().max()),name,v) for name,v in model.named_parameters() if v.grad is not None),key=lambda x:x[0])
  _,name,v=selected;index=int(v.grad.detach().abs().argmax());flat=v.detach().reshape(-1);old=flat[index].clone();analytic=float(v.grad.reshape(-1)[index]);checks=[]
  model.eval()
  try:
   with torch.no_grad():
    for eps in (1e-7,1e-8,1e-9,1e-10):
     flat[index]=old+eps;plus=float(model(tokens));flat[index]=old-eps;minus=float(model(tokens));flat[index]=old
     check=dict(parameter=name,index=index,epsilon=eps,analytic=analytic,finite_difference=(plus-minus)/(2*eps),plus=plus,minus=minus);checks.append(check);print('FINITE_DIFFERENCE',json.dumps(check),flush=True)
  finally:flat[index].copy_(old)
  results['finite_difference']=checks
  (work/f'reference-{a.length}-double.json').write_text(json.dumps(results,indent=2))

print('REFERENCE DIAGNOSTIC COMPLETE',flush=True)
