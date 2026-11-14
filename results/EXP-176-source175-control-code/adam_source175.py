import os,sys,json,time,math
from pathlib import Path
sys.path.insert(0,os.getcwd());os.environ.update(USE_TORCH='1',USE_TF='0',USE_FLAX='0')
import torch,numpy as np
from extent.torch_continue_176 import FullStudent,load_config
from extent.torch_clean_170 import microbatches
work=Path('../extent176/source175-diagnostic');root=Path(json.loads((work/'ram-workspace.json').read_text())['path'])
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(root/'inductor');os.environ['TRITON_CACHE_DIR']=str(root/'triton')
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
saved=torch.load(root/'CURRENT.pt',map_location='cpu',weights_only=True,mmap=True);assert saved['step']==511
model=FullStudent(load_config(root/'config'),backend='chunked-compiled');model.load_state_dict(saved['model'],assign=True);model.cuda()
source=Path('../engine-core/data-pack-edu-600m')
tr=np.memmap(source/'train.bin',dtype='<i4',mode='r');va=np.memmap(source/'validation.bin',dtype='<i4',mode='r')
tokens=torch.from_numpy(np.array(tr[saved['cursor']:saved['cursor']+16*1024],dtype=np.int64).reshape(16,1024)).cuda()
validation=torch.from_numpy(np.array(va[:128*1024],dtype=np.int64).reshape(128,1024)).cuda()
report=dict(step=511,cursor=saved['cursor'],context=1024,effective_batch=16,physical_batch=8,disposable=True,persisted_updates=0,trials=[])
def write(): (work/'adam-probe.json').write_text(json.dumps(report,indent=2))
def loss(data):
 model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return float(model(data))
def trainloss():return .5*(loss(tokens[:8])+loss(tokens[8:]))
def valid(count=8):return sum(loss(validation[i:i+1]) for i in range(count))/count
initial=valid(128);report['initial_validation128']=initial;print('VALIDATION128',initial,flush=True)
assert abs(initial-2.5398051738739014)<.02

def reset(rate):
 model.zero_grad(set_to_none=True)
 with torch.no_grad():
  for name,param in model.named_parameters():param.copy_(saved['model'][name].to(param.device))
 assert all(p.is_cuda for p in model.parameters())
 for group in saved['optimizer']['param_groups']:group.update(fused=True,foreach=False)
 opt=torch.optim.AdamW(model.parameters(),lr=rate,betas=(.9,.95),eps=1e-8,weight_decay=0.,fused=True)
 opt.load_state_dict(saved['optimizer'])
 for group in opt.param_groups:group['lr']=rate
 return opt

def update(opt):
 model.train();opt.zero_grad(set_to_none=True);ce=0.
 for at in (0,8):
  with torch.autocast('cuda',dtype=torch.bfloat16):l=model(tokens[at:at+8])
  (l*.5).backward();ce+=float(l.detach())*.5
 norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=True));opt.step();torch.cuda.synchronize()
 return ce,norm
for rate in (1e-6,1e-7,1e-8):
 opt=reset(rate);before=trainloss();vbefore=valid();t=time.monotonic();ce,norm=update(opt)
 dot=gg=uu=0.
 for name,param in model.named_parameters():
  if param.grad is None:continue
  delta=param.detach().double()-saved['model'][name].to('cuda').double();g=param.grad.detach().double()
  dot+=float((g*delta).sum());gg+=float(g.square().sum());uu+=float(delta.square().sum())
 row=dict(lr=rate,train_before=before,train_before_repeat=ce,train_after=trainloss(),validation8_before=vbefore,validation8_after=valid(),grad_norm_before_clip=norm,update_norm=math.sqrt(uu),clipped_gradient_dot_update=dot,gradient_update_cosine=dot/math.sqrt(gg*uu),seconds=time.monotonic()-t)
 report['trials'].append(row);write();print('ONE_STEP',json.dumps(row),flush=True)
 del opt
# Matched fixed-batch replay of the existing LR, all updates discarded.
opt=reset(1e-6);report['fixed_batch32']=dict(lr=1e-6,history=[],before=trainloss(),validation8_before=valid());write()
for i in range(32):
 ce,norm=update(opt)
 if (i+1)%4==0:
  row=dict(update=i+1,train_ce=trainloss(),raw_grad_norm=norm);report['fixed_batch32']['history'].append(row);write();print('FIXED_BATCH',json.dumps(row),flush=True)
report['fixed_batch32'].update(after=trainloss(),validation8_after=valid());report['completed']=True;write();print('ADAM PROBE COMPLETE; all updates discarded',flush=True)
