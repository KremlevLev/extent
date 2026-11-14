"""Independent sequential MIMO oracle for bounded EXP176 gradient diagnostics."""
import math
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sequential_oracle(x,z,b,c,dt,decay,trap,angles,m,*,dtype=torch.float64,block=32):
    batch,length,h,n=x.shape
    def normalize(v,w):
        return v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)*w
    def body(x,z,b,c,dt,decay,trap,angles,state,pk,pv,phase,bb,cb,mx,mz,mo,skip,rw):
        with torch.autocast(device_type=x.device.type,enabled=False):
            x,z,b,c,dt,decay,trap,angles=[v.to(dtype) for v in (x,z,b,c,dt,decay,trap,angles)]
            bb,cb,mx,mz,mo,skip,rw=[v.to(dtype) for v in (bb,cb,mx,mz,mo,skip,rw)]
            out=[]
            def rotate(v,p):
                co=F.pad(p.cos(),(0,n//4),value=1)[:,None]
                si=F.pad(p.sin(),(0,n//4))[:,None]
                a,q=v.chunk(2,-1)
                return torch.cat((a*co-q*si,q*co+a*si),-1)
            for t in range(x.shape[1]):
                alpha=(dt[:,t]*decay[:,t]).exp()
                f=trap[:,t].sigmoid()
                gamma=f*dt[:,t]; beta=(1-f)*dt[:,t]*alpha
                phase=phase+math.pi*angles[:,t,None].tanh()*dt[:,t,:,None]
                k=rotate(b[:,t].repeat_interleave(h//m.g,2)+bb.transpose(0,1)[None],phase)
                q=rotate(c[:,t].repeat_interleave(h//m.g,2)+cb.transpose(0,1)[None],phase)
                now=x[:,t,None]*mx.transpose(0,1)[None]
                prev=pv[:,None]*mx.transpose(0,1)[None]
                state=alpha[...,None,None]*state+torch.einsum('brhp,brhn->bhpn',gamma[:,None,:,None]*now,k)+torch.einsum('brhp,brhn->bhpn',beta[:,None,:,None]*prev,pk)
                y=torch.einsum('bhpn,brhn->brhp',state,q)+skip[None,None,:,None]*now
                y=normalize(y,rw[None,None])
                out.append((y*F.silu(z[:,t,None]*mz.transpose(0,1)[None])*mo.transpose(0,1)[None]).sum(1))
                pk,pv=k,x[:,t]
            return torch.stack(out,1),state,pk,pv,phase
    state=torch.zeros(batch,h,n,n,device=x.device,dtype=dtype)
    pk=torch.zeros(batch,m.r,h,n,device=x.device,dtype=dtype)
    pv=torch.zeros(batch,h,n,device=x.device,dtype=dtype)
    phase=torch.zeros(batch,h,n//4,device=x.device,dtype=dtype)
    outputs=[]
    for start in range(0,length,block):
        pieces=[v[:,start:start+block] for v in (x,z,b,c,dt,decay,trap,angles)]
        args=(*pieces,state,pk,pv,phase,m.b_bias,m.c_bias,m.mimo_x,m.mimo_z,m.mimo_o,m.D,m.readout_norm)
        y,state,pk,pv,phase=checkpoint(body,*args,use_reentrant=False) if torch.is_grad_enabled() else body(*args)
        outputs.append(y)
    return torch.cat(outputs,1).to(x.dtype)
