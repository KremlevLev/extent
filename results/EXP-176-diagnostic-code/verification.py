import sys,torch
from types import SimpleNamespace
sys.path.insert(0,'work/exp176-diagnostic')
from sequential_oracle import sequential_oracle
from extent.torch_clean_170 import chunked_scan

def test():
 torch.manual_seed(12)
 m=SimpleNamespace(g=1,r=2)
 for name,shape in [('b_bias',(2,2,8)),('c_bias',(2,2,8)),('mimo_x',(2,2,8)),('mimo_z',(2,2,8)),('mimo_o',(2,2,8)),('D',(2,)),('readout_norm',(2,8))]:
  setattr(m,name,torch.randn(*shape,requires_grad=True))
 shapes=[(1,9,2,8),(1,9,2,8),(1,9,2,1,8),(1,9,2,1,8),(1,9,2),(1,9,2),(1,9,2),(1,9,2)]
 vals=[torch.randn(*s,requires_grad=True) for s in shapes];vals[4]=torch.full(shapes[4],.02,requires_grad=True);vals[5]=torch.full(shapes[5],-1.,requires_grad=True)
 params=vals+[getattr(m,n) for n in ['b_bias','c_bias','mimo_x','mimo_z','mimo_o','D','readout_norm']]
 a=sequential_oracle(*vals,m,dtype=torch.float64,block=3);b=chunked_scan(*vals,m,block=4,checkpoint=False)
 ga=torch.autograd.grad(a.square().sum(),params,retain_graph=True);gb=torch.autograd.grad(b.square().sum(),params)
 torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
 for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-4,rtol=3e-3)
 print('Independent oracle outputs and all input/parameter gradients: PASS')
test()
