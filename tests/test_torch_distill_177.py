import torch
from torch.nn import functional as F
from extent.torch_distill_177 import causal_ce_kl

def test_causal_tiled_values_and_student_gradients_match_dense():
    torch.manual_seed(177)
    s=torch.randn(2,5,7,requires_grad=True); t=torch.randn(2,5,7,requires_grad=True)
    h=torch.randn(13,7,requires_grad=True); teacher_head=torch.randn(13,7,requires_grad=True)
    ids=torch.randint(0,13,(2,5))
    ce,kl=causal_ce_kl(s,t,h,teacher_head,ids,tile=3)
    sl=F.linear(s[:,:-1],h);tl=F.linear(t[:,:-1],teacher_head).detach()
    lp=F.log_softmax(tl,-1);refce=F.cross_entropy(sl.reshape(-1,13),ids[:,1:].reshape(-1));refkl=(lp.exp()*(lp-F.log_softmax(sl,-1))).sum(-1).mean()
    torch.testing.assert_close(ce,refce);torch.testing.assert_close(kl,refkl)
    actual=torch.autograd.grad(ce+kl,(s,h),retain_graph=True);expected=torch.autograd.grad(refce+refkl,(s,h))
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y)
    assert t.grad is None and teacher_head.grad is None

def test_budget_matches_updates_and_rejects_invalid_throughput():
    from extent.torch_distill_177 import matched_token_budget
    assert matched_token_budget(101,2,3,input_tokens_per_update=16)==(20,320)
    assert matched_token_budget(1,2,3)==(0,0)
    import pytest
    with pytest.raises(ValueError):matched_token_budget(10,float('nan'),3)

def test_teacher_receives_no_gradients_in_microbatch():
    from types import SimpleNamespace
    from torch import nn
    from extent.torch_distill_177 import distillation_microbatch
    class Student(nn.Module):
        def __init__(self):
            super().__init__();self.qwen=nn.Module();self.qwen.embed_tokens=nn.Embedding(11,4)
        def hidden(self,x):return self.qwen.embed_tokens(x)
    class Teacher(nn.Module):
        def __init__(self):super().__init__();self.embedding=nn.Embedding(11,4)
        def model(self,input_ids,use_cache=False):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        def get_output_embeddings(self):return self.embedding
    s=Student();t=Teacher();tokens=torch.tensor([[1,2,3,4]])
    ce,kl=distillation_microbatch(s,t,tokens,tile=2);(ce+kl).backward()
    assert s.qwen.embed_tokens.weight.grad is not None
    assert all(p.grad is None for p in t.parameters())
