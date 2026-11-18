import pytest
from engine_torch177 import choose_budget

def test_conservative_matched_budget_reserves_time():
    updates,tokens=choose_budget(10000,2,3,reserve=1000,safety=.8)
    assert updates==1440 and tokens==1440*16384
    assert choose_budget(20,2,3,reserve=30)==(0,0)

def test_resume_rejects_corruption_wrong_arm_and_skipped_tokens():
    from engine_torch177 import validate_branch_state,INPUTS
    contract={'exp177_arm':'kd','exp177_target_updates':20,'exp177_code':{'a':'b'},'exp177_origin_cursor':100}
    state={'step':2504,'cursor':100+2*INPUTS,'contract':contract}
    descriptor={'step':2504,'cursor':state['cursor'],'sha256':'valid'}
    assert validate_branch_state(state,descriptor,'valid','kd',20,{'a':'b'})==100
    for data,desc,sha,arm,target,code in [(state,descriptor,'bad','kd',20,{'a':'b'}),(state,descriptor,'valid','ce',20,{'a':'b'}),(dict(state,cursor=state['cursor']+1),descriptor,'valid','kd',20,{'a':'b'}),(state,descriptor,'valid','kd',21,{'a':'b'}),(state,descriptor,'valid','kd',20,{'a':'changed'})]:
        with pytest.raises(ValueError):validate_branch_state(data,desc,sha,arm,target,code)

def test_resume_discards_only_noncheckpointed_metrics():
    from engine_torch177 import trim_history
    report={'history':[{'step':2503},{'step':2506}], 'evaluations':[{'updates':1},{'updates':4}]}
    actual=trim_history(report,2504)
    assert actual['history']==[{'step':2503}]
    assert actual['evaluations']==[{'updates':1}]

def test_cpu_worker_saves_and_resumes_adam_without_extra_update(tmp_path,monkeypatch):
    import json,time
    import numpy as np
    import torch
    from torch import nn
    from types import SimpleNamespace
    import transformers
    import engine_torch177 as runner
    import extent.torch_continue_176 as impl
    import extent.weekend_storage_176 as storage
    from extent.broad_corpus import digest_file
    torch.set_num_threads(1)
    class ToyStudent(nn.Module):
        def __init__(self,*args,**kwargs):
            super().__init__();self.qwen=nn.Module();self.qwen.embed_tokens=nn.Embedding(11,4);self.qwen.layers=nn.ModuleList([]);self.qwen.norm=nn.Identity();self.qwen.rotary_emb=lambda h,p:None;self.mixers=nn.ModuleDict();self.retained=set();self.checkpoint_blocks=True
        def hidden(self,tokens):return self.qwen.embed_tokens(tokens)
        def forward(self,tokens):
            from torch.nn import functional as F
            h=self.hidden(tokens);return F.cross_entropy(F.linear(h[:,:-1],self.qwen.embed_tokens.weight).reshape(-1,11),tokens[:,1:].reshape(-1))
    class ToyTeacher(nn.Module):
        def __init__(self):super().__init__();self.embedding=nn.Embedding(11,4)
        def model(self,input_ids,use_cache=False):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        def get_output_embeddings(self):return self.embedding
    monkeypatch.setattr(impl,'FullStudent',ToyStudent);monkeypatch.setattr(impl,'load_config',lambda _:None)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:ToyTeacher())
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self);monkeypatch.setattr(nn.Module,'cuda',lambda self,*a,**k:self)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(100*2**30,100*2**30));monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a,**k:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(storage,'disk_guard',lambda *a,**k:None);monkeypatch.setattr(storage,'check_ram_growth',lambda *a,**k:0);monkeypatch.setattr(storage,'host_available',lambda:100*2**30);monkeypatch.setattr(runner.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(runner,'verify_source_nll',lambda *a,**k:None);monkeypatch.setattr(runner,'code_identity',lambda:{'test':'unchanged'})
    monkeypatch.setattr(runner,'LENGTH',4);monkeypatch.setattr(runner,'EFFECTIVE',2);monkeypatch.setattr(runner,'INPUTS',8)
    root=tmp_path/'root';work=tmp_path/'work';(root/'source').mkdir(parents=True);work.mkdir()
    model=ToyStudent();opt=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=0.)
    source=root/'source'/'CURRENT.pt';torch.save(dict(step=2502,cursor=0,contract={},model=model.state_dict(),optimizer=opt.state_dict()),source)
    monkeypatch.setattr(runner,'SOURCE_SHA',digest_file(source))
    for folder in ('edu','c4'):
        d=tmp_path/folder;d.mkdir();np.zeros(600,dtype='<i4').tofile(d/'validation.bin');np.arange(800,dtype='<i4').__mod__(11).tofile(d/'train.bin')
    args=SimpleNamespace(root=str(root),work_dir=str(work),arm='kd',smoke=False,probe=False,memory_gib=90,batch=1,tile=2,target_updates=1,corpus=str(tmp_path/'edu'),c4=str(tmp_path/'c4'),train_deadline=time.time()+180,hard_deadline=time.time()+600,sync=False,remote_prefix='unused')
    first=runner.worker(args);assert first['status']=='matched_complete' and first['step']==2503 and first['cursor']==8
    checkpoint=torch.load(root/'kd'/'CURRENT.pt',map_location='cpu',weights_only=True);steps=[int(x['step']) for x in checkpoint['optimizer']['state'].values()]
    assert steps and set(steps)=={1}
    second=runner.worker(args);assert second['step']==2503 and second['cursor']==8
    checkpoint2=torch.load(root/'kd'/'CURRENT.pt',map_location='cpu',weights_only=True);assert {int(x['step']) for x in checkpoint2['optimizer']['state'].values()}=={1}
    import engine_torch170
    def fail_upload(*a,**k):raise ConnectionError('simulated outage')
    monkeypatch.setattr(engine_torch170,'sync_remote',fail_upload);monkeypatch.setattr(runner,'notify',lambda _:None)
    args.arm='ce';args.sync=True
    with pytest.raises(ConnectionError):runner.worker(args)
    assert (root/'ce'/'CURRENT.pt').exists()
    failed=json.loads((work/'exp177-ce.json').read_text());assert failed['hf_verified'] is False and failed['hf_error']=='ConnectionError'

def test_exp177_status_shows_kl_and_c4():
    import engine_status
    report={'contract':{'experiment':177,'format':'torch-full-ce-v1'},'status':'running','branches':{'KD':{'step':2630,'source_step':2502,'accepted_input_tokens':2097152,'last_kl':.12,'last_c4_nll':2.85,'validation_step':2630,'last_validation_nll':2.54,'validation_windows':128}}}
    text=engine_status.format_report('test',report,None,active=True)
    assert 'Train KL: 0.12000' in text and 'NLL C4: 2.85000' in text and 'NLL Edu: 2.54000' in text
