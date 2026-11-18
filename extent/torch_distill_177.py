"""Memory-bounded causal CE and teacher-to-student KL on hidden states."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def causal_ce_kl(student_hidden, teacher_hidden, student_head, teacher_head, tokens, *, tile=128):
    if student_hidden.shape[:2] != tokens.shape or teacher_hidden.shape[:2] != tokens.shape:
        raise ValueError('hidden and token dimensions differ')
    if student_hidden.shape[1] < 2 or tile < 1:
        raise ValueError('sequence requires at least two tokens and a positive tile')
    student=student_hidden[:, :-1].reshape(-1,student_hidden.shape[-1])
    teacher=teacher_hidden[:, :-1].detach().reshape(-1,teacher_hidden.shape[-1])
    targets=tokens[:, 1:].reshape(-1)
    total_ce=student.new_zeros((),dtype=torch.float32)
    total_kl=student.new_zeros((),dtype=torch.float32)
    def tile_loss(s,t,head,target):
        student_logits=F.linear(s,head).float()
        with torch.no_grad():
            teacher_logits=F.linear(t,teacher_head.detach()).float()
            log_teacher=F.log_softmax(teacher_logits,dim=-1)
            probs=log_teacher.exp()
        log_student=F.log_softmax(student_logits,dim=-1)
        return F.nll_loss(log_student,target,reduction='sum'), (probs*(log_teacher-log_student)).sum()
    for start in range(0,targets.numel(),tile):
        args=(student[start:start+tile],teacher[start:start+tile],student_head,targets[start:start+tile])
        ce,kl=checkpoint(tile_loss,*args,use_reentrant=False) if torch.is_grad_enabled() else tile_loss(*args)
        total_ce=total_ce+ce;total_kl=total_kl+kl
    return total_ce/targets.numel(),total_kl/targets.numel()


def distillation_microbatch(student, teacher, tokens, *, tile=128):
    """Compute aligned next-token CE/KL with a frozen teacher.

    Both models return normalized last hidden states through `.hidden` and
    `.model` respectively; teacher weights and all activations stay detached.
    """
    with torch.no_grad():
        teacher_hidden=teacher.model(input_ids=tokens, use_cache=False).last_hidden_state
    student_hidden=student.hidden(tokens)
    return causal_ce_kl(student_hidden,teacher_hidden,student.qwen.embed_tokens.weight,
                        teacher.get_output_embeddings().weight,tokens,tile=tile)


def matched_token_budget(available_seconds, seconds_ce, seconds_kd, *, input_tokens_per_update=16384):
    """Conservative equal-update budget for sequential arms; zero means no run."""
    import math
    values=(available_seconds,seconds_ce,seconds_kd)
    if any(not math.isfinite(x) for x in values) or seconds_ce<=0 or seconds_kd<=0 or input_tokens_per_update<=0:
        raise ValueError('invalid budget or throughput')
    updates=max(0,int(available_seconds//(seconds_ce+seconds_kd)))
    return updates,updates*input_tokens_per_update
