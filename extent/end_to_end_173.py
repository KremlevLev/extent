"""EXP173 stage2: composed 7-attention hybrid, frozen original Qwen, CE + KL."""
import jax
import jax.numpy as jnp
from extent.clean_tpu_171 import teacher_block, student_block, rms

RETAINED = (3, 7, 11, 15, 19, 23, 27)


def hidden(q, teacher, last, embedding, finalnorm, tokens, c, tc, *, original=False):
    h = embedding[tokens].astype(jnp.bfloat16)
    mask = jnp.asarray([original or i in RETAINED for i in range(27)])
    def body(h, inputs):
        index, p, m = inputs
        h = jax.lax.cond(mask[index], lambda _: teacher_block(h,p,tc),
                         lambda _: student_block(h,p,m,c), None)
        return h, None
    h, _ = jax.lax.scan(jax.checkpoint(body), h, (jnp.arange(27),teacher,q))
    return rms(teacher_block(h,last,tc),finalnorm,c.rms_norm_eps)


def head_loss(student, teacher, embedding, labels, *, tile=32):
    """Exact full-vocabulary forward KL, T=1; bounded head logits, no top-k."""
    s=student.reshape(-1,student.shape[-1]);t=jax.lax.stop_gradient(teacher.reshape(s.shape))
    y=labels.reshape(-1);n=y.size;pad=(-n)%tile
    s=jnp.pad(s,((0,pad),(0,0))).reshape(-1,tile,s.shape[-1])
    t=jnp.pad(t,((0,pad),(0,0))).reshape(s.shape)
    y=jnp.pad(y,(0,pad)).reshape(-1,tile)
    weights=(jnp.arange(n+pad)<n).reshape(-1,tile)
    def body(total, inputs):
        a,b,label,w=inputs
        logp=jax.nn.log_softmax((a@embedding.T).astype(jnp.float32),-1)
        logt=jax.lax.stop_gradient(jax.nn.log_softmax((b@embedding.T).astype(jnp.float32),-1))
        ce=-jnp.take_along_axis(logp,label[:,None],-1)[:,0]
        kl=jnp.sum(jnp.exp(logt)*(logt-logp),-1)
        return total+jnp.asarray([jnp.sum(ce*w),jnp.sum(kl*w)]),None
    totals,_=jax.lax.scan(jax.checkpoint(body),jnp.zeros(2,jnp.float32),(s,t,y,weights))
    metrics=totals/n
    return .5*jnp.sum(metrics),metrics


def make_grad(c,tc):
    def objective(q,teacher,embedding,last,finalnorm,tokens):
        th=jax.lax.stop_gradient(hidden(q,teacher,last,embedding,finalnorm,tokens,c,tc,original=True))
        sh=hidden(q,teacher,last,embedding,finalnorm,tokens,c,tc)
        return head_loss(sh[:,:-1],th[:,:-1],embedding,tokens[:,1:])
    def gradient(q,teacher,embedding,last,finalnorm,tokens):
        (value,metrics),g=jax.value_and_grad(objective,has_aux=True)(q,teacher,embedding,last,finalnorm,tokens)
        return value,g,metrics
    return gradient


def learning_rate(step,total):
    warm=128
    progress=jnp.clip((step-warm)/max(1,total-warm-1),0.,1.)
    return 1e-4*jnp.minimum((step+1)/warm,1.)*(.1+.45*(1+jnp.cos(jnp.pi*progress)))
