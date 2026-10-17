"""Short GPU GEMM-only timing probe; never changes the training precision."""
import gc,time
import numpy as np
import jax
import jax.numpy as jnp


def probe(device,*,shapes=((256,2048,6144),(256,6144,2048)),repeats=5):
    rows=[]
    with jax.default_device(device):
        for m,k,n in shapes:
            rng=np.random.default_rng(146);x=rng.normal(0,.1,(m,k)).astype(np.float32);w=rng.normal(0,.02,(k,n)).astype(np.float32)
            timings={}
            for label,dtype in (('bf16',jnp.bfloat16),('fp16',jnp.float16),('fp32',jnp.float32)):
                a=jax.device_put(jnp.asarray(x,dtype),device);b=jax.device_put(jnp.asarray(w,dtype),device)
                fn=jax.jit(lambda a,b:jax.lax.dot_general(a,b,(((1,),(0,)),((),())),preferred_element_type=jnp.float32))
                begin=time.perf_counter();out=fn(a,b);jax.block_until_ready(out);compile_seconds=time.perf_counter()-begin
                samples=[]
                for _ in range(repeats):
                    begin=time.perf_counter();out=fn(a,b);jax.block_until_ready(out);samples.append(time.perf_counter()-begin)
                timings[label]=dict(median_seconds=float(np.median(samples)),compile_and_first_seconds=compile_seconds,finite=bool(jnp.all(jnp.isfinite(out))))
                del a,b,out,fn
            rows.append(dict(shape=[m,k,n],timings=timings,bf16_over_fp16=timings['bf16']['median_seconds']/timings['fp16']['median_seconds']))
    gc.collect()
    return dict(device_kind=device.device_kind,platform=device.platform,scope='GEMM forward ONLY; normalized synthetic inputs; no model/gradient parity or full training speed claim; training remainsBF16',rows=rows)
