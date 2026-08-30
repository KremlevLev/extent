# EXP-062B full-hybrid backward bring-up

- Duration: `0.16595` hours (`9.96` minutes)
- Context: `8`
- Loss: `23.203165`
- BF16 global gradient norm: `828,493,760`
- Maximum absolute BF16 gradient: `46,466,468`
- Finite gradients: `True`
- Non-finite gradient leaves: `0`
- Backward compile + execute: `96.46` seconds
- BF16 parameters per TPU: `3.4513 GiB`
- BF16 Lion state per TPU: `3.4513 GiB`

The complete 34-Mamba/6-MLA 14.766B graph produced a finite gradient for every
parameter leaf while Lion state remained allocated. The very large pre-clipping
norm quantifies the initialization shock; it is not evidence of recovered
quality and makes production FP32-norm clipping mandatory.
