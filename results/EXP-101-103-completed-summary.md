# EXP-101/103 completed; EXP-102 reported only

Reviewed full supplied JSONs, not rerun. Both completed, all branches8192,
no pending HF uploads, no nonfinite forward/gradient steps, no norm-only events.

101 Wiki CE /Qwen0.1 /Qwen0.5:7.177410 /7.197104 /7.288943 (123),
7.183887 /7.170213 /7.298713 (456). Main gate FALSE. Qwen0.1 advantage
-0.019694 /+0.013674, both below0.1. Qwen0.5 worsens Wiki versus recovered
start in both seeds, but improves PG19 against CE by0.371667 /0.174428.
This is a domain tradeoff, not universal anchoring failure.

103 Wiki protected /released:7.177410 /7.178490 and7.183887 /7.194813.
Released loses both, and PG19 worsens by0.022311 /0.313255. Main gate FALSE.
Shared protected controls match101 CE exactly; NOT independent replications.

102: user reports gate FALSE; no files received. Completion, branch metrics,
health and reason for failure UNVERIFIED. Do not infer that every LR arm failed.

Next hypothesis: the active parameter subspace, rather than rank or input-mask
alone, may limit full-model adaptation. Test paper-inspired full decoder CE
finetuning against matched adapter continuation, keeping embeddings/readout fixed.
This is not an exact reproduction of HedgeMamba or the paper's first stage.
