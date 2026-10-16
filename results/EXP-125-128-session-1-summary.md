# EXP125–128: first sessions, 16 October 2026

MEASURED from user-supplied raw JSON; SHA256 and derived metrics in
`EXP-125-128-session-1-analysis.json`. All eight workers stopped by registered
deadline with exit0, no numerical failure or pending fit/report sync. All current
cursors durable; final diagnostics completed. Full32768 horizons NOT completed.
One seed123, previously exposed A/B/C; source pin and actual token hashes match.
No independent seed/domain confirmation.

## Last common fixed endpoint:4096 updates

| Experiment/worker | Arm | A64 | B32 | C64 |
| --- | --- | ---: | ---: | ---: |
|125/0|R32-CONTROL|7.241708|7.287571|6.989294|
|125/1|R32-REL05|7.611802|7.590903|7.371157|
|126/0|R32-REL05-ABS10|7.368749|7.398731|7.191057|
|126/1|R32-REL05-ABS100|7.010170|6.992399|6.698316|
|127/0|R32-REL05-MLP32|7.431200|7.471572|7.165530|
|127/1|R32-REL05-ABS10-MLP32|7.244261|7.294528|7.005280|
|128/0|R32-CHANNEL-HARD-MLP32|15.748048|15.964280|15.682137|
|128/1|R32-CHANNEL-TANH-MLP32|9.748282|9.918175|9.645111|

## Observations and limits

125 relative mixer RMS cap.5 alone reduces huge residuals/absorbed additions but
worsens NLL at4096 versus control.126 adds token residual RMS ceilings after
BOTH additions in all28 blocks. RMS100 gives A7.01017/B6.99240/C6.69832 versus
control7.24171/7.28757/6.98929. Selected by A64, not C; cross-session comparison
has matching source/data/optimizer contracts, not independent confirmation.
RMS100 last7523 regresses to A7.83556/B7.78816/C7.59831. Numeric health remains
finite; no justification to attribute the quality regression to overflow.
RMS10 last7549 degrades to A10.75665. More aggressive bounds are not uniformly better.

127 enables MLP down-projection LoRA32 on top of protected Mamba R32.
With RMS10, A/B/C at4096 improve from1267.36875/7.39873/7.19106 to
7.24426/7.29453/7.00528; last7212 remains near R32 control at7.20828/7.29087/
7.02267. Absolute cap absent: MLP treatment still worse than control, lastA8.96413.
RMS100+MLP remains untested. MLP adaptation can help under a tested scale intervention,
but increasing trainable parameter count alone does not guarantee recovery.

128 hard component cap versus tanh, both with MLP32: at4096 A15.74805/9.74828;
last7435/7362 A13.39529/7.94774. Bounds and ordinary/profile gates pass. Limiter
fractions approach100% for Mamba components on inspected window. Component caps
change directions and are stricter than RMS caps of the same numerical value;
not a calibrated geometry-only experiment and not rejection of all clipping.
Eliminating absorbed additions does not suffice for low NLL.

At126 best4096 RMS100, same-tail127 NLL atcontext128/1024 is6.77097/6.77043;
longer-context utility not recovered on this probe. Context curves of damaged
128 arms are not evidence of useful memory restoration.

Important best/last distinction: best refers to A64 selection ONLY at registered
post-ramp endpoints2048/4096 here; unregistered final step is evaluated separately.
Thus128 last can be better than its stored selected-best, without an optimizer
or checkpoint corruption. Compare4096 for all arms; do not rank unequal last steps.

## Decision

User allocated remaining4h TPU and10h T4;129 chosen by user for T4, runner9.5h
including60min reserve leaves30min setup margin. New TPU experiment not launched.
Priority for TPU design: determine whether the126 RMS100 gain can be retained
from its selected-best4096 state through controlled continuation, LR/optimizer
comparison and/or zero-effect MLP adaptation. Source binary portability/parity
and available checkpoint revision must be verified before an executable protocol.
Do not automatically extend128.129 checks a different internal pre-gate norm.
