# 18 October2026 — EXP137 dt-only weak; MIMO-X-only transient gain

User supplies full137 aggregate. Exact bytes archived results/EXP-137-streamed-v2-session-1.json,
SHA/extract results/EXP-137-streamed-v2-analysis.json. Bothworkers exit0,
deadline_partial after10.6052h, durable12328/12002, pending[], no upload/error.
RESET-DT accepted3,155,968input positions, bestA-selected4608 A/B/C
7.203941/7.263066/6.971151; last12328 7.250804/7.275950/6.996569.
RESET-MIMO-X accepted3,072,512positions, best8192 6.827096/6.795481/6.454612;
last12002 6.997996/7.010704/6.726889, worse than its selected best.
At matched10240 with136: ALL6.156050/6.159662/5.784677;
PROJECTIONS6.197205/6.189168/5.861837; DT7.242575/7.293844/7.053530;
MIMO-X7.074269/7.121730/7.020657. Same ordering at8192. In this one seed,
projection reset is the strongest tested standalone intervention; DT alone leaves
A~7.2; MIMO-X alone helps transiently but is weaker and nonmonotonic. Fullreset's
modest additional advantage over projection-only does not isolate dt vsMIMO-X
interaction; do not claim a universal faulty parameter or plateau cause.
Only RESET-DT final-last layer26 profile fails engineering control(.0316456>.02);
ordinary NLL remains usable, that profile cannot establish activation mechanism.
All selected-best profiles pass; MIMO-X final profiles pass. Same-tail final
context256->1024 benefit DT~.00059, MIMO-X~.01413 over8windows; no broad long-context
recovery claim. Pre2048 A16 monitors must not be compared to A64 as identical panels.
Neither arm completed its131072 horizon; DT continuation lower priority than136
full/projection reset.137 is resumable under its unchanged streamed-v2 contract.
No138 result supplied this turn;135 replay root remains unknown, no code fixes claimed.

# 18 October2026 — EXP136 improves to NLL~6; EXP135 rejected gradient replay

User supplies full136 aggregate; read135 report directly at HF snapshot
 e5a25058a543ff5e5811a8e676b924a60a32d029. Archived exact bytes and SHA receipts:
results/EXP-135-136-streamed-v2-analysis.json. No training code changed this turn.
135 DID run (~22min notebook /~21min worker). Statusarm_failed,0 new accepted
updates, remains3434. First proposal all stages finite, CE7.085037 vslegacy7.086225,
error.001187<.03. Gradient replay relativeL2.530583>.15, cosine.999603>.98;
source replay passed. Reject-before-update worked. Root of gradient magnitude
mismatch UNKNOWN; direction close does not establish correct optimizer updates.
Do NOT loosen gate or claim135 continued. Parent treats arm_failed as exit0, so
Kaggle can finish without traceback even when no model progress was accepted.

136 bothworkers exit0/deadline_partial after10.6026h, fully saved, pending[],
no report/storage errors. RESET-ALL12586steps=3,222,016accepted input positions,
last A/B/C6.000789/6.004432/5.617432; A-selected registered best12288:
6.040121/6.020951/5.619627. RESET-PROJECTIONS12051steps=3,085,056inputpositions,
last6.112766/6.179138/5.728322; selectedbest10240:6.197205/6.189168/5.861837.
Compare matched10240: ALL6.156050/6.159662/5.784677 vs PROJ6.197205/6.189168/5.861837.
Both improve substantially beyond source~7.264/7.376/7.191; the all-reset advantage
at matched10240 is modest. One exposed-panel seed, no generic superiority claim.
2.819/2.929sec/update recent medians. Neither completed131072 horizon nor
established convergence; same streamed-v2 cell can resume each exact worker state.
All final136 layer profile checks pass. Same-tail context256->1024 gain only
~.00027 ALL /~.00550 PROJ at8windows: lower short-context NLL does not show useful
long-context recovery. No new results from137/138 supplied/retrieved this turn.
Next for135: diagnose same-input single/dual-device gradient scales by layer,
including whether norm mismatch comes from partitioning or BF16 execution;
no cause established yet and no rerun can be advertised as fixed.

# 17 October2026 — T4 first-step memory guard failures; streamed-v2 PREPARED

User reports all new T4 jobs failed. Read-only HF snapshot
 e227cc0c0860798a143e340c53ce8450430791b0 confirms135/worker0 and136/workers0,1
failed inside GuardedJit at the FIRST gradient executable: MemoryError
'EXP114 executable exceeds reported HBM'. Legacy helper name, not EXP114 launch.
0 accepted new updates;135 remains3434,136 remain0. Original135 source replay
passed all A/B/C: NLL6.74723/6.77329/6.48021; max window error.003858<.03.
This is an execution/memory failure, not a model-quality result. Old guard did not
persist memory-size numbers; exact byte-level culprit/allocator OOM not established.
137/138 available snapshot only has preflight reports; exact root errors unconfirmed.
Raw3 failures and receipt results/EXP-135-138-startup-failure-audit.json archived.

New isolated extent/reset_streamed_gpu.py + scripts/m3q_reset_t4_streamed_campaign.py.
GPU135-138 now use layer-by-layer VJP with HOST boundary activations, same full
chain-rule CE/KL/batch256/reset/optimizer. Resident base remains sharded;135 BOTH GPUs
model-shard per-layer executables;136-138 independent single-GPU workers as before.
No giant full28-layer backward executable; per-layer HBM guard kept with byte details.
135 source current3434+exactAdam and legacy gradient replay before accepting update
remain required. Source, data, recipe, selection, eval and checkpoint cadence unchanged.
New separate experiments/expID-reset-streamed-v2 namespace, no overwrite/reinterpretation
of failed v1 states. Restart fallen jobs with results/EXP-135-138-streamed-v2-launch.md;
individual135-138 launch files updated. Old135-142 code/contracts verified unchanged,
so running TPU139-142 need no changes. New parent prints failed child exception/log tail
and embeds it in ONE aggregate extent-m3q-reset-streamed-v2-ID.json.
13 local tests passed85.70s incl7 recipe full-gradient parity cases,2-device135 execution,
BF16 legacy-gradient replay, restored source Adam/cursor, rejection-before-update,
actual parent traceback surfacing and numeric memory-guard rejection. No physical
T4 streamed-v2 full model HBM/throughput verified yet. First restart135 and inspect
8 accepted-step throughput before spending all4 sessions again. No new cloud job launched.

# 17 October2026 — EXP142 PREPARED; one additional TPU9h session

User authorizes full-decoder CE from pinned EXP130 CURRENT3434 as a comparison to
EXP139 dense-Mamba. New isolated extent/full_decoder_reset.py and
scripts/m3q_full_decoder_142_campaign.py; existing135-141 contracts verified unchanged.
Launch results/EXP-142-launch.md; full protocol results/EXP-142-protocol.md;
frozen plan results/EXP-142-engineering-plan.json. No accelerator run launched here.
Single seed123; source130 folded, fresh Adam/cursor0, normalization+tanh retained.
ALL1,675,340,928 params trainable including one tied embedding/readout; no LoRA.
Shared139 rates: Mamba3e-4 incl internal norm scales, pregate1e-4. Newly unlocked
MLP/GQA/embedding3e-5; other original norm scales1e-5. Same batch1/context256,
data permutation, stableclip1, Adam.9/.999,256warm flat. Not a112 resume.
8.5h includes60min reserve, ceiling131072, no step/time guarantee. Local512,
remote4h+exit,80GB upload cap. Abstract full q/Adam/best payload26.805GB;
two remote generations about53.61GB plus small metadata, no frozen-base uploads.
Six parameter-group gradient/master/BF16-change diagnostics; source A/B/C replay.03,
first-proposal finite gate, ordinary NLL/profiles/context probes, exact checkpoint
resume. Eight tiny CPU tests passed51.55s, four runner/rate checks repeated7.50s
on final code; virtual8 devices, BF16 actual gradients, payload restore/next-step
identity with restored layouts. Physical1.7B TPU HBM/speed/results still pending.

# 17 October2026 — EXP135–141 PREPARED; EXP133 failed before training

User allocates4 T4x2 sessions12h and3 TPUv5e8 sessions9h. User explicitly requests
both T4 GPUs on ONE EXP130 winning trajectory. No new accelerator run launched here.
New scripts/m3q_reset_followup_campaign.py + extent/reset_followup.py; protocol
results/EXP-135-141-protocol.md; all7 standalone cells results/EXP-135-141-launches.md.
135 model-shards one trajectory across BOTH GPUs in one fresh process, resident
weights/full remat/tiled vocabulary; batch1 remains256, no data-parallel batch change.
Source130 CURRENT3434 inclAdam/schedule/data, first goalABSOLUTE32768 (29,334 new updates), then continues within wall budget to ceiling131072.
Pin results/EXP-135-source-pin.json at HF b85a8fbfb2832ad7cc4f7442defaafb454417e6a;
all487 chunks114,979,740B checked; real optimizer template restored; current hash matches,
best3072; all6 Adam/schedule counters3434. Receipt EXP-135-source-audit.json.
Physical replay EACH A/B/C window<=.03 is required before training. Throughput/ETA after8
accepted new steps; NO promise32k fits one12h session; checkpoint/resume supported.
Before accepting first135 proposal each invocation, compare legacy streamed130 actual-batch
gradient vsfullcompiled: CEerr<=.03,relativeL2<=.15,cos>=.98; failed probe rejects proposal.
No bitwise/long-trajectory equivalence claim. Probe tokens/time charged separately.
136 fullreset vsprojection-only;137 dt_bias-only vsmimo_x-only, same111 start/ramp2048.
138 CE vsfullforwardKL1 fromFOLDED130 current3434; freshAdam BOTH, noMLP expansion.
139 R32 vsallMamba dense fromsameFOLDED130, freshAdam BOTH; MLP/GQA/embedding frozen.
140 R32 and141dense cross clean balancedQKVO vsEXP072ONPOLICY BEFORE111 training;
zero-effect new corrections, no inherited111 weights/Adam. Fullreset+norm+tanh active
AT INITIALIZATION in BOTH140/141 starts; no ramp, so dense dynamics remain trainable
instead of repeatedly overwritten by surgery. This is a common new normalized setting,
not an exact historical097 replay or the competitor4B mapping. BF16 storage both starts.
TPU runs IN NOTEBOOK kernel; GPU135 one2GPUchild,136–138 two1GPUchildren. GPU11.5h/TPU8.5h
INCLUDES60min reserve plus30min notebook setup margin. Local128R32/512dense; remote120min
+exit, also4096steps for135. GPU1356GB,136–1384GB TOTAL/session; TPU13924GB/1404GB/14140GB.
Frozen base reconstructed, never newly uploaded. All3 paired comparisons keepbatch1ctx256
Adam.9/.999,LR3e-4Mamba/1e-4norm,warm256;135 preserves original schedule3434.
Source/registeredA/B/C/last/best,profiles,matched-tailcontext128..1024; A-only best.
First proposal health persisted BEFORE acceptance, stages:objective,oldq/Adam,grads,updates,newq/Adam.
All new outputs use experiments/expID-reset and extent-m3q-reset-ID.json, old130prefix read-only.
EXP133 raw results/EXP-133-failed-session-1.json:0 accepted updates,3 rejected first proposals,
0.5483h; source7.01233/6.99394/6.70001 reproduced. NOT a method-quality result;133 not
allocated a session. Detailed133 diagnostic mode prepared locally, not run; no root cause
is proven. Old134 outcome still not supplied. Local engineering receipt:
results/EXP-135-141-local-validation.md. No physical GPU/TPU/throughput result from local tests.

# Extent / Mamba-3 in the Qwen: передача проекта в новый чат

**16 October2026 — EXP130–134 PREPARED, five user-allocated sessions.**
User explicitly requests3 T4x2 sessions7h each, TPU4h and9h, strongly based on
arianraje recipe;129 already assigned10h T4. No new cloud training launched.
Separate extent/competitor_recovery.py + competitor_tpu.py; GPU runner
scripts/m3q_t4_competitor_campaign.py and in-kernel sharded TPU runner
scripts/m3q_peer_tpu_campaign.py. Full protocol results/EXP-130-134-protocol.md,
five standalone cells results/EXP-130-launch.md through EXP-134-launch.md;
frozen new plans EXP-130-134-engineering-plan.json. All10 old125–129 worker
contracts unchanged (plus124 source untouched). One seed123, reused panels.
130: norm+tanh control versus same+zero dynamics/loguniform dt/simplex reset.
131: norm+tanh+unprotected dynamics in_projR32 versus same+MLP-downLoRA32.
132: paired CE versus pure exact full-vocab forwardKL T1, same131/GPU1 capacity.
All start111 MAIN32768, freshAdam .9/.999, R32 LR3e-4/norm+MLP1e-4,
2048 identity ramp, no residual caps; GPU horizon32768. Clock6.6h INCLUDES45min
reserve,24min setup margin; cap4GB TOTAL per session, split between2 workers.
133: TPU from126 selected4096 (C6.69832), fixed relative.5/residualRMS100;
3 arms R32 LR3e-4 /3e-5 /3e-5+MLP32 LR1e-5, freshAdam/warmup256 ALL arms.
Pinned HF revision68b6879f73207f23b440f3dfc68df153a09a4a0b and manifest in
EXP-133-source-pin.json. Locally downloaded/checksummed96 best leaves,7,145,472
finite coordinates. Best weights only, not Adam at4096. Replay EACH A/B/C
window <=.03 on TPU before updates. Training permutation continues4096.
Clock3.6h incl45min reserve,4GB cap. Original111 reconstruction still verified.
134: TPU all Mamba params dense (264,102,016 total trainable inclnorm+MLP32),
paired CE versus full-vocab KL T1, norm+tanh ramp2048, start111. Context1024
(same pool packed4096 blocks),1023 targets/update. LR2e-4 Mamba/2e-5 norm+MLP,
Adam .9/.95,492warmup, cosine floor.1 at16384. GQA/embedding/MLP base frozen,
MLP-downLoRA32 ALL28 layers. NOT full-decoder replication. Clock8.5h incl60min
reserve,24GB cap. Dense payload4.23GB/arm,8.45GB/paired save,~17GB two uploads.
TPU horizon16384, alternate128 blocks, best/current/Adam save before diagnostics.
Nested multi_transform Adam mu/nu sharded like parameters; empty coordinate
nodes preserved (legacy helper would replicate nested moments). Local128,
remote120min133/240min134+exit. GPU remote2048-or120min. No frozen model copies.
More frequent post-ramp A64/B32/C64/profile endpoints2048/2560/3072/3584/4096/
4608/5120/6144 etc; A-only selection, last/best contexts128..1024. Source0 eligible
best for133. Missing interrupted endpoint retried at restored cursor. No extra
seeds in screening. Source inspection: pre-gate norm,tanh,dynamics zero/dt,
simplex,KL and mixer/rest LRs/betas. T1/cosine curve explicitly OUR choices where
trainer source absent. Historical architecture/QKV mapping and long preparation
not reproduced; reset130 is on learned weights, not fresh faithful inheritance.
Local engineering checks and physical limitations: EXP-130-134-local-validation.md.


**16 October2026 — EXP125–128 FIRST RESULTS archived;129 user launching T4.**
Raw results/EXP-125/126/127/128-session-1.json, SHA/derived comparison in
EXP-125-128-session-1-analysis.json; readable EXP-125-128-session-1-summary.md.
All8 workers deadline_partial, exit0, durable current steps, no pending fits or
report-sync errors, finalization completed; source/hook checks pass, same token
hashes/source pin. Common4096: R32 control A/B/C7.24171/7.28757/6.98929;
126 relative.5+residualRMS100 gives7.01017/6.99240/6.69832 (selected by A64),
but last7523 worsens7.83556/7.78816/7.59831.126 RMS10 last7549 A10.75665.
127 RMS10+MLP32 helps versus126 RMS10 at4096;7.24426/7.29453/7.00528,
last7212 remains7.20828/7.29087/7.02267, roughly control quality. MLP32 without
absolute residual cap remains worse (last7282 A8.96413). RMS100+MLP UNTESTED.
128 component hard/tanh both damage quality:4096 A15.74805/9.74828,
last7435/7362 A13.39529/7.94774; finite/bound/profile gates pass. Component caps
are stricter and change direction; do not reject all clipping or infer unique
cause. Removed absorbed additions alone does not suffice.126 best128→1024
context same-tail127 NLL6.77097→6.77043: no recovered context gain on probe.
"Best" selects ONLY fixed post-ramp endpoints; final arbitrary last is separate,
so128 last can outperform stored best. Do not mistake that for checkpoint fault.
User has4h TPU and10h T4;129 launching T4 with max-wall9.5h INCLUDING60min
reserve,30min setup allowance. TPU design priority: retain126 selected4096 gain,
diagnose continuation degradation via controlled LR/optimizer/MLP comparison.
No new TPU runner or actual TPU run yet; verify binary source/parity first.

**16 October2026 — EXP129 PREPARED: competitor-informed pre-gate RMSNorm.**
User authorized adapting/testing the inspected competitor methods. Read-only
arianraje HF source pinned b946069890837bbe9b5dd75e2d118a7e98d2d193,
Apache-2.0 metadata; SHA/source attribution in results/EXP-129-pregate-protocol.md.
Separate extent/pregate_recovery.py and scripts/m3q_t4_pregate_129_campaign.py;
legacy124–128 core/runners/contracts untouched. One T4x2: R32 control versus
R32 plus shared H,P learned RMSNorm weights, applied separately per rank/head
AFTER state readout+D skip, BEFORE SiLU gate and rank reduction. Eps1e-5,
geometric identity ramp2048; standard upstream normalization, not new invention.
Source111 seed123 MAIN32768; same data/order/freshAdam, R32 LR3e-4/norm1e-4,
stableclip1/warmup256; no MLP adaptation, output gains or residual caps.
Horizon32768,11.5h including60min reserve, local128/HF2048-or120min/exit,
4GB TOTAL upload cap/invocation. Source A/B/C/context and full-strength FIT8
shock/profile; A16 before ramp, A64 selects only full-strength endpoints;
last/best A/B/C/context and residual/readout/gate/absorption profiles.
Treatment adds36,864 norm coordinates; not parameter-count-matched ablation.
This is continuation from a model trained WITHOUT the norm, not a reproduction
of competitor conversion/full-model long-context KD; a failure cannot reject
that full recipe. Other differences (tanh angle, dynamics init, rank factors,
dimensions/training) recorded for separate audits, no causal conclusion.
Physical1.7B GPU outcome pending. Existing T4 notebook last cell only.

**16 октября2026 — EXP128 PREPARED: fourth T4x2 session, per-component limits.**
Separate `extent/channel_bound_recovery.py` and
`scripts/m3q_t4_channel_bound_128_campaign.py`;125–127 modules/contracts UNCHANGED
and compared against frozen engineering plan. GPU0 hard clipping each component,
GPU1 smooth tanh; dynamic Mamba component bound.5*max(token RMS(input),.001),
absolute residual component bound10 after both additions/all28 layers. Both
have protectedR32 + MLP-down-proj LoRA32, same initialization/source as127/GPU1.
Geometric ramp2048, true gradients, post-ramp A64 best, same endpoints/data,
source A/B/C controls and profiles, local128/HF2048-or120min,4GB total cap,
11.5h includes60min reserve, horizon32768. Distinct question: whole-vector RMS
scaling shrinks small components due to large neighbors; component caps preserve
uncapped hard components. Tanh also changes small components gently. Component
caps are STRICTER than RMS caps at the same numerical threshold; not a clean
geometry-only ablation with equal removed signal. Profile activation fractions
count COMPONENTS, not tokens; peak bound explicitly checked within2% rounding.
Physical1.7B GPU outcome pending. Protocol/cell results/EXP-128-t4-launch.md.

**16 октября2026 — EXP125–127 PREPARED, three T4x2 sessions;124 user launching.**
New independent `extent/dynamic_recovery.py`, `scripts/m3q_t4_dynamic_campaign.py`.
125: R32 control vs per-token relative Mamba output RMS cap rho.5.
126: same cap + absolute residual RMS10 vs100 after BOTH additions in ALL28
layers.127: relative cap with MLP-down-proj LoRA32, vs same+absolute10;
compare to125/GPU1 and126/GPU0 respectively. All same111 seed123 MAIN32768,
freshAdam, MambaR32 LR3e-4, MLP1e-4, jointstableclip1, LRwarmup256.
True gradients through input/output RMS; causal current-token channels only,
stable logRMS numerical floor~1e-12, input reference floor.001. Geometric
identity→full ramp2048; best eligible ONLY after full ramp, selected byA64.
Before-ramp A16 monitor only. Source A64/B32/C64/context/profile explicitly
recorded; full-bound initial FIT8 diagnosis; bounds/inspect engineering gate.
32768 horizon/arm,11.5h includes60min reserve. Local128, HF2048/120min/exit,
4GB total upload cap/session. Resume current+Adam+best, strength fromcursor,
future metric rollback; best params evaluation-only.30min progress notices.
Same old T4 notebook LAST cell; protocols/cells results/EXP-125-127-t4-launch.md.
Physical1.7B GPU feasibility/quality pending.124 result not yet received.

**16 октября2026 — EXP120–123 results received and archived.**
Raw JSON results/EXP-120/121/122/123-session-1.json; derived matched4096 and
source controls in EXP-120-123-session-1-analysis.json.120 completed461/461,
8.6826h; best layer16*.03 A64 7.26410→7.26094, no plateau breakthrough.
121 deadline_partial durable8098/8020, gains only; no material improvement.
122 durable7775/7871, R32 vsR32+gains: same4096 B32 7.28712/7.28742.
Same1024 best A64 7.20104/7.20226; C64 7.03723/7.03083; last C64
7.01876/7.02754 at unequal steps.123 durable5697/5681, KL.1/.5;
matched4096 B32 7.29271/7.44105 versus122 CE+gains7.28742. KL.5 beststep0:
source C64 7.19110 gives a neighboring-run baseline for122 C improvement~.16;
not independent confirmation or exact same-executable source control for122.
All source/hook gates pass, no pending fit states, finals completed despite
training deadline_partial. Do not call untested full65k horizon a failure.
Profiles on one A window show near-total absorbed late MLP additions;
free gains/KL can increase residual scale. No unique causal diagnosis.
Do not automatically continue121–123/107–109.112 remains partial12288/13850,
111/113 completed32768;110 latest Download completed65536/65536.

**15 октября2026 — EXP124 PREPARED; EXP115 results archived.**
EXP124 (`scripts/m3q_context_recovery_124_campaign.py`): one TPU v5e8,8.8h
including60min reserve. Paired SHORT4x256 vs LONG1x1024, identical inputs and
1020 CE targets/update (LONG boundary losses255/511/767 omitted). Source111
seed123 MAIN32768; protectedrank32 +24 learned Mamba output gains, fresh Adam
both arms, stableclip1, LR3e-4/.01 and256warmup. Horizon32768/arm; alternate256
blocks, same4096-block permutation. Last common pair evaluated on identical
last127 targets at contexts128/256/512/1024, source/Qwen controls, layer scale
and absorbed-addition profiles. Exposed115 panels, not independent tests.
Snapshots local128, HF4096-pair/90min/exit,2GB upload cap; same cell resumes.
One aggregate `extent-m3q-context-124.json`; launch/protocol in
`results/EXP-124-kaggle-launch.md`. Actual1.7B TPU run pending.
EXP115 completed7.10h, source raw/summary/analysis archived under
`results/EXP-115-session-1*`. At16384 updates R32 Wiki7.28643/PG199.48182;
G256 7.28229/9.39644, delta-.00414/-.08538. ADD256 worse+.11258/+.12478.
Other arms only4096, do not infer long-run failure. No plateau breakthrough;
large activation scale remains a hypothesis, not established unique cause.

**15 октября2026 — EXP120–123 PREPARED: long T4 scale/CE/KL campaigns.**
User просит4 существенно более длинных сессии. НЕ считать причину плато
установленной: activation scale — hypothesis; protectedrank32/stableclip
помогает recovery, но не доказан optimal rank или activation-normalization method.
Source111 seed123 MAIN32768/данные114–115 сохранены. Новый независимый runner
`scripts/m3q_t4_scale_campaign.py`, `extent/t4_scale_recovery.py`, notebook
`notebooks/extent-t4-long.ipynb`. Старые116–119 code/contracts не менялись.
120:461 gain conditions ×A64, top3/worker B32/C64/context, coverage/partial.
121:24 learned output gains CE vs CE+.01 log-RMS excess penalty.
122:trained rank32 CE control vs rank32+gains; freshAdam у обоих.
123:rank32+gains end-to-end exact-vocab T2 KL mixtures .1/.5, compare122 CE.
121–123 planned65536 updates/GPU, не local fits; actual full decoder chain rule
через streamed VJP, host inputs/checkpointing, head32 tiles. 256warmup, LR gains.01/
rank32 3e-4, jointstableclip1. A16 selection, B32 monitor fixed endpoints;
last/best A64/B32/C64/context127/profile in45min reserve. No independent test.
11.5h includes load/save;65k horizon may require multiple sessions. Same cell
resumes local/remote state. Snapshots128 local, remote4096 or90min +deadline.
Cap2GB/invocation total; storage_partial explicit; step/durable_step distinct.
Current+Adam+best payload, exact cursor rollback, future endpoint invalidation,
state-SHA keyed final evaluations. Completed-horizon but incomplete validation
gets finalization_partial and resumes. SourceNLL<=.03, hookbackground<=.01;
inspection relative output discrepancy<=.02 separately labelled, old116 trace
acceptance never relaxed. CPU engineering tests cover all6 recipe gradients
versus full autodiff, CE/KL/penalty, optimizer restore, nonfinite and clean
deadline/resume. **Actual1.7B GPU training HBM/throughput/outcome pending.**
Launch/protocol `results/EXP-120-123-t4-launch.md`, frozen engineering contracts
`results/EXP-120-123-engineering-plan.json`.

**15 октября2026 — EXP116–119 RESULTS received and archived.**
Raw JSON `results/EXP-116/117/118/119-session-1.json`, source SHA/status manifest
`EXP-116-119-session-1-analysis.json`. All completed/no HF pending.116 .4105h,
117 .2866h,118 .2826h,119 .7300h. GPU sourceNLL7.2641 matches TPU7.26461.
116 all28 layer profiles; residual RMS jumps~941 after0,6.7e5 after3,6e8 after16,
1.53e11 after26; allfinite.24/24 detailed trace parity rejected: no accepted
state dynamics.117 same tail127: hybrid7.24384→7.24391 when128→1024 context,
Qwen3.43932→2.51573, reset effects<.001, carrycontrols exact.118 FP32 effects
<.0005, FP16 nonfinite; storedweights/readoutBF16, inference-only.119 all18 fits
1024 durable. Layer0 tiny NLL effects;12 local MSE~8.8e-5 on one trainwindow
but no fullNLLgain;26 evenFREE256 local MSE.00147 yet fullNLL~16.1 vs7.2.
This motivates causal scale intervention/end-to-end objective, not proof of
activation explosion as unique cause, nor rejection of end-to-end rank growth.

**15 октября2026 — T4 setup fix, no experiment updates in failed run.**
User cloud log: nvidia-smi exit12 in setup before pip/runner. Notebook previously
cleared LD_LIBRARY_PATH before querying NVML; this can hide Kaggle driver libs.
User confirms T4 allocation and requests no preliminary hardware checks.
Removed nvidia-smi query/driver-version/vendor/count checks, preserved library
paths in notebook and both worker subprocesses. Mathematical source/parity
controls remain. Reimport updated notebook; git pull alone does not rewrite
old setup cell. EXP116–119 launch cells unchanged, GPU outcome still pending.

**15 октября2026 — EXP116–119 T4 PREPARED, GPU outcomes pending.**
Отдельный `notebooks/extent-t4.ipynb`, CUDA12 pinned JAX0.11/flax0.12.8/optax0.2.8.
User deploy-key clone cell сохранён, setup T4x2/Internet/secrets, subprocess
engineering preflight. `scripts/m3q_t4_campaign.py`: независимые GPU workers,
host BF16 patient, one layer on GPU, source111 seed123 MAIN32768 pinned114,
parameter-only restore без Adam111; исходный base SHA и все chunks checked.
Все64 A source replay <=0.03 per-window против TPU; before probes обязательный
gate. Tiny streamed/monolithic <=0.01 token NLL (BF16 fusion boundaries, не
bitwise parity). 116 all-layer error/dynamics profile;117 same-target context
128–1024/reset32–128 с traced-carry control;118 precision compute interventions;
119 fixed layers0/12/26 teacher-mixer local fitting на hybrid-prefix inputs,
ADD32/64/128/256(new extra correction)/FREE256/ALL-MAMBA,1024updates/128windows,
после каждого candidate full-model NLL A/B/C. Reused validation и FIT overlap115
явно маркированы; НЕ independent test/seed confirmation. 11.5h from runner,
45min reserve; cap2GB total/invocation, fit local128 snapshots, remote branch-end/
deadline, pending/durable cursors отдельно. Telegram started/end/failure.
Протокол и4 ячейки: results/EXP-116-119-t4-launch.md. CPU engineering tests
проверяют forward, SHA restore, zero-effect/init protection, finite update и
exact next-step restore; actual1.7B GPU memory/throughput/runtime untested.

**14 октября2026 — EXP115 PREPARED: user-authorized aggressive single-seed search.**
Новый аккаунт9h. Пользователь явно просит сначала искать на одном seed, затем
подтвердить лучший метод другими seeds. Seed123 из pinned111 MAIN32768;
7 arms: R32/G64/G128/G256/ADD256(frozen old32)/FAST256(newLR1e-3)/LADDER256.
Старые coordinates/Adam/count32768 сохранены; новые cohorts с собственными
moment/count и128-step warmup, zero-effect start, joint stable clip1. Все4096,
top2 по mean validation3072/4096 + R32 продолжаются до16384. Это exploratory
screening, возможен late bloomer; не выдавать одинseed за confirmation.
Новые данные train74973184/16384, selection reused122880/64; lockedWiki
test262144/64, PG19524288/128 после durable choices. Real token SHA/capacity и
8 virtual8 CPU tests + cursor-reconciliation regression passed; actual1.7B TPU
memory/throughput/results pending.
Runner m3q_rank_growth_115_campaign, ячейка EXP-115-kaggle-launch.md.
8.8h включает60min reserve, upload cap6GB на вызов; local512 snapshots,
remote active states в конце. Kill доsync теряет local steps, paid replay
selection-reference parity checked. Старые111–114 contracts/runners не изменены.

**14 октября2026 — EXP114 first session failed in dynamics, earlier modules saved.**
Получен полный JSON:8338 tasks,1.72734h, completed profile/sweep/FIT/temperature/
readout/optimizer; оба readout4096/endpoints durable, pending[]. Original111 replay
exact per-window. Trace parity2.0 в первой dynamics task; никаких accepted traces
или locked C. Итоговый patient hash check не достигнут. Анализ/исходный SHA:
results/EXP-114-session-1-analysis.json и EXP-114-session-1-summary.md.
Selected mixer/rollback B effects<0.007 NLL, readout B ухудшается0.034/0.090.
Boundary26 bypasses весь upstream prefix; НЕ диагноз виновного26 слоя.
Trace operation order repaired без ослабления1e-4;19 virtual8 CPU tests passed,
actual repaired TPU parity pending. Для сохранения completed probes и exact
readout states — обновлённый main и явный --repair-trace, узкие source/code/data/
status/absence-of-C guards, migration history. Wall budget уточнить по остаточной
квоте, не брать прежние6.1h автоматически. Следом dynamics/precision/locked.

**13 октября2026 — короткий112: network preflight failure, без новых updates.**
Пользовательский Kaggle лог: HF429 `maximum queue size reached` на pinned PG19
`data/test_files.txt`; fail в load_tokens до model restore/training. SIGTERM
в teardown не первичная причина. Сохранённые MAIN12288/8642 этим run не заменены.
Transport-only fix: entry112 заполняет HF cache точной bundled копией100-file
manifest (1475bytes), SHA c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c.
Legacy scientific files/contract hashes не изменены; mismatched cache/bundle
отвергается.4 regression tests passed, original PG19 loader offline воспроизводит
registered tokens SHA8c0fda1eafbd9c8b064391bce5c42a429103fbba929a0d369edbf0f06d95f1e2.
Это устраняет запрос только для этого manifest, не всех сетевых assets/checkpoints.
По последним полученным JSON113 completed32768/32768 durable,112 partial12288/8642.

**12 октября2026 — EXP111 completed /113 partial, DESIGN114:** полученные полные
пользовательские JSON111/113 проверены офлайн.111 MAIN32768/32768 durable,
fresh Wiki7.345073/7.309848, PG199.934455/9.978602; substantial Qwen gap.
113 MAIN26662/24576 durable, штатный training deadline; LR1e-4, сильные spikes.
На24576 MAIN обеихseeds111 лучше113, но pilots/подготовка различаются:
total pipeline/изолированная initializer superiority не подтверждены.
Пользователь сообщил, что112 уже запущен: не предлагать повтор112 из-за
отсутствия результата.113 дозавершить старой ячейкой;110 сейчас не продлевать.
Пользователь разрешил подробное обследование плато EXP114 на одном аккаунте.
Протокол results/EXP-114-plateau-forensics-protocol.md: frozen111 final32768,
все-layer rollback/teacher-mixer/boundary rescues, held-out verification,
MIMO dynamics/context, Adam shadow proposals и output-only readout alignment.
СТАТУС PREPARED locally: runner `scripts/m3q_plateau_forensics_114_campaign.py`,
ячейка `results/EXP-114-kaggle-launch.md`. Patient revision закреплена:
f839c2a898c3e9eff9907ea3cf1e022fe16f3e09; оба32768 manifests и580 chunk paths
проверены read-only, binary SHA проверяется при restore. Реальные data capacity,
SHA/provenance в EXP-114-data-preflight.json; B/C Wiki token-disjoint, но имеют
одну общую статью, article independence не утверждается.
18 инженерных тестов passed на8 virtual CPU devices: functional hooks, recurrence,
неизменность пациента, exact next update после cloud restore, A-only selection,
incomplete aggregation,429/cap/pending и tiny полный цикл. CSV/plots export smoke
и --plan-only passed. Это НЕ1.7B TPU HBM/throughput или научные результаты.
BF16 hook background зарегистрирован до TPU: same-executable control, предел
0.01 NLL; reset сравнивается с теми же selected layers при normal carry period0.
Original32768 canonical replay требует1e-4. Practical signal0.1 не изменён.
Readout accepted params/Adam приводятся к стабильным sharding layouts; snapshots
256-step, endpoints1024/4096; C только после durable choices. Cap1GB на вызов,
8.8h включает60min save reserve; deadline_partial продолжается той же ячейкой.
Старые111–113 contracts сохраняются. Новые большие HF copies не нужны.

**11 октября2026 — результаты108–110 и DESIGN111–113:** получены все3 JSON,
все deadline_partial/durable, pending[]. Общий paired endpoint24576/seed.
Batch4 не дал устойчивого преимущества batch1; AGC исправляет часть ухудшений
109 у456, но110 не превосходит108 устойчиво. Оригиналы архивированы
results/EXP-108/109/110-session-1.json; contract-checked comparisons в
results/EXP-108-110-partial-comparisons.json. Final gatesNULL: кампании не закончены.
Пользователь просит1–3 эксперимента с диагностикой и честный контроль скорости12→7.
Выбраны3 аккаунта:111 protectedrank32 из исходного072;112 ordinary full-decoder
CE из того же072;113 ordinary CE со случайными Mamba и сохранёнными Qwen остальными
весами. Во всех одинаковые диагностические замеры по слоям/позициям/BF16 updates.
112/113 получают LR pilots, затраты поиска учитываются. Скорость marginal recovery
и total pipeline (включая069/072) строго отдельно. Полный протокол:
results/EXP-111-113-recovery-efficiency-protocol.md. Теперь PREPARED locally:
новые entry points m3q_efficiency_campaign / m3q_efficiency_112_campaign /
m3q_efficiency_113_campaign; ячейки results/EXP-111-113-launch.md. Реальные token
capacity/SHA и96 source manifests проверены без model weight download. Test
offset301056 не поместился: ДО TPU зарегистрирован отдельный locked validation
offset147456/64. Source pinned9c031f6e615d7afb0133f5611ace78f8cc5cc5fc.
HF read-only audit: current tree1.87155TB, API usedStorage4.08611TB; account quota
не доступна без auth, эти значения не утверждение physical billed storage.
Промежуточные states локально, одна remote generation/active slot в конце вызова;
caps2GB для111,35GB для112/113 НА ВЫЗОВ. Завершённые LR pilots metrics-only,
выбранный replay оплачивается и проверяется validation parity. При внезапном
уничтожении VM до sync локальные шаги теряются; metrics откатываются к durable
cursor, затраты сохраняются. Старые HF файлы/история не удалялись.
Старый108 runner НЕ подходит. Ничего нового на TPU не запущено; full-model TPU
memory/throughput ещё не измерены.

**10 октября2026 — EXP108/109/110 PREPARED:** три аккаунта, по9h.
Получен полный106: обе ветви32768, durable; gain Wiki0.014/0.065,
PG190.012/0.087. Значимого выхода из плато нет. Имеющийся присланный107
partial27799/24576; на matched24576 эффект AGC смешанный, <0.011.
Логи: results/EXP-106-completed.json и EXP-107-session-1.json.
Новые кампании из pinned098:108 batch1/global,109 accumulation4/global,
110 accumulation4/layer-group. Одинаковые данные/seed/token-clock LR,
65536 окон/seed; число Adam updates различается. Новые train/PG19 срезы.
Протокол и ячейки: results/EXP-108-110-batch-recovery-protocol.md,
results/EXP-108-110-kaggle-launch.md. Default8.8h включает90min save reserve,
12min внешний setup запас. Endpoint может требовать resume; deadline_partial
штатен. Старые106/107 контракты не изменены; новых TPU результатов нет.

**Git author correction,9 октября2026:** пользователь требует GitHub attribution
именно KremlevLev. Правильные author И committer: `KremlevLev
<kremlevlev888@gmail.com>` — как у привязанных GitHub коммитов627c9d9/8b2eef3.
Имя без правильного email недостаточно. Repo-local user.name/user.email и
user.useConfigOnly закреплены; перед commit проверять оба git var IDENT.
Недавние коммиты с lkremlev@phoenixit.ru не привязаны к профилю. Опубликованная
история не переписана; для её исправления требуется отдельное решение пользователя.

**Короткие continuation-сессии106/107:** --max-wall-hours теперь принимает <1.5h,
--save-reserve-minutes задаёт резерв >=10min и меньше wall budget. Default reserve
min(90min,25% wall). Точные измеренные v1 контракты сохранены; новый операционный
таймер НЕ сбрасывает параметры/Adam/data/LR. AST/math и dependency SHA guards.
Примеры оставшейся квоты1h/2h:0.9h+15min reserve /1.9h+20min reserve;
setup время нужно дополнительно вычесть. Большие HF restore/compile/save входят
в budget, поэтому1h может не хватить для завершения. Те же notebook imports.

**8 октября2026, следующий шаг:** пользователь подтвердил уничтожение облачных
runtime104/105: незагруженные финальные состояния потеряны, метрики сохранены.
Разрешены EXP106/107 на двух аккаунтах: одинаковое длинное full-decoder CE,
глобальный clip1 против layer-group relative clip0.01*max(weightnorm,0.001).
Оба начинаются из pinned098, НЕ из потерянных104/105; дваseed/аккаунт,
32768 шагов/seed, warmup512 и новое длинное расписание. Единственное различие
между аккаунтами — клиппинг. Общий gate по свежему PG19 требует обаJSON;
индивидуальный gateNULL ожидаем. Wiki test reused, exploratory. См.
`results/EXP-106-107-layer-clipping-protocol.md`, готовые ячейки в
`results/EXP-106-107-kaggle-launch.md`; новые runner/core не меняют104/105.
TARGET6–6.5h loading/compile/training+1.5h save, endpoint может требовать resume.
Добавлены per-layer gradient/clipping/post-Adam update diagnostics, бинарный HF
roundtrip перед model allocation и initial full checkpoint до обучения с
remote manifest/sample verification. Новые успехи sync проверяют manifest/sample,
финальный cooldown/retry уважает429. Это НЕ полный remote reload16GB каждого
checkpoint и не TPU evidence. Подготовка/CPU проверки отдельно от результатов.

**8 октября2026:** EXP104/105 действительно completed, все4 ветви2048/конечные,
оба gatesFALSE. Runner-hours1.071/0.970; остановились по фиксированному endpoint,
а не8h wall cap. Пользователь недоволен коротким использованием сессий после
очереди4–24h: будущие длительные протоколы рассчитывать по измеренной скорости,
с достаточно большим заранее закреплённым горизонтом/данными/LR на6–6.5h работы.
Не продлевать завершённые2048 ветви молча и не считать unused cap фактическими8h.
ИсходныеJSON `EXP-104-completed.json`, `EXP-105-completed.json`: всефинальные
checkpoint slots PENDING; cloud durability НЕ подтверждена. Не удалять живой
runtime/RAM до sync. Ранее Telegram в runner отсутствовал; теперь подключён
start/finished/failed/sync notifications с сохранением статуса отправки. Добавлено
логирование типа/HTTP-кода ошибок checkpoint sync; старые логи не содержат причины.
Для105 тоже закреплён точный v1 контракт, чтобы sync после фикса не сбрасывал state.

**Обновление 7 октября2026:** получен полный EXP102: completed, все6 ветвей8192,
gateFALSE; меньший LR лучше Wiki, больший лучше PG19, устойчивого выигрыша>=0.1
нет. Полные данные/разбор: `EXP-102-completed.json`, `EXP-102-and-104-update.md`.
EXP104 остановился на стартовой NLL-parity проверке: контроль123 сохранён512,
DECODER123=0. Это инженерная остановка, gatesNULL, не отрицательный результат.
Исправление104-v2: канонические BF16 materialization/evaluation, прежний1e-4;
точный исходный контракт/новые моменты сохранены, старые старты архивированы и
пересчитаны из исходного098, не step512. RAM-checkpoint default для Kaggle.
CPU/virtual8 проверки НЕ подтверждают исправление на TPU. Пользователь разрешил
ещё EXP105 для второго аккаунта: все Mamba веса открыты, остальные Qwen слои
заморожены, против protectedrank32. `EXP-105-mamba-capacity-protocol.md`;
runner `scripts/m3q_mamba_capacity_recovery_campaign.py`, два8h запуска в
`results/EXP-104-105-kaggle-launch.md`. Результатов105 нет. Ниже записи исторические.

**Обновление 6 октября 2026:** получены полные результаты EXP-101 и103,
оба completed, основные gates FALSE; разбор и JSON в results. Слабая поздняя
дистилляция не даёт устойчивого Wiki выигрыша; сильная даёт PG19/Wiki tradeoff.
Снятие защиты103 не помогает. По102 есть только сообщение пользователя gate=false;
полных файлов нет, его метрики и причину не считать установленными.
Пользователь запросил ОДИН следующий эксперимент с опорой на Attention to Mamba.
EXP-104 — подготовленный запуск полного CE-дообучения decoder, embeddings/head
заморожены, против adapter-CE; `results/EXP-104-paper-decoder-recovery-protocol.md`.
Runner104: `scripts/m3q_paper_decoder_recovery_campaign.py`; ячейка и ограничения
— `results/EXP-104-kaggle-launch.md`. Проверены CPU-шаги, точный resume и mesh8
на виртуальных устройствах; реальные диапазоны/хеши текстов зафиксированы.
Внутри decoder1,364,139,136 обучаемых параметров; состояние с Adam ~16.37GB.
Реальные TPU HBM/скорость, большие исходные HF payloads и upload-timing этим
локальным тестом НЕ подтверждены: запуск проверяет их сам. Результатов104 нет;
завершение четырёх ветвей за8h не гарантировано.
Это адаптация stage2 статьи, не точная реплика HedgeMamba/10B-token recipe.

**Обновление 5 октября 2026:** EXP-098–100 завершены; все основные gates FALSE.
Защищённый rank32 остаётся рабочим кандидатом: rank64 не пробил плато, слабый
Qwen anchor с начала восстановления не дал устойчивого выигрыша. Полные JSON и
разбор: `results/EXP-098-100-completed-summary.md`, `EXP-098/099/100-completed.json`.
Пользователь разрешил EXP-101–103: три отдельные 8h TPU v5e8 кампании продолжения.
Общий фиксированный старт — EXP098 SAFE-PROTECTED-R32 final16384, оба seed;
новый Adam одинаково для всех ветвей, потом обычный точный resume состояния.
Протокол: `results/EXP-101-103-three-account-protocol.md`. Это подготовленные
гипотезы, результатов TPU EXP-101–103 ещё нет. Нижние записи исторические.

**Обновление 3 октября 2026 после передачи:** EXP-097 завершён на TPU;
зарегистрированный диагностический gate PASS. Полный JSON и разбор:
`results/EXP-097-numerical-stability.json`, `results/EXP-097-completed-summary.md`.
Четыре NAIVE-ветви остановились из-за переполнения нормы при конечных градиентах;
восемь SAFE завершены. Protected rank32 CE при LR3e-4 — перспективный кандидат,
не конечный подтверждённый метод. Пользователь разрешил три новые кампании
EXP-098–100 под три отдельных восьмичасовых TPU v5e8-сессии:
`results/EXP-098-100-three-account-protocol.md` и отдельные протоколы.
Ниже исходная передача сохранена исторически: её «у EXP-097 ещё нет результатов»
уже перекрыто этим обновлением и последними записями технического отчёта.

Состояние на 3 октября 2026. Это инструкция для нового исследовательского/кодового
ассистента, а не статья и не новый результат эксперимента.
Последний кодовый коммит перед созданием этой инструкции:
`8b2eef3 feat: diagnose gradient norm overflow in EXP-097`.

## 0. Что пользователь просит от нового ассистента

Продолжить уже существующий исследовательский проект Extent. Не начинать заново,
не выдавать его за готовую модель и не повторять десятки закрытых экспериментов.
Сначала прочитать перечисленные ниже файлы, понять результаты и текущий протокол,
затем помогать с анализом новых результатов и реализацией следующих экспериментов.

Пользователь делает эксперименты на TPU в Kaggle, присылает небольшие сводки и
полные JSON/текстовые логи. Ассистент пишет и проверяет код локально, документирует
результаты, коммитит и отправляет изменения в приватный GitHub. Локально доступного
TPU у ассистента нет: CPU-проверку нельзя называть проверкой на реальном TPU.

Язык общения — русский, простой и прямой. Объяснять, что проверяем и почему это
может помочь, без лавины английских терминов и без чрезмерно оптимистичных обещаний.
Пользователь устал от двух месяцев исследований и мелких запусков. Это не повод
приукрашивать отрицательные результаты, но важно показывать конкретные выводы.

## 1. Где всё хранится

- Приватный код: `https://github.com/KremlevLev/extent`, основная ветка `main`.
- HF dataset: `https://huggingface.co/datasets/levkremlev888/extent`.
  Он публичный; там лежат большие состояния, чекпоинты, манифесты и сводки.
- Основной научный журнал: `extent technical report.md` в корне репозитория.
- Компактные протоколы, разборы и зарегистрированные критерии: `results/`.
- Основной актуальный Python package: `extent/`; запуски: `scripts/`; тесты: `tests/`.
- README — полезная карта проекта, но некоторые слова «next» и верхняя сводка
  устарели. Последние записи журнала и отдельные протоколы важнее.
- `kaggle operations.md` — эксплуатационная документация, не текст статьи.
  Раннее упоминание 10 часов не отменяет текущего требования пользователя:
  для EXP-098–100 рассчитывать на 8 часов одного вызова, включая 45 минут
  запаса на сохранение; учитывать время верхних ячеек отдельно.
- Старое имя проекта — `singularity`; новое — `extent`. Старые логи с прежним именем
  относятся к тому же проекту. Не переименовывать обратно.
- На старом ноутбуке checkout был в
  `C:\Users\Utest\Documents\Codex\2026-08-15\referenced-chatgpt-conversation-this-is-an\extent`.
  На новом пути будут другими; никогда не зашивать старый путь в код.

Старые пользовательские вложения из Downloads не обязательно имеются на новом ПК
или в Git. Не утверждать, что прочитал отсутствующий исходный JSON. Репозиторий
содержит документированные разборы и хеши; HF — часть исходных артефактов.
Секретов в этом документе нет и быть не должно.

## 2. Как читать репозиторий, чтобы восстановить контекст

### Первый проход: цель и актуальное состояние

1. Прочитать этот `HANDOFF.md` полностью.
2. Проверить `git status`, ветку, remote и последние коммиты. Не стирать чужие изменения.
3. Прочитать начало `extent technical report.md`: цель, точные источники,
   архитектуру, границы утверждений и правила фиксации результатов.
4. Прочитать `results/EXP-094-096-first-results.md` полностью, включая full-file addendum.
5. Прочитать `results/EXP-094-096-parallel-recovery-protocol.md`.
6. Прочитать `results/EXP-097-numerical-stability-protocol.md` полностью.
7. Прочитать в техрепорте EXP-061, EXP-062/062B, EXP-064, EXP-070–072,
   затем последние разделы EXP-089–097. Эти записи объясняют текущие решения.

### Второй проход: история механизма

Прочитать существующие `results/*summary.md` и `results/*protocol.md` для:

- EXP-071/072 — последовательное восстановление на входах уже изменённой модели;
- EXP-081/082 — принятие локальных обновлений по поведению собранной модели и провал репликации;
- EXP-085/086/087/088 — парные/совместные предложения и downstream-чувствительность;
- EXP-089/090 — длинный горизонт и заморозка dt;
- EXP-091/092/093 — защита скопированных весов, глобальная интерполяция, коэффициенты ветвей.

Использовать `rg --files results` для точных имён. Не выдумывать название файла,
если оно отличается. Для полной хронологии прочитать весь журнал по частям;
его ранняя дата «Last updated» устарела, свежие записи находятся в конце.

### Третий проход: текущая реализация

Перед изменением EXP-097 прочитать:

- `scripts/m3q_numerical_stability_campaign.py` — новый эксперимент;
- `scripts/m3q_subspace_engine.py` — общий resumable двигатель EXP-094–097;
- `extent/stable_gradient_clip.py` — устойчивое вычисление нормы/клиппинга;
- `extent/recovery_subspace.py` — обучаемые поправки к замороженной модели;
- `extent/hf_artifact_sync.py`, `extent/hf_checkpoint_sync.py` — отправка на HF;
- `extent/campaign_checkpoint.py`, `extent/endpoint_checkpoint.py` — состояния и provenance;
- `extent/notifications.py` — Telegram;
- `tests/test_stable_gradient_clip.py`, `tests/test_recovery_subspace.py`,
  `tests/test_subspace_campaign_workflow.py` и относящиеся к загрузке/сохранению тесты.

Для архитектуры и восстановления:

- `extent/qwen_source.py`, `extent/qwen3_teacher.py`, `extent/qwen3_parity.py`;
- `extent/config.py`, `extent/model.py`, `extent/layers/`;
- `extent/mamba3_transplant.py`, `extent/initialization.py`, `extent/hybrid_transplant.py`;
- `extent/rorope_bkv_conversion.py`, `extent/rorope.py`, `extent/mla_conversion.py`;
- `extent/sequential_recovery.py`, `extent/full_model_distillation.py`;
- `extent/sharding.py`, `extent/optimizer.py`, `extent/train_step.py`;
- `extent/teacher_activation_cache.py`, `extent/offline_distillation.py`,
  `extent/streamed_lm_eval.py`, `extent/artifact_summary.py`.

В корне есть более ранние `model/`, `modeling/`, `training/` и `config1.py`.
Не считать их автоматически основной реализацией: проследить реальные imports
актуальных campaign scripts. Legacy `tests/test_qwen_base.py` скачивает большую
модель/запускает работу на import и исключён из обычного offline pytest.

## 3. Научная задача — не просто собрать гибрид

Итоговая цель: превратить предобученную Qwen3 в преимущественно Mamba-3 MIMO модель
без обучения с нуля, сохранить знания и восстановить качество с ограниченным
числом токенов/TPU-часов. Сохранённое GQA-внимание потом сжать в MLA-подобный cache.
Нужны реальные преимущества по качеству/памяти/скорости, а не только конфиг.

Пользователь хочет сильную научную работу, arXiv и потенциально ICLR-подобную
конференцию. Рабочее название метода — «Mamba-3 in the Qwen», сокращение M3Q.
Сильного подтверждённого конечного метода пока нет. Не обещать принятие статьи.

Научный вклад должен быть в переносе и восстановлении архитектуры: новый механизм,
объяснение, сильные контролируемые сравнения, воспроизводимость и экономия compute.
Само сочетание SSM и attention не является достаточной новизной.

После восстановления пользователь хочет reasoning SFT, возможно на примерах
рассуждений Claude, и большое контекстное окно. Это будущая отдельная стадия;
не смешивать улучшение от SFT с восстановлением после замены архитектуры.

## 4. Точные модели и архитектуры

### 14B — окончательная целевая модель

Источник именно `Qwen/Qwen3-14B`, НЕ Qwen2.5, НЕ автоматически `Qwen3-14B-Base`.
Immutable revision: `40c069824f4251a91eefaf281ebe4c544efd3e18`.

- 40 слоёв, hidden 5120, intermediate 17408;
- 40 Q heads, 8 KV heads, head_dim 128, per-head Q/K RMSNorm;
- vocab 151936, RoPE theta 1000000, опубликованные max positions 40960;
- untied embeddings; источник содержит 443 тензора;
- teacher mapping: 14,768,307,200 параметров, coverage 443/443;
- direct mapping: 203 тензора / 12,251,714,560 параметров.

Цель строго 34 Mamba + 6 attention = 85%/15%.
Сохранённое attention на zero-based `[5,12,19,25,32,39]`.
Пользователь прямо отверг 25% attention как полноразмерный baseline.
Ошибочные ранние 48-слойные схемы/12 attention не являются текущим контрактом.

Production attention выбран по EXP-019: RoRoPE fold1 + BKV activation PCA,
rank448 latent cache + один128-элементный rotary key. На сохранённом attention
cache сокращается с2048 до576 элементов/token, то есть71.875%.
Для шести слоёв:12288→3456 элементов/token. Это не измерение полного HBM,
не отношение всей модели и не обещание ускорения decode.
Ранний MaxText-style MLA остался scaffold/ablation, не production attention.
Extent не нужно объявлять готовым drop-in upstream MaxText decoder.

Production Mamba первоначально выбран `INIT-K-balanced-qkvo-lift`.
Это инженерный выбранный builder, а не утверждение о победе метода в full-model recovery.

### 1.7B — текущий исследовательский стенд

Источник `Qwen/Qwen3-1.7B-Base` revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.

- 28 слоёв, hidden2048, intermediate6144;
- 16 Q heads,8 KV heads,head_dim128, vocab151936;
- tied embeddings, max positions32768;310 source tensors;
- hybrid:24 Mamba +4 неизменённых GQA на `[6,13,20,27]`.

Это около85.7% Mamba, дискретная малая аналогия целевых85%.
Сохранённое внимание сейчас GQA намеренно: изолируем перенос Mamba от MLA.
Не добавлять MLA в текущий recovery-протокол незаметно.
14B — проверка масштабирования лучшего метода позже, не каждодневный стенд.

## 5. Что уже действительно проверено

### Корректность и инженерная выполнимость

- JAX Qwen decoder layer сверялся с official PyTorch; layer0 FP32 на2T4:
  max_abs≈4.05e-6, relative_l2≈9.86e-8. Это layer parity, не full-model quality.
- Mamba-3 JAX scan сверялся с независимой Torch reference и shapes/names.
  Это корректность конкретной reference-реализации, не доказательство быстрого kernel.
- Малые BF16 sharded train steps прошли на TPU v5e1 и v5e8 с конечными градиентами.
- EXP-062 собрал полный14.766B hybrid:34 Mamba/6 выбранных attention,
  443 source tensors,203 direct mappings, реальный sharded Lion, конечный forward8/32/128.
- Реальные weights и Lion momentum по3.4513GiB/device, вместе6.9026GiB/device.
  Не включены gradients, activations, temporaries, executable buffers.
- EXP-062B backward context8 прошёл: конечные loss/grads, но raw norm≈8.28e8.
  Это numerical/HBM feasibility, не восстановленная модель, не качественное обучение.

Reference recurrence — `lax.scan`, FP32 recurrent state. Для убедительных speed claims
нужны оптимизированные kernels (например Pallas), корректный streaming decode,
сопоставимый Transformer baseline и реальные измерения.

### Перенос матриц и exact lift

Прямое копирование Q/K/V/O в Mamba не оказалось универсально хорошим.
Изучали prior reuse, scale matching, readout calibration, разные MIMO allocations,
linear/complex bridge, RoPE преобразования и несколько objectives.

Exact balanced lift раскладывает исходную линейную операцию по MIMO-каналам,
сохраняя её в соответствующем подвыражении. Он НЕ делает softmax attention
математически эквивалентным всей Mamba-3 recurrence.

EXP-056–061 дал реальный позитивный результат на локальном восстановлении и
частичных композициях. EXP-061:4/8/12 production-aligned замен,3paired seeds;
exact выигрывает3/3 seeds на каждой глубине; mean exact-random NLL
−0.206008/−0.122781/−0.164937. Все upper bootstrap bounds ниже нуля.
На12 слоях это≈3.12% total-NLL improvement и14.51% уменьшения excess NLL
над исходным Qwen относительно random. Не говорить «модель стала лучше на15%»
без указания метрики, знаменателя и условий.

Hidden-state L2 мог становиться хуже, хотя NLL улучшался. Локальный MSE и NLL
собранной модели не взаимозаменяемые конечные критерии.

### Почему локальная победа не перенеслась на полную модель

В full-model Qwen3-1.7B сравнениях random мог обгонять exact.
Подтверждение длинным горизонтом EXP-089: в завершённой paired seed exact
проиграл random после25.17M training tokens; зарегистрированный gate стал невозможен.
Нельзя описывать ранние layerwise успехи как подтверждённую full-model superiority.

Ключевой механизм: ранние замены меняют residual streams для последующих слоёв.
Блок, обученный на входах чистого Qwen, получает другие входы в собранном hybrid.
Это проверяли не только рассуждением, но matched-input/композиционными опытами.

### Последовательное восстановление — важный подтверждённый эффект

EXP-071: последовательно добавляем Mamba, каждый новый блок учим на входах,
которые реально создаёт уже изменённый prefix. ONPOLICY лучше teacher-input обучения.

EXP-072-v2 подтвердил на свежем тексте, всех24 Mamba positions и большем бюджете:
ONPOLICY final NLL10.545254/10.266002 против22.346362/22.037049 у TEACHER,
дваpaired seeds123/456, gate проходит. Разница≈−11.8NLL в каждой seed.
Снижение total NLL≈53%, excess NLL≈61–62% относительно TEACHER-условия.
Исходный Qwen≈3.0675 на том же оценочном наборе: восстановление далеко не полное.
Особенно заметен провал последних замен:depth20≈6.9–7.0 →depth24≈10.3–10.5.

Это улучшение процедуры восстановления относительно matched TEACHER, НЕ
победа над random при равном полном compute. Затраты на подготовку warm start
должны входить в окончательное сравнение метода.

Completed EXP-072-v2 ONPOLICY checkpoints — текущие исходные модели поздних опытов.
Восстанавливать их непосредственно с HF по проверенным контрактам и hashes.
Не просить пользователя заново обучать всё или передавать большие файлы вручную.

### Что не стало надёжным решением после EXP-072

- EXP-073–080: обычное совместное обучение, варианты защиты backbone/оптимизатора,
  depth segments и второй coordinate sweep не дали надёжного восстановления.
- EXP-081: assembled-model trust interpolation улучшил start в обеих seeds,
  но эффект seed456 мал. EXP-082 с8replications не подтвердил matched-control gate.
  Отбор на WikiText не гарантировал улучшение PG-19.
- EXP-083–088: dual-domain/robust selection, paired lookahead/joint pair/downstream
  не превратились в подтверждённый конечный метод. Прочитать отдельные summaries;
  не свести все опыты к «вообще ничего не работает» и не выбрать лучший seed.
- EXP-090: ранняя заморозка dt до8.39M tokens; endpoint25.17M.
  Seed123 небольшое улучшение excess NLL−0.224476/KL−0.282565;
  seed456 ухудшение+2.374740/+3.372097. Two-seed gate FAIL.
- EXP-091: ONPOLICY warm start + post-Lion backbone updates×0.03 противPLAIN.
  Seed123 уже опроверг gate; оба варианта ухудшались от общего старта.
  Не тратить сессию только на завершение mathematically impossible gate.
- EXP-092: straight whole-model interpolation черезEXP-091 направления.
  Все3available branches выбралиalpha0. Малые положительные alpha тоже ухудшалиNLL.
- EXP-093: обучаем только bounded output gains, без изменения остального.
  Per-layer выигрывает global gain в обеихseeds, но ухудшает start seed456.
  Gate FAIL. Start/пер-layer NLL13.261727→12.451787 и9.161675→10.170573.

Не сравнивать числа между экспериментами без проверки одинаковых текстов,
offsets, checkpoints и того, absolute это NLL или excess NLL.

## 6. Последние законченные EXP-094/095/096

Три параллельных кампании проверяли не новые инициализации, а маленькие обучаемые
FP32 поправки к уже собранным замороженным BF16 ONPOLICY hybrids.
База: 24 Mamba + 4 GQA; seeds 123/456; batch 1; context 256; Adam 3e-4; clip 1.
Поправки начинают с нулевого изменения функции. Low-rank A×B прибавляется к kernel
и по контракту может быть свёрнут для inference. Реальный экспорт надо проверять.

- EXP-094: OUT rank 8, INOUT rank 8, MLP readout rank 8, head gains; всё CE.
- EXP-095: одинаковый INOUT rank 8, цели CE / teacher KD / self-anchor / delta-then-KD.
- EXP-096: INOUT CE rank 8 / rank 32 / rank 64 / protected rank 32.

Protected запрещает поправки к in_proj columns, производящим dt/raw_a/trap/angle.
Но входы от предыдущих слоёв меняются: динамика не целиком «заморожена».

Общие тексты и порядок; 8192 steps на траекторию; горизонты 2048→4096→8192.
Validation диагностический; locked test только на final и start.
Сводки строятся автоматически; полный JSON остаётся для аудита.

Результаты:

- Shared INOUT-rank8 CE повторён byte-equivalent в трёх кампаниях. Это дублированный
  контроль, не три независимых репликации. Seed 123 завершён: NLL 12.968006→7.349788.
  Seed 456 падает на attempted step 4228; last-good step 4227.
- OUT rank 8 падает в обеих seeds. MLP readout завершён, но NLL ≈19.87/19.14.
  Head gains дают смешанный слабый эффект.
- Teacher KD и self-anchor в seed 123 завершены с NLL 7.983263/8.881876, хуже CE;
  в seed 456 падают. Delta-then-KD падает на attempted steps 91/44.
- Ordinary rank32 CE завершён в обеих seeds:
  NLL 12.968006→7.643549 и 12.276975→9.342761.
- Rank 64 падает в обеих seeds.
- Protected32 seed 123 падает на attempted step 4025; seed 456 завершён с NLL
  7.315458, лучше ordinary32 seed 456 (9.342761). Это выигрыш одной seed, не PASS.
- Все три кампании имеют branch_failure и scientific_gate=false.

Новая реальная зацепка: адаптация небольшого input/output subspace способна
улучшать whole-model NLL в обеих seeds при rank 32. Но это не полное восстановление,
не победа инициализации и не устойчивая траектория. Даже успешные rank32 validation
curves сильно скачут; последние нормы ≈1.66e7/4.15e10.

Полные JSON прочитаны. Старый guard объединяет loss, градиенты, naive norm,
новые параметры и Adam state. Из FAIL нельзя узнать, какая величина испортилась.
Перед некоторыми failures norms ≈1.64e17/6.37e17. Это повод проверить переполнение
sum-of-squares нормы, НЕ доказательство этой причины. В FP32 квадрат большого
конечного градиента может переполниться при ещё конечных gradient elements.

Полный разбор и SHA256: `results/EXP-094-096-first-results.md`.

## 7. Текущий следующий эксперимент: EXP-097

EXP-097 реализован, локально проверен, закоммичен и отправлен в GitHub.
В передаваемом контексте реальных TPU-результатов ещё НЕТ.
Если пользователь пришлёт новую сводку, сначала установить её свежесть и статус.

Цель: отделить переполнение вычисления нормы от настоящего forward/backward/Adam
срыва и проверить устойчивый clipping / пониженный LR. Это не новый initializer.

12 траекторий: seeds 123/456 × шесть arms:

| Arm | Rank | Поправка к dynamics projections | Norm/clip | LR |
|---|---:|---|---|---:|
| NAIVE-R8 | 8 | разрешена | старый sum-of-squares | 3e-4 |
| SAFE-R8 | 8 | разрешена | max-scaled | 3e-4 |
| SAFE-R8-LOWLR | 8 | разрешена | max-scaled | 3e-5 |
| NAIVE-PROTECTED-R32 | 32 | запрещена | старый | 3e-4 |
| SAFE-PROTECTED-R32 | 32 | запрещена | max-scaled | 3e-4 |
| SAFE-PROTECTED-R32-LOWLR | 32 | запрещена | max-scaled | 3e-5 |

Все стартуют от неизменённых EXP-072-v2 ONPOLICY. Не использовать optimizer states
проваленных EXP-094–096. Base BF16 frozen; coordinates/moments FP32;
CE; batch 1; sequence 256; clip 1.

Оценка повторяет EXP-094–096: это reused evaluation, не fresh confirmation.
Train WikiText103 offset 8,388,608, 8192 windows;
validation offset 49,152 / 32 windows; test offset 32,768 / 64 windows.
Горизонты 2048/4096/8192; validation каждые 1024; test на start и final 8192.
Максимум 98,304 student steps / 25,165,824 input tokens.

Stable norm: m=max|g|, s=sqrt(sum((g/m)^2)); clipping через (g/m)/s без g²
и без обязательного формирования m×s. Zero case обработан. NaN/Inf не заменяются
nan_to_num и не маскируются. Авторитетная диагностика — log10(norm);
display norm может насыщаться на 1e38, это явно обозначено.

Каждый step вычисляет NAIVE и SAFE proposals на ОДНОМ gradient и ОДНОМ Adam state.
Неиспользуемый shadow proposal не обновляет optimizer. Отдельные trajectories сами
по себе не доказывают norm-overflow causality: rounding может раньше менять путь.
Norm-only event: finite forward/loss/gradient elements + nonfinite naive norm +
finite safe proposal.

Диагностика разделяет forward activations/logits, loss, gradient elements, norm,
current parameters/moments и proposal parameters/moments. Пишутся layer maxima и
coordinate gradient health. До первого norm-only update сохраняется event checkpoint;
при failure сохраняется last-good state. Исходный Qwen отдельно оценивается на том
же locked test для остаточного quality gap, не используется для выбора LR или steps.

Зарегистрированный gate: все 12 arms terminal; хотя бы один norm-only event;
SAFE-R8 final test NLL лучше start в обеих seeds. Controls могут numerically fail
как наблюдаемый диагностический исход, но это не successful recovery.
Protected и LOWLR — заранее заданные вторичные сравнения.

Default cap 8.25 часа, включая 30 минут резерва; это не обещание runtime.
Local atomic binary save каждые 256 steps; bundled HF не чаще одного commit / 10 min,
плюс final. HF 429 → cooldown 1 h; обучение локально продолжается.
Внешний VM kill может потерять unsynced tail.

HF namespace: `experiments/exp097-numerical-stability`.
Summary stem: `extent-m3q-numerical-stability`.

У общего engine есть optional hooks для EXP-097; legacy contracts EXP-094–096
сохранены. Legacy engine digest:
`c3e17745b12e2d1bbab225ead5db3db705343327c1caa5241e10a8512dfb36e6`.
Не менять его произвольно и не обходить resume checks.

Локально проверены нормы до 1e38, zero/nonfinite handling, matched proposal overflow,
compiled steps всех шести arms, восемь virtual CPU shards, binary optimizer resume,
event checkpoints и 429/interruption workflows. В предыдущем ответе отмечены
32 targeted CPU tests и шесть sharded step tests. Это НЕ real TPU throughput
и НЕ наблюдённый numerical win.

Последняя ячейка пользователя, когда setup уже выше:

```python
from scripts.m3q_numerical_stability_campaign import main as run_exp097
result = run_exp097([])
```

Попросить small summary + JSON, а не передачу checkpoint через пользовательский ПК.
При budget-partial повтор той же ячейки возобновляет совместимые HF states.

## 8. Правила Kaggle/Colab и сохранения

Основной accelerator: Kaggle TPU v5e8; mesh data/fsdp/tensor = 1/4/2.
Colab v5e1 подходит для single-device correctness, не для multi-device sharding
или полного 14B. На двух T4 старый BF16 compute давал NaN; FP32 compute проходил.
Использовать аппаратный auto dtype и не переносить GPU-specific failure на TPU
без проверки.

TPU эксклюзивен одному Python process. После jax.devices() в notebook kernel
не запускать `!python`, `!pytest` или subprocess с JAX: возникают busy TPU/PID/libtpu.
Campaign запускается через import main([]) в kernel. Pytest при необходимости —
pytest.main([...]) в том же процессе. Установка dependencies должна предшествовать
инициализации JAX; после его замены pip может потребоваться restart kernel.

Пользователь часто запускает Save Version и уходит. Не полагаться на интерактивные
переменные, ручное сохранение или живую сессию через несколько часов.
Реально теряется локальный /kaggle/working после окончания сессии.
Durable HF save отличается от существующего локального checkpoint.
Нельзя гарантировать callback при внешнем kill: описывать recovery granularity честно.

Ранее уже были ошибки: missing JSON keys; NumPy conversion внутри jit;
FrozenDict/dict mask mismatch; NumPy scalar JSON serialization; donation alias
teacher/student; недостаточный размер токенизированного корпуса; disk full; HF429.
Эти пути нужно проверять локально ДО многочасовой очереди TPU.

Большие Qwen shards/cache можно хранить в RAM-backed /dev/shm при достаточной RAM.
Не забивать примерно 20 GB working disk полным 14B checkpoint.
RAM budget проверяется; наличие 300 GB на одной сессии не универсальная гарантия.

HF uploads должны быть пакетными. Repository commit limit уже ломал запуски:
128 commits/hour. Не делать отдельный remote commit на каждый metric/JSON/step.
Несколько аккаунтов вместе нагружают один и тот же HF repository.

Секреты и провайдеры:

- EXTENT_DEPLOY_KEY_B64 — read-only Git deploy key в Base64;
- HF_TOKEN — авторизация HF;
- EXTENT_HF_CHECKPOINT_REPO — если используется соответствующим runner;
- TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID;
- Kaggle: UserSecretsClient; Colab: google.colab.userdata. Это разные механизмы.

Секреты не печатать и не коммитить. Telegram должен сообщать start, end,
budget-partial и caught failure. NOTIFIER_ENABLED позволяет отключить его.
Никакой notifier не запускается до фактического старта kernel в очереди.

Пользователь не хочет manual uploads больших файлов, Google Drive или перенос
checkpoint через ноутбук. Всё напрямую HF↔runtime. Скачать нужно только небольшие
summary/JSON. HF dataset накопил примерно 500 GB; пользователь хочет периодический
retention audit, а не ограничение использования HF. Ничего не удалять без отдельного
одобрения точных целей. Сохранять зависимости, важные endpoints, configs/hashes/raw
metrics. Удаление current files не обязательно освобождает исторические HF versions.

## 9. Как проектировать следующие эксперименты

Пользователь ждёт TPU часами и хочет 5–8 часов полезной работы в пределах девятичасовой
сессии, не отдельную 15-minute пробу. Разрешены несколько гипотез/arms в одной
кампании. Не тратить compute только ради длительности: нужны comparators,
balanced horizons и остановка при уже доказанном провале зарегистрированного gate.
Runtime estimate брать из measured throughput, не объявлять 8192 steps восьми часами.
Длинный научный run можно продолжать несколькими resumable sessions.

До запуска:

- plan-only и проверки без скачивания моделей;
- реальные compiled tiny steps, все arms и objectives;
- восемь virtual CPU devices для sharding, если возможно;
- serialization/resume model+optimizer+data cursor;
- aggregation с отсутствующими/failed/partial branches;
- cloud execution без интерактивных запросов и случайных переменных из прошлой ячейки;
- HF429, transient network error, auth failure, budget stop;
- token capacity guards, наличие checkpoint dependencies, memory/disk budget.

Не скрывать structural exception под «научным отрицательным результатом».
Не заменять missing metrics нулём ради завершения сводки.

До наблюдения outcomes нужен protocol в results: source pin, methods, paired seeds,
data hashes и offsets, train/validation/test domains, steps/tokens, optimizer,
primary metric/gate, secondary exploratory metrics, budget и resume contract.
Test не использовать для выбора alpha/LR/checkpoint.
Если test уже обсудили и принимали решения по нему, следующая confirmatory phase
требует нового test либо честной отметки reused evaluation.

Equal steps/tokens ≠ equal FLOPs. Учитывать layerwise preparation, offline cache
и teacher passes. Две seeds сами по себе не делают результат conference-level.
Нужны свежие домены, больше seeds, длинные контексты, сильные предыдущие методы
и сравнения при сопоставимом compute.

Related work: Mamba in the Llama, MOHAWK, MHA2MLA, SSM hybrids, обсуждавшиеся
Apple attention-to-Mamba и attention-transfer papers. Некоторые названия/даты
пришли от другой LLM: проверить originals и bibliography в техрепорте перед
ссылкой или утверждением новизны. Не говорить «linear attention = softmax = Mamba3»
без точных условий. Mamba-aware linear/complex/MIMO bridge уже пробовали;
новый вариант должен объяснять отличие от EXP-065/066 и предыдущих мостов.

После EXP-097 дальнейшее решение зависит от результата:

- Norm-only event подтверждён: оценить реальный эффект stable clipping,
  затем проверить перспективный рецепт на свежем наборе.
- Forward/gradient-element failure: работать с layer diagnostics;
  не обещать, что clipping всё исправит.
- LOWLR улучшил устойчивость: отдельно измерять progress и remaining teacher gap.
- Всё устойчиво, но качество плохое: диагностировать capacity/objective/input shift.
- До перехода на14B нужен устойчивый whole-model1.7B trend и random control
  при сопоставимых ПОЛНЫХ затратах, включая warm-start preparation.

Не обещать число токенов полного восстановления8B/14B: это не измерено.
Пользователь допускал3–6месяцев TPU;25–50B tokens были лишь предположением.
Без реальной скорости и recovery curves это не обоснованный прогноз.

## 10. Документирование и взаимодействие

Каждый результат добавлять в `extent technical report.md`: дата, источник,
hash при наличии, MEASURED/DERIVED/TARGET/HYPOTHESIS, решение и ограничения.
Отдельный компактный `results/EXP-XXX-...summary.md`; raw JSON оставлять аудируемым.
Агрегатор должен проверяемо вычислять выводы, а не просто выбрасывать поля.

Не менять gate постфактум ради PASS. Лучший intermediate не заменяет конечный
registered endpoint. Различать operational failure, numerical failure,
budget partial и scientific gate FAIL. Partial иногда уже делает gate невозможным;
тогда не обязательно тратить квоту на оставшиеся controls.

Перед новой гипотезой прочитать историю и проверить, не тестировали ли именно её.
Не предлагать пользователю заново собирать прошлые кеши, если есть durable HF source.

После изменений сообщать:

1. Что изменено и какую гипотезу проверяет.
2. Что проверено локально; что не проверено на TPU.
3. Название и hash коммита; успешен ли push.
4. Одну готовую финальную ячейку для Kaggle.
5. Какую небольшую сводку прислать.

Setup clone/install/secrets у пользователя обычно уже выше в ноутбуке.
Не повторять весь setup, когда он просит только эксперимент.
При сохраняющейся авторизации коммитить и отправлять код, протокол и журнал;
чистый Git и local commit не означают успешный push — проверять.

В прошлом были отдельные просьбы о backdated commits ради GitHub streak.
Это не научное требование и не повод искажать даты измерений или provenance.

## 11. Первый ответ нового чата

После чтения файлов коротко сверить понимание:

1. Цель — надёжный Qwen→Mamba3 recovery method; потом MLA,14B и efficiency.
2. Уже работало: exact lift в частичных композициях; ONPOLICY против TEACHER
   full-depth; recent rank32 corrections улучшают NLL, но нестабильны.
3. Не подтверждено: полное восстановление, whole-model initializer superiority,
   long context, inference speed, tokens-to-recovery для14B.
4. Текущий подготовленный код — EXP-097; запросить результат только если его нет.
5. Не предлагать снова random-vs-copy на одном слое, Qwen2.5 вместоQwen3,
   полгода обучения14B без малого подтверждения или только план без runner.

Уважать амбицию пользователя, оставаясь исследователем:
факты → гипотеза → контролируемый эксперимент → измерение → обновление вывода.
