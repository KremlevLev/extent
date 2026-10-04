# EXP-098--100: completed results and next decision

Reviewed from all three user-supplied full JSONs; no rerun performed.
All campaigns completed with no pending HF states: EXP098 5.6758h,
EXP099 6.3785h, EXP100 4.6649h. All registered scientific gates FALSE.

| Experiment | Wiki final seed123 | Wiki final seed456 | Paired conclusion |
|---|---|---|---|
|098 ordinary32 / protected32|13.6214 /7.3058|7.3971 /7.3415|Protection advantage6.3155 /0.0556; second below registered0.1|
|099 protected16 /32 /64|7.3472 /7.2885 /7.2897|7.4117 /7.2851 /7.3931|Rank64 does not improve Wiki; rank32 best endpoint in both|
|100 CE /Qwen0.1 /self0.1|7.4959 /7.3376 /7.4717|7.6848 /8.0524 /7.9499|Qwen helps one, harms other; PG19 worsens both|

098 PG19 ordinary/protected:14.2448 /9.4160 and10.0845 /8.8472.
Protection passes the PG19 paired comparison, but overall cross-domain gate
requires the failed Wiki gate. Ordinary32 raw-norm overflow counts89 /46;
protected32 zero. 099 rank64 has one norm-only event; no nonfinite forward/gradient
steps in any 098--100 branch. Stable clipping is still required.

Source seeds are the same original EXP072 seeds. Repeated protected32 prefixes
are NOT independent replications. In particular 098/099 validation trajectories
match through their shared horizon; differences with100 appear after shared early points; exact cause of cross-campaign
trajectory differences is not established. Do not claim bitwise reproducible trajectories across
campaigns or identical locked endpoints at different horizons. Test data are shared
across098--100, and comparisons informed the next design.

Decision: fixed098 protected32 FINAL16384 as common next-stage source, not the
best099 endpoint. Rank32/CE remains the working candidate, not a recovered model.
Test late teacher anchoring, late LR reduction, and late dynamics release separately.
Fresh token ranges and pinned source/checkpoint hashes in101--103.

## Original artifact provenance
- EXP-098 JSON SHA256 `03ca7a1e175ebc72196f8f8f674171885141dfe3501c344b3147432631c9216e`.
- EXP-099 JSON SHA256 `d245c2865f0752035768536e36626e463fec527c4bbd84422d7e1c8848ec18b8`.
- EXP-100 JSON SHA256 `b036453f8c78905e7619266937d9766f64d48410d8eac6885861cb3e5c7b4b40`.
