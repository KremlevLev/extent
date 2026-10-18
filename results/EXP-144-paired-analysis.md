# EXP-144: original learning rate versus LR / 3

Worker 0 stops normally at the registered work deadline after 11.29449 h; no reported error. Source138-CE current11965, original optimizer moments/counts preserved. Accepted 11225 new updates / 2,873,600 input tokens. Both workers use identical data and shared contract (excluding worker and recipes). Source replay passes on all160 windows.

## Matched new-update comparison

| New updates | Original A | LR/3 A | Original B | LR/3 B | Original C | LR/3 C |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.000904 | 6.000852 | 5.992092 | 5.992129 | 5.566415 | 5.566408 |
| 128 | 6.005409 | 5.962322 | 5.982779 | 5.946403 | 5.574080 | 5.529323 |
| 512 | 5.960432 | 5.930963 | 5.937431 | 5.919762 | 5.537997 | 5.502199 |
| 1024 | 5.939740 | 5.919194 | 5.936334 | 5.911280 | 5.526054 | 5.505813 |
| 2048 | 5.923913 | 5.902842 | 5.942006 | 5.914729 | 5.583893 | 5.536391 |
| 2560 | 5.925408 | 5.900343 | 5.911442 | 5.899538 | 5.538842 | 5.500326 |
| 3072 | 5.877583 | 5.881522 | 5.890975 | 5.892926 | 5.479839 | 5.487645 |
| 3434 | 5.882288 | 5.878039 | 5.890181 | 5.886096 | 5.516064 | 5.503922 |
| 3584 | 5.878232 | 5.863221 | 5.888491 | 5.891818 | 5.522485 | 5.495707 |
| 4096 | 5.887993 | 5.881199 | 5.918462 | 5.899956 | 5.483513 | 5.482735 |
| 4419 | 5.874940 | 5.870281 | 5.908197 | 5.899137 | 5.468700 | 5.465112 |
| 4608 | 5.857401 | 5.860165 | 5.871226 | 5.891300 | 5.464398 | 5.451702 |
| 5120 | 5.858249 | 5.860774 | 5.850708 | 5.867446 | 5.459480 | 5.472171 |
| 6144 | 5.874570 | 5.889542 | 5.866796 | 5.879281 | 5.488674 | 5.499803 |
| 8192 | 5.789894 | 5.818954 | 5.839864 | 5.875902 | 5.384389 | 5.429943 |
| 8515 | 5.832533 | 5.856356 | 5.815621 | 5.857535 | 5.395046 | 5.440320 |
| 10240 | 5.781944 | 5.832523 | 5.812152 | 5.850099 | 5.413618 | 5.443914 |

Original LR has worse early values, but wins all three panels at matched8192 and10240. At10240 the advantage is A0.050579/B0.037947/C0.030296 NLL. This is a paired single-seed result; A selects checkpoints, B/C remain repeatedly exposed descriptive validation panels. No independent test or universal optimal-LR claim.

## Final and registered checkpoints

Original last global23190 (new11225): {'A': 5.754783913493156, 'B': 5.784173473715782, 'C': 5.365548655390739}. Registered A-selected best global22205: {'A': 5.781943768262863, 'B': 5.812152191996574, 'C': 5.413618318736553}.
LR/3 last global23854 (new11889): {'A': 5.803474850952625, 'B': 5.824707701802254, 'C': 5.409201122820377}. Final endpoints have different update counts; use the matched table for the learning-rate effect.

Original A at8192:5.789894;10240:5.781944;last11225:5.754784. From8192 to last: A improves0.035110/B0.055691/C0.018841. Nonmonotonic trajectory; no demonstrated hard plateau. These improvements require neither additional rank nor an optimizer restart. They do not establish that capacity or activation scale are irrelevant.

Both final layer-profile consistency checks pass. Tiny context audit (8 windows,127 targets) gives256:5.537693 versus1024:5.550043; no useful long-context improvement established.

## Persistence

Report pending=['CE-CONTINUE'], durable step=21437. Targeted read-only HF manifest audit confirms final23190 and exact report contract plus arm. Full payload restore not performed. See EXP-144-fixed-worker0-remote-audit.json.

Keep existing147/148 screen: it starts145 current and asks whether rates help at a different conditioning point. This result does not replace that comparison.
