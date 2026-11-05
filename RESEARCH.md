# Extent: research documents

Updated5 November2026. Start with the current ledger, then follow its evidence links.

| Document | Role |
|---|---|
| [Technical report, Volume II](extent%20technical%20report%20v2.md) | Current scientific ledger: received129–168 and clean171, provenance correction, prepared170/172. |
| [Technical report, Volume I](extent%20technical%20report.md) | Historical ledger through measured155 / prepared156. Existing scientific entries are preserved. |
| [HANDOFF](HANDOFF.md) | Current context and dated operational handoffs. Earlier dated states may be superseded. |
| [English paper draft](paper/draft.md) / [Russian draft](paper/draft.ru.md) | Narrative manuscripts derived from evidence; not substitutes for the ledger. |
| [Paper figures and provenance](paper/README.md) | Scientific figures, numeric extracts and rebuild instructions. |
| [Result artifacts](results/) | Per-experiment protocols, received summaries, analyses and audits. |
| [Clean-start audit170](results/EXP-170-clean-start-audit.md) | Why the old trained lineage is not direct Q/K/V initialization. |
| [PyTorch170 launch and limits](results/EXP-170-pytorch-launch.md) | New head-preserving stage1, setup, planner, stops and verification limits. |
| [Clean QKVO TPU171](results/EXP-171-launch.md) | Separate original-Qwen TPU alignment, phase-aware speed planner, no historical bank reconstruction. |
| [Frozen171 quality benchmark172](results/EXP-172-protocol-and-launch.md) | Matched teacher/gqa4/gqa7 T4 evaluation, fixed CURRENT5120, precision gates and partial-resume limits. |

`MEASURED` describes received observations; `ENGINEERING` describes implementation
checks; `PREPARED` describes code/plans without scientific outcomes. Missing
reports are not automatically failed experiments. No single NLL trajectory joins
different text panels or silently substitutes BEST for CURRENT.

Example: EXP168 has a measured final CURRENT260386 and153,896,960 new accepted
input positions. EXP170 has tested code and a100M-input target, not an observed
100M training result. Read each experiment's source and status before citing it.
