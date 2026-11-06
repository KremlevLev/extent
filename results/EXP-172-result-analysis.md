# EXP172 received — frozen clean QKVO CURRENT5120 quality

Received6 November2026. Complete:6000 questions (1000 each),192 NLL windows,
24 greedy48-token prompts, all scored on identical teacher/gqa4/gqa7 inputs.
Both workers completed3108 jobs, returncode0. Frozen171 source83,886,080 inputs.
No new training. Raw artifact SHA and contract in EXP-172-received-summary.json.

| Task / primary metric | Original Qwen | 4 attention /24 Mamba | 7 attention /21 Mamba |
|---|---:|---:|---:|
| ARC-Easy acc_norm |67.0%|74.2%|74.6%|
| ARC-Challenge acc_norm |46.3%|46.9%|47.2%|
| PIQA acc_norm |77.7%|76.3%|77.2%|
| WinoGrande acc |63.9%|62.9%|62.5%|
| HellaSwag acc_norm |67.3%|62.2%|62.6%|
| MMLU acc |62.4%|45.1%|51.9%|

Raw-row metrics independently recomputed, match summary. Gold/prompt hashes match
across models; no truncated question contexts. Separate EXP156-derived evaluator,
zero-shot sampled tasks, NOT official full-split/five-shot leaderboard scores.
MMLU sample balances57 subjects (17–18 questions/subject), not full-split weighting.

Paired95% bootstrap intervals: ARC-Easy improvement over teacher +7.2pp[4.9,9.5]
and+7.6pp[5.4,9.8]. HellaSwag declines -5.1pp[-6.8,-3.3]/-4.7pp[-6.3,-3.0].
MMLU declines -17.3pp[-20.7,-13.9]/-10.5pp[-13.5,-7.6].
Gqa7 exceeds gqa4 MMLU by6.8pp[4.2,9.4], independently computed paired bootstrap,
3000 resamples/seed172. Exploratory intervals, no multiple-comparison correction.

| NLL panel/context1024 | Original | gqa4 | gqa7 |
|---|---:|---:|---:|
| Edu |2.435852|2.558359|2.548330|
| English TinyStories |1.340851|1.539719|1.532551|
| Russian Wikipedia |1.565064|1.709873|1.695655|

32 windows/32736 targets each; different-length256 panels also recorded. English
stories are simple text, not broad web or long-context evidence. Edu includes
previously monitored validation; other panels independent of171 training, but
not claimed absent from Qwen pretraining. Hybrids larger (~2.13/2.08B vs1.72B).

Qualitative generation is coherent on many prompts and simple code, but errors
remain: both hybrids continue Russian freezing-temperature prompt with -18C,
versus teacher0C; gqa7 repeats a file-recovery sentence. Some correct answers
continue invented dialogue; original is also a BASE model with continuation
artifacts.48-token cap truncates some code; no code execution/pass@1 measured.

Conclusion: clean bank retains substantial task ability and language quality;
full restoration NOT established. Nearly equal NLL of masks conceals substantial
MMLU difference.7-attention is the stronger tested MMLU composition; not universally
better (Wino slightly lower). No speed/memory advantage established by quality
scores. No controlled attribution to QKVO alone: geometry, initialization,
normalization, objective and data differ from historical lineage. Next candidate
is broad-data end-to-end recovery/distillation from this frozen source, with
these results as baseline; this report does not launch or prescribe a new run.
