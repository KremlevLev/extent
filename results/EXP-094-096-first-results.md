# EXP-094--096: first submitted summaries

Reviewed on 2026-10-03 from the three user-submitted `extent-m3q-subspace*summary.txt` files.
This entry records summary evidence only; full JSONs and numerical failure traces
have not been inspected. All three campaigns report `branch_failure`, scientific
gate false, and no pending HF upload. Reported cumulative invocation times are
4.400h /3.811h /4.380h respectively.

## Observations

- The shared rank8 input/output CE control is identical across all three jobs:
  seed123 completes8192 steps and reduces locked-test NLL12.968006 ->7.349788;
  seed456 fails after4227 steps. These are duplicate controls, not three replications.
- EXP094: output-only correction fails in both seeds. MLP-readout corrections
  complete but worsen test NLL to19.867574 /19.137341. Head gains give mixed changes:
  seed123 12.968006 ->13.221407; seed456 12.276975 ->11.855398.
- EXP095: seed123 teacher-KD and self-anchor complete, giving test NLL7.983263
  and8.881876, both worse than its CE control. Their seed456 branches fail before
  the final endpoint. Delta-then-KD fails at steps90 /43; no final comparison exists.
- EXP096: rank32 CE completes in both seeds, with test NLL12.968006 ->7.643549
  and12.276975 ->9.342761. Rank64 fails in both seeds. Protected rank32 fails at4024
  for seed123, but completes for seed456 and gives NLL7.315458 versus ordinary
  rank32's9.342761. This single-seed benefit does not establish the registered claim.

## Interpretation and limits

Rank32 input/output corrections show recovery at both available source seeds,
unlike the earlier scalar-gain intervention. However, the capacity campaign's
primary protected-vs-unprotected comparison is incomplete, and all registered
campaign gates fail. No claim of full Qwen recovery, new initialization superiority,
or inference speed follows. Numerical instability is a major unresolved issue;
its cause cannot be diagnosed from these summaries alone. Failed endpoints must
not be replaced by their best intermediate validation values.

The next research decision requires full failure diagnostics and unchanged-start
teacher metrics; no additional experiment or runtime fix is authorized by this
documentation-only request.
