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

At this initial documentation review, full diagnostics were unavailable.

## Full-file addendum (2026-10-03)

The subsequently submitted three full JSON files confirm the same final numbers
and identical rank8 CE checkpoint hashes across campaigns. The first failed
updates are OUT-R8 steps5761/3900; INOUT-R8 seed456 step4228; delta bridge91/44;
teacher KD seed4567358; self-anchor seed4565555; rank64 steps4772/6338;
protected rank32 seed1234025. The recorded last-good norms include1.6374e17
before the rank8 failure and6.3729e17 before the seed123 rank64 failure.

The guard combines loss, gradients, their naive norm, new coordinates and Adam
state, so these files **do not identify the offending quantity**. Even successful
ordinary rank32 endpoints have last norms1.6565e7 /4.1451e10. Their validation
curves oscillate substantially; survival must not be called numerical stability.
Protected rank32 seed456 is smoother after3072 steps, but its other seed fails.

EXP097 is now separately authorized and preregistered to diagnose same-gradient
norm overflow, compare stable clipping, and test reduced learning rate without
silently replacing nonfinite gradients. Original experiment results remain fixed.

Full submitted file SHA256 (subspace /objectives /capacity respectively):

- `25cf4e209db3c8d395047450e24b0a37a3087e086fb2c0a8e3ed5e875057ad88`
- `375280b13cde346924e1a0326a1fcb6958c481da5a49d573d38de968d3902bf3`
- `35061281c1af59decc7a200de31940f1c8a0884c3cec4d2bb4a8dfc569a6b113`
