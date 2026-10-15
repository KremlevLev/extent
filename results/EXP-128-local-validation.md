# EXP128 engineering validation —16 October2026

PREPARED, no physical1.7B GPU training result.

- Four CPU tests passed in47.38s: small hard-capped components remain unchanged
  beside huge outliers; tanh value/zero-gradient/identity checks; both streamed
  full-chain gradients match monolithic autodiff; RMS/peak/profile gates;
  actual clean timeout/Adam restore continues2→4 accepted updates.
- Two additional BF16 cases passed in15.98s: source identity token-NLL≤.01,
  full-strength hard/tanh profile and component/RMS bounds within2% rounding,
  finite full-chain gradients.
- Plan-only generated both128 contracts without GPU/patient loading.
- All six frozen125–127 contracts compared equal against their committed plan
  after JSON canonicalization. Existing modules were not modified.
- Component bounds are stricter than RMS-only bounds at the same numeric
  ceiling. Neither same-seed screening nor an improved exposed C panel would
  establish independent confirmation or a universal cause of the plateau.

Physical T4 memory, throughput, number of updates in a session and quality
remain pending. Old sessions retain their original code/contracts on resume.
