# EXP-080 — second on-policy coordinate sweep (partial, conclusive negative)

- Status: operational failure after `0.684051` hours at seed 456, forward position 16 / layer 17.
- Failure cause: the local checkpoint was valid, but its Hugging Face commit still returned `HfHubHTTPError` after the complete retry window. This is not a numerical failure.
- Completed registered arm: seed 123 forward changes full-model NLL from `10.916678` to `12.245170` (`+1.328492`, `+12.17%`). Its best registered point is step zero.
- Matched control: seed 123 reverse ends at `12.317666` (`+1.400988`, `+12.83%`). Forward is only `0.072496` NLL better than reverse and neither improves the model.
- Independent adverse trajectory: seed 456 reverse changes NLL from `9.692907` to `14.008691` (`+4.315784`, `+44.53%`) and violates the registered `1.25x` excursion bound.
- Partial seed-456 forward trajectory: `9.692907 -> 11.828470 -> 11.259406` after 0/6/12 coordinates. It had not reached the next registered evaluation when upload failed.
- Scientific decision: the primary gate is already impossible because it requires forward improvement at both seeds and seed 123 forward is complete and worse than its start. Do not spend TPU quota merely to turn this conclusive partial result into a formally complete failed result.
- Mechanistic conclusion: revisiting locally trained Mamba blocks can continue reducing their conditional block objective while worsening the assembled model. Coordinate order alone does not solve composition drift.
- Next intervention: a teacher-guided trust-region sweep must evaluate each proposed block update in the assembled model on a separate calibration split and retain the unchanged block as an explicit zero-step candidate.
- Full artifact SHA-256: `9ea93988d960e566731ba104fbb0a58bd0f3546af04b478ad0fc67a704fb2e92`.
- User summary SHA-256: `3e6acd940d352b8f6d2fb3a1998858c140eef280ef64b411006f509889acb8bc`.
