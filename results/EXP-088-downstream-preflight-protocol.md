# EXP-088 downstream-sensitive recovery engineering preflight

Purpose: establish a compilable candidate-only KL training step through a frozen
hybrid suffix before assigning a multi-hour scientific recovery budget.

The suffix reuses production decoder parameters, checkpoints each decoder layer,
and starts from a cached prefix hidden state. Only the two chosen Mamba subtrees
are differentiated as parameters and receive optimizer slots. Frozen downstream
layers still differentiate with respect to their inputs. The target is detached
full-vocabulary next-token logits; KL is computed in FP32 at temperature 1.

The default probe initializes the 1.7B architecture directly in device shards,
uses context 32 and three updates on the earliest two Mamba coordinates. Targets
are synthetic uniform distributions, not Qwen teacher outputs. This is a resource
and numerical check only, not evidence for a transplant method. No checkpoint
downloads or old artifact uploads are required.

Output includes compile duration, compiler buffer accounting (not observed peak
HBM), synchronized step durations and finite-gradient metrics. Local tiny tests
verify suffix/full-model forward parity and updates to both candidate subtrees
while frozen parameter values remain unchanged.

The later scientific comparison must restore identical pretrained endpoints,
compare local segment-MSE against downstream KL on fresh train/calibration/test
ranges, and report actual compute: downstream backward is not equal-FLOP local
training. No claim or long-campaign budget is fixed from synthetic losses.
