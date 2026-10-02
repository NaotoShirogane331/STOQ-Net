# C two-hour development pilot — source only

This directory copies the approved 2026-10-02 pilot's policy and thin adapter.
It does not include the cloud launcher, prepared data, private source bindings,
checkpoints or results, and cannot launch the production campaign by itself.

| Variant | Weight decay | Shared router temperature |
| --- | ---: | ---: |
| C0 | 0.0001 | 1 |
| C-Reg | 0.001 | 1 |
| C-Temp | 0.0001 | 1.25 |

All use seed 13, one complete epoch, batch 64, learning rate 0.001 and
AMP float16, with the original ResearchTrainer and separate state optimizer.
Only the original train/validation scope is allowed; no test/calibration reads,
old-checkpoint warm start, seed selection or metric change.
The 7200-second allowance was for the entire original round, not each variant.
Copying this source grants no additional execution budget.

`pilot_adapter.py` imports the unchanged numerical library and expects an
explicit qualified input signature. Its production construction requires the
original native backend; no substitute backend or private fixture is bundled.
For direct imports put this directory on the Python module search path.

`test_pilot_policy.py` checks the candidate roster, input scope, once-only
execution and bounded child-process lifetime, without model training or GPU.
The process supervision includes a Linux parent-death binding; Windows CPU
tests do not independently validate that Linux-only mechanism.
