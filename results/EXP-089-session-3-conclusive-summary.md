# EXP-089 — session 3, conclusive registered-gate failure

Reviewed 2026-09-22 from user-supplied full and compact artifacts.

- Status: `deadline_partial`; session duration `7.7883` hours.
- Seed 123 is complete for both arms at the registered 98,304-step / 25,165,824-token endpoint.
- `RANDOM`: excess NLL `9.763291`, prediction KL `14.612423`.
- `EXACT-LIFT`: excess NLL `15.541040`, prediction KL `20.611743`.
- Exact-minus-random: `+5.777749` excess NLL, `+5.999321` prediction KL. Positive is worse.
- Seed 456 exact lift reached step 51,008; its paired random control has not started. The session's 14 HF checkpoint uploads all passed and the partial state is durable.

The pre-registered gate requires exact lift to beat random in both metrics at *both* seeds. Seed 123 has completed and fails both. No possible seed-456 result can reverse the conjunctive gate. Stop EXP-089 for registered-gate futility; completing seed 456 would consume TPU without changing the decision. This does not prove that every exact-lift variant fails universally, only that the tested initializer does not deliver the promised long-horizon full-model advantage under this locked recipe. There is no crossover at any matched seed-123 registered checkpoint.

The later partial seed-456 exact trajectory is not used to infer a seed-456 paired effect. Do not reinterpret its intermediate low loss at step 3,072 as a positive endpoint.

Full artifact SHA-256: `9b6db63b0a6c749fb70878d64e9bdbc048e6e9ddd82c121601ff1b6ec00018d0`.
Run revision: `00ac38b3d3f853c487ecdd5457621b84dfb9dbcc`.
