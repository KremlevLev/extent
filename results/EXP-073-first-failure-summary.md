# EXP-073 — first engineering failure

Reviewed 2026-09-08. Status: stopped before the first optimizer update; the scientific gate is unconsumed.

The supplied full artifact has SHA-256 `8f14c7bc63131d2a8ad428cc87f5ac6fe6c2964b9cad547461408f78e9c6ac5a`; its compact summary has SHA-256 `a576eccc2703ede61c2b80adc1374a25a437f015a2ddf0b31af9b5d78286312e`. Runtime was `0.064536` TPU hours at revision `5c3dc46`.

Only the seed-123 `TEACHER` step-0 evaluation and checkpoint completed (`NLL 21.987584`, teacher NLL `2.815887`). The first update raised JAX's donated-buffer alias guard: composed student parameters reused unchanged Qwen device arrays that were also passed as frozen teacher parameters. The train step donates student buffers, so the same physical buffer cannot simultaneously appear in the teacher argument.

This is an ownership error, not numerical instability or an experimental result. The fix makes a non-aliasing sharding-preserving copy of the composed student tree before optimizer creation. Values, endpoint hashes, data, objective, arm order and scientific contract remain unchanged. The already uploaded step-0 state is resumable; no EXP-072 work must repeat.
