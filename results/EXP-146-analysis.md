# EXP146: second preparation seed reaches NLL near6

Both GPU workers stopped normally at work deadline after11.30h, exit0.

| arm | accepted new updates | new input tokens | final A | final B | final C |
|---|---:|---:|---:|---:|---:|
| FULL |13089|3350784|5.976876|6.002024|5.614221|
| PROJECTIONS |13171|3371776|6.118156|6.122296|5.740983|

These are arbitrary time-stop endpoints at different update counts, descriptive.
Registered A-selected best is10240 both: FULL6.097788/6.098570/5.753431 vs
PROJ6.222167/6.150277/5.836028. At10240 this matched control favors FULL on all
panels; earlier PROJ often wins, including8192. Both worsen at12288 and improve
again by final endpoint. Do not claim monotonic decrease or a proven new floor.

Model/preparation seed456 recovers to approximately the same level as136 on123.
Data/reset RNG are shared, and panels reused, so scope this as sensitivity
to preparation seed. Not independent-corpus replication or teacher recovery.
New input tokens exclude inherited prep. All source/final profile checks pass;
no unique activation mechanism established. No meaningful context-length gain.

User report final remote verification is pending (last confirmed12672). Live
read-only HF snapshot229e32d5c6fb2674cf68e239ab7a9845e6277de9 has final manifests
13089/13171. Optional additional listing/sample check hit TLS EOF; full checkpoint
restore remains unverified. Retry-work deadline had blocked save-reserve reads;
transport-only fix now separates source/work and remote-verification/save limits.
20 retry/artifact tests pass; original numerical/runner contracts unchanged.
No new cloud job launched. Raw JSON bytes and live manifest audit are archived.
