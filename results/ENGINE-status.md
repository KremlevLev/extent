# Telegram /check: read-only session status

One separate process receives commands for the existing notification bot. No
new bot/token, W&B account, web server, inbound ports or training restart.
No JAX imports or accepted updates. Existing sendMessage notifications continue
independently. The listener is silent until the owner requests a status.

## Start on the existing server

```bash
cd /data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core-src
git pull --ff-only
set -a
source ../engine-core.env
set +a
nohup ../engine-core-env/bin/python -u -m engine_status \
  --work-dir "$PWD/../engine-core" \
  > ../engine-core/status-bot.log 2>&1 &
tail -n 10 ../engine-core/status-bot.log
```

The existing training environment already has filelock. The observer only uses
stdlib, filelock, and stdlib-only dashboard helpers; importing engine_dashboard
does NOT import wandb or JAX. If using a new small environment instead, install
requirements-status.txt there. No training environment update is necessary.

Now send the SAME bot in a private chat:

```
/check
/check blackwell
/sessions
/help
```

Default session blackwell reads WORK/results/runtime-b6.json (EXP161). A missing
file is reported as unavailable, not successful completion. The listener remains
running after a training session stops and serves the next report at the same path.

Preview locally without sending or receiving Telegram messages:

```bash
../engine-core-env/bin/python -m engine_status \
  --work-dir "$PWD/../engine-core" --preview
```

## Multiple sessions: one listener and an explicit registry

Create WORK/status-sessions.json OUTSIDE the checkout. Example of two local
work directories (replace the second with the actual session directory):

```json
{
  "sessions": [
    {"id": "blackwell", "work_dir": "/data/workdir/_biometric-old/biometric-linking-guarded-v4/engine-core", "report": "results/runtime-b6.json"},
    {"id": "second", "work_dir": "/absolute/path/to/second-session", "report": "results/runtime-b6.json"}
  ]
}
```

Restart ONLY the listener with --config. Do NOT run a second listener alongside
the default one. Find its PID with `pgrep -af 'python.*-m engine_status'`, inspect
the result and `kill PID` for that listener only. Training processes are untouched.

```bash
nohup ../engine-core-env/bin/python -u -m engine_status \
  --work-dir "$PWD/../engine-core" \
  --config "$PWD/../engine-core/status-sessions.json" \
  >> ../engine-core/status-bot.log 2>&1 &
```

Then /check lists both; /check second selects the second. Config is loaded on
listener startup, not training startup. Up to20 aliases, no credentials in JSON.

For a REMOTE session, use a small report already uploaded to HF instead of
work_dir/report. Example of the registered EXP160 T4 aggregate report path:

```json
{"id": "t4", "repo_id": "levkremlev888/extent", "repo_type": "dataset", "hf_file": "experiments/exp160-corpus-recovery-v1/latest.json"}
```

This is an example of an existing report location, NOT a claim EXP160 is still
running. For another experiment use its actual report repository path. EXP161
remote report is experiments/exp161-batch6-v1/latest.json. Separate simultaneous
runs MUST use different report paths; an overwritten shared path cannot identify
two sessions. No repository scan, checkpoint downloads or Hub writes. HF_TOKEN
is read from the same environment for private reports. Small JSON download happens
only on /check, cached60s. Requests have12s timeout. Last successful snapshot is
retained in memory on a temporary HF error; reported explicitly as stale cache.
No cached snapshot survives listener restart; if unavailable, the reply says so.

## Meaning of the reply

- Status is copied from the report; stale running is NOT a live-process claim.
- Local /proc/locks checks the work directory's trainer lock, read only. This
  verifies a process holds that lock, not GPU progress or the health of each branch.
- Remote process liveness is unknown. File age uses Last-Modified if provided by
  HF; otherwise unavailable. Files older15min are explicitly marked. HF upload
  cadence may be hours even while training is healthy.
- Per-branch CURRENT NLL A/B/C/D has its measured global step and difference from
  the previous observation on the SAME panel. A late older measurement cannot
  replace a newer value. BEST is labelled separately, never substituted for CURRENT.
- EN/RU fresh sets include window count to keep16/128-window panels distinct.
- New tokens are (step-source_step)*batch*256, per branch, only where explicit
  batch/source clocks exist. This is not preparation cost, unique tokens or total
  lineage cost. Current161 and160 input length is256.
- Train CE/KL/loss is distinct from held-out NLL. Speed is mean recent training
  kernel tokens/s excluding startup audit, not end-to-end wall throughput.
- Existing reports have no wall-clock start timestamp; live elapsed/ETA is not
  invented. Final launcher_elapsed_hours is shown if available.

## Safety, collisions, network failures

Private configured chat only: TELEGRAM_CHAT_ID or TELEGRAM_USER_ID. Group commands
and other chats are silently ignored. Command arguments select registered aliases;
they never become file paths, shell commands, URLs or code. No /stop, /exec or
training mutation. Messages are plain text, split below Telegram size limits.

Only ONE getUpdates consumer per bot. Per-user/host token-hash lock prevents
accidental duplicate listeners even in different work directories. Cross-host
listeners are not globally locked; Telegram409 stops this listener rather than
retrying forever and fighting another process. Existing webhooks are detected
and left unchanged; if one exists, integrate /check there instead. Do not run
--show-chat-ids discovery while this listener is polling.

Temporary Telegram errors, including429, retry with exponential backoff, jitter
and retry_after.401/403/409 exit with sanitized errors. getWebhookInfo bootstrap
is retried too. Update cursor is atomic and persistent; messages older5min are
not replayed after downtime. Delivery is at-least-once: interrupted send/cursor
write can duplicate one reply, but cannot affect training. There are no automatic
start/progress notices from this listener. If its host/process is down, /check
cannot answer. No remote deployment or live token test was performed here.

Reference: https://core.telegram.org/bots/api#getupdates
