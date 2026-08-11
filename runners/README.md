# Modal CI runners

Self-hosted GitHub Actions runners for this repo, deployed as
[runner-modal](https://github.com/modal-projects/runner-modal) pools:
GitHub `workflow_job` webhook → ephemeral Modal Sandbox → one-shot JIT runner.

## Pools

| Pool | App | Labels | Jobs |
|---|---|---|---|
| `ci` | `nordlys-ci` | `self-hosted, modal, ci` | `nordlys-test.yml` linux legs (py 3.11/3.12/3.13) |
| `ci_core` | `nordlys-ci-core` | `self-hosted, modal, ci-core` | `nordlys-core-test.yml` linux legs (conan/cmake; conan cache Volume at `/cache`) |
| `ci_gpu` | `nordlys-ci-gpu` | `self-hosted, modal, ci-gpu` | `nordlys-test.yml` linux-gpu leg (T4, `--extra cu12`) |

One Modal App per pool (one resource profile per `Runner.create`). Workflow
jobs pin a unique `job-<run_id>-<job>-<index>` label so each JIT runner is
claimed by exactly one job leg.

## One-time setup

1. Fine-grained PAT: owner `Nordlys-Labs`, access to `nordlys` only,
   **Administration: Read & write** (needed for `generate-jitconfig`).
2. Secrets — never merge these two into one Secret (see runner-modal SECURITY.md):

   ```sh
   modal secret create github-token GITHUB_TOKEN=github_pat_xxx
   modal secret create github-webhook WEBHOOK_SECRET=$(openssl rand -hex 32)
   ```

3. Deploy all pools:

   ```sh
   uv sync
   uv run modal deploy cpu_app.py
   uv run modal deploy core_app.py
   uv run modal deploy gpu_app.py
   ```

4. Webhooks — one per pool, same secret, events = **Workflow jobs** only,
   content type `application/json`:

   ```sh
   uv run python -c "from runner_modal import Runner; \
     print('ci     ', Runner.from_name('ci').url); \
     print('ci_core', Runner.from_name('ci_core').url); \
     print('ci_gpu ', Runner.from_name('ci_gpu').url)"
   ```

   Payload URL is `{url}/github`.

5. Repo security (public repo — fork PR code runs on Modal):
   Settings → Actions → *Require approval for all outside collaborators* and
   read-only default workflow permissions.

## Ops

- Health: `curl {url}/health` (runner name, active/max concurrency).
- Ignored deliveries return HTTP 204 with an `X-Runner-Modal-Reason` header.
- Logs: `modal app logs nordlys-ci` (webhook), sandbox logs for the job itself.
- GitHub-side runner registration is ephemeral by design — runners appear only
   while a job executes.
- Teardown a pool: `uv run python -c "from runner_modal import Runner; Runner.objects.delete('<name>')"`,
  then remove its repo webhook and `modal app stop <app>`.

## Cost knobs

- Per-pool: `cpu`/`memory`/`gpu`, soft `max_concurrent`, `idle_timeout` (sandbox
  linger), `min_containers`/`buffer_containers` (webhook warm pool, kept at 0/1).
- `ci_core` conan cache: bump nothing if `/cache` is cold — first build is slow,
  steady state restores the `~/.conan2` duration.
