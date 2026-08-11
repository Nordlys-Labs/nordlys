"""CPU CI pool for the nordlys Python test jobs.

One-time setup (see README.md):
    modal secret create github-token GITHUB_TOKEN=github_pat_xxx
    modal secret create github-webhook WEBHOOK_SECRET=$(openssl rand -hex 32)

Deploy:
    uv run modal deploy cpu_app.py

Webhook URL (Settings -> Webhooks, "Workflow jobs" events only):
    python -c "from runner_modal import Runner; print(Runner.from_name('ci').url)"/github
"""

import modal
from runner_modal import Runner

app = modal.App("nordlys-ci")

github_secret = modal.Secret.from_name("github-token", required_keys=["GITHUB_TOKEN"])
webhook_secret = modal.Secret.from_name(
    "github-webhook", required_keys=["WEBHOOK_SECRET"]
)

Runner.create(
    app=app,
    name="ci",
    github_secret=github_secret,
    webhook_secret=webhook_secret,
    repositories=["Nordlys-Labs/nordlys"],
    labels=["self-hosted", "modal", "ci"],
    region="us-east",
    cpu=4.0,
    memory=8192,
    max_concurrent=9,
    min_containers=0,
    idle_timeout=900,
)
