"""GPU CI pool (T4) for the CUDA-enabled nordlys test leg.

CUDA userspace comes from the cu12 pip wheels (torch/cupy/cuml); the sandbox
only needs the host driver Modal provides, so the default job image is enough.

Deploy:
    uv run modal deploy gpu_app.py
"""

import modal
from runner_modal import Runner

app = modal.App("nordlys-ci-gpu")

github_secret = modal.Secret.from_name("github-token", required_keys=["GITHUB_TOKEN"])
webhook_secret = modal.Secret.from_name(
    "github-webhook", required_keys=["WEBHOOK_SECRET"]
)

Runner.create(
    app=app,
    name="ci_gpu",
    github_secret=github_secret,
    webhook_secret=webhook_secret,
    repositories=["Nordlys-Labs/nordlys"],
    labels=["self-hosted", "modal", "ci-gpu"],
    region="us-east",
    cpu=4.0,
    memory=16384,
    gpu="T4",
    max_concurrent=2,
    min_containers=0,
    idle_timeout=900,
)
