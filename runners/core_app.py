"""C++ core CI pool for the nordlys-core conan/cmake jobs.

The default runner-modal job image has no C++ toolchain, so we extend it with
build-essential, cmake (via pip, upstream new), ninja and conan, and publish it
over the `ci_core-job` Named Image that Runner.create just built. Both happen
sequentially at deploy time; last publish wins, so every `modal deploy` keeps
the custom toolchain image.

cache=True mounts one shared Volume at /cache for every job in the pool; the
workflow points CONAN_HOME at it so `conan install --build=missing` is paid
once per dependency bump, not once per run.

Deploy:
    uv run modal deploy core_app.py
"""

import modal
from runner_modal import Runner

APP_NAME = "nordlys-ci-core"
RUNNER_NAME = "ci_core"

app = modal.App(APP_NAME)

github_secret = modal.Secret.from_name("github-token", required_keys=["GITHUB_TOKEN"])
webhook_secret = modal.Secret.from_name(
    "github-webhook", required_keys=["WEBHOOK_SECRET"]
)

core_job_image = (
    Runner.default_image()
    .apt_install("build-essential", "ninja-build", "pkg-config")
    .uv_pip_install("cmake>=3.28", "conan>=2")
)

Runner.create(
    app=app,
    name=RUNNER_NAME,
    github_secret=github_secret,
    webhook_secret=webhook_secret,
    repositories=["Nordlys-Labs/nordlys"],
    labels=["self-hosted", "modal", "ci-core"],
    region="us-east",
    cpu=4.0,
    memory=8192,
    max_concurrent=4,
    cache=True,
    min_containers=0,
    idle_timeout=900,
)

# Overwrite the default recipe Runner.create published as "ci_core-job".
build_app = modal.App.lookup(APP_NAME, create_if_missing=True)
core_job_image.build(build_app).publish(f"{RUNNER_NAME}-job")
