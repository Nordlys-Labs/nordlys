"""C++ core CI pool for the nordlys-core conan/cmake jobs.

The job image rebuilds runner-modal's default recipe on Ubuntu 24.04 instead
of Debian 12: nordlys-core uses C++20 <format>, which Debian's gcc 12 lacks
(gcc 13 ships with Ubuntu 24.04's build-essential). On top it adds the build
toolchain (gcc-13, ninja, cmake + conan via pip) and is published over the
`ci_core-job` Named Image that Runner.create builds first; both happen
sequentially at deploy time and last publish wins, so every `modal deploy`
keeps the custom toolchain image.

cache=True mounts one shared Volume at /cache for every job in the pool; the
workflow points CONAN_HOME at it so `conan install --build=missing` is paid
once per dependency bump, not once per run.

Deploy:
    uv run modal deploy core_app.py
"""

import modal
from runner_modal import Runner
from runner_modal.meta import JOB_DEPS

APP_NAME = "nordlys-ci-core"
RUNNER_NAME = "ci_core"

app = modal.App(APP_NAME)

github_secret = modal.Secret.from_name("github-token", required_keys=["GITHUB_TOKEN"])
webhook_secret = modal.Secret.from_name(
    "github-webhook", required_keys=["WEBHOOK_SECRET"]
)

# Mirror of runner-modal's default_image() recipe on Ubuntu 24.04 (gcc 13)
# plus the C++ toolchain. Ubuntu t64 suffixes (time64 transition) for
# liblttng-ust1/libssl3.
core_job_image = Runner.install_actions_runner(
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install(
        "curl",
        "ca-certificates",
        "git",
        "libicu-dev",
        "liblttng-ust1t64",
        "libssl3t64",
        "tar",
        "unzip",
        "zip",
        "build-essential",
        "gcc-13",
        "g++-13",
        "ninja-build",
        "pkg-config",
    )
    .uv_pip_install(*JOB_DEPS)
    .uv_pip_install("cmake>=3.28", "conan>=2")
    .add_local_python_source("runner_modal", copy=True)
    .env({"RUNNER_ALLOW_RUNASROOT": "1", "CC": "gcc-13", "CXX": "g++-13"})
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
