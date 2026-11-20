"""Benchmarks that cannot run in this container. Each raises ``DependencyUnavailable`` with the concrete blocker
and the setup procedure, instead of silently substituting another environment. Status was checked on 2026-11-20
against the projects' READMEs / docs (sources in docs/CAM2_ENVIRONMENTS.md)."""
from __future__ import annotations

from agentrs.cam2.envs.base import DependencyUnavailable

REQUIREMENTS = {
    "webarena": (
        "BrowserGym WebArena needs six self-hosted web applications (shopping, shopping-admin, GitLab, Reddit, Wikipedia, "
        "map) served from the WebArena Docker images; this is an unprivileged container without Docker-in-Docker.",
        "On a host with Docker: follow https://github.com/web-arena-x/webarena/tree/main/environment_docker, then\n"
        "  uv pip install browsergym-webarena && python -c 'import nltk; nltk.download(\"punkt_tab\")'\n"
        "  export WA_SHOPPING=... WA_SHOPPING_ADMIN=... WA_REDDIT=... WA_GITLAB=... WA_WIKIPEDIA=... WA_MAP=... WA_HOMEPAGE=...\n"
        "and add a 'webarena' adapter mirroring MiniWoBAdapter (task ids browsergym/webarena.<n>)."),
    "theagentcompany": (
        "TheAgentCompany runs GitLab, Plane, ownCloud and RocketChat servers with Docker Compose (30+ GB, a mounted "
        "docker socket) and every one of its 175 tasks is its own Docker image; no Docker daemon is available here. "
        "Its graders mix deterministic checks and LLM-based evaluators.",
        "On a Docker host: bash <(curl -fsSL https://github.com/TheAgentCompany/the-agent-company-backup-data/releases/download/setup-script-20241208/setup.sh)\n"
        "then run each task image (ghcr.io/theagentcompany/<task>-image) and drive it through its OpenHands-style API;\n"
        "the per-task evaluator (/utils/eval.py) is the official checker."),
    "osworld": (
        "OSWorld executes tasks in full desktop virtual machines. Local providers are VMware, VirtualBox and Docker "
        "(Docker runs a qcow2 VM and is recommended with KVM); this container has no Docker daemon and no KVM. The hosted "
        "providers (AWS, Modal, Daytona) avoid local virtualization but need external cloud accounts, which autonomous "
        "training here does not use.",
        "On a KVM-capable host: git clone https://github.com/xlang-ai/OSWorld && pip install -r requirements.txt;\n"
        "use provider 'docker' (with /dev/kvm) or 'vmware'/'virtualbox', and the official evaluators in desktop_env/evaluators."),
    "gym-anything": (
        "Gym-Anything (cmu-l3/gym-anything, MIT) launches environments through Docker (general case), QEMU + Apptainer "
        "(rootless Slurm) or Apple Virtualization (macOS); none is available here (no Docker daemon, no KVM, no Slurm). "
        "Most tasks are verified by a Python verifier.py; some use a VLM checklist.",
        "On a Docker host: git clone https://github.com/cmu-l3/gym-anything && uv pip install -e '.[all]' && gym-anything doctor;\n"
        "tasks are folders with task.json / setup / export scripts and verifier.py (programmatic; optional VLM checklist)."),
    "webgym": (
        "WebGym (microsoft/webgym, code MIT) has ~292k tasks over 127,645 real websites and scores episodes with "
        "rubric-based LLM evaluators, not deterministic state-based verifiers. Its project pages describe no sandboxed "
        "copy of the sites, so training on it means interacting with the live web, which is outside this project's "
        "autonomous-training policy (allowlisted, controlled environments only).",
        "If approved: clone https://github.com/microsoft/webgym (branch 'webgym'; tasks: huggingface.co/datasets/microsoft/webgym_tasks),\n"
        "restrict browsing to an explicit allowlist of read-only sites, and report LLM-judge scores separately from state-based ones."),
}


def make_unavailable(kind: str, kwargs: dict):
    if kind in REQUIREMENTS:
        why, setup = REQUIREMENTS[kind]
        raise DependencyUnavailable(f"{kind} is not available: {why}\nSetup:\n{setup}")
    raise ValueError(f"unknown environment {kind!r}")
