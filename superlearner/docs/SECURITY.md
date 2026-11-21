# Security and experimental integrity

Two kinds of generated code exist and neither runs unconfined on the host.

## World code (written by the inner learner)

Code the agent writes runs only inside Crucible, through the world's own tools: it is committed to a repository in
the world and executed by `code.run` / deployed by `cloud.deploy` in Crucible's sandbox (unprivileged uid, Landlock,
seccomp, no network, CPU and memory limits; see `crucible/docs/SECURITY.md`). `ToolLibrary.register` parses code
tools with `ast` and never executes them. The frozen LLM's output is data: it is parsed as JSON or stored as file
content, never evaluated in the learner process.

The world is the agent's whole internet. The learner process has no tool that reaches the real network; the
frozen model is loaded with `local_files_only=True`.

## Research code (written by the Meta-Researcher)

A candidate is a copy of the package in its own directory. Before anything runs:

1. only files under `cam/`, `algo/` and `train/curriculum.py` may be replaced (`create_candidate` refuses others);
2. the hash of all other files must equal the evaluator tree's (`verify`); a changed protected file rejects the
   candidate.

Execution (`meta/workspace.py run_isolated` → `meta/isolate.py`, taken from the protected tree, standard library
only):

| Layer | Effect |
|---|---|
| privileges | `setgroups([])`, `setgid`, `setuid` to `nobody`; `PR_SET_NO_NEW_PRIVS`; refuses to continue as root |
| limits | processes, file size, open files, CPU seconds; wall-clock timeout by the parent; the whole process session is killed afterwards |
| Landlock | read: Python runtime, the candidate's own tree, the Crucible library, the shared demonstration file. write: its own work directory. Nothing else: no home directory, credentials, model weights, other candidates, evaluator results. TCP bind/connect denied |
| seccomp | no INET/INET6/netlink/packet sockets; no ptrace, mount, namespaces, bpf, keyrings, module loading, identity changes |

`probe_isolation` starts a confined process and checks from the inside that reading outside the allowlist, writing
outside the work directory, writing its own source, TCP and UDP sockets and `setuid(0)` all fail, and that the
permitted unix socket works (`tests/test_meta.py`, `superlearner doctor`). If the probe fails,
`MetaResearcher.__init__` raises `IsolationUnavailable`; there is no weaker fallback.

**What a candidate can reach:** two unix sockets — the env hub (experience, rewards) and, if configured, the
frozen-LLM server (text and embeddings).

**What it cannot do, and why:**

| Threat | Defence |
|---|---|
| edit reward definitions or locked tests | they live in protected files (hash-checked) and, for scoring, in the hub process, which runs the evaluator tree's code, not the candidate's |
| inspect hidden evaluation answers / read evaluation seeds | held-out tasks are started by index on the hub; the learner never receives their specs; summaries hide their identity |
| tamper with world state outside permitted operations | the world lives in the hub process; the candidate has only the six-call protocol |
| substitute fabricated metrics | scores are computed from the hub's own results file (root-owned, outside the candidate's allowlist) |
| use more experience than the budget | the hub counts training decisions and refuses new training tasks beyond the budget |
| replace the evaluator | `isolate.py` and the hub are started from the protected tree |
| validate itself with a changed methodology | comparison code and criteria are protected; a candidate cannot change them |

## Missing on this host (documented, not worked around)

* **No VM- or container-grade isolation.** The instance is an unprivileged Docker container: no nested containers,
  no user namespaces, no gVisor/Firecracker. Confinement is uid + rlimits + Landlock + seccomp in the same kernel.
  A kernel exploit would escape it.
* **No GPU inside the confinement.** CUDA does not initialise under the allowlist, so candidate training runs on
  CPU (this is why the Meta-Researcher's inner task uses a small controller).
* **Shared unprivileged account.** Candidate processes and Crucible's own sandbox workers both run as `nobody`;
  Landlock separates their files, but they share the account's process limit.
* **Resource exhaustion is bounded, not prevented**: a candidate can use its CPU and time budget unproductively.
* In-process training (the champion's own runs, `superlearner train`) is not confined: that code is the
  repository's own, not generated.
