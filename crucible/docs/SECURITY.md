# Security and isolation

Crucible executes code written by learning agents. That code is untrusted. The research infrastructure (kernel,
evaluator, reward definitions, generators, other agents' memory, training state) must stay out of its reach, and
nothing an agent does may touch the real world.

## 1. Threat model

| Asset | Adversary | Protection |
|---|---|---|
| Host filesystem, credentials (`/root`, `/workspace`, `/etc/environment`, env vars) | agent programs in the sandbox | unprivileged uid, Landlock filesystem allow-list, empty environment |
| Real network | agent programs | seccomp: no `socket`/`connect`/`bind`; Landlock ABI ≥ 4 denies TCP bind/connect as a second layer |
| Container process budget (`pids.max` 2816) | fork bombs, thread storms | seccomp blocks `fork`/`vfork`/non-thread `clone`; `RLIMIT_NPROC` per uid; worker pool size cap |
| CPU, memory, disk | runaway programs | `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`, `RLIMIT_NOFILE`, no core dumps, per-job wall-clock timeout with process-group kill |
| Simulation kernel and authoritative state | agents (tools, programs) | agents only issue tool calls; programs reach the world only through the brokered `crucible_sdk.call`, authorised as the program's principal and recorded like any command |
| Operator powers | agents | `admin.*` tools are rejected for every in-world principal (kernel check, tested) |
| Evaluator, locked test partition, reward definitions | agents, training code | evaluator state (sustain counters, goal verifiers) lives outside the world; locked test goals require an acknowledgement string and are access-logged; agents never see evaluator-only metrics or hidden parameters |
| Other agents' private data | agents | event visibility (`vis`) on every event; `World.visible_events` filters by principal; private docs/mail/tickets return `not_found`; tested |
| Money, goods, compute | agents | double-entry ledger (sum of balances = 0), authorisation on every debit, escrow accounts not payable directly, conserved inventory, oversubscription checks, invariant checks after every test |
| Researchers | prompt injection in simulated content | page/content text is data; the web frontend escapes all user content (XSS test); nothing in the kernel interprets text as instructions |

## 2. The sandbox (`crucible/src/crucible/sandbox/`)

`runner.Worker` starts `python -I -S -B boot.py <workdir> <config>` in a new session with rlimits set in the
parent's `preexec_fn`. `boot.py` (standard library only, ~250 lines, auditable) then, before any user code runs:

1. drops privileges: `setgroups([])`, `setgid`, `setuid` to `nobody` (refuses to continue as root);
2. sets `PR_SET_NO_NEW_PRIVS`;
3. applies a **Landlock** ruleset: read+execute on the Python standard library (excluding `site-packages`), the
   approved packages (`numpy` only), and system libraries; read-write only on the job's working directory;
   `/dev/null`, `/dev/urandom`; TCP bind/connect handled (denied) on ABI ≥ 4;
4. loads a **seccomp** filter (libseccomp via ctypes): EPERM for `socket`, `socketpair`, `connect`, `bind`,
   `listen`, `accept*`, `execve`, `execveat`, `fork`, `vfork`, non-thread `clone`, `ptrace`, `process_vm_*`,
   `mount`, `umount2`, `pivot_root`, `chroot`, `unshare`, `setns`, keyrings, `bpf`, `perf_event_open`,
   `userfaultfd`, `io_uring_*`, kernel-module and reboot syscalls, identity changes; ENOSYS for `clone3` (so
   threads still work through glibc's fallback);
5. replaces `sys.path`, installs the in-sandbox SDK and serves JSON-lines jobs on a pipe. User `print` output is
   captured, never mixed with the protocol.

Measured on this host (`tests/test_sandbox.py`): uid 65534; reading `/root`, `/workspace`, `/etc/environment`
→ PermissionError; `socket()` → PermissionError; `subprocess`, `os.system`, `os.fork`, `multiprocessing` →
blocked; writing outside the workdir → PermissionError; importing non-approved packages → ModuleNotFoundError;
3 GiB allocation → MemoryError; infinite loop → killed at the wall-clock limit; threads work.

If Landlock or seccomp is unavailable, `Worker.start` raises `SandboxUnavailable` and the code/cloud services
return `service_unavailable` with an explicit reason; weak isolation requires an explicit `allow_weak=True`
world parameter and is never the default.

**Limitations.** This is not a VM or gVisor boundary: the sandbox shares the host kernel, so a kernel exploit
reachable through the allowed syscalls would escape it. The container has no user namespaces, so
bubblewrap/nsjail-style mount namespaces are unavailable. CPU accounting in the default cost model is measured
time (noisy); in the deterministic cost model native-code work (e.g. numpy) is undercounted (see
LIMITATIONS.md). Sandboxed programs can read the approved Python/system library files (not secrets).

### In-process ("trusted") execution

For speed, the incumbents' *genesis* code (reference implementations written by world generation) runs
in-process. Trust is a property of **code provenance**, not of the organisation: a repository stays trusted only
while every commit to it was written by world generation itself (`code.commit_files(..., trusted=True)` from the
system context). Any commit through the public tools - including an NPC firm adopting a licensed agent-written
repository - permanently marks it untrusted, and its deployments run in the sandbox
(`tests/test_economy.py::test_agent_code_never_runs_in_process_even_when_adopted_by_trusted_firm`). This rule was
added after the lifetime demonstration showed an NPC firm adopting agent code (it ran in-process before the fix).

## 3. Network containment

The synthetic internet has no sockets at all: services are kernel endpoints; agent-deployed services are
sandboxed workers reached by the kernel; programs call the world through the broker. The HTTP/MCP/OpenEnv
servers bind 127.0.0.1 by default and authenticate every actor with a bearer token (hashed at rest). The browser
mode maps `*.crucible` to the local server and aborts every other request (tested). No component performs real
payments, sends real email, or contacts external services during autonomous training.

## 4. Protected research infrastructure

* Kernel code, evaluator code, world generators and reward definitions are Python modules outside any agent's
  reach (agents have no filesystem access to them and no tool that modifies them).
* World mechanics changes (new modules) are code changes reviewed by humans (see SYNTHETIC_INTERNET.md §
  "Extending the internet"); agent-created software runs *inside* the synthetic internet only, as a sandboxed
  service with a service principal holding only grants its operator gave it.
* Training controllers and outer learners talk to Crucible through the same API as agents (plus operator
  endpoints that require an operator token on the HTTP server).

## 5. Untrusted content

Content created by agents (posts, mail, docs, code, listings) is never executed by the kernel and never
interpreted as instructions to the operator. LLM-backed actors receive it as observation text; prompt injection
can therefore influence *their* in-world behaviour (a realistic phenomenon), but never the kernel, the ledger or
the evaluator.
