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

## Open-world surfaces (feat/crucible-open-world)

* **Delegation is scoped and revocable.** Installing a capability creates `id.grant`s from the installer to the
  service principal with the scopes the descriptor declares, bound to resources the installer owns; uninstalling
  revokes them (invariant checked). A service can act on another organisation's resources only within those grants
  (tested: a service with `cloud.manage` on one instance cannot resize another).
* **Brokered identity.** A sandboxed program's RPCs always run as its own service principal (`Ctx.call(...,
  brokered=True)` is used only by the broker), so callers cannot lend it their authority and it cannot act as them.
* **Institutional rules are enforced at authorisation** (`identity.authorize` -> `gov.check`): owners, admins and
  delegates alike are refused governed actions unless a proposal was approved or a delegated budget covers them.
* **Operator tools** (`admin.*`) remain unreachable from agents, services, triggers and flows (tested).
* **Private state of simulated minds** (reservation wages, beliefs, trust) is written only in internal events and
  exposed by no tool (tested).
* **Isolation backends** (`sandbox/backends.py`): `detect()` reports what the host provides; requesting an
  unavailable backend (gVisor, bubblewrap, Firecracker) raises `SandboxUnavailable`; there is no fallback to
  weaker isolation. On this host only the default Landlock/seccomp process sandbox is available - not VM-grade.

## Synthetic internet surfaces (feat/crucible-internet, Crucible 3.0)

* **Operating modes are separate** (`crucible/modes.py`).
  * **Mode A (synthetic training)** is the only mode used for training. The browser resolves only `*.crucible`
    (`MAP * ~NOTFOUND`) and routes every other request to a block. Sandboxed programs have no network.
  * **Mode B (offline grounding)** reads local, read-only reference files. `load_reference` refuses network
    locations.
  * **Mode C (supervised real-world evaluation)** is refused unless the operator sets
    `CRUCIBLE_MODE_C_APPROVAL=<evaluation>:<token>` *and* a human approval callback accepts each action. The
    `RealWorldGateway` only gates; it performs nothing itself.
  * Worlds cannot reach the environment or the gateway, so no in-world policy can approve its own escalation.
  * Tested: `tests/test_internet_adversarial.py::test_operating_modes_are_separated_and_mode_c_is_refused_by_default`,
    `::test_hermetic_browser_cannot_reach_outside_hosts`.
* **Web applications run as their service principal, inside the sandbox.**
  * Route handlers get a request dictionary and return a response. They cannot read other applications' `appdb`
    namespaces.
  * A site learns a visitor's identity only after the visitor consents (`net.login`).
  * Every request is a recorded `net.fetch` command, rate-limited per actor and per site.
* **Adversarial content is data.**
  * Injection pages, a fraudulent capability, spam and false documentation are generated in-world
    (`worldgen/adversarial.py`). None of it is addressed to, or read by, the operator's session.
  * Text an agent reads cannot change verifiers, metrics or the evaluator. Those are evaluator-side objects, and
    metrics read authoritative state only.
  * `evaluation/trust_boundary.assess` reports payments, grants and installs that went to adversary principals,
    and calls of tools that appeared only in injected text.
  * A naive reader policy is caught and a careful one is not. No simulated firm kept the fraudulent capability
    active after 40 days, and spam without verifiable references never became a belief
    (`tests/test_internet_adversarial.py`).
* **Minimum isolation fails closed** (19H). A workload can require an isolation level through
  `CRUCIBLE_MIN_ISOLATION` or `SandboxConfig.min_isolation`; the levels are
  `process < namespaces < user-space-kernel < vm`. If the selected backend does not provide that level, the
  worker refuses to start with `SandboxUnavailable` and names the backends that would. It never falls back to a
  weaker level.
  * On this host only `landlock-seccomp` is available (`process`), so any `vm` requirement is refused (tested in
    `tests/test_open_security.py`).
  * **Crucible does not provide VM-grade isolation here.**
* **Kernel contract found by testing.** An emitted event payload must never be mutated afterwards.
  * The NPC innovator routine first stored a dictionary that it kept mutating. Replaying the log then broke the
    hash chain, so the kernel's integrity check caught it.
  * The routine now stores deep copies, and its test asserts exact replay and regeneration.
  * The kernel does not defensively copy payloads, for speed. New routines must follow the contract.
