"""cam_live.harness.auth — the independent authorization layer (workstream EA).

Other workstreams use it only through ``contracts.PolicyEngine`` (``PolicyService`` in-process, ``AuthClient`` over
the authorization socket) and ``ExecContext.secrets`` (``PolicyService.secrets_handle()`` / ``AuthClient.secrets``).
See docs/EA/README.md for the integration notes and docs/EA/SECURITY.md for what is enforced by what.
"""
from .audit import AuditLog, AuditVerification, verify_audit_log
from .policy import PolicyService, is_resource_denial
from .schema import DEFAULT_POLICY_PATH, PolicyError, load_policy
from .untrusted import Untrusted, UntrustedUseError, detect_instructions, label_observation, label_result, policy_context
from .vault import ExecutorSecrets, SecretBroker, SecretError, scrub_record

__all__ = ["PolicyService", "is_resource_denial", "AuditLog", "AuditVerification", "verify_audit_log",
           "DEFAULT_POLICY_PATH", "PolicyError", "load_policy", "Untrusted", "UntrustedUseError",
           "detect_instructions", "label_observation", "label_result", "policy_context", "ExecutorSecrets",
           "SecretBroker", "SecretError", "scrub_record"]
