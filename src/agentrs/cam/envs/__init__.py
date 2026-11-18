"""CAM strategic environments (executable simulators with hidden parameters and verified outcomes)."""
from agentrs.cam.envs.base import NBINS, CamEnv, Grammar, StepResult  # noqa: F401
from agentrs.cam.envs.bizdev import BizDevEnv
from agentrs.cam.envs.negotiation import NegotiationEnv
from agentrs.cam.envs.product import ProductEnv

ENVS = {"negotiation": NegotiationEnv, "product": ProductEnv, "bizdev": BizDevEnv}
