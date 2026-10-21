"""faultgate-zero: agent failure checks that run inside your existing pytest tests."""
__version__ = "0.0.0"

from .patch import install, uninstall  # noqa: E402
from .policy import Policy, PolicyError  # noqa: E402
from .recorder import FaultgateZeroWarning, Run, current_run, run  # noqa: E402
from .runner import AgentCheckFailed, RunResult  # noqa: E402
from .manual import record_model_call  # noqa: E402
from .tools import tool  # noqa: E402

__all__ = ["AgentCheckFailed", "FaultgateZeroWarning", "Policy", "PolicyError", "Run", "RunResult", "__version__",
           "current_run", "install", "record_model_call", "run", "tool", "uninstall"]
