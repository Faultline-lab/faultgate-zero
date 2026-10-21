"""faultgate-zero: agent failure checks that run inside your existing pytest tests."""
__version__ = "0.0.0"

from .patch import install, uninstall  # noqa: E402
from .recorder import FaultgateZeroWarning, Run, current_run, run  # noqa: E402

__all__ = ["FaultgateZeroWarning", "Run", "__version__", "current_run", "install", "run", "uninstall"]
