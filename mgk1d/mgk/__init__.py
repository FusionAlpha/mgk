"""MGK gyrokinetic eigenvalue solver."""

from importlib.metadata import PackageNotFoundError, version

from ._struct import Struct
from .solver import solve
from .vmec import VMECWout, read_vmec_wout, sample_vmec_fieldline, write_vmec_profile

try:
    __version__ = version("mgk")
except PackageNotFoundError:  # Source tree without installed metadata.
    __version__ = "0.1.2"

__all__ = [
    "Struct", "solve", "__version__", "VMECWout", "read_vmec_wout",
    "sample_vmec_fieldline", "write_vmec_profile",
]
