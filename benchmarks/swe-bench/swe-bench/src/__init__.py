"""SWE-bench benchmark with Nordlys Router."""

from .nordlys_provider import NordlysProvider, register_nordlys_provider

__all__ = ["NordlysProvider", "register_nordlys_provider"]
