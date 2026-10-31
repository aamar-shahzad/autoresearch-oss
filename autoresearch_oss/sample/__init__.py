"""SAMPLE DATA ONLY. A small synthetic maintenance-diagnostics world used for the quickstart, tests
and docs. It is not part of the product's core: nothing outside `cli quickstart` and the tests
imports it. Real use starts from your own episodes (see docs/data-format.md)."""
from .cnc import *  # noqa: F401,F403
from .live import simulate_live  # noqa: F401
