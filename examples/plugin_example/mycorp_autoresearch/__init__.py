"""Example third-party plugin. Install it (`pip install -e examples/plugin_example`) and the new
types become usable in config.json with no change to autoresearch-oss itself:

  "data":    {"format": "mes", "url": "..."}
  "outcome": {"type": "sql_check", "table": "repairs"}
"""
import pandas as pd

from autoresearch_oss.plugins import register


@register("connector", "mes")
def read_mes(spec, wf):
    """Pull episodes from your MES/CMMS. Return the standard episode columns."""
    raise NotImplementedError("replace with your system's API call; must return a DataFrame")


@register("outcome", "sql_check")
def sql_check(spec):
    """Score proposed actions against your own ground truth (a twin, a rules engine, a lookup...)."""
    def fn(df: pd.DataFrame, actions):
        return [1.0 if a == "inspect_pump" else 0.0 for a in actions]      # placeholder
    fn.verifiable = True
    return fn
