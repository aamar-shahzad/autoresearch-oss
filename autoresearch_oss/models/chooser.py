"""Terminal model chooser: shows what fits THIS machine and lets the user pick. I/O is injected,
so it is scriptable (`--select N`) and unit-testable."""
from __future__ import annotations

_ICON = {"fits": "OK  ", "tight": "TIGHT", "no": "NO  "}


def table(plans, limit: int = 25, show_all: bool = False) -> list:
    rows = []
    for i, p in enumerate(plans[: (None if show_all else limit)], 1):
        b = p.mode(p.best) if p.best else None
        e = p.entry
        fit = f"{p.best} {b.required_gb:.0f}/{b.available_gb:.0f}GB {_ICON[b.verdict].strip()}" if b else "does not fit"
        flags = []
        if e.gated:
            flags.append("gated")
        if not p.disk_ok:
            flags.append("low disk")
        rows.append((i, e.short, f"{e.params_b:g}B" + (f" ({e.active_params_b:g}B act)" if e.active_params_b else ""),
                     e.license_id, e.commercial_ok, fit, ",".join(flags)))
    return rows


def render(plans, **kw) -> str:
    rows = table(plans, **kw)
    head = ("#", "model", "size", "license", "commercial", "best training mode (need/have)", "flags")
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))] if rows else [1] * len(head)
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    return "\n".join([line(head), line(tuple("-" * w for w in widths))] + [line(r) for r in rows])


def choose(plans, input_fn=input, print_fn=print, limit: int = 20):
    """Return the chosen Plan or None. Accepts a number, a repo/short name, 'a' (show all), 'q'."""
    show_all = False
    while True:
        shown = plans if show_all else plans[:limit]
        print_fn(render(plans, limit=limit, show_all=show_all))
        raw = input_fn("\nPick a model by number or name  [a = show all, q = quit]: ").strip()
        if raw.lower() in ("q", "quit", ""):
            return None
        if raw.lower() == "a":
            show_all = True
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(shown):
            return shown[int(raw) - 1]
        for p in plans:
            if raw in (p.entry.id, p.entry.short):
                return p
        print_fn(f"'{raw}' is not in the list.")
