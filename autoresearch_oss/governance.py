"""Tenant scope, audit log, and simple API-key RBAC for the platform."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiPrincipal:
    name: str
    role: str
    scopes: frozenset[str]


def parse_rbac_env(raw: str | None = None) -> dict[str, ApiPrincipal]:
    """Format: name:secret:scope1,scope2  (scope * = all). Multiple entries separated by ;"""
    raw = raw if raw is not None else os.environ.get("AUTORESEARCH_PLATFORM_RBAC", "")
    out: dict[str, ApiPrincipal] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) < 3:
            continue
        name, secret, scopes_s = bits[0], bits[1], ":".join(bits[2:])
        scopes = frozenset("*" if s.strip() == "*" else s.strip() for s in scopes_s.split(",") if s.strip())
        role = "admin" if "*" in scopes else "operator"
        out[secret] = ApiPrincipal(name=name, role=role, scopes=scopes)
    return out


def authorize(principal: ApiPrincipal | None, method: str, path: str) -> bool:
    if principal is None:
        return method == "GET" and path in ("/health", "/", "/static/app.js", "/static/styles.css")
    if "*" in principal.scopes:
        return True
    if method == "GET" and "read" in principal.scopes:
        return True
    if method in ("POST", "PUT", "PATCH", "DELETE") and "write" in principal.scopes:
        return True
    return False


def record_audit(
    conn,
    *,
    tenant_id: str,
    actor: str,
    action: str,
    detail: dict | None = None,
) -> None:
    conn.execute(
        "INSERT INTO audit_log(ts, tenant_id, actor, action, detail) VALUES (?,?,?,?,?)",
        (time.time(), tenant_id, actor, action, json.dumps(detail or {})),
    )
    conn.commit()


def list_audit(conn, tenant_id: str, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, tenant_id, actor, action, detail FROM audit_log "
        "WHERE tenant_id=? ORDER BY rowid DESC LIMIT ?",
        (tenant_id, limit),
    ).fetchall()
    return [
        {
            "ts": r[0],
            "tenant_id": r[1],
            "actor": r[2],
            "action": r[3],
            "detail": json.loads(r[4] or "{}"),
        }
        for r in rows
    ]
