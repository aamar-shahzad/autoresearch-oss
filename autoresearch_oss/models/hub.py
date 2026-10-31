"""Search, inspect and download open models from the Hugging Face Hub -- with the guard rails a
real product needs: disk check, license/gating confirmation, revision pinning, resumable
download, and a manifest registered in the registry.

The Hub client is injectable (`api=`, `downloader=`) so everything is testable offline and so
other hubs (ModelScope, an internal mirror) can be plugged in via the plugin system."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field

DEFAULT_ROOT = os.path.expanduser(os.environ.get("AUTORESEARCH_MODELS", "~/.autoresearch/models"))
MANIFEST = ".autoresearch.json"
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.tiktoken", "*.jinja"]
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "openrail"}


class HubError(RuntimeError):
    pass


@dataclass
class RepoInfo:
    repo: str
    sha: str
    license_id: str
    gated: bool
    private: bool
    size_gb: float
    tags: list = field(default_factory=list)
    pipeline: str = ""
    has_safetensors: bool = True

    @property
    def permissive(self) -> bool:
        return self.license_id.lower() in PERMISSIVE


class HubClient:
    def __init__(self, token: str | None = None, api=None, downloader=None, offline: bool | None = None):
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.offline = bool(os.environ.get("HF_HUB_OFFLINE")) if offline is None else offline
        self._api, self._dl = api, downloader

    # lazily import so the package works without huggingface_hub installed
    @property
    def api(self):
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as e:
                raise HubError('huggingface_hub not installed: pip install "autoresearch-oss[llm]"') from e
            self._api = HfApi(token=self.token)
        return self._api

    @property
    def downloader(self):
        if self._dl is None:
            from huggingface_hub import snapshot_download
            self._dl = snapshot_download
        return self._dl

    def _guard(self):
        if self.offline:
            raise HubError("offline mode (HF_HUB_OFFLINE): cannot reach the Hub")

    def info(self, repo: str, revision: str | None = None) -> RepoInfo:
        self._guard()
        try:
            m = self.api.model_info(repo, revision=revision, files_metadata=True)
        except Exception as e:  # network, 404, auth
            raise HubError(f"cannot read {repo}: {e}") from e
        card = getattr(m, "card_data", None) or getattr(m, "cardData", None) or {}
        get = (lambda k, d=None: card.get(k, d)) if isinstance(card, dict) else (lambda k, d=None: getattr(card, k, d))
        lic = get("license") or next((t.split(":", 1)[1] for t in (m.tags or []) if t.startswith("license:")), "unknown")
        sib = getattr(m, "siblings", None) or []
        st = [s for s in sib if str(getattr(s, "rfilename", "")).endswith(".safetensors")]
        size = sum((getattr(s, "size", 0) or 0) for s in (st or sib))
        return RepoInfo(repo, getattr(m, "sha", "") or "", str(lic).lower(), bool(getattr(m, "gated", False)),
                        bool(getattr(m, "private", False)), round(size / 1024 ** 3, 2),
                        list(m.tags or []), getattr(m, "pipeline_tag", "") or "", bool(st))

    def search(self, query: str, limit: int = 20, task: str = "text-generation") -> list:
        self._guard()
        try:
            res = self.api.list_models(search=query, filter=task, sort="downloads", limit=limit)
        except Exception as e:
            raise HubError(f"hub search failed: {e}") from e
        return [{"repo": m.id, "downloads": getattr(m, "downloads", None), "likes": getattr(m, "likes", None),
                 "gated": getattr(m, "gated", False)} for m in res]

    def download(self, repo: str, root: str = DEFAULT_ROOT, revision: str | None = None,
                 accept_license: bool = False, registry=None, confirm=None) -> dict:
        """confirm: optional callable(info)->bool for interactive license acceptance."""
        info = self.info(repo, revision)
        if info.private:
            raise HubError(f"{repo} is private; refusing to copy it into a shared model store")
        if not info.has_safetensors:
            raise HubError(f"{repo} has no .safetensors weights (pickle-based .bin files are refused for safety)")
        if info.gated and not self.token:
            raise HubError(f"{repo} is gated. Accept the terms on its Hub page, then set HF_TOKEN "
                           f"(huggingface.co/settings/tokens) and retry.")
        if not info.permissive and not accept_license:
            ok = confirm(info) if confirm else False
            if not ok:
                raise HubError(f"{repo} uses license '{info.license_id}', which has conditions. Read it on the Hub, "
                               f"then re-run with --accept-license to proceed.")
        os.makedirs(root, exist_ok=True)
        free = shutil.disk_usage(root).free / 1024 ** 3
        need = info.size_gb * 1.1 + 0.5
        if free < need:
            raise HubError(f"not enough disk in {root}: need ~{need:.1f} GB, have {free:.1f} GB")
        dest = os.path.join(root, repo.replace("/", "__"))
        path = self.downloader(repo_id=repo, revision=info.sha or revision, local_dir=dest,
                               allow_patterns=WEIGHT_PATTERNS, token=self.token)
        files = []
        for dp, _, fn in os.walk(dest):
            for f in fn:
                if f != MANIFEST:
                    fp = os.path.join(dp, f)
                    files.append({"file": os.path.relpath(fp, dest), "bytes": os.path.getsize(fp)})
        man = {"repo": repo, "revision": info.sha, "license": info.license_id, "gated": info.gated,
               "path": path or dest, "bytes": sum(x["bytes"] for x in files), "files": files,
               "downloaded_at": time.time(), "license_accepted": bool(accept_license or info.permissive)}
        json.dump(man, open(os.path.join(dest, MANIFEST), "w"), indent=1)
        if registry is not None:
            man["registry_id"] = registry.put("basemodel", repo.replace("/", "__"),
                                              {k: v for k, v in man.items() if k != "files"})
        return man


def local_models(root: str = DEFAULT_ROOT) -> list:
    out = []
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            mf = os.path.join(root, d, MANIFEST)
            if os.path.exists(mf):
                out.append(json.load(open(mf)))
    return out


def verify_local(manifest: dict) -> list:
    """Return a list of problems (missing/size-changed files). Empty list = intact."""
    bad = []
    for f in manifest.get("files", []):
        p = os.path.join(manifest["path"], f["file"])
        if not os.path.exists(p):
            bad.append(f"missing {f['file']}")
        elif os.path.getsize(p) != f["bytes"]:
            bad.append(f"size changed {f['file']}")
    return bad
