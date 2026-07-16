"""Project-controlled MLX-LM plugin registration for DS4 fine-tuning.

This module deliberately avoids editing installed site-packages.  When imported
from the editable `python-envs/mlx` environment it appends the project-owned
vendor directory to `mlx_lm.models.__path__`, making
`import mlx_lm.models.deepseek_v4` resolve to this repository's scaffold.
"""

from __future__ import annotations

import importlib
import sysconfig
from pathlib import Path


def plugin_models_path() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "mlx_lm_models"


def install_deepseek_v4_plugin() -> bool:
    """Register the vendored MLX-LM model path if MLX-LM is importable.

    Returns True when the path is present after registration and the vendored
    module resolves to this repository.  Returns False when MLX-LM is absent or
    the project-controlled module file is missing. Unexpected MLX-LM import
    errors are intentionally allowed to surface.
    """

    vendor_path = plugin_models_path()
    module_path = vendor_path / "deepseek_v4.py"
    if not module_path.is_file():
        return False

    try:
        import mlx_lm.models as models  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return False

    path = str(vendor_path)
    current = list(getattr(models, "__path__", []))
    if path not in current:
        current.append(path)
        models.__path__ = current

    module = importlib.import_module("mlx_lm.models.deepseek_v4")
    resolved = Path(getattr(module, "__file__", "")).resolve()
    return module_path.resolve() == resolved


def install_startup_pth_hook(site_packages: str | Path | None = None) -> Path:
    """Install a reproducible `.pth` startup hook for editable MLX envs.

    A host/global `sitecustomize.py` may shadow project-local sitecustomize
    modules.  A `.pth` import hook is the reliable editable-install mechanism
    for this plugin; setup-env writes it explicitly instead of relying on manual
    site-packages edits.
    """

    target_dir = Path(site_packages) if site_packages is not None else Path(sysconfig.get_paths()["purelib"])
    target_dir.mkdir(parents=True, exist_ok=True)
    hook = target_dir / "ds4_ft_mlx_mlx_lm_plugin.pth"
    hook.write_text("import ds4_ft_mlx.mlx_lm_plugin as _p; _p.install_deepseek_v4_plugin()\n", encoding="utf-8")
    return hook
