"""Editable-environment startup hooks for DS4 fine-tuning.

Python imports `sitecustomize` automatically when this source tree is installed
editable.  Keep this hook tiny: it only registers the project-controlled
MLX-LM model plugin path, and it silently no-ops when MLX-LM is not installed.
"""

try:
    from ds4_ft_mlx.mlx_lm_plugin import install_deepseek_v4_plugin

    install_deepseek_v4_plugin()
except Exception:
    # Startup hooks must not make unrelated Python commands fail.  Explicit
    # tests cover registration behavior and report actionable failures.
    pass
