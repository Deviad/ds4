#!/usr/bin/env python3
"""Sentinel: any invocation is a Slice 1 safety failure."""

from __future__ import annotations

import os
from pathlib import Path

marker = os.environ.get("SENTINEL_MARKER")
if marker:
    Path(marker).write_text("launched\n")
raise SystemExit(91)
