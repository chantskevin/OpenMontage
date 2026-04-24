"""pytest config for the render-matrix suite.

These tests actually invoke video_compose end-to-end against
ffmpeg-generated test sources. They're slower than the unit tests
in tests/tools/ and are gated behind explicit pytest markers
(`render_matrix_fast` / `render_matrix_full`) so they don't run on
every PR by default.

Run via:
  make test-renders-fast   # 5 cells, ~2 min
  make test-renders-full   # 12 cells, requires Remotion + ~10 min
"""

from __future__ import annotations

import shutil

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "render_matrix_fast: smoke render-matrix tests (ffmpeg-only, ~2 min)",
    )
    config.addinivalue_line(
        "markers",
        "render_matrix_full: full render-matrix tests (requires Remotion + Node, ~10 min)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip Remotion-requiring tests when the runtime isn't available.

    Cells marked render_matrix_full need npx + a built remotion-composer.
    Skip them gracefully on machines that don't have Node, rather than
    failing in confusing ways."""
    has_node = shutil.which("npx") is not None
    skip_no_node = pytest.mark.skip(reason="Remotion render requires npx (Node.js)")
    for item in items:
        if "render_matrix_full" in item.keywords and not has_node:
            item.add_marker(skip_no_node)
