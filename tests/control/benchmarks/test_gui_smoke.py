import importlib
import sys

import pytest


def _tk_available() -> bool:
    try:
        import tkinter as _tk  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tk_available(), reason="Tkinter not available on this environment")
def test_gui_import_and_mainloop_constructible(monkeypatch):
    """
    Non-interactive smoke: verify the GUI module loads and the MorphogenesisGUI
    class can be constructed without starting the blocking mainloop.

    Skips automatically if Tkinter is not available.
    """
    mod = importlib.import_module("harness.gui.app")
    assert hasattr(mod, "MorphogenesisGUI")

    # Construct and immediately destroy to avoid entering mainloop.
    app = mod.MorphogenesisGUI()
    # Verify core widgets exist
    assert hasattr(app, "_build_controls")
    assert hasattr(app, "_build_plots")
    assert hasattr(app, "_build_metrics_table")
    # Destroy the root window to clean up
    app.destroy()