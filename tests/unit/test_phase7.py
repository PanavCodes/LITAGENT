"""Unit tests for Phase 7: Gradio UI construction, Trace Viewer, and polish.

Per AGENTS.md §4 and PHASES.md Phase 7:
- Tests Gradio app assembly
- Tests trace recording and viewer formatting
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts.view_trace import pretty_print_trace

from litagent.tracing import Tracer
from litagent.ui.app_gradio import build_app


def test_gradio_app_structure():
    """Verify Gradio app builds with expected blocks and elements."""
    # Test that build_app returns a Gradio Blocks instance without launching network port
    app = build_app()
    assert app is not None
    assert hasattr(app, "blocks")


def test_tracer_and_viewer():
    """Verify tracer logs events and view_trace pretty-prints without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tracer = Tracer(run_id="test_run", traces_dir=tmp_path)
        tracer.record(
            node="router",
            start_time=100.0,
            input_summary="Test question",
            output_summary="Route: graph",
        )
        assert tracer.trace_file.exists()

        # Run viewer on generated trace
        pretty_print_trace(tracer.trace_file)
