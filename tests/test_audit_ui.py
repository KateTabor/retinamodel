"""Tests for the interactive audit interface helpers."""

from retinamodel.audit_ui import (
    KEY_TO_LABEL,
    backend_is_interactive,
)


def test_audit_key_map():
    """Check the human-label keyboard mapping."""

    assert KEY_TO_LABEL == {
        "1": "gray_d",
        "2": "gray_l",
        "3": "red",
        "4": "green",
        "5": "blue",
        "6": "yellow",
    }


def test_backend_detection():
    """Check common interactive and non-interactive backends."""

    assert backend_is_interactive("MacOSX")
    assert backend_is_interactive("TkAgg")
    assert not backend_is_interactive("Agg")
    assert not backend_is_interactive(
        "module://matplotlib_inline.backend_inline"
    )