"""
Unit and regression test for the dpolfit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import dpolfit


def test_dpolfit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dpolfit" in sys.modules
