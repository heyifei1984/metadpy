# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import tempfile


def pytest_configure():
    compiledir = os.path.join(tempfile.gettempdir(), "pytensor")
    mpldir = os.path.join(tempfile.gettempdir(), "mpl")

    os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={compiledir}")
    os.environ.setdefault("MPLCONFIGDIR", mpldir)
