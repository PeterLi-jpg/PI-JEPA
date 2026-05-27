#!/usr/bin/env bash
# =============================================================================
# Install OPM Flow + pyopmspe11 on a Ubuntu instance (Brev L40S).
#
# Required by scripts/generate_newwell_dataset.py for the New-Well
# Generalization dataset (reviewer-requested, dataset #5 of the 5-dataset
# resubmission plan).
#
# Run this ONCE on the Brev instance when you want to generate the
# new-well dataset. It is idempotent — re-running is safe.
#
# Why this lives as a separate script (not in setup.sh):
#   - OPM is only needed for ONE of the 5 datasets
#   - It pulls in ~500 MB of C++ deps via apt
#   - Skipping it lets the main `setup.sh` stay lean for the four
#     datasets that don't need a reservoir simulator
#
# Usage on Brev:
#   ssh into the L40S, cd into the repo, then:
#       bash scripts/setup_opm_flow.sh
#   Verify:
#       flow --version
#       python -c "import pyopmspe11; print(pyopmspe11.__version__)"
# =============================================================================

set -euo pipefail

echo "=== OPM Flow + pyopmspe11 installer ==="

# Detect Ubuntu (this script does not support macOS/other OS)
if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: apt-get not found. This script is Ubuntu-only."
    echo "On macOS, install Docker Desktop and pull openporousmedia/opm-flow"
    echo "instead, or run this on the Brev L40S instance."
    exit 1
fi

# ----- 1. Add OPM PPA + install opm-simulators (provides `flow` binary) -----
if command -v flow >/dev/null 2>&1; then
    echo "  flow already installed: $(flow --version 2>&1 | head -1)"
else
    echo "  installing software-properties-common (for add-apt-repository)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common
    echo "  adding ppa:opm/ppa ..."
    sudo add-apt-repository -y ppa:opm/ppa
    sudo apt-get update -qq
    echo "  installing opm-simulators (this is the slow part, ~5-10 min)..."
    sudo apt-get install -y -qq mpi-default-bin opm-simulators
    echo "  flow installed: $(flow --version 2>&1 | head -1)"
fi

# ----- 2. Install pyopmspe11 (Python driver) in the repo .venv -----
VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: .venv/bin/python not found. Run bash setup.sh first."
    exit 1
fi
if "$VENV_PY" -c "import pyopmspe11" 2>/dev/null; then
    echo "  pyopmspe11 already installed in .venv"
else
    echo "  installing pyopmspe11 (from GitHub — not on PyPI)..."
    "$VENV_PY" -m pip install --quiet git+https://github.com/OPM/pyopmspe11.git
    echo "  pyopmspe11 installed."
fi

# ----- 3. Final verification -----
echo ""
echo "=== Verification ==="
echo -n "  flow:         "
flow --version 2>&1 | head -1 || echo "FAIL"
echo -n "  pyopmspe11:   "
"$VENV_PY" -c "import pyopmspe11; print(pyopmspe11.__version__)" || echo "FAIL"
echo ""
echo "Ready to run: python scripts/generate_newwell_dataset.py ..."
