#!/usr/bin/env python
"""
Download publication-grade benchmark datasets for PI-JEPA revision.

Downloads:
1. SPE10 Model 2 — permeability and porosity (60×220×85, ~18MB compressed)
   Source: SPE Comparative Solution Project
   URL: https://www.spe.org/web/csp/datasets/set02.htm

2. Norne field — full reservoir model with production history
   Source: OPM Project (open-source, GitHub)
   URL: https://github.com/OPM/opm-data

3. Sleipner CO2 storage — 2019 benchmark model
   Source: CO2DataShare (requires manual download due to license agreement)
   URL: https://co2datashare.org/dataset/sleipner-2019-benchmark-model

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --spe10
    python scripts/download_data.py --norne
    python scripts/download_data.py --sleipner  (prints instructions)
"""

import os
import sys
import argparse
import zipfile
import shutil
import urllib.request
import subprocess
from pathlib import Path


DATA_DIR = Path("data")

# =============================================================================
# SPE10 Model 2
# =============================================================================

SPE10_URL = "https://www.spe.org/web/csp/datasets/por_perm_case2a.zip"
SPE10_DIR = DATA_DIR / "spe10"


def download_spe10():
    """Download SPE10 Model 2 permeability and porosity data.

    The dataset contains:
    - File 1: Porosity (60 × 220 × 85) — 1,122,000 values
    - File 2: Kx, Ky, Kz permeability (60 × 220 × 85 each)

    Total compressed size: ~18.5 MB
    """
    SPE10_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SPE10_DIR / "por_perm_case2a.zip"

    # Check if already downloaded
    marker = SPE10_DIR / "spe_perm.dat"
    if marker.exists():
        print(f"  SPE10 data already exists at {SPE10_DIR}")
        return True

    print(f"  Downloading SPE10 Model 2 from {SPE10_URL}...")
    print(f"  (Corrected dataset — por_perm_case2a.zip, ~18.5 MB)")

    try:
        urllib.request.urlretrieve(SPE10_URL, str(zip_path), _progress_hook)
        print()
    except Exception as e:
        print(f"\n  ERROR: Failed to download SPE10 data: {e}")
        print(f"  Manual download: {SPE10_URL}")
        print(f"  Extract to: {SPE10_DIR}/")
        return False

    # Extract
    print(f"  Extracting to {SPE10_DIR}/...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(SPE10_DIR))
    except zipfile.BadZipFile:
        print(f"  ERROR: Downloaded file is not a valid zip. The SPE website may")
        print(f"  require browser download. Please download manually:")
        print(f"    URL: {SPE10_URL}")
        print(f"    Save to: {SPE10_DIR}/")
        zip_path.unlink(missing_ok=True)
        return False

    # Rename extracted files to standard names
    # The zip typically contains files like "spe_phi.dat" and "spe_perm.dat"
    # or similar. List what we got:
    extracted = list(SPE10_DIR.iterdir())
    print(f"  Extracted files: {[f.name for f in extracted if f.is_file()]}")

    # Clean up zip
    zip_path.unlink(missing_ok=True)

    print(f"  SPE10 data ready at {SPE10_DIR}/")
    print(f"  Grid: 60 × 220 × 85 cells (1.122M cells)")
    print(f"  Tarbert formation: layers 1-35 (top 70 ft)")
    print(f"  Upper Ness: layers 36-85 (bottom 100 ft)")
    return True


# =============================================================================
# Norne Field (OPM)
# =============================================================================

NORNE_REPO = "https://github.com/OPM/opm-data.git"
NORNE_DIR = DATA_DIR / "norne"


def download_norne():
    """Download Norne field data from OPM project.

    The OPM opm-data repository contains the full Norne reservoir model
    including grid, petrophysical properties, well schedules, and
    production history. This is a standard benchmark in reservoir simulation.

    Uses sparse checkout to only get the norne/ subdirectory (~50 MB).
    """
    NORNE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    marker = NORNE_DIR / "NORNE_ATW2013.DATA"
    if marker.exists():
        print(f"  Norne data already exists at {NORNE_DIR}")
        return True

    print(f"  Downloading Norne field data from OPM project...")
    print(f"  Repository: {NORNE_REPO}")

    # Try git sparse checkout for efficiency
    opm_data_dir = DATA_DIR / "opm-data"

    try:
        if not (opm_data_dir / ".git").exists():
            # Initialize sparse checkout
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--sparse",
                 NORNE_REPO, str(opm_data_dir)],
                check=True, capture_output=True, text=True
            )
            subprocess.run(
                ["git", "sparse-checkout", "set", "norne"],
                cwd=str(opm_data_dir),
                check=True, capture_output=True, text=True
            )
        else:
            print(f"  Using existing clone at {opm_data_dir}")

        # Copy norne data to our data directory
        norne_src = opm_data_dir / "norne"
        if norne_src.exists():
            # Copy the main DATA file and INCLUDE directory
            for item in norne_src.iterdir():
                dest = NORNE_DIR / item.name
                if item.is_file():
                    shutil.copy2(str(item), str(dest))
                elif item.is_dir() and item.name == "INCLUDE":
                    if dest.exists():
                        shutil.rmtree(str(dest))
                    shutil.copytree(str(item), str(dest))

            print(f"  Norne data ready at {NORNE_DIR}/")
            print(f"  Contains: grid, petrophysical properties, well schedules,")
            print(f"  production history, PVT data")
            return True
        else:
            print(f"  ERROR: norne/ directory not found in cloned repo")
            return False

    except FileNotFoundError:
        print(f"  ERROR: git not found. Install git or download manually:")
        print(f"    git clone --sparse {NORNE_REPO}")
        print(f"    cd opm-data && git sparse-checkout set norne")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: git command failed: {e.stderr}")
        print(f"  Try manual download from: {NORNE_REPO}")
        return False


# =============================================================================
# Sleipner CO2 Storage
# =============================================================================

SLEIPNER_URL = "https://co2datashare.org/dataset/sleipner-2019-benchmark-model"


def download_sleipner():
    """Print instructions for downloading Sleipner data.

    The Sleipner 2019 Benchmark Model is hosted on CO2DataShare and requires
    accepting a license agreement before download. It cannot be automated.

    The dataset includes:
    - Simulation grid (Eclipse format)
    - Petrophysical properties
    - CO2 injection history
    - Time-lapse seismic interpretations
    """
    sleipner_dir = DATA_DIR / "sleipner"
    sleipner_dir.mkdir(parents=True, exist_ok=True)

    marker = sleipner_dir / "README_downloaded.txt"
    if marker.exists():
        print(f"  Sleipner data already exists at {sleipner_dir}")
        return True

    print(f"  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  SLEIPNER DATA — MANUAL DOWNLOAD REQUIRED                   ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  The Sleipner 2019 Benchmark Model requires accepting a     ║")
    print(f"  ║  license agreement on CO2DataShare before download.          ║")
    print(f"  ║                                                              ║")
    print(f"  ║  1. Visit: {SLEIPNER_URL}")
    print(f"  ║  2. Register / log in                                        ║")
    print(f"  ║  3. Accept the data license                                  ║")
    print(f"  ║  4. Download the benchmark model files                       ║")
    print(f"  ║  5. Extract to: {sleipner_dir}/              ║")
    print(f"  ║                                                              ║")
    print(f"  ║  Also available (optional):                                  ║")
    print(f"  ║  - Sleipner 4D Seismic Dataset:                              ║")
    print(f"  ║    https://co2datashare.org/dataset/sleipner-4d-seismic-dataset║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  After downloading, create a marker file:")
    print(f"    echo 'downloaded' > {sleipner_dir}/README_downloaded.txt")
    print()

    return False


# =============================================================================
# Utilities
# =============================================================================

_last_percent = -1


def _progress_hook(block_num, block_size, total_size):
    """Progress bar for urllib downloads."""
    global _last_percent
    if total_size > 0:
        percent = int(block_num * block_size * 100 / total_size)
        percent = min(percent, 100)
        if percent != _last_percent:
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            mb_done = block_num * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [{bar}] {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
            _last_percent = percent


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download publication-grade benchmark datasets for PI-JEPA"
    )
    parser.add_argument("--all", action="store_true",
                        help="Download all available datasets")
    parser.add_argument("--spe10", action="store_true",
                        help="Download SPE10 Model 2 (permeability + porosity)")
    parser.add_argument("--norne", action="store_true",
                        help="Download Norne field data (OPM project)")
    parser.add_argument("--sleipner", action="store_true",
                        help="Print Sleipner download instructions")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Base output directory (default: data/)")
    args = parser.parse_args()

    global DATA_DIR, SPE10_DIR, NORNE_DIR
    DATA_DIR = Path(args.output_dir)
    SPE10_DIR = DATA_DIR / "spe10"
    NORNE_DIR = DATA_DIR / "norne"

    if not any([args.all, args.spe10, args.norne, args.sleipner]):
        print("No dataset specified. Use --all or specify individual datasets.")
        print("Run with --help for options.")
        return

    print("=" * 60)
    print("PI-JEPA Publication Data Downloader")
    print("=" * 60)

    results = {}

    if args.all or args.spe10:
        print("\n[1/3] SPE10 Model 2")
        results['spe10'] = download_spe10()

    if args.all or args.norne:
        print("\n[2/3] Norne Field (OPM)")
        results['norne'] = download_norne()

    if args.all or args.sleipner:
        print("\n[3/3] Sleipner CO2 Storage")
        results['sleipner'] = download_sleipner()

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    for name, success in results.items():
        status = "✓ Ready" if success else "✗ Action needed"
        print(f"  {name:12s} {status}")

    print("\nNext steps:")
    print("  1. python scripts/generate_sgs_corpus.py --n-realizations 10000")
    print("  2. python scripts/generate_spe10_data.py --spe10-path data/spe10/spe_perm.dat")
    print("  3. python scripts/generate_compositional_data.py")
    print("  4. python scripts/run_full_benchmarks.py --n-seeds 5")
    print("=" * 60)


if __name__ == "__main__":
    main()
