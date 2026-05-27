#!/usr/bin/env bash
# =============================================================================
# Bundle a focused-paper output directory into a portable tarball for
# easy export off Brev (or any remote machine).
#
# Usage:
#   bash scripts/export_results.sh outputs_focused/v1 [export_dir]
#
# Bundles:
#   - all *.json files (results, configs, metrics)
#   - *.png/*.pdf figures
#   - all checkpoint_final.pt (one per pretrain)
#   - checkpoint_latest.pt (mid-run safety snapshot; useful if a run
#     crashed and you want to resume)
#   - all *.yaml configs that were materialized for each variant
#
# Drops the heavy stuff:
#   - checkpoint_best.pt (we keep checkpoint_final.pt which mirrors it)
#   - optimizer state, logs, intermediate epoch snapshots
#
# Outputs a single .tar.gz to <export_dir>/<basename>_<timestamp>.tar.gz
# so you can `brev copy` or `scp` it back in one shot, or upload to
# Google Drive / Dropbox.
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <output_root> [export_dir=./exports]"
    exit 1
fi

OUTPUT_ROOT="$1"
EXPORT_DIR="${2:-./exports}"

if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "ERROR: $OUTPUT_ROOT does not exist"
    exit 1
fi

mkdir -p "$EXPORT_DIR"
BASENAME=$(basename "$OUTPUT_ROOT")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="$EXPORT_DIR/${BASENAME}_${TIMESTAMP}.tar.gz"

echo "Bundling $OUTPUT_ROOT into $ARCHIVE ..."
echo "  this preserves: JSON results + figures + checkpoint_final.pt"
echo "                  + checkpoint_latest.pt (mid-run safety)"
echo "                  + variant configs (.yaml)"
echo "  this excludes:  checkpoint_best.pt (mirrored by _final), logs"
echo ""

# Build a list of files to include
TMP_LIST=$(mktemp)
trap "rm -f $TMP_LIST" EXIT

find "$OUTPUT_ROOT" -type f \( \
       -name '*.json' \
    -o -name '*.png' \
    -o -name '*.pdf' \
    -o -name 'checkpoint_final.pt' \
    -o -name 'checkpoint_latest.pt' \
    -o -name '*.yaml' \
    \) -print > "$TMP_LIST"

N=$(wc -l < "$TMP_LIST" | tr -d ' ')
if [ "$N" = "0" ]; then
    echo "ERROR: no exportable files found under $OUTPUT_ROOT"
    exit 1
fi

echo "  including $N files"

# Compute total size before tarring so user knows what to expect
TOTAL_KB=$(du -ck $(cat "$TMP_LIST") 2>/dev/null | tail -1 | awk '{print $1}')
echo "  uncompressed size: $((TOTAL_KB / 1024)) MB"

tar -czf "$ARCHIVE" -T "$TMP_LIST"
ARCHIVE_MB=$(du -k "$ARCHIVE" | awk '{print int($1 / 1024) "." int(($1 % 1024) * 10 / 1024) " MB"}')

echo ""
echo "=== Done ==="
echo "  Archive: $ARCHIVE ($ARCHIVE_MB)"
echo ""
echo "To pull off Brev to your Mac:"
echo "  brev copy pi-jepa-train:~/PI-JEPA/$ARCHIVE ~/Downloads/"
echo ""
echo "Or use scp / rsync if you have the SSH config Brev wrote."
