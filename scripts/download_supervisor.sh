#!/usr/bin/env bash
# Resilient gdown supervisor: restarts gdown if it dies, with exponential
# backoff between retries. Logs to /tmp/<label>_resume.log.
#
# Usage:  scripts/download_supervisor.sh <label> <gdrive_folder_id> <output_dir>
#
# Example:
#   scripts/download_supervisor.sh ccsnet 1SVZFkaxkAIjcGKew3rzGTmKW5tSBUGf7 data/ccsnet_resume
#
# Designed to be launched with nohup so it survives shell exit:
#   nohup scripts/download_supervisor.sh ccsnet 1SVZFkaxkAIjcGKew3rzGTmKW5tSBUGf7 data/ccsnet_resume > /dev/null 2>&1 &

LABEL="$1"
FOLDER_ID="$2"
OUTPUT_DIR="$3"

if [ -z "$LABEL" ] || [ -z "$FOLDER_ID" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <label> <gdrive_folder_id> <output_dir>" >&2
    exit 1
fi

GDOWN="/Users/anqipeterli/Desktop/Research/PI-JEPA/.venv/bin/gdown"
LOG="/tmp/${LABEL}_supervisor.log"
SLEEP=10
MAX_SLEEP=300
MAX_ATTEMPTS=50

mkdir -p "$OUTPUT_DIR"
echo "[$(date)] supervisor START label=$LABEL output=$OUTPUT_DIR" > "$LOG"

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "[$(date)] attempt $attempt/$MAX_ATTEMPTS — launching gdown" >> "$LOG"
    "$GDOWN" --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$OUTPUT_DIR" >> "$LOG" 2>&1
    RC=$?
    echo "[$(date)] attempt $attempt finished with rc=$RC" >> "$LOG"

    if [ $RC -eq 0 ]; then
        echo "[$(date)] supervisor SUCCESS" >> "$LOG"
        exit 0
    fi

    # Exponential backoff with cap
    echo "[$(date)] sleeping ${SLEEP}s before retry" >> "$LOG"
    sleep "$SLEEP"
    SLEEP=$((SLEEP * 2))
    if [ $SLEEP -gt $MAX_SLEEP ]; then
        SLEEP=$MAX_SLEEP
    fi
done

echo "[$(date)] supervisor GIVING UP after $MAX_ATTEMPTS attempts" >> "$LOG"
exit 1
