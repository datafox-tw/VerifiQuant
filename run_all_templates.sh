#!/bin/bash

# Log file for batch processing status
LOG_FILE="batch_processing_status.log"
echo "Starting batch processing at $(date)" > "$LOG_FILE"

# Find all python files in the templates directory
TEMPLATE_DIR="finchain/data/templates"
FILES=$(find "$TEMPLATE_DIR" -name "*.py")

# Function to run command with timeout
run_with_timeout() {
    local duration=$1
    shift
    # Remove 's' suffix if present for perl compatibility
    local duration_sec=${duration%s}
    
    if command -v gtimeout &> /dev/null; then
        gtimeout "$duration" "$@"
    elif command -v timeout &> /dev/null; then
        timeout "$duration" "$@"
    else
        # Perl fallback
        # alarm triggers SIGALRM (14), resulting in exit code 142 (128+14)
        perl -e 'alarm shift; exec @ARGV' "$duration_sec" "$@"
    fi
}

echo "Starting batch processing..."

# Loop through each file
for file in $FILES; do
    # Construct expected output path to check for existence
    # Assumes structure: .../templates/<domain>/<topic>.py -> data/<domain>/<topic>.json
    DIR=$(dirname "$file")
    DOMAIN=$(basename "$DIR")
    FILENAME=$(basename "$file")
    TOPIC="${FILENAME%.*}"
    JSON_OUTPUT="data/$DOMAIN/$TOPIC.json"

    if [ -f "$JSON_OUTPUT" ]; then
        echo "Skipping $file (Output $JSON_OUTPUT already exists)"
        echo "$(date) - File: $file - Status: SKIPPED (Exists)" >> "$LOG_FILE"
        echo "----------------------------------------"
        continue
    fi

    echo "Processing $file..."
    
    # Run the python script with a timeout (e.g., 300 seconds = 5 minutes)
    run_with_timeout 300 python3 preprocessing/finchain_def_to_jsoncard.py --input "$file"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        STATUS="SUCCESS"
        echo "Successfully processed $file"
    elif [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 142 ]; then
        STATUS="TIMEOUT"
        echo "Timeout processing $file"
    else
        STATUS="FAILURE (Exit Code: $EXIT_CODE)"
        echo "Failed to process $file"
    fi

    # Log the result
    echo "$(date) - File: $file - Status: $STATUS" >> "$LOG_FILE"
    echo "----------------------------------------"
done

echo "Batch processing complete. Check $LOG_FILE for details."
