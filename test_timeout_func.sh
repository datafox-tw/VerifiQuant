#!/bin/bash
run_with_timeout() {
    local duration=$1
    shift
    local duration_sec=${duration%s}
    
    if command -v gtimeout &> /dev/null; then
        gtimeout "$duration" "$@"
    elif command -v timeout &> /dev/null; then
        timeout "$duration" "$@"
    else
        perl -e 'alarm shift; exec @ARGV' "$duration_sec" "$@"
    fi
}

echo "Testing timeout..."
run_with_timeout 2 sleep 5
EXIT_CODE=$?
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 142 ]; then
    echo "Timeout works!"
else
    echo "Timeout failed!"
fi
