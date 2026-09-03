#!/bin/bash

ROOT="."
PATTERN="apologize. I can only provide safe, medically relevant information about my symptoms"

find "$ROOT" -type f -name "dialogue_history_*.json" | while read -r dialog; do
    if grep -qF "$PATTERN" "$dialog"; then
        dir=$(dirname "$dialog")
        base=$(basename "$dialog")
        id=${base#dialogue_history_}

        judge="$dir/judge_evaluation_$id"

        echo "Deleting:"
        echo "  $dialog"

        rm -f "$dialog"

        if [ -f "$judge" ]; then
            echo "  $judge"
            rm -f "$judge"
        fi
    fi
done