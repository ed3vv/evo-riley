#!/bin/bash
# Count new cards sorted today

echo "Cards with new images sorted today (2025-12-27):"
echo "================================================"

for dir in sorted/*/; do
    card_name=$(basename "$dir")
    new_count=$(find "$dir" -name "*.png" -type f -newermt "2025-12-27 00:00:00" 2>/dev/null | wc -l)
    total_count=$(find "$dir" -name "*.png" -type f 2>/dev/null | wc -l)

    if [ $new_count -gt 0 ]; then
        printf "%-20s +%-4d new (total: %d)\n" "$card_name" "$new_count" "$total_count"
    fi
done | sort -t+ -k2 -rn
