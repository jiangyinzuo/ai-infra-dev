#!/usr/bin/env -S gawk -f
# Usage:
#   LC_ALL=C ./log_time_range.awk \
#     -v start='2026-10-07 12:50:58.000000' \
#     -v end='2026-10-07 12:51:00.000000' trace.log > window.log
# Selects [start, end). Each line must begin with YYYY-MM-DD HH:MM:SS.ffffff.
# Assumes chronological order; use -v unordered=1 to scan without early exit.
# Reads stdin if no file is supplied. For multiple files, filters each separately.

BEGIN {
    timestamp = "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{6}$"
    if (start !~ timestamp || end !~ timestamp) {
        print "error: set -v start and -v end using YYYY-MM-DD HH:MM:SS.ffffff" > "/dev/stderr"
        exit 2
    }
    if (start >= end) {
        print "error: start must be earlier than end" > "/dev/stderr"
        exit 2
    }
}

{
    t = substr($0, 1, 26)
    if (t !~ timestamp || t < start)
        next
    if (t >= end) {
        if (!unordered)
            nextfile
        next
    }
    print
}
