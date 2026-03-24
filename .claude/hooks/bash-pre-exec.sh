#!/bin/bash
# File: /home/barattts/lavoltabuona/BA/.claude/hooks/activate-env.sh

if [ -n "$CLAUDE_ENV_FILE" ]; then
    # Your exact venv path
    VENV_PATH="/home/barattts/lavoltabuona/BA/.venv/env"
    
    # Force this venv to the front of Claude's PATH
    echo "export PATH=\"$VENV_PATH/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
    echo "export VIRTUAL_ENV=\"$VENV_PATH\"" >> "$CLAUDE_ENV_FILE"
fi
exit 0