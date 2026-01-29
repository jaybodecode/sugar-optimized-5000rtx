#!/bin/bash
# Quick script to check GPU memory from training environment

cd /home/jason/GITHUB/SugarV3/SuGaR

echo "Running inline memory check..."
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, '/home/jason/GITHUB/SugarV3/SuGaR/TOOLS')
from inline_memory_check import check_memory

check_memory(show_full_summary=True)
EOF
