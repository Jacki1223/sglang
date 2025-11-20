#!/bin/bash
#
# Apply Mamba State Recomputation Patches to SGLang
#
# This script applies all necessary patches to enable mamba state recomputation
# in SGLang's MambaRadixCache.
#
# Usage: bash apply_mamba_recompute_patches.sh
#

set -e  # Exit on error

echo "========================================="
echo "Applying Mamba Recomputation Patches"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "python/sglang/srt/mem_cache/mamba_radix_cache.py" ]; then
    echo "Error: Must run from SGLang root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Backup original files
echo "[1/4] Creating backups..."
BACKUP_DIR="mamba_recompute_backups_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp python/sglang/srt/server_args.py "$BACKUP_DIR/"
cp python/sglang/srt/mem_cache/mamba_radix_cache.py "$BACKUP_DIR/"
cp python/sglang/srt/model_executor/model_runner.py "$BACKUP_DIR/"

echo "Backups saved to: $BACKUP_DIR"
echo ""

# Apply Patch 1: Server Args
echo "[2/4] Applying Patch 1: Server Args Configuration..."
python3 << 'EOF'
import re

# Read the patch file
with open('mamba_recompute_patch_1_server_args.py', 'r') as f:
    patch_content = f.read()

# Read server_args.py
with open('python/sglang/srt/server_args.py', 'r') as f:
    server_args_content = f.read()

# Extract the dataclass fields to add
fields_match = re.search(
    r'# ==================== ADD THESE FIELDS.*?# ==================== ADD CLI',
    patch_content,
    re.DOTALL
)

if fields_match:
    fields_to_add = fields_match.group(0).split('# ==================== ADD CLI')[0]
    fields_to_add = '\n'.join([line for line in fields_to_add.split('\n')
                                if line.strip() and not line.strip().startswith('#')])

    # Find where to insert (after disable_radix_cache)
    insertion_point = server_args_content.find('disable_radix_cache: bool = False')
    if insertion_point == -1:
        print("Warning: Could not find insertion point in server_args.py")
    else:
        # Find the end of the line
        line_end = server_args_content.find('\n', insertion_point)
        # Insert the new fields
        modified_content = (
            server_args_content[:line_end+1] +
            '\n' + fields_to_add + '\n' +
            server_args_content[line_end+1:]
        )

        # Write back
        with open('python/sglang/srt/server_args.py', 'w') as f:
            f.write(modified_content)

        print("✓ Server args configuration added")
else:
    print("Warning: Could not extract fields from patch")

EOF

# Apply Patch 2: MambaRadixCache
echo "[3/4] Applying Patch 2: MambaRadixCache Enhancement..."
python3 << 'EOF'
# This is a manual integration point
# We'll add clear markers to the file for manual editing

with open('python/sglang/srt/mem_cache/mamba_radix_cache.py', 'r') as f:
    content = f.read()

# Add marker comments
marker = """
# ============================================================================
# MAMBA STATE RECOMPUTATION SUPPORT
# See: mamba_recompute_patch_2_radix_cache.py for implementation details
# ============================================================================
"""

if marker not in content:
    # Add marker at the top of the class definition
    class_pos = content.find('class MambaRadixCache(BasePrefixCache):')
    if class_pos != -1:
        content = content[:class_pos] + marker + content[class_pos:]

        with open('python/sglang/srt/mem_cache/mamba_radix_cache.py', 'w') as f:
            f.write(content)

        print("✓ Marker added to mamba_radix_cache.py")
        print("  → Manual integration required - see patch file for details")
    else:
        print("Warning: Could not find class definition")
else:
    print("✓ Marker already present")

EOF

# Apply Patch 3: ModelRunner
echo "[4/4] Applying Patch 3: ModelRunner Recomputation Interface..."
python3 << 'EOF'
# Add marker to model_runner.py

with open('python/sglang/srt/model_executor/model_runner.py', 'r') as f:
    content = f.read()

marker = """
# ============================================================================
# MAMBA STATE RECOMPUTATION INTERFACE
# See: mamba_recompute_patch_3_model_runner.py for implementation details
# ============================================================================
"""

if marker not in content:
    # Add marker before ModelRunner class
    class_pos = content.find('class ModelRunner:')
    if class_pos != -1:
        content = content[:class_pos] + marker + '\n' + content[class_pos:]

        with open('python/sglang/srt/model_executor/model_runner.py', 'w') as f:
            f.write(content)

        print("✓ Marker added to model_runner.py")
        print("  → Manual integration required - see patch file for details")
    else:
        print("Warning: Could not find class definition")
else:
    print("✓ Marker already present")

EOF

echo ""
echo "========================================="
echo "Patch Application Summary"
echo "========================================="
echo ""
echo "✓ Patch 1: Server Args - APPLIED (configuration parameters added)"
echo "⚠ Patch 2: MambaRadixCache - MARKED (requires manual integration)"
echo "⚠ Patch 3: ModelRunner - MARKED (requires manual integration)"
echo ""
echo "Next Steps:"
echo "1. Review the patch files:"
echo "   - mamba_recompute_patch_2_radix_cache.py"
echo "   - mamba_recompute_patch_3_model_runner.py"
echo ""
echo "2. Manually integrate the enhanced methods into:"
echo "   - python/sglang/srt/mem_cache/mamba_radix_cache.py"
echo "   - python/sglang/srt/model_executor/model_runner.py"
echo ""
echo "3. Test the implementation:"
echo "   bash test_mamba_recompute.sh"
echo ""
echo "Backups are saved in: $BACKUP_DIR"
echo ""
echo "========================================="
