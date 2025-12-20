#!/bin/bash

TASKS=(
    "pick-place"
    "door-open"
    "drawer-open"
    "button-press"
    "reach"
)

ROOT_DIR="../data/"
AUGMENT_RATIO=0.2  # 20% augmentation
SEED=42

echo "=================================="
echo "Augmenting Multi-Task Datasets"
echo "=================================="
echo "Tasks: ${TASKS[@]}"
echo "Augmentation ratio: ${AUGMENT_RATIO} (${AUGMENT_RATIO}*100%))"
echo "=================================="

for TASK in "${TASKS[@]}"
do
    INPUT_PATH="${ROOT_DIR}metaworld_${TASK}_expert.zarr"
    OUTPUT_PATH="${ROOT_DIR}metaworld_${TASK}_expert_augmented.zarr"
    
    echo ""
    echo "Augmenting: $TASK"
    echo "----------------------------------"
    
    if [ ! -d "$INPUT_PATH" ]; then
        echo "✗ Input dataset not found: $INPUT_PATH"
        continue
    fi
    
    python augment.py \
        --input_zarr $INPUT_PATH \
        --output_zarr $OUTPUT_PATH \
        --augment_ratio $AUGMENT_RATIO \
        --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully augmented $TASK"
    else
        echo "✗ Failed to augment $TASK"
        exit 1
    fi
done

echo ""
echo "=================================="
echo "Augmentation complete!"
echo "=================================="
echo ""
echo "Augmented datasets:"
for TASK in "${TASKS[@]}"
do
    ZARR_PATH="${ROOT_DIR}metaworld_${TASK}_expert_augmented.zarr"
    if [ -d "$ZARR_PATH" ]; then
        echo "  ✓ $ZARR_PATH"
    fi
done
echo ""