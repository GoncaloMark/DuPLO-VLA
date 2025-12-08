#!/bin/bash

TASKS=(
    "pick-place"
    "door-open"
    "drawer-open"
    "button-press"
    "reach"
)

NUM_EPISODES=90  # Per task
ROOT_DIR="data/"

echo "=================================="
echo "Generating Multi-Task Dataset"
echo "=================================="
echo "Tasks: ${TASKS[@]}"
echo "Episodes per task: ${NUM_EPISODES}"
echo "=================================="

for TASK in "${TASKS[@]}"
do
    echo ""
    echo "Generating data for: $TASK"
    echo "----------------------------------"
    
    python scripts/generate_dataset.py \
        --env_name $TASK \
        --num_episodes $NUM_EPISODES \
        --root_dir $ROOT_DIR
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated $TASK"
    else
        echo "✗ Failed to generate $TASK"
        exit 1
    fi
done

echo ""
echo "=================================="
echo "Multi-task dataset generation complete!"
echo "=================================="
echo ""
echo "Generated datasets:"
for TASK in "${TASKS[@]}"
do
    ZARR_PATH="${ROOT_DIR}metaworld_${TASK}_expert.zarr"
    if [ -d "$ZARR_PATH" ]; then
        echo "  ✓ $ZARR_PATH"
    fi
done
echo ""