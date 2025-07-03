#!/bin/bash

# You can choose to use multi-node training by changing the --nnodes and echo the configuration below. For demo purposes, we use 1 GPU on a single node.
#echo "Using configuration:"
#echo "  MASTER_ADDR:   $MASTER_ADDR"
#echo "  MASTER_PORT:   $MASTER_PORT"
#echo "  WORLD_SIZE:    $WORLD_SIZE"
#echo "  PET_NPROC_PER_NODE: $PET_NPROC_PER_NODE"

torchrun --nproc_per_node=1 \
  --nnodes=1 \
  distributed.py
