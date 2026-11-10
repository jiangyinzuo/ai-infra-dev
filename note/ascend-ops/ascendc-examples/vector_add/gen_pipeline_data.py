#!/usr/bin/env python3
import os
import numpy as np


NUM_BLOCKS = 8
BLOCK_LENGTH = 16 * 1024
TOTAL_LENGTH = NUM_BLOCKS * BLOCK_LENGTH


def generate(seed=20261028):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, TOTAL_LENGTH).astype(np.float32)
    y = rng.uniform(1, 10, TOTAL_LENGTH).astype(np.float32)
    golden = ((x + y) * np.float32(1.25)).astype(np.float32)
    os.makedirs("pipeline_input", exist_ok=True)
    os.makedirs("pipeline_output", exist_ok=True)
    x.tofile("pipeline_input/input_x.bin")
    y.tofile("pipeline_input/input_y.bin")
    golden.tofile("pipeline_output/golden.bin")


if __name__ == "__main__":
    generate()
