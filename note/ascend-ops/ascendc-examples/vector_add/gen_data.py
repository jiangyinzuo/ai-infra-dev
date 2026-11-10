#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np


def gen_golden_data_simple(seed=20261028):
    rng = np.random.default_rng(seed)
    input_x = rng.uniform(1, 10, [8, 2048]).astype(np.float32)
    input_y = rng.uniform(1, 10, [8, 2048]).astype(np.float32)
    golden = (input_x + input_y).astype(np.float32)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
