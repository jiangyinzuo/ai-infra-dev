from __future__ import annotations

import argparse
from collections import Counter

from huggingface_hub import HfApi


MODEL_ID = "zai-org/GLM-5.3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print official GLM-5.3 checkpoint tensor metadata without downloading weights."
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only print tensor names starting with this prefix; totals cover the whole checkpoint.",
    )
    args = parser.parse_args()

    api = HfApi(token=False)
    print(f"repository: {MODEL_ID}", flush=True)
    print(f"endpoint: {api.endpoint}", flush=True)
    print("source: Hub safetensors metadata API (main); no weight data requested", flush=True)
    metadata = api.get_sharded_safetensors_metadata(MODEL_ID, timeout=30)

    print(f"shards: {len(metadata.files_metadata)}")
    print(f"checkpoint tensors: {len(metadata.weight_map)}")
    print(f"index metadata: {metadata.metadata}")
    print("Tensor element totals include quantization scales and buffers, not just model parameters.")
    for dtype, numel in sorted(metadata.parameter_count.items()):
        print(f"dtype={dtype}: numel={numel:,}")

    print("\n=== Checkpoint tensors ===")
    selected_tensors = 0
    selected_numel = 0
    storage_bytes = Counter()
    for name, shard in sorted(metadata.weight_map.items()):
        tensor = metadata.files_metadata[shard].tensors[name]
        nbytes = tensor.data_offsets[1] - tensor.data_offsets[0]
        storage_bytes[tensor.dtype] += nbytes
        if not name.startswith(args.prefix):
            continue
        selected_tensors += 1
        selected_numel += tensor.parameter_count
        print(
            f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"numel={tensor.parameter_count:,}, bytes={nbytes:,}, shard={shard}"
        )

    print(f"\nmatched tensors: {selected_tensors:,}")
    print(f"matched tensor elements: {selected_numel:,}")
    print("\n=== Whole checkpoint tensor data bytes (not downloaded) ===")
    for dtype, nbytes in sorted(storage_bytes.items()):
        print(f"dtype={dtype}: bytes={nbytes:,}")
    print(f"total tensor data bytes: {sum(storage_bytes.values()):,}")


if __name__ == "__main__":
    main()
