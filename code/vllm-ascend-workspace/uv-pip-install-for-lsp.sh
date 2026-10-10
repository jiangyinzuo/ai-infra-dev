#!/bin/bash
uv pip install -r vllm/requirements/common.txt -r vllm-ascend/requirements.txt --excludes requirements-exclude-lsp.txt  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --system-certs
