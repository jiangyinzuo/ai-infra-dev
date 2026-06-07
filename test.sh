#!/bin/bash

URL="http://0.0.0.0:8000/v1/chat/completions"

curl -N -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "你的模型名称", 
    "messages": [{"role": "user", "content": "写一首关于人工智能的诗，长一点"}],
    "max_tokens": 200,
    "stream": true
  }'
