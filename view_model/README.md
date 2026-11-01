uv venv --python 3.12 --seed --managed-python
source gitee-proxy.sh

```bash
euporie-preview --run --save linear.ipynb
jupyter lab --ip=0.0.0.0 --custom-css
```
