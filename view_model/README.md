uv venv --python 3.12 --seed --managed-python
source gitee-proxy.sh

```bash
# 安装 ipykernel
pip install ipykernel

# 注册当前环境，取一个清晰的名字，例如 "view_model"
python -m ipykernel install --user --name view_model --display-name "Python (view_model)"
euporie-preview --run --save linear.ipynb --kernel view_model
```
