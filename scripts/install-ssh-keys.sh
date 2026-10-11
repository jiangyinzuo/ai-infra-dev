python3 install-ssh-keys.py \
  --config ~/.ssh/config \
  --ssh /usr/bin/ssh \
  --public-key ~/.ssh/id_ed25519.pub \
  --host-glob 'xiaoqiaoling*' \
  --timeout 120
