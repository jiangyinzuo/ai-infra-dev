#!/bin/bash
pushd $HOME
# 如果当前$HOME目录已经有vimrc，则git pull拉取最新
# git clone https://github.com/jiangyinzuo/vimrc.git
pushd vimrc
./install/setup.sh
./install/openeuler-ascend.sh
popd
popd
