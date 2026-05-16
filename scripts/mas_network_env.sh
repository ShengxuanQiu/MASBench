#!/usr/bin/env bash
# MAS 实验环境的网络变量修正脚本。
# 当前服务器的 all_proxy 指向 SOCKS 代理，uv/pip 在解析大量 wheel 时可能长时间无输出。
# 这里显式使用 HTTP/HTTPS 代理，并禁用 all_proxy，避免走不稳定的 SOCKS/IPv6 路径。

export HTTP_PROXY="${HTTP_PROXY:-http://10.7.47.157:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://10.7.47.157:7890}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"

unset ALL_PROXY
unset all_proxy

# 给大 wheel 下载更长的读超时，避免 torch/cudnn/cublas 等包下载中途误判失败。
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
