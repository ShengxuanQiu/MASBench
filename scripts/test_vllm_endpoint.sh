#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
MODEL="${MODEL:-local-mas-model}"
URL="${URL:-http://127.0.0.1:${PORT}/v1/chat/completions}"
LOG_HINT="${LOG_HINT:-/data/home/shegnxuanqiu/mas-serving/logs/vllm_server.log}"

# 本地 vLLM endpoint 必须直连，避免被 http_proxy/https_proxy 转发到外部代理。
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

echo "测试 endpoint：${URL}"
echo "模型名：${MODEL}"

TMP_BODY="$(mktemp)"
TMP_ERR="$(mktemp)"
trap 'rm -f "${TMP_BODY}" "${TMP_ERR}"' EXIT

HTTP_CODE="$(
  curl -sS \
    -o "${TMP_BODY}" \
    -w "%{http_code}" \
    -X POST "${URL}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"请用一句话说明你可以作为本地模型服务使用。\"}
      ],
      \"temperature\": 0.2,
      \"max_tokens\": 128
    }" 2>"${TMP_ERR}" || true
)"

if [[ "${HTTP_CODE}" =~ ^2 ]]; then
  echo "请求成功，HTTP ${HTTP_CODE}"
  cat "${TMP_BODY}"
  echo
else
  echo "请求失败，HTTP ${HTTP_CODE}" >&2
  echo "curl 错误：" >&2
  cat "${TMP_ERR}" >&2
  echo "响应内容：" >&2
  cat "${TMP_BODY}" >&2
  echo >&2
  echo "请检查服务日志：${LOG_HINT}" >&2
  exit 1
fi
