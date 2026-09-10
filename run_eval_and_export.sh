#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Server host: run evaluation inside Docker, then export results
# ============================================================

CONTAINER="${CONTAINER:-456dd0f41cca}"

CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace/unsloth/unsloth_test_data}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval_vlm_server_answer_only.py}"

CONTAINER_RESULTS="${CONTAINER_RESULTS:-/workspace/results}"
HOST_RESULTS="${HOST_RESULTS:-/home/user/workspace/swkim/unsloth_results}"

mkdir -p "${HOST_RESULTS}"

echo "[1/3] Docker evaluation start"
docker exec "${CONTAINER}" bash -lc \
  "cd '${CONTAINER_WORKDIR}' && python3 '${EVAL_SCRIPT}'"

echo "[2/3] Evaluation completed successfully"

# docker cp의 SOURCE 뒤에 /. 를 붙이면 results 폴더 자체가 아니라
# 내부 파일/폴더만 HOST_RESULTS 아래로 복사됩니다.
echo "[3/3] Exporting results from container to server host"
docker cp "${CONTAINER}:${CONTAINER_RESULTS}/." "${HOST_RESULTS}/"

# 로컬 자동 동기화 측에서 완료 시점을 확인할 수 있는 marker.
date -Is > "${HOST_RESULTS}/.export_complete"

echo
echo "DONE"
echo "Server result directory:"
echo "  ${HOST_RESULTS}"
echo "Completion marker:"
echo "  ${HOST_RESULTS}/.export_complete"
