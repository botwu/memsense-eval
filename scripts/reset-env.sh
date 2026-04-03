#!/usr/bin/env bash
# reset-env.sh — 评测前环境重置脚本
# 每次运行完整评测流程前执行，确保干净状态
#
# Usage:
#   bash scripts/reset-env.sh          # 重置并重启
#   bash scripts/reset-env.sh --no-start  # 仅重置，不启动 gateway
set -euo pipefail

MEMSENSE_DIR="${MEMSENSE_DIR:-$HOME/.openclaw/extensions/memsense}"
SCHEMA_SQL="$MEMSENSE_DIR/src/server/db/schema.sql"
DB_HOST="${DB_HOST:-127.0.0.1}"
DB_NAME="${DB_NAME:-memsense}"
NO_START=false

for arg in "$@"; do
  case "$arg" in
    --no-start) NO_START=true ;;
  esac
done

echo "========================================="
echo "  Memsense 评测环境重置"
echo "========================================="

# ---- 1. 停止 memsense 子进程 ----
echo ""
echo "[1/6] 停止 memsense 子进程..."
if [[ -f "$MEMSENSE_DIR/scripts/stop-bash.sh" ]]; then
  bash "$MEMSENSE_DIR/scripts/stop-bash.sh" 2>/dev/null || true
else
  echo "  (stop-bash.sh 不存在，跳过)"
fi

# ---- 2. 停止 openclaw gateway ----
echo ""
echo "[2/6] 停止 openclaw gateway..."
pkill -f "openclaw" 2>/dev/null || true
sleep 2
if pgrep -f "openclaw" > /dev/null 2>&1; then
  echo "  警告: 仍有 openclaw 进程残留，强制杀掉..."
  pkill -9 -f "openclaw" 2>/dev/null || true
  sleep 1
fi
echo "  所有 openclaw 进程已停止"

# ---- 3. 清空数据库数据（保留表结构） ----
echo ""
echo "[3/6] 清空数据库 $DB_NAME..."
psql -h "$DB_HOST" -d "$DB_NAME" -q -c "
  TRUNCATE
    embedding_dlq,
    tag_dlq,
    embedding_jobs,
    tag_jobs,
    memory_events,
    memory_chunk_embeddings,
    memory_chunks
  CASCADE;
" 2>/dev/null && echo "  数据库已清空（表结构保留）" || {
  echo "  TRUNCATE 失败，尝试 DROP + 重建..."
  psql -h "$DB_HOST" -d postgres -q -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$DB_NAME' AND pid <> pg_backend_pid();" \
    > /dev/null 2>&1 || true
  psql -h "$DB_HOST" -d postgres -q -c "DROP DATABASE IF EXISTS $DB_NAME;"
  psql -h "$DB_HOST" -d postgres -q -c "CREATE DATABASE $DB_NAME;"
  if [[ -f "$SCHEMA_SQL" ]]; then
    psql -h "$DB_HOST" -d "$DB_NAME" -q -f "$SCHEMA_SQL" > /dev/null 2>&1
    echo "  数据库已重建 (schema: $SCHEMA_SQL)"
  else
    echo "  错误: schema 文件不存在: $SCHEMA_SQL" >&2
    exit 1
  fi
}

# ---- 4. 清理 OpenClaw 缓存和会话 ----
echo ""
echo "[4/6] 清理 OpenClaw 缓存..."
rm -rf ~/.openclaw/agents/main/qmd/xdg-cache/qmd/index.sqlite 2>/dev/null && echo "  - qmd index.sqlite 已删除" || echo "  - qmd index.sqlite 不存在（跳过）"
rm -rf ~/.openclaw/agents/main/sessions/* 2>/dev/null && echo "  - sessions 已清空" || echo "  - sessions 目录不存在（跳过）"
rm -rf ~/.openclaw/workspace/memory/* 2>/dev/null && echo "  - workspace/memory 已清空" || echo "  - workspace/memory 目录不存在（跳过）"

# ---- 5. 清理评测输出缓存（可选） ----
echo ""
echo "[5/6] 清理评测输出..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "$SCRIPT_DIR/output" ]]; then
  rm -rf "$SCRIPT_DIR/output"/*
  echo "  - output/ 已清空"
fi
if [[ -d "$SCRIPT_DIR/.cache" ]]; then
  rm -rf "$SCRIPT_DIR/.cache"/*
  echo "  - .cache/ 已清空"
fi

# ---- 6. 重启 OpenClaw gateway ----
echo ""
if [[ "$NO_START" == "true" ]]; then
  echo "[6/6] 跳过 gateway 启动 (--no-start)"
else
  echo "[6/6] 启动 OpenClaw gateway..."
  openclaw gateway --force > /dev/null 2>&1 &
  GW_PID=$!

  # 等待 gateway 就绪
  for i in $(seq 1 15); do
    if curl -s http://127.0.0.1:18789/health > /dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  # 验证 memsense 服务
  sleep 3
  if curl -s http://127.0.0.1:8787/health > /dev/null 2>&1; then
    echo "  gateway + memsense 已启动 (gateway PID: $GW_PID)"
  else
    echo "  gateway 已启动，memsense 服务可能需要几秒钟..."
  fi
fi

# ---- 验证 ----
echo ""
echo "========================================="
echo "  验证"
echo "========================================="
COUNTS=$(psql -h "$DB_HOST" -d "$DB_NAME" -t -A -c \
  "SELECT count(*) FROM memory_chunks;" 2>/dev/null || echo "ERR")
if [[ "$COUNTS" == "0" ]]; then
  echo "  ✓ 数据库为空 (chunks=0)"
else
  echo "  ✗ 数据库未清空 (chunks=$COUNTS)"
fi

if [[ "$NO_START" == "false" ]] && curl -s http://127.0.0.1:18789/health > /dev/null 2>&1; then
  echo "  ✓ Gateway 正常运行"
else
  [[ "$NO_START" == "false" ]] && echo "  ✗ Gateway 未响应"
fi

echo ""
echo "环境重置完成。可以开始评测："
echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "  uv run python -m memsense_eval configs/full_pipeline.yaml"
