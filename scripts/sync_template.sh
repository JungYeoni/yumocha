#!/usr/bin/env bash
# da-template의 .claude/ 설정을 등록된 프로젝트에 동기화한다

set -euo pipefail

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECTS_FILE="$TEMPLATE_DIR/scripts/projects.txt"

# .claude/ 하위 동기화 대상 (CLAUDE.md 제외 — 프로젝트별 커스텀 유지)
SYNC_DIRS=("rules" "agents" "commands" "skills")

# 루트 기준 전체 경로 동기화 대상 (삭제도 전파됨)
SYNC_PATHS=(".github/workflows")

if [ ! -f "$PROJECTS_FILE" ]; then
  echo "❌ projects.txt 없음: $PROJECTS_FILE"
  exit 1
fi

echo "🔄 da-template 동기화 시작"
echo "   템플릿: $TEMPLATE_DIR"
echo ""

success=0
skip=0
fail=0

while IFS= read -r project || [ -n "$project" ]; do
  # 빈 줄, 주석 무시
  [[ -z "$project" || "$project" == \#* ]] && continue

  if [ ! -d "$project" ]; then
    echo "⚠️  건너뜀 (경로 없음): $project"
    ((skip++)) || true
    continue
  fi

  echo "📁 $project"
  for dir in "${SYNC_DIRS[@]}"; do
    src="$TEMPLATE_DIR/.claude/$dir"
    dst="$project/.claude/$dir"

    if [ ! -d "$src" ]; then
      continue
    fi

    mkdir -p "$dst"
    rsync -a --delete "$src/" "$dst/"
    echo "   ✓ .claude/$dir"
  done

  # 루트 기준 경로 동기화 (workflows 등 — 삭제도 전파)
  for path in "${SYNC_PATHS[@]}"; do
    src="$TEMPLATE_DIR/$path"
    dst="$project/$path"

    if [ ! -d "$src" ]; then
      continue
    fi

    mkdir -p "$dst"
    rsync -a --delete "$src/" "$dst/"
    echo "   ✓ $path"
  done

  # Git hooks 동기화
  if [ -d "$project/.git" ]; then
    for hook in "$TEMPLATE_DIR/scripts/hooks/"*; do
      hook_name=$(basename "$hook")
      cp "$hook" "$project/.git/hooks/$hook_name"
      chmod +x "$project/.git/hooks/$hook_name"
    done
    echo "   ✓ .git/hooks"
  fi

  # pre-commit 훅 설치 (git repo이고 설정 파일이 있는 경우)
  if [ -d "$project/.git" ] && [ -f "$project/.pre-commit-config.yaml" ] && command -v pre-commit &>/dev/null; then
    (cd "$project" && pre-commit install 2>/dev/null) || true
    echo "   ✓ pre-commit install"
  fi

  ((success++)) || true
done < "$PROJECTS_FILE"

echo ""
echo "✅ 완료 — 성공: $success | 건너뜀: $skip | 실패: $fail"
