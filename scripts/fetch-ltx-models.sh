#!/usr/bin/env bash
# 下载 LTX-2.5 视频模型（ComfyUI 官方 LTX-2.5 工作流所需）
#
# 用法：
#   bash scripts/fetch-ltx-models.sh                    # 只下核心 5 个（约 39.7GB）
#   INCLUDE_OPTIONAL=1 bash scripts/fetch-ltx-models.sh # 连 Prompt Enhancer 一起下
#   MODELS_DIR=/path/to/models bash scripts/fetch-ltx-models.sh
#
# 特性：断点续传、精确字节校验、.part 原子改名、已完整则直接转正。

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/ComfyUI-Shared/models}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
RETRIES="${RETRIES:-8}"

MS="https://www.modelscope.cn/models"
HF="https://hf-mirror.com"

# 字段：目标子目录|仓库|文件路径|精确字节数|必需(1)/可选(0)|源(ms/hf)
FILES=(
  "diffusion_models|Lightricks/LTX-2.5|diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|21504034224|1|ms"
  "text_encoders|Lightricks/LTX-2.5|text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|15372969374|1|ms"
  "vae|Lightricks/LTX-2.5|vae/ltx-2.5-video-vae-bf16.safetensors|1472223346|1|ms"
  "vae|Lightricks/LTX-2.5|vae/ltx-2.5-audio-vae-bf16.safetensors|364866540|1|ms"
  "latent_upscale_models|Lightricks/LTX-2.5|latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|995778752|1|ms"
  "text_encoders|Comfy-Org/gemma-4|text_encoders/gemma4_e2b_it_int8_convrot.safetensors|5199462336|0|hf"
)

human() { awk -v b="$1" 'BEGIN{printf "%.2f GB", b/1e9}'; }

build_url() {
  local repo="$1" path="$2" src="$3"
  if [ "$src" = "hf" ]; then
    echo "$HF/$repo/resolve/main/$path"
  else
    echo "$MS/$repo/resolve/master/$path"
  fi
}

fetch() {
  local sub="$1" repo="$2" rel="$3" expect="$4" src="$5"
  local name dest url tmp have
  name="$(basename "$rel")"
  dest="$MODELS_DIR/$sub"
  url="$(build_url "$repo" "$rel" "$src")"
  mkdir -p "$dest"

  if [ -f "$dest/$name" ]; then
    have=$(stat -f%z "$dest/$name" 2>/dev/null || echo 0)
    if [ "$have" -ge "$expect" ]; then
      echo "✓ 已存在：$sub/$name ($(human "$have"))"
      return 0
    fi
  fi

  tmp="$dest/$name.part"
  if [ -f "$tmp" ]; then
    have=$(stat -f%z "$tmp" 2>/dev/null || echo 0)
    if [ "$have" -ge "$expect" ]; then
      mv "$tmp" "$dest/$name"
      echo "✓ 续传文件已完整，转正：$sub/$name ($(human "$have"))"
      return 0
    fi
    echo "… 发现未完成文件（$(human "$have")），断点续传"
  fi

  echo "↓ $sub/$name  ←  $( [ "$src" = hf ] && echo hf-mirror || echo ModelScope )"
  curl -L --fail --retry "$RETRIES" --retry-delay 3 --retry-all-errors \
       -C - --progress-bar -o "$tmp" "$url" || {
    echo "✗ 下载失败：$name（重跑本脚本会接着下）"
    return 1
  }

  have=$(stat -f%z "$tmp" 2>/dev/null || echo 0)
  if [ "$have" -lt "$expect" ]; then
    echo "✗ 文件不完整（$(human "$have") / $(human "$expect")），保留 .part 供续传"
    return 1
  fi
  mv "$tmp" "$dest/$name"
  echo "✓ 完成：$sub/$name ($(human "$have"))"
}

echo "目标目录：$MODELS_DIR"
echo "可选模型：$([ "$INCLUDE_OPTIONAL" = "1" ] && echo 包含 || echo 跳过)"
echo

fail=0
for entry in "${FILES[@]}"; do
  IFS='|' read -r sub repo rel expect required src <<< "$entry"
  if [ "$required" = "0" ] && [ "$INCLUDE_OPTIONAL" != "1" ]; then
    echo "· 跳过可选：$(basename "$rel")"
    continue
  fi
  fetch "$sub" "$repo" "$rel" "$expect" "$src" || fail=1
done

echo
echo "================== 结果 =================="
for entry in "${FILES[@]}"; do
  IFS='|' read -r sub repo rel expect required src <<< "$entry"
  p="$MODELS_DIR/$sub/$(basename "$rel")"
  if [ -f "$p" ]; then
    printf "  ✓ %-58s %s\n" "$sub/$(basename "$rel")" "$(human "$(stat -f%z "$p")")"
  else
    printf "  ✗ %-58s 缺失%s\n" "$sub/$(basename "$rel")" "$([ "$required" = 0 ] && echo "（可选）")"
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "核心模型就绪。回到 ComfyUI 按 R 刷新，模板里的加载节点会自动识别。"
else
  echo "有文件没下完，重跑本脚本即可续传。"
fi
exit "$fail"
