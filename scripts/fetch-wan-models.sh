#!/usr/bin/env bash
# 下载 Wan 2.2 TI2V-5B 三件套到 ComfyUI 的模型目录
#
# 用法：
#   bash scripts/fetch-wan-models.sh              # 默认装到 ComfyUI 桌面版共享目录
#   MODELS_DIR=/path/to/models bash scripts/fetch-wan-models.sh
#   SOURCE=hf bash scripts/fetch-wan-models.sh    # 换 hf-mirror 源
#
# 特性：断点续传（中断后再跑一次会接着下）、下完校验大小、失败自动重试。

set -uo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/ComfyUI-Shared/models}"
SOURCE="${SOURCE:-modelscope}"
RETRIES="${RETRIES:-5}"

MS_BASE="https://www.modelscope.cn/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/master"
HF_BASE="https://hf-mirror.com/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main"

# 路径|精确字节数（取自 ModelScope 仓库元数据，用于校验完整性）
FILES=(
  "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors|9999658848"
  "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|6735906897"
  "split_files/vae/wan2.2_vae.safetensors|1409400960"
)

if [ "$SOURCE" = "hf" ]; then BASE="$HF_BASE"; else BASE="$MS_BASE"; fi

echo "目标目录：$MODELS_DIR"
echo "下载源  ：$SOURCE ($BASE)"
echo

human() { awk -v b="$1" 'BEGIN{printf "%.2f GB", b/1e9}'; }

download_one() {
  local rel="$1" min="$2"
  local name sub dest tmp url
  name="$(basename "$rel")"
  sub="$(echo "$rel" | sed -E 's#split_files/([^/]+)/.*#\1#')"
  dest="$MODELS_DIR/$sub"
  url="$BASE/$rel"

  mkdir -p "$dest"

  if [ -f "$dest/$name" ]; then
    local have
    have=$(stat -f%z "$dest/$name" 2>/dev/null || echo 0)
    if [ "$have" -ge "$min" ]; then
      echo "✓ 已存在，跳过：$sub/$name ($(human "$have"))"
      return 0
    fi
    echo "… 发现未完成的文件（$(human "$have")），继续断点续传"
  fi

  # 上次跑到 100% 但校验没过留下的 .part：若其实已经完整，直接转正，不重下
  tmp="$dest/$name.part"
  if [ -f "$tmp" ]; then
    local psize
    psize=$(stat -f%z "$tmp" 2>/dev/null || echo 0)
    if [ "$psize" -ge "$min" ]; then
      mv "$tmp" "$dest/$name"
      echo "✓ 续传文件已完整，直接转正：$sub/$name ($(human "$psize"))"
      return 0
    fi
  fi

  echo "↓ 下载 $sub/$name"
  # 先落 .part，完整后再改名，避免中断留下“看起来像好文件”的半截货
  tmp="$dest/$name.part"
  curl -L --fail --retry "$RETRIES" --retry-delay 3 --retry-all-errors \
       -C - --progress-bar -o "$tmp" "$url"
  if [ $? -ne 0 ]; then
    echo "✗ 下载失败：$name（重跑本脚本会接着下）"
    return 1
  fi

  local size
  size=$(stat -f%z "$tmp" 2>/dev/null || echo 0)
  if [ "$size" -lt "$min" ]; then
    echo "✗ 文件偏小（$(human "$size")），可能被截断，保留 .part 供续传"
    return 1
  fi

  mv "$tmp" "$dest/$name"
  echo "✓ 完成：$sub/$name ($(human "$size"))"
}

fail=0
for entry in "${FILES[@]}"; do
  download_one "${entry%%|*}" "${entry##*|}" || fail=1
done

echo
echo "================== 结果 =================="
for f in "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors" \
         "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
         "vae/wan2.2_vae.safetensors"; do
  p="$MODELS_DIR/$f"
  if [ -f "$p" ]; then
    printf "  ✓ %-45s %s\n" "$f" "$(human "$(stat -f%z "$p")")"
  else
    printf "  ✗ %-45s 缺失\n" "$f"
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "全部就绪。回到 ComfyUI 按 R 刷新，或重启 App，模型才会出现在下拉框里。"
else
  echo "有文件没下完，重新执行本脚本即可续传（不会重复下载已完成的）。"
fi
exit "$fail"
