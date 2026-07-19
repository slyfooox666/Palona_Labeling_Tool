# Palona Depth Prior：黄海老师 Linux/CUDA 部署与整段视频验收

本文用于在 Linux/NVIDIA 机器上部署 `Depth_Prior` 分支，并用整段 CCTV 视频验证：

```text
原始视频
  ├─ vision_pipeline native SAM3 whole/split ─> 完整 Control JSON
  ├─ companion ai-models / DA3 ──────────────> relative-depth sidecar
  └─ 安全转码（必要时）───────────────────────> 浏览器可播放 MP4

视频 + Control JSON + depth sidecar ─────────> Palona Labeling Tool
```

这里有两个必须分开的运行时：

1. **整段视频 SAM3** 复用同级仓库 `vision_pipeline/scripts/vp_cli.py` 的 Meta native whole-video 路径。Palona 只增加安全 wrapper、完整性验证和部署入口，不重写 SAM3。
2. **Depth sidecar** 由 `scripts/depth-preprocess.sh` 调用 companion `ai-models` DA3 服务。开元当前的 `/Users/kaiyuanhu/AI_Runtime` 是个人 Mac/MPS 环境，**不在本仓库中，也不是 Linux/CUDA 一键安装包**。黄老师若要用 CUDA 跑 DA3，需要另行部署兼容的 `ai-models` runtime；只有 compatible runtime 而没有 CUDA backend 时才可显式改用 CPU。完全没有 `ai-models` CLI 时，Depth 命令会明确失败，不会凭空回退。

SAM3 输出 mask；DA3 只提供 estimated relative depth 的空间—时间 cue。`depth_rank` 的 `0` 表示相对近、`1` 表示相对远，**不是米制距离，也不是自动交互标签**。

## 1. 机器与账户前提

建议使用一个普通 Linux 用户完成部署，模型缓存和研究数据不与系统其他账户共享。需要：

- Linux x86-64、NVIDIA GPU、可用的 NVIDIA driver/CUDA；
- 原生 SAM3 所需的 CUDA 版本及相匹配的 CUDA PyTorch；
- Python 3.12 和 `uv`；
- Node.js `>=22.13.0`、pnpm `11.7.0`；
- `git`、`curl`、`jq`、`ffmpeg` 和 `ffprobe`；
- Hugging Face 账户已接受 `facebook/sam3` 的访问条款；
- 已取得 `vision_pipeline` 源码及其依赖访问权限；
- 视频、输出和模型缓存有足够的私有磁盘空间。

Ubuntu/Debian 基础工具可这样安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl jq ffmpeg

ffmpeg -version | head -n 1
ffprobe -version | head -n 1
nvidia-smi
```

不要用 `--allow-non-cuda` 绕过正式 SAM3 验收。该 flag 只用于 wrapper 的合成测试；Meta native whole-video 路径本身仍需要 CUDA。

## 2. 拉取 `Depth_Prior` 分支

首次部署：

```bash
export PALONA_HOME="$HOME/palona"
mkdir -p "$PALONA_HOME"

git clone \
  --branch Depth_Prior \
  --single-branch \
  https://github.com/slyfooox666/Palona_Labeling_Tool.git \
  "$PALONA_HOME/Palona_Labeling_Tool"

export LABELING_ROOT="$PALONA_HOME/Palona_Labeling_Tool"
cd "$LABELING_ROOT"
git branch --show-current
git log -1 --oneline
```

已有 clone 时只做 fast-forward 更新，避免覆盖本地改动：

```bash
cd "$LABELING_ROOT"
git fetch origin Depth_Prior
git switch Depth_Prior
git pull --ff-only origin Depth_Prior
```

仓库中的 LFS 样例不是本次真实视频的依赖。除非已获授权且确实需要，不要运行 `git lfs pull`。

## 3. 安装前端与 Palona Python 环境

先确认 Node 版本：

```bash
node --version
npm --version
```

若 Node 不存在或低于 `22.13.0`，先通过服务器认可的 Node 22 安装方式升级。Node 安装好以后再安装 pnpm：

```bash
if command -v corepack >/dev/null 2>&1; then
  corepack enable
  corepack prepare pnpm@11.7.0 --activate
else
  npm config set prefix "$HOME/.local"
  npm install -g pnpm@11.7.0
  export PATH="$HOME/.local/bin:$PATH"
fi

pnpm --version
cd "$LABELING_ROOT"
pnpm install --frozen-lockfile
```

`corepack: command not found` 不是项目缺文件；直接使用上面的 `npm install -g pnpm@11.7.0` fallback 即可。

安装 `uv` 和项目固定的 Python 3.12 依赖：

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv --version
uv python install 3.12
uv sync \
  --project "$LABELING_ROOT/depth_pipeline" \
  --python 3.12 \
  --frozen \
  --extra dev
```

Palona 的 `depth_pipeline` 环境不安装 SAM3、DA3 或 PyTorch；模型依赖分别留在 `vision_pipeline` 和 companion Runtime 中，以避免冲突。

## 4. 准备 `vision_pipeline` 的 native SAM3 CUDA 环境

把已经取得的 `vision_pipeline` 源码放在任意私有代码目录，然后设置绝对路径。下面假设它与 Labeling Tool 同级：

```bash
export VISION_PIPELINE_ROOT="$PALONA_HOME/vision_pipeline"
export VISION_PIPELINE_PYTHON="$VISION_PIPELINE_ROOT/.venv/bin/python"

test -f "$VISION_PIPELINE_ROOT/scripts/vp_cli.py"
test -f "$VISION_PIPELINE_ROOT/configs/models/sam3.yaml"
```

如果该仓库还没有 Python 环境，可按当前 `vision_pipeline/README.md` 建立隔离环境。CUDA PyTorch wheel 必须与服务器 driver/CUDA 匹配；如果管理员已经准备好环境，不要重复覆盖：

```bash
uv venv --python 3.12 "$VISION_PIPELINE_ROOT/.venv"

# 先按服务器 CUDA 版本安装相匹配的 CUDA PyTorch；不要装成 CPU-only wheel。
uv pip install --python "$VISION_PIPELINE_PYTHON" torch torchvision

uv pip install --python "$VISION_PIPELINE_PYTHON" \
  pillow accelerate "transformers>=5.0.0" kernels huggingface_hub \
  numpy opencv-python-headless pyyaml
uv pip install --python "$VISION_PIPELINE_PYTHON" \
  "git+https://github.com/facebookresearch/sam3.git"
```

如 `vision_pipeline` 自己还有 requirements/锁文件，以它的说明为准。安装后必须在**同一个 Python** 中确认 CUDA：

```bash
"$VISION_PIPELINE_PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

输出必须包含 `cuda available: True`。若为 `False`，先修复 driver/CUDA/PyTorch wheel，不能继续把 CPU 环境当作完整 SAM3 环境。

在 Hugging Face 网页接受 `facebook/sam3` 条款后，交互式登录。不要把 token 写进仓库、命令参数或日志：

```bash
"$VISION_PIPELINE_ROOT/.venv/bin/hf" auth login
"$VISION_PIPELINE_ROOT/.venv/bin/hf" auth whoami
```

建议把共享模型缓存放到用户私有目录，而不是项目工作树：

```bash
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$HF_HOME"
chmod 700 "$HF_HOME"
```

## 5. 设置视频和私有输出目录

开元 Mac 上的原始路径是：

```text
/Users/kaiyuanhu/Desktop/Research/Palona/dataset/2026-07-09_00-00-27.mkv
```

这个路径在 Linux 服务器上不存在。先通过获授权的安全方式把视频放到服务器私有数据盘，再把 `VIDEO` 改成服务器真实路径：

```bash
export VIDEO="/srv/palona-private/dataset/2026-07-09_00-00-27.mkv"
export RUN="/srv/palona-private/outputs/2026-07-09_00-00-27"

umask 077
install -d -m 700 "$RUN" "$RUN/sam3-full"
test -f "$VIDEO"

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration \
  -of json \
  "$VIDEO" | jq .
```

`RUN` 必须在 Git worktree 之外。不要把视频、Control、Depth arrays、权重或生成视频写入 `Palona_Labeling_Tool` 后再提交。

## 6. 整段视频 SAM3：推荐 wrapper 路径

### 6.1 先做 preflight

```bash
cd "$LABELING_ROOT"

./scripts/full-video-preprocess.sh \
  --video "$VIDEO" \
  --control "$RUN/clip.control.json" \
  --output-dir "$RUN/sam3-full" \
  --prompt person \
  --prompt "cashier machine" \
  --require-label person \
  --split-seconds 12 \
  --overlap-seconds 2 \
  --check-only
```

wrapper 默认读取 `VISION_PIPELINE_ROOT` 和 `VISION_PIPELINE_PYTHON`。也可以显式传入：

```bash
--vision-pipeline-root "$VISION_PIPELINE_ROOT" \
--pipeline-python "$VISION_PIPELINE_PYTHON"
```

preflight 会检查视频、`vp_cli.py`、model config、`ffmpeg`、`ffprobe`、Python import 和 CUDA。任何一项失败都应先修复。

### 6.2 运行完整视频

```bash
cd "$LABELING_ROOT"

./scripts/full-video-preprocess.sh \
  --video "$VIDEO" \
  --control "$RUN/clip.control.json" \
  --output-dir "$RUN/sam3-full" \
  --prompt person \
  --prompt "cashier machine" \
  --require-label person \
  --split-seconds 12 \
  --overlap-seconds 2
```

等价的 pnpm 入口是下面这个形式；pnpm 参数直接跟在 script 名后，不要额外插入一个 `--`：

```bash
pnpm run full:preprocess \
  --video "$VIDEO" \
  --control "$RUN/clip.control.json" \
  --output-dir "$RUN/sam3-full" \
  --prompt person \
  --prompt "cashier machine" \
  --require-label person \
  --split-seconds 12 \
  --overlap-seconds 2
```

关键语义：

- `--video-mode whole` 由 wrapper 固定启用；
- `12/2` 表示整段视频内部按 12 秒 chunk、2 秒 overlap 运行和拼接，**不是只处理前 12 秒**；
- native whole 路径按原视频 FPS 处理 mask；
- 这个完整命令**不传 `--max-frames`**；
- overlap 内会去重，并尝试将相邻 chunk 的同类实例 ID 拼接为全局 ID；拥挤/遮挡场景仍需在 UI 用 ID alias 人工修复；
- wrapper 先写临时文件，只有命令成功且起点、终点、FPS、轮廓范围和所需 label 验证通过后，才原子发布 `clip.control.json`；
- stdout/stderr、实际 argv 和 coverage validation 分别保存在 `sam3-full/` 下。

整段推理耗时长是正常现象。可在另一个终端查看日志：

```bash
tail -f "$RUN/sam3-full/full-video-preprocess.log"
```

如果出现 CUDA OOM，保持整段模式，只缩短 chunk：

```bash
./scripts/full-video-preprocess.sh \
  --video "$VIDEO" \
  --control "$RUN/clip.control.json" \
  --output-dir "$RUN/sam3-full" \
  --prompt person \
  --prompt "cashier machine" \
  --require-label person \
  --split-seconds 6 \
  --overlap-seconds 1
```

若已有成功的 `clip.control.json`，wrapper 默认拒绝覆盖。只有确认旧文件可替换时才加 `--force`；原视频永远不会被覆盖。

### 6.3 检查整段覆盖

不要用一次 `jq` 把巨大的 Control JSON 展开到终端；wrapper 已生成小型 coverage manifest：

```bash
jq . "$RUN/sam3-full/full-video-preprocess.validation.json"

jq -e '
  (.frame_count > 0) and
  (.first_frame_index == 0) and
  (.first_timestamp_seconds <= 0.1) and
  ((.video_duration_seconds - .last_timestamp_seconds) <= ((2 / .video_fps) + 0.101)) and
  (.labels | index("person") != null)
' "$RUN/sam3-full/full-video-preprocess.validation.json"

jq -e '(.argv | index("--video-mode")) != null and (.argv | index("whole")) != null' \
  "$RUN/sam3-full/full-video-preprocess.command.json"
jq -e '(.argv | index("--max-frames")) == null' \
  "$RUN/sam3-full/full-video-preprocess.command.json"
```

再确认实际 GPU 环境：

```bash
nvidia-smi
"$VISION_PIPELINE_PYTHON" -c \
  'import torch; print({"cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})'
```

只有 `clip.control.json` 和 validation 都存在、起止时间覆盖整段，才把它交给 UI。短 smoke manifest（例如曾经只处理 8 帧的文件）不能冒充整段 Control；新版 UI 在 cue 覆盖范围外会停止显示旧 mask，而不是让最后一帧 mask 冻结到视频结尾。

## 7. 直接调用 `vp_cli.py` 的兜底命令

wrapper 是推荐入口，因为它包含 preflight、原子输出和 coverage 验证。若需要隔离 wrapper 问题，可直接调用现有 pipeline：

```bash
cd "$(dirname "$VISION_PIPELINE_ROOT")"

"$VISION_PIPELINE_PYTHON" \
  "$VISION_PIPELINE_ROOT/scripts/vp_cli.py" sam3 \
  "$VIDEO" \
  --model-config "$VISION_PIPELINE_ROOT/configs/models/sam3.yaml" \
  --video-mode whole \
  --prompt person "cashier machine" \
  --output-dir "$RUN/sam3-full-direct" \
  --contour-json "$RUN/clip.direct.control.json" \
  --contour-epsilon-px 2 \
  --split-seconds 12 \
  --overlap-seconds 2 \
  --timeit
```

该命令同样不传 `--max-frames`。需要额外 QA 轮廓视频时可加 `--draw-contours`，但会增加时间和磁盘。直接路径不包含 Palona wrapper 的最终 coverage/atomic publish 保护，完成后仍建议用 wrapper 内的 validator 或人工检查起点、终点和 labels。

## 8. 准备 companion `ai-models` / DA3

### 8.1 明确部署边界

本分支只包含 **DA3 client 和 depth-feature builder**，不包含 DA3 权重、CUDA worker 或 `ai-models` 安装器。不要把开元 Mac 上 `/Users/kaiyuanhu/AI_Runtime` 的绝对路径硬编码到 Linux。

黄老师部署的 companion runtime 必须满足当前 client 契约：

- 提供可执行的 `ai-models` CLI，支持 `use da3 --device cuda|cpu`、`status` 和 `stop`；
- HTTP 只监听 `127.0.0.1:8765`；
- 支持 `GET /v1/health`、`POST /v1/jobs`、`GET /v1/jobs/{job_id}`；
- 支持任务 `da3.depth_image`；
- DA3 revision 必须是 client 固定的 `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`；
- 返回 finite `float32 .npy` relative depth、shape、model/device/dtype 和 manifest；
- 不把 HF token、视频帧或模型输出写入 Git。

Runtime 来源和 CUDA 安装步骤应由其维护者单独提供；本指南不假称 `pnpm install` 或 `uv sync` 会安装它。

把实际 CLI 加到 PATH，或显式设置：

```bash
export AI_MODELS_CLI="/opt/ai-runtime/bin/ai-models"
test -x "$AI_MODELS_CLI"

"$AI_MODELS_CLI" use da3 --device cuda | jq .
"$AI_MODELS_CLI" status
curl -fsS http://127.0.0.1:8765/v1/health | jq \
  '{model: .model.model_name, revision: .model.model_revision, device, dtype}'
```

必须查看实际 `device`，不要因为服务器有 GPU 就推断 DA3 已在 CUDA 上。如果 CUDA-compatible runtime 尚未部署，有两个诚实选项：

1. 部署 compatible runtime 后用 `--device cuda`；
2. 已有 compatible runtime 但只有 CPU backend 时，用 `--device cpu`，接受显著更慢的速度。

如果 `ai-models` 完全不存在，Depth 会报 `ai-models CLI was not found`；这不是 pnpm 或 Palona Python 依赖问题。也可以把完整 SAM3 Control 安全传回已经安装 Mac/MPS Runtime 的工作站，再在那里生成 portable depth sidecar。

### 8.2 为整段视频生成 Depth sidecar

这个视频的 target prompt 是 `cashier machine`，所以 `--target-labels` 必须与 Control 中的 label 一致。整段运行不要传 `--max-frames`：

```bash
cd "$LABELING_ROOT"

./scripts/depth-preprocess.sh \
  --video "$VIDEO" \
  --contour "$RUN/clip.control.json" \
  --output "$RUN/clip.depth-features.json" \
  --sample-fps 5 \
  --device cuda \
  --person-labels person \
  --target-labels "cashier machine" \
  --stop-runtime
```

对于 table 场景，把 SAM3 prompt 和 Depth target 同时改为 `table`，不要只改其中一处。当前 exact minimal exporter 对接的是 table-interaction schema：它要求至少一个 person 和恰好一个 `table` label。因此本视频的 `cashier machine` 路径可验证整段 mask、Depth cue 和 rich project 保存，但 `occupy_table` / `table_touch` 的 minimal JSON 验收必须使用包含 `table` label 的 Control。`cashier machine` 事件目前不要宣称已经兼容 table-only minimal exporter。

Depth 默认按 5 FPS 生成 cue，这不等于 SAM3 只按 5 FPS 生成 mask：SAM3 Control 仍覆盖原始 FPS；Depth 是额外的低频空间—时间证据。若只想做一次有界诊断，可另写到 `clip.smoke.depth-features.json` 并加 `--max-frames 8`，但**不要把 smoke sidecar 当整段 cue 文件交付**。

检查最终 sidecar 的模型、revision、device、语义和时间覆盖：

```bash
jq '{
  schema_version,
  video,
  contour,
  model: .depth_metadata.model,
  revision: .depth_metadata.model_revision,
  device: .depth_metadata.device,
  dtype: .depth_metadata.dtype,
  metric: .depth_metadata.metric,
  semantics: .depth_metadata.depth_semantics,
  sample_fps: .depth_metadata.sample_fps,
  sample_count: .depth_metadata.sample_count,
  source_duration: .source.video_duration_seconds,
  first: .frames[0].timestamp_seconds,
  last: .frames[-1].timestamp_seconds,
  boundary_candidates: (.boundary_candidates | length)
}' "$RUN/clip.depth-features.json"

jq -e '
  .schema_version == "palona.depth-features/v1" and
  .depth_metadata.metric == false and
  .depth_metadata.model_revision == "f4a6c9b3c95e41c82048423d3493a81ec3fa810e" and
  (.frames | length) == .depth_metadata.sample_count and
  (.frames | length) > 0 and
  (.frames[0].timestamp_seconds <= (.depth_metadata.max_alignment_error_seconds + (1 / .depth_metadata.sample_fps))) and
  ((.source.video_duration_seconds - .frames[-1].timestamp_seconds)
    <= ((1 / .depth_metadata.sample_fps) + .depth_metadata.max_alignment_error_seconds + 0.1))
' "$RUN/clip.depth-features.json"
```

若要求 CUDA 验收，再加：

```bash
jq -e '.depth_metadata.device == "cuda"' "$RUN/clip.depth-features.json"
```

如需调试 DA3 原始 NPY/PNG/manifest，可在 Depth 命令中增加：

```bash
--keep-depth-artifacts "$RUN/da3-debug-artifacts"
```

这会占用更多私有磁盘；正常交付只需要最终 `.depth-features.json`。

## 9. 生成浏览器播放副本

浏览器不能稳定播放所有 MKV/HEVC。建议生成同 stem 的 H.264/yuv420p MP4：

```bash
cd "$LABELING_ROOT"

./scripts/video-convert.sh \
  --input "$VIDEO" \
  --output "$RUN/2026-07-09_00-00-27.mp4"
```

转换器会验证宽高、FPS、帧序、时长和 H.264 pixel format，并且不会覆盖源 MKV。输出必须保留源 stem，所以上面的文件名不能随意改成 `playback.mp4`。音频不会保留。

快速检查：

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames:format=duration \
  -of json \
  "$RUN/2026-07-09_00-00-27.mp4" | jq .
```

## 10. 启动和使用 Labeling Tool

先运行不依赖真实模型的检查：

```bash
cd "$LABELING_ROOT"
pnpm run test:project
pnpm run build
```

只绑定 loopback 启动：

```bash
cd "$LABELING_ROOT"
pnpm run dev --hostname 127.0.0.1
```

默认打开 `http://127.0.0.1:3000`。不要把标注服务或 DA3 的 `8765` 端口绑定到 `0.0.0.0` 或暴露到公网。

若 UI 在有桌面的同一台工作站运行，按顺序选择：

1. **Video clip**：`$RUN/2026-07-09_00-00-27.mp4`（或浏览器能直接播放时选择原 MKV）；
2. **Control JSON**：`$RUN/clip.control.json`；
3. **Depth cues JSON**：`$RUN/clip.depth-features.json`。

加载后应看到：

- mask/contour 随视频前进，在开头、中间和结尾都对齐；
- Control 未覆盖的时间显示 `Outside Control coverage`，不会冻结最后一帧 mask；
- track label/ID 可点击、隐藏和加入 interaction；
- Depth 面板显示 instance `zᵣ`、person–target relative-depth gap、2D gap、trend、quality 和 boundary candidates；
- Depth 文案明确标为 relative/non-metric；
- 可创建、编辑、删除 interaction 并保存/重开 project；使用 person + table Control 时还可导出 exact minimal event JSON。

### 远程 GPU 服务器的浏览器文件边界

HTML file picker 读取的是**运行浏览器的机器**，不是提供网页的远程服务器。因此：

- 如果浏览器就在服务器桌面上，可以直接选择 `$RUN` 中三个文件；
- 如果浏览器在笔记本上，即使通过 SSH tunnel 打开远程 `pnpm dev`，file picker 也看不到服务器 `$RUN`。需要先通过授权的 `scp`/SFTP/加密盘把 MP4、Control 和 Depth sidecar 复制到笔记本私有目录，或者直接在笔记本运行 UI；
- 不要为方便而把私有视频放进 Git 或临时公网 HTTP 目录。

需要 SSH tunnel 时，先确认 dev server 的实际端口，然后从本地机器执行：

```bash
ssh -N -L 3000:127.0.0.1:3000 USER@GPU_SERVER
```

随后在本地浏览器打开 `http://127.0.0.1:3000`；要选择的三个文件仍必须位于本地浏览器可访问的文件系统。

## 11. 一次复制执行的主流程

完成前置安装、HF 授权和 companion Runtime 部署后，主流程可以归纳为：

```bash
export LABELING_ROOT="$HOME/palona/Palona_Labeling_Tool"
export VISION_PIPELINE_ROOT="$HOME/palona/vision_pipeline"
export VISION_PIPELINE_PYTHON="$VISION_PIPELINE_ROOT/.venv/bin/python"
export AI_MODELS_CLI="/opt/ai-runtime/bin/ai-models"
export HF_HOME="$HOME/.cache/huggingface"

# 必须改为服务器真实私有路径。
export VIDEO="/srv/palona-private/dataset/2026-07-09_00-00-27.mkv"
export RUN="/srv/palona-private/outputs/2026-07-09_00-00-27"

umask 077
install -d -m 700 "$RUN" "$RUN/sam3-full"
cd "$LABELING_ROOT"

./scripts/full-video-preprocess.sh \
  --video "$VIDEO" \
  --control "$RUN/clip.control.json" \
  --output-dir "$RUN/sam3-full" \
  --prompt person \
  --prompt "cashier machine" \
  --require-label person \
  --split-seconds 12 \
  --overlap-seconds 2 \
  --check-only

./scripts/full-video-preprocess.sh \
  --video "$VIDEO" \
  --control "$RUN/clip.control.json" \
  --output-dir "$RUN/sam3-full" \
  --prompt person \
  --prompt "cashier machine" \
  --require-label person \
  --split-seconds 12 \
  --overlap-seconds 2

./scripts/depth-preprocess.sh \
  --video "$VIDEO" \
  --contour "$RUN/clip.control.json" \
  --output "$RUN/clip.depth-features.json" \
  --sample-fps 5 \
  --device cuda \
  --person-labels person \
  --target-labels "cashier machine" \
  --stop-runtime

./scripts/video-convert.sh \
  --input "$VIDEO" \
  --output "$RUN/2026-07-09_00-00-27.mp4"

jq . "$RUN/sam3-full/full-video-preprocess.validation.json"
jq '{device: .depth_metadata.device, revision: .depth_metadata.model_revision,
     frames: (.frames | length), first: .frames[0].timestamp_seconds,
     last: .frames[-1].timestamp_seconds, metric: .depth_metadata.metric}' \
  "$RUN/clip.depth-features.json"

pnpm run dev --hostname 127.0.0.1
```

## 12. 常见故障

### `corepack: command not found` / `pnpm: command not found`

先确认 Node `>=22.13.0`，再运行：

```bash
npm config set prefix "$HOME/.local"
npm install -g pnpm@11.7.0
export PATH="$HOME/.local/bin:$PATH"
```

### `ffprobe was not found on PATH`

Ubuntu/Debian 安装 `ffmpeg` 包；它同时提供 `ffmpeg` 和 `ffprobe`：

```bash
sudo apt-get install -y ffmpeg
command -v ffmpeg
command -v ffprobe
```

### `CUDA preflight failed` / `torch.cuda.is_available() is false`

依次检查 `nvidia-smi`、`torch.version.cuda`、PyTorch wheel 和执行 wrapper 的 Python 是否就是 `$VISION_PIPELINE_PYTHON`。最常见问题是在另一个 venv 里装了 CUDA PyTorch，而 wrapper 实际找到了 CPU-only Python。

### SAM3 报 401/403、gated model 或找不到权重

先在网页接受 `facebook/sam3` 条款，再用运行 pipeline 的 Linux 用户执行 `hf auth login`。不要把 token 作为 CLI 参数或提交到 `.env`。

### SAM3 很慢

整段 native whole 模式按原 FPS 运行，本来就明显慢于 `--max-frames 8` 的 smoke test。查看 `full-video-preprocess.log` 和 GPU 利用率；只要 frame 持续前进就不是 UI 卡死。完整交付不能用 smoke output 代替。

### SAM3 CUDA OOM

先停止同 GPU 上无关 worker，确认显存，再把 `12/2` 改为 `6/1`。不要增加 `--max-frames` 来“解决”整段覆盖问题。

### wrapper 报 Control 未覆盖视频结尾

最终 `clip.control.json` 不会被原子发布。检查 log 中失败的 chunk、ffmpeg split、CUDA OOM 和源视频时间轴；不要直接把临时/部分 JSON 加载到 UI。

### mask 开头正常，后面不动或消失

先看 coverage manifest 的 `last_timestamp_seconds`。如果是旧的 8 帧/短 smoke Control，必须重跑第 6 节完整命令。新版 UI 会在 coverage 外停止显示旧 mask；它不会用一帧静止 mask 假装后续已处理。

### `ai-models CLI was not found`

Palona 仓库不分发 AI Runtime。部署 compatible runtime，然后把 CLI 放进 PATH 或设置 `AI_MODELS_CLI`。单独 `pip install depth-anything` 不能替代当前 CLI/API 契约。

### DA3 revision mismatch

当前 client 固定 revision 为 `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`。部署相同 revision；不要为了跳过检查而改 sidecar JSON。

### DA3 没有使用 CUDA

先看 `/v1/health` 和最终 `.depth_metadata.device`。如果 compatible runtime 没有 CUDA backend，改用 `--device cpu` 可以运行但会慢；不能把 CPU 结果报告成 CUDA 验收通过。

### MKV 在浏览器无法播放

运行第 9 节转换命令，输出必须与源视频同 stem。不要用会改变尺寸、FPS 或帧序的随意转码命令，否则 contour 会错位。

### 浏览器看不到服务器文件

这是浏览器沙箱的预期行为。把三个交付文件安全复制到运行浏览器的机器，或在有桌面的服务器上使用浏览器；不要暴露私有数据目录到公网。

## 13. 验收清单

- [ ] 当前 Git 分支是 `Depth_Prior`，依赖按 lockfile 安装。
- [ ] `ffmpeg` 和 `ffprobe` 都在 PATH。
- [ ] `$VISION_PIPELINE_PYTHON` 中 `torch.cuda.is_available()` 为 `True`。
- [ ] 当前 Linux 用户已接受并登录 `facebook/sam3`。
- [ ] full-video command 使用 native `whole`、`split=12`、`overlap=2`，且没有 `--max-frames`。
- [ ] OOM 时使用 `6/1` 重跑，而不是只交付前几帧。
- [ ] coverage manifest 从第 0 帧覆盖到视频结尾，并包含 `person`。
- [ ] Control 的开头、中间、结尾 mask 都与画面同步；没有最后一帧 mask 长时间冻结。
- [ ] companion DA3 实际 device 已检查；CUDA 验收时 sidecar 显示 `device=cuda`。
- [ ] Depth sidecar 是 `palona.depth-features/v1`、固定 revision、`metric=false`，并覆盖整段采样时间。
- [ ] 浏览器加载顺序为 video → Control → Depth，三者匹配且无 schema/尺寸/时长错误。
- [ ] 可选择 person/target、建立 interaction并保存/重开 project；table 场景可导出 exact minimal event JSON。
- [ ] 生成物全部位于私有输出目录；`git status --short` 不包含 CCTV、Control、Depth artifacts、模型权重或凭据。
- [ ] UI 与 DA3 只监听 `127.0.0.1`，没有公网暴露。

最终可交付给标注端的三个文件是：同 stem 的浏览器 MP4（或可播放原视频）、完整 `clip.control.json`、完整 `clip.depth-features.json`。项目保存文件由 UI 在标注过程中生成；person + table 项目还可生成当前训练契约要求的 minimal event JSON。
