# Vision Pipeline

Config-driven computer-vision pipeline for restaurant operations use cases.

The design separates reusable model adapters from per-use-case YAML configs:

- RF-DETR handles object/person detection.
- SAM3 handles promptable segmentation/tracking experiments.
- ByteTrack handles realtime object tracking.
- BoT-SORT handles people/line tracking.
- Qwen3-VL handles semantic classification, intent review, and prompt-grounded
  frame analysis.

OCR/barcode support is intentionally omitted for now.

## Implementation Status

Implemented:

- Config folders for use cases, models, cameras, and VLM prompts.
- Use-case YAMLs for xiaolongbao table dwell, line velocity, and takeout bagging.
- RF-DETR local inference adapter for image paths and RGB frames.
- SAM3 local adapter for promptable image segmentation, sampled-frame video
  tracking, and native whole-video tracking with contour rendering.
- BoT-SORT online person tracking adapter with motion prediction, IoU association,
  and color-histogram appearance reactivation.
- Tracklet evidence/frame-selection logic for VLM classification.
- Config validation and dry-run scripts.
- `vp_cli.py rfdetr` and `vp_cli.py sam3` commands for image inputs,
  sampled MP4 video frames, and native whole-video SAM3 sessions.
- `vp_cli.py sam3-live` for windowed SAM3 tracking, global track stitching,
  dwell alerts, and optional evidence videos from ROI frames.
- `vp_cli.py qwen3-vl` command for Qwen3-VL image inputs and sampled MP4
  frame analysis through an OpenAI-compatible endpoint.

Not implemented yet:

- ByteTrack runtime integration.
- Neural BoT-SORT ReID and camera-motion compensation.
- Production multi-process RTSP capture/inference workers.
- Crop image persistence and contact-sheet generation.
- Qwen3-VL server lifecycle management.
- Realtime signaling sinks such as webhook, Pub/Sub, Redis, or database writes.

Next steps:

1. Install the GPU inference stack on the VM and run RF-DETR on sample images/videos.
2. Run SAM3 on the same samples and compare ID stability against BoT-SORT.
3. Tune `sam3-live` ROI/window settings on sample clips, then smoke test against
   the RTSP camera.
4. Wire ByteTrack and optional neural BoT-SORT ReID/runtime implementations behind the existing adapters.
5. Add crop persistence/contact sheets for Qwen3-VL evidence bundles.
6. Add realtime output sinks for downstream alerts and dashboards.

## Layout

```text
vision_pipeline/
  configs/
    use_cases/     # One YAML per business use case.
    models/        # Shared model/runtime defaults.
    cameras/       # Camera sources, ROIs, zones, geometry.
    prompts/       # VLM prompts and expected JSON schemas.
  src/
    vision_pipeline/
      core/        # Pipeline, config, events, crops, tracklet evidence.
      models/      # One adapter folder per model family.
      use_cases/   # Use-case specific logic hooks.
      utils/       # Geometry, video, logging, time helpers.
  scripts/         # CLI helpers for config validation and dry runs.
  tests/           # Lightweight tests for config and tracklet logic.
```

## GCP VM

For the current RF-DETR + Qwen3-VL pilot, use a GPU VM with at least an
**A100 40GB**. An A100 80GB gives more headroom for larger Qwen3-VL variants,
higher crop throughput, or running detector and VLM services on the same VM.

Create or reuse the VM in GCP project `gen-lang-client-0554757211`. From a
machine with the Google Cloud CLI installed and authenticated, connect with:

```bash
gcloud compute ssh --zone "us-central1-f" "research-vision" --project "gen-lang-client-0554757211"
```

Minimum setup on the VM:

```bash
sudo apt update
sudo apt install -y git gh git-lfs ffmpeg
git lfs install
gh auth login
git config --global user.email "<email>"
gh repo clone Proactive-AI-Lab/pal-research
sudo apt install nvidia-driver-550-server -y
sudo reboot
```

After cloning, follow the root repository setup to create the Python
environment and install dependencies.

The `ffmpeg` package provides both `ffmpeg` and `ffprobe`, which the sample
video scripts use for MP4 metadata and frame extraction.

### RF-DETR Inference

For RF-DETR inference on the VM, install the model runtime in a Python 3.10+
environment:

```bash
pip install rfdetr supervision opencv-python
```

### SAM3 Inference

For sampled-frame SAM3 inference, install a Transformers build with SAM3
support. For native whole-video mode, install Meta's SAM3 package as well. In
both cases, make sure your Hugging Face account has access to `facebook/sam3`:

```bash
pip install torch pillow accelerate "transformers>=5.0.0" kernels
pip install "git+https://github.com/facebookresearch/sam3.git"
pip install huggingface_hub
hf auth login
```

SAM3 is released under Meta's SAM license rather than a permissive MIT/Apache
license, so confirm the license terms before using it in a commercial product.

Run sampled-frame tracking, which is cheaper and matches the historical eval
path:

```bash
python vision_pipeline/scripts/vp_cli.py sam3 path/to/video.mp4 \
  --prompt person \
  --sample-fps 2 \
  --draw-boxes \
  --output-dir outputs/sam3_sampled
```

Run native whole-video tracking and write contour overlays from SAM3 masks:

```bash
python vision_pipeline/scripts/vp_cli.py sam3 path/to/video.mp4 \
  --video-mode whole \
  --prompt person \
  --draw-contours \
  --output-dir outputs/sam3_whole
```

Use `--roi x1,y1,x2,y2` with whole-video mode to crop the video before SAM3
tracking; contour overlays are offset back onto the original video frames.

### Qwen3-VL Inference

For Qwen3-VL inference through vLLM:

```bash
sudo apt install python3-dev
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12
echo 'export PATH=/usr/local/cuda-12/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
pip install vllm
vllm serve "Qwen/Qwen3-VL-8B-Instruct" --port 8080 --max-model-len 122144
export QWEN3_VL_ENDPOINT=http://localhost:8080
```

`Qwen/Qwen3-VL-8B-Instruct` is Apache-2.0 licensed. The CLI sends local images
as base64 `image_url` payloads to the OpenAI-compatible `/v1/chat/completions`
endpoint.

### Connect VS Code to the VM

Install the VS Code **Remote - SSH** extension, then generate SSH config
entries for your GCP instances:

```bash
gcloud auth login
gcloud config set project gen-lang-client-0554757211
gcloud compute config-ssh
```

In VS Code, open `Remote Explorer`. Under
`REMOTES (TUNNELS/SSH) > SSH`, select
`research-vision.us-central1-f` and click `Connect in New Window`.

Once connected, open the cloned repository directory on the VM:

```text
~/pal-research
```

## Validate a Use Case

```bash
python vision_pipeline/scripts/validate_config.py \
  vision_pipeline/configs/use_cases/line_velocity.yaml
```

## Dry Run

```bash
python vision_pipeline/scripts/run_use_case.py \
  vision_pipeline/configs/use_cases/xlb_table_dwell.yaml \
  --dry-run
```

## RF-DETR on Images or Videos

The repo includes sample line-velocity videos under
`agent/vision/pokeworks/line-velocity/`. Run RF-DETR on sampled frames with:

```bash
python vision_pipeline/scripts/vp_cli.py rfdetr \
  agent/vision/pokeworks/line-velocity/2026-05-27_18-07-07.mp4 \
  --model-config vision_pipeline/configs/models/rf_detr.yaml \
  --sample-fps 2 \
  --max-frames 120 \
  --number-worker-threads 8 \
  --start-from 0 \
  --classes person \
  --tracker bot-sort \
  --tracker-config vision_pipeline/configs/models/bot_sort.yaml \
  --draw-boxes \
  --tracked-object-clips \
  --output-dir /tmp/vision_pipeline_samples
```

Use `--draw-boxes` to write annotated images with labels such as
`person-92.34%`. When `--tracker bot-sort` is enabled, annotated labels include
the track ID, such as `track-1 person-92.34%`. Use `--extract-only` to validate
frame extraction without loading RF-DETR. Add `--timeit` to print average
per-frame inference latency after the command output.

Use `--number-worker-threads` to parallelize ffmpeg timestamp extraction
(default: `8`) and `--start-from` to skip the first N seconds of a video.
With `--tracked-object-clips`, the CLI writes one MP4 per tracked object
continuous appearance under `tracked_object_clips/` in the output directory. If
`--output-dir` is omitted, clip export creates
`<source_stem>_vision_pipeline_outputs/` in the current working directory.
Reappearing track IDs after one or more missing sampled frames become new
appearance clips. If the source filename is formatted like
`2026-06-24_13-20-00.mp4`, each clip overlays timestamps like
`2026-06-24 13:20:05`, and only the clip's target track is boxed.
Add `--single-appearance` to write only one clip per track ID; that clip spans
from the first visible frame through the last tracked frame and includes
intervening hidden/missing frames without drawing a box on those frames.

The same script also accepts image inputs:

```bash
python vision_pipeline/scripts/vp_cli.py rfdetr \
  /tmp/vision_pipeline_samples/frame_0000_20.00s.jpg \
  --model-config vision_pipeline/configs/models/rf_detr.yaml \
  --classes person cup chair \
  --draw-boxes \
  --output-dir /tmp/vision_pipeline_samples
```

## SAM3 on Images or Videos

SAM3 uses text prompts instead of fixed class names. Run promptable tracking on
the same sampled line-velocity videos with:

```bash
python vision_pipeline/scripts/vp_cli.py sam3 \
  agent/vision/pokeworks/line-velocity/2026-05-27_18-07-07.mp4 \
  --model-config vision_pipeline/configs/models/sam3.yaml \
  --sample-fps 2 \
  --max-frames 120 \
  --prompt person \
  --draw-boxes \
  --output-dir /tmp/vision_pipeline_sam3
```

For object-specific use cases, quote multi-word prompts:

```bash
python vision_pipeline/scripts/vp_cli.py sam3 \
  /tmp/xlb_table_frame.jpg \
  --prompt "bamboo steamer" \
  --draw-boxes \
  --output-dir /tmp/vision_pipeline_sam3
```

Use `--extract-only` to validate video probing and frame extraction without
loading SAM3. Add `--timeit` to print average per-frame tracking latency after
the command output. SAM3 video tracking also supports `--tracked-object-clips`
to write one target-only boxed MP4 per continuous SAM track appearance. When
`--roi` is set, those clips are written from the ROI crop frames instead of the
full source frames. Add `--single-appearance` to produce one span clip per SAM
track ID, including hidden/missing frames between visible observations.

For high-resolution videos, use `--roi x1,y1,x2,y2` to run SAM3 only inside a
full-frame bounding box while keeping the crop at native resolution. Output
boxes are mapped back to the original frame coordinates:

```bash
python vision_pipeline/scripts/vp_cli.py sam3 \
  agent/vision/dintaifung/expo/2026-06-10_16-30-09.mp4 \
  --prompt basket \
  --roi 500,500,1800,1800 \
  --sample-fps 2 \
  --max-frames 120 \
  --draw-boxes \
  --output-dir /tmp/vision_pipeline_sam3
```

With `--roi --draw-boxes`, the full frame gets mapped boxes in `*_boxes.jpg`,
and the ROI crop `*_roi.jpg` is also annotated with crop-local boxes.

## SAM3 Live Dwell Tracking

Use `sam3-live` when SAM should own the visual tracking accuracy, while the
pipeline owns global IDs, dwell timers, and evidence output. The default Expo
config uses the accuracy-first strategy:

```yaml
sam3_live:
  strategy: long_window
  sample_fps: 1
  window_seconds: 30
  stride_seconds: 20
  roi: [500, 500, 1800, 1800]
  dwell_threshold_seconds: 120
```

`window_seconds` controls how much temporal context SAM sees. `stride_seconds`
controls overlap; with the settings above, each 30-second window overlaps the
previous one by 10 seconds. Overlapped frames are used to stitch SAM-local IDs
onto long-lived global IDs, but they are not committed twice.

Smoke test the live logic on an existing clip without loading SAM3:

```bash
python vision_pipeline/scripts/vp_cli.py sam3-live \
  agent/vision/dintaifung/expo/2026-06-10_16-30-09.mp4 \
  --config vision_pipeline/configs/use_cases/expo_basket_dwell_sam3.yaml \
  --max-runtime-seconds 60 \
  --extract-only \
  --output-dir /tmp/vision_pipeline_sam3_live
```

Run SAM3 and write annotated ROI evidence frames/video:

```bash
python vision_pipeline/scripts/vp_cli.py sam3-live \
  agent/vision/dintaifung/expo/2026-06-10_16-30-09.mp4 \
  --config vision_pipeline/configs/use_cases/expo_basket_dwell_sam3.yaml \
  --draw-boxes \
  --evidence-video \
  --output-dir /tmp/vision_pipeline_sam3_live
```

For RTSP, export the camera URL referenced by the camera YAML, then omit the
file input:

```bash
export STORE_001_EXPO_RTSP_URL='rtsp://user:pass@camera-host/stream'

python vision_pipeline/scripts/vp_cli.py sam3-live \
  --config vision_pipeline/configs/use_cases/expo_basket_dwell_sam3.yaml \
  --rtsp-smoke-test \
  --smoke-test-frames 3 \
  --output-dir /tmp/vision_pipeline_sam3_live
```

The smoke test only opens the stream, decodes a few frames, writes the ROI crops
if configured, and exits without loading SAM3 or waiting for a full window.

Then run the windowed tracker:

```bash
python vision_pipeline/scripts/vp_cli.py sam3-live \
  --config vision_pipeline/configs/use_cases/expo_basket_dwell_sam3.yaml \
  --max-windows 3 \
  --draw-boxes \
  --evidence-video \
  --output-dir /tmp/vision_pipeline_sam3_live
```

The current RTSP path is a single-process sampler meant for low-FPS smoke tests.
For production, keep this same config surface but split RTSP capture and SAM
inference into separate workers so slow model windows do not interrupt frame
ingest.

Switching strategy is a config-only change:

```yaml
sam3_live:
  strategy: rolling_window  # smaller window/stride for lower latency
```

or:

```yaml
sam3_live:
  strategy: per_frame       # lowest latency; less temporal context
```

## Qwen3-VL on Images or Videos

Qwen3-VL is useful for semantic checks while waiting on SAM3 access. It can ask
for prompt-grounded boxes per image/frame, but it does not maintain persistent
video track IDs by itself.

Run per-frame prompt-grounded detection:

```bash
python vision_pipeline/scripts/vp_cli.py qwen3-vl \
  agent/vision/pokeworks/line-velocity/2026-05-27_18-07-07.mp4 \
  --model-config vision_pipeline/configs/models/qwen3_vl.yaml \
  --sample-fps 1 \
  --max-frames 20 \
  --prompt person \
  --draw-boxes \
  --output-dir /tmp/vision_pipeline_qwen3_vl
```

Use the short alias if you prefer:

```bash
python vision_pipeline/scripts/vp_cli.py qwen3 \
  /tmp/xlb_table_frame.jpg \
  --task detect \
  --prompt "bamboo steamer" \
  --draw-boxes \
  --output-dir /tmp/vision_pipeline_qwen3_vl
```

For arbitrary visual reasoning/classification, use `--task ask`:

```bash
python vision_pipeline/scripts/vp_cli.py qwen3-vl \
  /tmp/line_frame.jpg \
  --task ask \
  --prompt "Return JSON describing whether the customer is being served by an employee."
```

If Qwen returns normalized boxes instead of absolute pixel coordinates, retry
with `--bbox-format qwen1000`, `--bbox-format normalized`, or
`--bbox-format auto`. Add `--timeit` to print average per-frame API latency
after the command output.

If Qwen3-VL emits a long response that is truncated before the closing JSON
braces, the parser recovers any complete detection objects and marks the frame
with `parse_status: partial_recovery`. Increase `request.max_tokens` in
`configs/models/qwen3_vl.yaml` if this happens often.
