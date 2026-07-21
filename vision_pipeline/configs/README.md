# Vision Pipeline Configs

Configs are declarative. They describe the use case, model choices, camera geometry, prompts, thresholds, and output behavior.

Use environment-variable references such as `uri_env` and `endpoint_env` for secrets or deployment-specific values. Do not put RTSP credentials, API keys, or private endpoints directly in YAML.

Config groups:

- `use_cases/`: business workflow configs.
- `models/`: reusable model runtime defaults.
- `cameras/`: camera streams, ROIs, zones, and geometry.
- `prompts/`: VLM prompts and expected structured outputs.

SAM-backed dwell use cases can define a `models.sam3` block plus `sam3_live`
settings. The `sam3_live.strategy` value can be `long_window`,
`rolling_window`, or `per_frame`, so the same use case can move between
accuracy-first and latency-first modes without changing Python code.

For clip experiments, `vp_cli.py sam3 --video-mode whole` uses Meta's native
SAM3 video session against the source MP4/MOV instead of pre-sampled frames.
Use `--draw-contours` to write an annotated MP4 from SAM3 masks.
