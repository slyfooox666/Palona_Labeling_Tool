# Palona Labeling Tool

A local web interface for inspecting frame-level contours over video clips.

## Run locally

Requires Node.js 22.13 or newer and pnpm.

```bash
pnpm install
pnpm dev
```

Then open `http://localhost:3000`.

## First-version workflow

1. Choose a video clip from disk.
2. Choose its matching control JSON file.
3. Play, scrub, pause, or step one frame at a time.
4. Use the object-type and track-ID checkboxes to control visible contours.
5. Hover inside a visible contour to highlight it; click it to hide that track.

The sample files are in `assets/chica/table`. Control files are parsed in a Web Worker so the interface remains responsive while large JSON files load.

MKV/HEVC decoding is provided by the browser and operating system. If a browser cannot decode a selected clip, use a browser with native HEVC support or convert the clip to H.264 MP4 before loading it.
