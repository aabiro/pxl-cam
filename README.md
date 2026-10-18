# pxl-cam

AI-powered studio camera & streaming suite with Android hardware ISP, macOS Metal clean feed receiver, retro/synthwave HUD, and OBS integration.

## Overview

`pxl-cam` transforms connected Android devices (such as Moto G / Pixel hardware) into dedicated high-throughput, ultra-low-latency 4K/2K/1080p studio webcams and clean-feed streaming sources for macOS and OBS Studio.

### Highlights

- **Hardware Camera2 ISP Server (`android/`)**: Custom low-overhead server running directly via `app_process` / ADB forward, streaming raw H.264 / zero-latency feeds without requiring full Android UI rendering.
- **macOS Metal Receiver (`mac/`)**: Zero-copy Metal texture rendering pipeline with live studio ISP:
  - Low-light shadow lifting & exposure gain
  - Contrast, saturation, sharpening & noise reduction
  - Real-time rotation & mirroring controls
  - 80s Retro / Synthwave HUD overlay
  - Apple Vision framework integration for face tracking & auto-exposure
- **OBS Clean Feed Integration (`obs/`)**: OBS Python scripts and custom docks to manage clean feeds, PTZ virtual control, and direct canvas ingestion.
- **Android App Companion (`app/`)**: Full Android app with customizable launcher assets, permissions management, and background keepalive.

## Architecture

```
pxl-cam/
├── android/            # Android Camera2 ISP server (Java) & jar
├── app/                # Android companion app & resources (Kotlin + Jetpack)
├── mac/                # Native macOS Metal receiver app (Swift + Metal)
├── obs/                # OBS Studio dock & clean feed integration scripts
├── assets/             # Brand logos, synthwave badges, vectors & palettes
├── deploy/             # Deployment scripts, dock pinning & macOS app packager
└── run_pxl_cam.sh      # One-click startup script (ADB forward + server + Metal app)
```

## Getting Started

### Prerequisites

- macOS 12+ (Apple Silicon recommended for Metal compute)
- Android device with USB debugging enabled
- `adb` and `scrcpy` (installed via Homebrew: `brew install scrcpy android-platform-tools`)
- Xcode Command Line Tools (`swiftc`)

### Quick Start

1. Connect your Android device via USB with USB debugging enabled.
2. Launch the camera suite:
   ```bash
   ./run_pxl_cam.sh
   ```
   This will forward TCP port `9876`, push and start the Android camera server, and launch the macOS Metal receiver.

3. To build the macOS application bundle:
   ```bash
   ./deploy/make_macos_app.sh
   ```

4. For OBS integration:
   Add `obs/configure_obs_pxl_cam.py` or dock `obs/pxl_cam_obs_dock.py` within OBS Studio Scripts / Docks.

## License

MIT License. See individual modules for third-party component licenses.
