#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

PXL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_JAR="$PXL_DIR/android/pxl-cam-server.jar"
SCRCPY_SERVER="/opt/homebrew/Cellar/scrcpy/4.1/share/scrcpy/scrcpy-server"

# Auto-detect first connected ADB device if SERIAL not specified
SERIAL="${1:-}"
if [[ -z "$SERIAL" ]]; then
    SERIAL="$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
fi

if [[ -z "$SERIAL" ]]; then
    exit 0
fi

adb -s "$SERIAL" forward tcp:9876 tcp:9876 >/dev/null 2>&1 || true

# Check if PxlCamServer is already running on the phone
if adb -s "$SERIAL" shell "pgrep -f 'com.pxlcam.server.PxlCamServer'" >/dev/null 2>&1; then
    exit 0
fi

if [[ -f "$SCRCPY_SERVER" ]]; then
    adb -s "$SERIAL" push "$SCRCPY_SERVER" /data/local/tmp/scrcpy-server.jar >/dev/null 2>&1 || true
fi
adb -s "$SERIAL" push "$SERVER_JAR" /data/local/tmp/pxl-cam-server.jar >/dev/null 2>&1 || true

adb -s "$SERIAL" shell "pkill -f '[c]om.genymobile.scrcpy.Server' || true; svc power stayon usb; settings put system screen_brightness 0" >/dev/null 2>&1 || true

nohup adb -s "$SERIAL" shell "CLASSPATH=/data/local/tmp/scrcpy-server.jar:/data/local/tmp/pxl-cam-server.jar app_process / com.pxlcam.server.PxlCamServer 0 1920 1080" >/tmp/pxl_cam_android.log 2>&1 &
