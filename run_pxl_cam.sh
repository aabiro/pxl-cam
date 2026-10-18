#!/usr/bin/env bash
set -euo pipefail

SERIAL="${1:-ZY22H9HQNL}"
DEFAULT_LENS="${2:-0}"
DEFAULT_WIDTH="${3:-1920}"
DEFAULT_HEIGHT="${4:-1080}"

PXL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_JAR="$PXL_DIR/android/pxl-cam-server.jar"
SCRCPY_SERVER="/opt/homebrew/Cellar/scrcpy/4.1/share/scrcpy/scrcpy-server"
if [ -x "/Applications/Pxl Cam.app/Contents/MacOS/PxlCam" ]; then
    MAC_APP="/Applications/Pxl Cam.app/Contents/MacOS/PxlCam"
elif [ -x "$HOME/Applications/Pxl Cam.app/Contents/MacOS/PxlCam" ]; then
    MAC_APP="$HOME/Applications/Pxl Cam.app/Contents/MacOS/PxlCam"
else
    MAC_APP="$PXL_DIR/mac/PxlCamApp"
fi

echo "[pxl-cam] Preparing Moto G Pure ($SERIAL)..."
adb -s "$SERIAL" push "$SCRCPY_SERVER" /data/local/tmp/scrcpy-server.jar >/dev/null
adb -s "$SERIAL" push "$SERVER_JAR" /data/local/tmp/pxl-cam-server.jar >/dev/null
adb -s "$SERIAL" forward tcp:9876 tcp:9876

# Release any previous camera holder on device
adb -s "$SERIAL" shell "pkill -f '[c]om.pxlcam.server.PxlCamServer' || true; pkill -f '[c]om.genymobile.scrcpy.Server' || true; svc power stayon usb; settings put system screen_brightness 0"

echo "[pxl-cam] Starting Android Camera2 4K/2K Hardware ISP Server on $SERIAL (Lens=$DEFAULT_LENS ${DEFAULT_WIDTH}x${DEFAULT_HEIGHT})..."
adb -s "$SERIAL" shell "CLASSPATH=/data/local/tmp/scrcpy-server.jar:/data/local/tmp/pxl-cam-server.jar app_process / com.pxlcam.server.PxlCamServer $DEFAULT_LENS $DEFAULT_WIDTH $DEFAULT_HEIGHT" &
ANDROID_PID=$!

cleanup() {
    echo "[pxl-cam] Shutting down..."
    kill "$ANDROID_PID" 2>/dev/null || true
    adb -s "$SERIAL" shell "pkill -f '[c]om.pxlcam.server.PxlCamServer' || true" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1.2
echo "[pxl-cam] Launching macOS Metal Receiver + OBS Clean Feed HUD..."
"$MAC_APP"
