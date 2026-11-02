#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

PXL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_JAR="$PXL_DIR/android/pxl-cam-server.jar"
SCRCPY_SERVER="/opt/homebrew/Cellar/scrcpy/4.1/share/scrcpy/scrcpy-server"
RESOLVER="$PXL_DIR/resolve_android_device.sh"

# An explicit adb serial still wins. With no argument, discover the phone.
# USB wins inside the resolver. A saved address is not dialed.
SERIAL="${1:-}"
if [[ -z "$SERIAL" ]]; then
    if ! SERIAL="$("$RESOLVER")"; then
        SERIAL=""
    fi
fi

# No adb transport: bridge the on-phone camera port if exactly one LAN host
# is listening on 9876. Do not reuse a stale address from .env.
if [[ -z "$SERIAL" ]]; then
    WIFI_HOST="$("$RESOLVER" --lan-port 9876 2>/dev/null || true)"
    if [[ "$WIFI_HOST" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] && nc -z -G 1 "$WIFI_HOST" 9876 >/dev/null 2>&1; then
        if ! nc -z -G 1 127.0.0.1 9876 >/dev/null 2>&1; then
            nohup python3 -c "
import socket, threading
def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d: break
            b.sendall(d)
    except Exception: pass
    finally:
        try: a.close()
        except Exception: pass
        try: b.close()
        except Exception: pass
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(('127.0.0.1', 9876))
srv.listen(8)
while True:
    c, _ = srv.accept()
    try:
        r = socket.create_connection(('$WIFI_HOST', 9876), timeout=2.0)
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        r.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=pipe, args=(c, r), daemon=True).start()
        threading.Thread(target=pipe, args=(r, c), daemon=True).start()
    except Exception:
        c.close()
" >/dev/null 2>&1 &
        fi
    fi
    exit 0
fi

# Stop any standalone localhost Wi-Fi relay before setting up adb forward
pkill -f "srv.bind\(\('127.0.0.1', 9876\)\)" >/dev/null 2>&1 || true
adb -s "$SERIAL" forward tcp:9876 tcp:9876 >/dev/null 2>&1 || true

# Stop any competing scrcpy server holding Camera 0
adb -s "$SERIAL" shell "pkill -f 'com.genymobile.scrcpy.Server' || true" >/dev/null 2>&1 || true

# If the native Android Pxl Cam app (com.pxlcam.app) is open on the phone,
# let it own the camera and TCP :9876 directly (kill any competing headless PxlCamServer)
if adb -s "$SERIAL" shell "pidof com.pxlcam.app" >/dev/null 2>&1; then
    adb -s "$SERIAL" shell "pkill -f '[c]om.pxlcam.server.PxlCamServer' || true" >/dev/null 2>&1 || true
    adb -s "$SERIAL" shell "am start-foreground-service -n com.pxlcam.app/.PxlCamStreamingService -a com.pxlcam.app.ACTION_START >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
    exit 0
fi

# Check if headless PxlCamServer is already running on the phone
if adb -s "$SERIAL" shell "pgrep -f '[c]om.pxlcam.server.PxlCamServer'" >/dev/null 2>&1; then
    exit 0
fi

if [[ -f "$SCRCPY_SERVER" ]]; then
    adb -s "$SERIAL" push "$SCRCPY_SERVER" /data/local/tmp/scrcpy-server.jar >/dev/null 2>&1 || true
fi
adb -s "$SERIAL" push "$SERVER_JAR" /data/local/tmp/pxl-cam-server.jar >/dev/null 2>&1 || true

adb -s "$SERIAL" shell "dumpsys deviceidle disable; svc power stayon true; settings put global wifi_sleep_policy 2" >/dev/null 2>&1 || true

nohup adb -s "$SERIAL" shell "CLASSPATH=/data/local/tmp/scrcpy-server.jar:/data/local/tmp/pxl-cam-server.jar app_process / com.pxlcam.server.PxlCamServer 0 1920 1080" </dev/null >/tmp/pxl_cam_android.log 2>&1 &
