#!/usr/bin/env python3
"""
Auto-reconnects Wireless ADB + 1080p scrcpy camera from Motorola Moto G Pure
and dynamically binds the new scrcpy CGWindowID to OBS 'Wireless scrcpy Camera (1080p)'.
Also disables Android Doze / Wi-Fi sleep on connect so the stream does not drop after 25 minutes.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import websocket

OBS_WS_URL = "ws://127.0.0.1:4455"
OBS_WS_PASS = "o99FcV6xXBaWWZyF"
OBS_SOURCE_NAME = "Wireless scrcpy Camera (1080p)"
OBS_SCENE_NAME = "Scene"
WINDOW_TITLE = "MotoG_OBS_Camera"


def obs_req(ws, req_type, req_data=None, req_id="1"):
    payload = {"op": 6, "d": {"requestType": req_type, "requestId": req_id}}
    if req_data:
        payload["d"]["requestData"] = req_data
    ws.send(json.dumps(payload))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("op") == 7 and msg["d"].get("requestId") == req_id:
            return msg["d"]


def bind_obs_window():
    try:
        ws = websocket.create_connection(OBS_WS_URL, timeout=5)
        hello = json.loads(ws.recv())
        if "authentication" in hello.get("d", {}):
            salt = hello["d"]["authentication"]["salt"]
            challenge = hello["d"]["authentication"]["challenge"]
            sec = base64.b64encode(hashlib.sha256((OBS_WS_PASS + salt).encode()).digest()).decode()
            auth = base64.b64encode(hashlib.sha256((sec + challenge).encode()).digest()).decode()
            ws.send(json.dumps({"op": 1, "d": {"rpcVersion": 1, "authentication": auth}}))
        else:
            ws.send(json.dumps({"op": 1, "d": {"rpcVersion": 1}}))
        ws.recv()

        props = obs_req(ws, "GetInputPropertiesListPropertyItems", {
            "inputName": OBS_SOURCE_NAME,
            "propertyName": "window"
        }, "win_list")
        items = props.get("responseData", {}).get("propertyItems", [])
        scrcpy_win = None
        for it in items:
            name = it.get("itemName", "")
            if WINDOW_TITLE.lower() in name.lower() or "scrcpy" in name.lower():
                scrcpy_win = it["itemValue"]
                break

        if scrcpy_win is None:
            ws.close()
            return False

        obs_req(ws, "SetInputSettings", {
            "inputName": OBS_SOURCE_NAME,
            "inputSettings": {
                "type": 1,
                "window": scrcpy_win,
                "show_cursor": False,
                "show_empty_names": False,
                "show_hidden_windows": True,
                "hide_obs": False
            },
            "overlay": False
        }, "set_win")

        scene_items = obs_req(ws, "GetSceneItemList", {"sceneName": OBS_SCENE_NAME}, "items")["responseData"]["sceneItems"]
        for si in scene_items:
            sid = si["sceneItemId"]
            sname = si["sourceName"]
            should_enable = (sname == OBS_SOURCE_NAME)
            obs_req(ws, "SetSceneItemEnabled", {
                "sceneName": OBS_SCENE_NAME,
                "sceneItemId": sid,
                "sceneItemEnabled": should_enable
            }, f"en_{sid}")

        vc = obs_req(ws, "GetVirtualCamStatus", req_id="vc")
        if not vc.get("responseData", {}).get("outputActive"):
            obs_req(ws, "StartVirtualCam", req_id="start_vc")

        ws.close()
        print(f"[OBS] Bound '{OBS_SOURCE_NAME}' to window ID {scrcpy_win}", flush=True)
        return True
    except Exception as e:
        print(f"[OBS] Error binding window: {e}", flush=True)
        return False


def get_active_adb_device():
    """Look the phone up. A saved IP is not dialed."""
    resolver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolve_android_device.sh")
    result = subprocess.run(["/bin/bash", resolver], capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or "").strip()
        if message and message != getattr(get_active_adb_device, "_last_error", None):
            print(message, file=sys.stderr, flush=True)
            get_active_adb_device._last_error = message
        return None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def is_scrcpy_running():
    res = subprocess.run(["pgrep", "-x", "scrcpy"], capture_output=True, text=True)
    return res.returncode == 0


def keep_phone_wifi_awake(serial):
    cmds = [
        "dumpsys deviceidle disable",
        "svc power stayon true",
        "settings put global wifi_sleep_policy 2",
        "pkill -f PxlCamServer || true"
    ]
    for cmd in cmds:
        subprocess.run(["/opt/homebrew/bin/adb", "-s", serial, "shell", cmd],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    once = "--once" in sys.argv
    print("[Watcher] Watching for Motorola Moto G Pure Wireless ADB & scrcpy...", flush=True)
    while True:
        dev = get_active_adb_device()
        if not dev:
            time.sleep(2)
            continue

        if not is_scrcpy_running():
            print(f"[Watcher] Device {dev} online! Configuring anti-Doze & starting 1080p scrcpy...", flush=True)
            keep_phone_wifi_awake(dev)
            log_file = open("/tmp/scrcpy_obs.log", "w")
            subprocess.Popen([
                "/opt/homebrew/bin/scrcpy",
                "-s", dev,
                "--video-source=camera",
                "--camera-id=0",
                "--camera-size=1920x1080",
                "--camera-fps=30",
                "--video-bit-rate=18M",
                "--video-codec=h264",
                "--no-audio",
                f"--window-title={WINDOW_TITLE}"
            ], stdout=log_file, stderr=subprocess.STDOUT)
            for _ in range(10):
                time.sleep(1.0)
                if bind_obs_window():
                    break
            if once:
                return

        if once:
            bind_obs_window()
            return
        time.sleep(3)


if __name__ == "__main__":
    main()

