#!/usr/bin/env python3
"""
Configure OBS Studio on macOS for Pxl Cam:
  1. Sets OutputCX=1920 and OutputCY=1080 (fixing 720p downscale blur) in basic.ini.
  2. Registers the "Pxl Cam Studio" Custom Dock (http://127.0.0.1:9877/dock) in global.ini.
  3. Adds "Pxl Cam (1080p Direct)" (minimize-safe 1920x1080 30fps stream) and fixes
     the macOS Screen Capture transform to fit 1920x1080 in Untitled.json.
"""
import configparser
import json
import uuid
from pathlib import Path

OBS_DIR = Path.home() / "Library" / "Application Support" / "obs-studio"
GLOBAL_INI = OBS_DIR / "global.ini"
BASIC_INI = OBS_DIR / "basic" / "profiles" / "Untitled" / "basic.ini"
SCENE_JSON = OBS_DIR / "basic" / "scenes" / "Untitled.json"


def update_basic_ini() -> None:
    if not BASIC_INI.exists():
        return
    text = BASIC_INI.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    out = []
    in_video = False
    replaced_cx = False
    replaced_cy = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_video and not replaced_cx:
                out.append("OutputCX=1920")
                replaced_cx = True
            if in_video and not replaced_cy:
                out.append("OutputCY=1080")
                replaced_cy = True
            in_video = (stripped == "[Video]")
            out.append(line)
            continue

        if in_video and stripped.startswith("OutputCX="):
            out.append("OutputCX=1920")
            replaced_cx = True
        elif in_video and stripped.startswith("OutputCY="):
            out.append("OutputCY=1080")
            replaced_cy = True
        elif in_video and stripped.startswith("BaseCX="):
            out.append("BaseCX=1920")
        elif in_video and stripped.startswith("BaseCY="):
            out.append("BaseCY=1080")
        else:
            out.append(line)

    BASIC_INI.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("Updated OBS [Video] Base/Output resolution to 1920x1080 in basic.ini")


def update_global_ini() -> None:
    if not GLOBAL_INI.exists():
        return
    text = GLOBAL_INI.read_text(encoding="utf-8", errors="ignore")
    dock_entry = '{"title": "Pxl Cam Studio", "url": "http://127.0.0.1:9877/dock", "uuid": "pxlcam-studio-dock-0001"}'

    lines = text.splitlines()
    out = []
    in_basic_window = False
    found_extra_docks = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_basic_window and not found_extra_docks:
                out.append(f"ExtraBrowserDocks=[{dock_entry}]")
                found_extra_docks = True
            in_basic_window = (stripped == "[BasicWindow]")
            out.append(line)
            continue

        if in_basic_window and stripped.startswith("ExtraBrowserDocks="):
            out.append(f"ExtraBrowserDocks=[{dock_entry}]")
            found_extra_docks = True
        else:
            out.append(line)

    if not found_extra_docks:
        out.append("\n[BasicWindow]")
        out.append(f"ExtraBrowserDocks=[{dock_entry}]")

    GLOBAL_INI.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("Registered 'Pxl Cam Studio' Custom Dock in OBS global.ini")


def update_scene_json() -> None:
    if not SCENE_JSON.exists():
        return
    data = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
    sources = data.get("sources", [])

    direct_source_name = "Pxl Cam (1080p Direct)"
    for s in sources:
        if s.get("id") == "browser_source" and s.get("settings", {}).get("url") == "http://127.0.0.1:9877/player":
            s["name"] = direct_source_name

    has_direct_source = any(s.get("name") == direct_source_name for s in sources)

    if not has_direct_source:
        direct_source = {
            "prev_ver": 536870914,
            "name": direct_source_name,
            "uuid": str(uuid.uuid4()),
            "id": "browser_source",
            "versioned_id": "browser_source",
            "settings": {
                "url": "http://127.0.0.1:9877/player",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "fps_custom": True,
                "reroute_audio": False,
                "restart_when_active": False,
                "shutdown": False,
                "css": "body { background-color: rgba(0, 0, 0, 0); margin: 0px auto; overflow: hidden; }"
            },
            "mixers": 255,
            "sync": 0,
            "flags": 0,
            "volume": 1.0,
            "balance": 0.5,
            "enabled": True,
            "muted": True,
            "push-to-mute": False,
            "push-to-mute-delay": 0,
            "push-to-talk": False,
            "push-to-talk-delay": 0,
            "hotkeys": {}
        }
        sources.append(direct_source)

    direct_uuid = next(s["uuid"] for s in sources if s["name"] == direct_source_name)

    for src in sources:
        if src.get("id") == "scene":
            items = src.get("settings", {}).get("items", [])
            # Fix transform on existing macOS Screen Capture so it fits 1920x1080 cleanly
            for item in items:
                if item.get("source_uuid") == direct_uuid or "1080p Direct" in str(item.get("name", "")):
                    item["name"] = direct_source_name
                    item["source_uuid"] = direct_uuid
                if item.get("name") == "macOS Screen Capture":
                    item["pos"] = {"x": 0.0, "y": 0.0}
                    item["scale"] = {"x": 1.0, "y": 1.0}
                    item["bounds_type"] = 2
                    item["bounds_align"] = 0
                    item["bounds"] = {"x": 1920.0, "y": 1080.0}

            if not any(it.get("name") == direct_source_name for it in items):
                max_id = max([it.get("id", 0) for it in items] + [1])
                items.append({
                    "name": direct_source_name,
                    "source_uuid": direct_uuid,
                    "visible": True,
                    "locked": False,
                    "rot": 0.0,
                    "scale_ref": {"x": 1920.0, "y": 1080.0},
                    "align": 5,
                    "bounds_type": 2,
                    "bounds_align": 0,
                    "bounds_crop": False,
                    "crop_left": 0,
                    "crop_top": 0,
                    "crop_right": 0,
                    "crop_bottom": 0,
                    "id": max_id + 1,
                    "group_item_backup": False,
                    "pos": {"x": 0.0, "y": 0.0},
                    "pos_rel": {"x": -1.7777777910232544, "y": -1.0},
                    "scale": {"x": 1.0, "y": 1.0},
                    "scale_rel": {"x": 1.0, "y": 1.0},
                    "bounds": {"x": 1920.0, "y": 1080.0},
                    "bounds_rel": {"x": 3.5555555820465088, "y": 2.0},
                    "scale_filter": "disable",
                    "blend_method": "default",
                    "blend_type": "normal",
                    "show_transition": {"duration": 0},
                    "hide_transition": {"duration": 0},
                    "private_settings": {}
                })
            src["settings"]["id_counter"] = max([it.get("id", 0) for it in items] + [2]) + 1

    SCENE_JSON.write_text(json.dumps(data, indent=4), encoding="utf-8")
    print("Updated OBS scene collection with 'Pxl Cam (1080p Direct)' (minimize-safe 1920x1080).")


if __name__ == "__main__":
    update_basic_ini()
    update_global_ini()
    update_scene_json()

