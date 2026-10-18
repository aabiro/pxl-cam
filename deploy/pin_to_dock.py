#!/usr/bin/env python3
"""
Idempotently pin an app to the macOS Dock — removing any existing copies first.

Why not `defaults write ... -array-add`: it can't dedupe (the path is stored
URL-encoded) so repeated runs multiply the icon, and a bare file-data entry
lacks the keys the Dock needs to render the icon while the app is closed
(showing a blank/white tile). This reads the Dock plist, strips every entry for
the app, appends one well-formed file-tile, and restarts the Dock.

Usage: pin_to_dock.py "/Applications/Pxl Cam.app" "Pxl Cam" <bundle-id>
"""

import plistlib
import subprocess
import sys
import urllib.parse
import uuid


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: pin_to_dock.py <app_path> <label> [bundle_id]")
        return 2
    app_path = sys.argv[1].rstrip("/")
    label = sys.argv[2]
    bundle_id = sys.argv[3] if len(sys.argv) > 3 else ""

    url = "file://" + urllib.parse.quote(app_path) + "/"

    xml = subprocess.run(["defaults", "export", "com.apple.dock", "-"],
                         capture_output=True).stdout
    dock = plistlib.loads(xml)
    apps = dock.get("persistent-apps", [])

    def is_target(entry):
        try:
            u = entry["tile-data"]["file-data"]["_CFURLString"]
            lbl = entry["tile-data"].get("file-label", "")
        except Exception:
            return False
        return (label.replace(" ", "%20") in u or label in u or app_path in u
                or "Pxl%20Cam" in u or lbl == label)

    kept = [a for a in apps if not is_target(a)]
    removed = len(apps) - len(kept)

    kept.append({
        "GUID": uuid.uuid4().int >> 96,
        "tile-type": "file-tile",
        "tile-data": {
            "file-data": {"_CFURLString": url, "_CFURLStringType": 15},
            "file-label": label,
            "file-type": 41,
            **({"bundle-identifier": bundle_id} if bundle_id else {}),
        },
    })
    dock["persistent-apps"] = kept

    subprocess.run(["defaults", "import", "com.apple.dock", "-"],
                   input=plistlib.dumps(dock))

    # Re-register the bundle so Launch Services has a fresh icon, then restart.
    lsreg = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
             "LaunchServices.framework/Support/lsregister")
    subprocess.run([lsreg, "-f", app_path], capture_output=True)
    subprocess.run(["killall", "Dock"])

    print(f"Removed {removed} old entr{'y' if removed == 1 else 'ies'}; pinned 1 → {app_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
