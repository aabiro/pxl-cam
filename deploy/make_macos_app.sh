#!/bin/bash
#
# Build a double-clickable macOS .app bundle for Pxl Cam.
#
# Why: launching via terminal shows a generic binary icon in the Dock that
# disappears on quit. A real .app bundle gives our custom collection icon, a
# one-click launch (automatically starting the Android ADB server if needed),
# and pins cleanly to the macOS Dock just like Stash Viewer.
#
# Usage:
#   deploy/make_macos_app.sh [ICON_PNG] [--candidate 1..4] [--variant rounded_transparent] [--pin]
#
#   ICON_PNG              explicit source icon PNG (optional)
#   --candidate <1|2|3|4> activate candidate 1..4 before building
#   --variant <name>      border variant: rounded_transparent (default), rounded,
#                         noborder, transparent, square, square_transparent
#   --pin                 also pin the app to the macOS Dock

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PXL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$PXL_DIR"

APP_NAME="Pxl Cam"
BUNDLE_ID="com.pxlcam.app"
EXEC_NAME="PxlCam"

CANDIDATE=""
VARIANT="rounded_transparent"
CUSTOM_ICON=""
PIN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --pin)
      PIN=1
      shift
      ;;
    --candidate)
      CANDIDATE="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    *)
      if [ -f "$1" ]; then
        CUSTOM_ICON="$1"
      fi
      shift
      ;;
  esac
done

if [ -n "$CANDIDATE" ]; then
  python3 "$PXL_DIR/assets/generate_pxl_cam_assets.py" --active "$CANDIDATE"
fi

if [ -n "$CUSTOM_ICON" ]; then
  ICON_SRC="$CUSTOM_ICON"
else
  ICON_SRC="$PXL_DIR/assets/pxl_cam_logo_${VARIANT}.png"
fi

# Install to /Applications if writable, else ~/Applications.
if [ -w /Applications ]; then
  DEST_DIR="/Applications"
else
  DEST_DIR="$HOME/Applications"
  mkdir -p "$DEST_DIR"
fi
APP="$DEST_DIR/$APP_NAME.app"

echo "Project : $PXL_DIR"
echo "Icon    : $ICON_SRC"
echo "Target  : $APP"

[ -f "$ICON_SRC" ] || { echo "ERROR: icon not found: $ICON_SRC"; exit 1; }

# --- 1a. Compile Android Camera2 Hardware ISP Server (PxlCamServer.java) ---
ANDROID_JAR="$HOME/Library/Android/sdk/platforms/android-36/android.jar"
D8_BIN="$HOME/Library/Android/sdk/build-tools/36.0.0/d8"
if [ -f "$ANDROID_JAR" ] && [ -x "$D8_BIN" ]; then
  echo "🔨 Compiling Android Camera2 Server (PxlCamServer.java -> pxl-cam-server.jar)..."
  mkdir -p "$PXL_DIR/android/build"
  rm -rf "$PXL_DIR/android/build/"*
  javac -source 11 -target 11 -cp "$ANDROID_JAR" -d "$PXL_DIR/android/build" "$PXL_DIR/android/PxlCamServer.java"
  "$D8_BIN" --min-api 26 --output "$PXL_DIR/android/pxl-cam-server.jar" $(find "$PXL_DIR/android/build" -name "*.class")
fi

# --- 1b. Compile native Swift + Metal viewer ---
echo "🔨 Compiling native Metal viewer (PxlCamApp.swift)..."
swiftc -O \
  "$PXL_DIR/mac/PxlCamApp.swift" \
  -o "$PXL_DIR/mac/PxlCamApp" \
  -framework AppKit \
  -framework Metal \
  -framework MetalKit \
  -framework CoreVideo \
  -framework VideoToolbox \
  -framework CoreMedia \
  -framework AVFoundation \
  -framework CoreImage \
  -framework QuartzCore

# --- 2. Bundle skeleton ---
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- 3. Icon: PNG -> AppIcon.icns ---
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -s format png -z "$size" "$size"             "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}.png"    >/dev/null
  sips -s format png -z "$((size*2))" "$((size*2))" "$ICON_SRC" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
cp -f "$ICON_SRC" "$APP/Contents/Resources/AppIcon.png"

# --- 4. Copy compiled native Mach-O executable directly into bundle ---
# Putting the real Mach-O binary at Contents/MacOS/PxlCam ensures macOS
# Dock binds the running window to the pinned Dock tile (no second icon).
cp -f "$PXL_DIR/mac/PxlCamApp" "$APP/Contents/MacOS/$EXEC_NAME"
chmod +x "$APP/Contents/MacOS/$EXEC_NAME"

# --- 5. Info.plist ---
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>$EXEC_NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

# --- 6. Refresh Launch Services so the icon shows immediately ---
touch "$APP"
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREG" ] && "$LSREG" -f "$APP" >/dev/null 2>&1 || true

echo "✅ Built $APP"

# --- 7. Optionally pin to Dock (idempotent; removes any existing copies) ---
if [ "$PIN" = "1" ]; then
  python3 "$SCRIPT_DIR/pin_to_dock.py" "$APP" "$APP_NAME" "$BUNDLE_ID"
fi
