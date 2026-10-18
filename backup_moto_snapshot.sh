#!/bin/zsh
set -euo pipefail

SERIAL="${1:-ZY22H9HQNL}"
BACKUP_ROOT="/Users/aaryn/Projects/pxl-cam/backups/moto_g_pure_snapshot"
mkdir -p "$BACKUP_ROOT/sdcard" "$BACKUP_ROOT/apks" "$BACKUP_ROOT/settings"

echo "=== [1/4] Backing up System / Secure / Global Settings & Device Props ==="
adb -s "$SERIAL" shell getprop > "$BACKUP_ROOT/settings/device_getprop.txt"
adb -s "$SERIAL" shell settings list system > "$BACKUP_ROOT/settings/settings_system.txt"
adb -s "$SERIAL" shell settings list secure > "$BACKUP_ROOT/settings/settings_secure.txt"
adb -s "$SERIAL" shell settings list global > "$BACKUP_ROOT/settings/settings_global.txt"
adb -s "$SERIAL" shell pm list packages -3 > "$BACKUP_ROOT/settings/user_packages.txt"

echo "=== [2/4] Pulling /sdcard/ Media & Documents (~205 MB) ==="
for dir in DCIM Pictures Movies Download Documents Music Podcasts Recordings Ringtones Alarms Notifications; do
    adb -s "$SERIAL" pull "/sdcard/$dir" "$BACKUP_ROOT/sdcard/" 2>/dev/null || true
done
adb -s "$SERIAL" shell 'ls /sdcard/*.mp4 /sdcard/*.xml 2>/dev/null' | tr -d '\r' | while read -r f; do
    [ -n "$f" ] && adb -s "$SERIAL" pull "$f" "$BACKUP_ROOT/sdcard/" 2>/dev/null || true
done

echo "=== [3/4] Archiving User APKs & Generating 1-Click Restore Script ==="
RESTORE_SCRIPT="$BACKUP_ROOT/restore_all_apks.sh"
cat << 'EOF' > "$RESTORE_SCRIPT"
#!/bin/zsh
set -e
SERIAL="${1:-ZY22H9HQNL}"
SCRIPT_DIR="${0:A:h}"
for pkg_dir in "$SCRIPT_DIR/apks"/*(/); do
    pkg_name="${pkg_dir:t}"
    echo "Restoring $pkg_name..."
    adb -s "$SERIAL" install-multiple "$pkg_dir"/*.apk || true
done
echo "All APKs restored!"
EOF
chmod +x "$RESTORE_SCRIPT"

adb -s "$SERIAL" shell pm list packages -3 | tr -d '\r' | sed 's/^package://' | while read -r pkg; do
    [ -z "$pkg" ] && continue
    mkdir -p "$BACKUP_ROOT/apks/$pkg"
    adb -s "$SERIAL" shell pm path "$pkg" | tr -d '\r' | sed 's/^package://' | while read -r apk_path; do
        [ -z "$apk_path" ] && continue
        apk_name=$(basename "$apk_path")
        if [ ! -f "$BACKUP_ROOT/apks/$pkg/$apk_name" ]; then
            adb -s "$SERIAL" pull "$apk_path" "$BACKUP_ROOT/apks/$pkg/$apk_name" >/dev/null
        fi
    done
done

echo "=== [4/4] Triggering Android Backup Manager (Cloud / Local Transport) ==="
adb -s "$SERIAL" shell bmgr run >/dev/null 2>&1 || true

echo "Snapshot complete at: $BACKUP_ROOT"
du -sh "$BACKUP_ROOT"/*
