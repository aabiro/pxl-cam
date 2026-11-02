#!/usr/bin/env bash
# Print one adb serial for the known phone.
#
# USB wins when that transport is up and its serial matches. With no cable,
# discover wireless ADB from mDNS (the TLS port rotates) and, only if that
# finds nothing, from TCP 5555 on local non-virtual subnets. A saved IP is
# not an address to dial. This file is the same on the tower
# (pixelspark/scripts) and on the Mac (pxl-cam/resolve_android_device.sh).
#
# stdout: one serial for `adb -s`
# stderr: diagnostics
# exit 0 found, 1 not found, 2 unauthorized or ambiguous
#
# Compatible with macOS bash 3.2 (no associative arrays, no empty-array
# expansion under `set -u`).

if [[ -z "${BASH_VERSION:-}" ]]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -u

export PATH="${PATH:-}:/opt/homebrew/bin:/usr/local/bin:${HOME}/Android/Sdk/platform-tools"
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
  printf 'resolve_android_device: %s\n' "$*" >&2
}

die() {
  log "$*"
  exit 1
}

refuse() {
  log "$*"
  exit 2
}

env_value() {
  local file="$1" key="$2"
  [[ -f "$file" ]] || return 0
  awk -F= -v k="$key" '
    $1 == k {
      v = $2
      gsub(/["'\'' \r]/, "", v)
      print v
      exit
    }
  ' "$file"
}

load_expected() {
  if [[ -n "${ANDROID_USB_SERIAL+x}" ]]; then
    printf '%s' "$ANDROID_USB_SERIAL"
    return 0
  fi
  local value=""
  value="$(env_value "$SCRIPT_DIR/.env" ADB_USB_SERIAL)"
  if [[ -z "$value" ]]; then
    value="$(env_value "$SCRIPT_DIR/../.env" ADB_USB_SERIAL)"
  fi
  if [[ -z "$value" ]]; then
    value="ZY22H9HQNL"
  fi
  printf '%s' "$value"
}

EXPECTED="$(load_expected)"
TLS_ENDPOINTS=""

env_file_for_cache() {
  local candidate
  for candidate in "$SCRIPT_DIR/.env" "$SCRIPT_DIR/../.env"; do
    if [[ -f "$candidate" ]] && grep -q '^ADB_WIRELESS_HOST=' "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

list_devices() {
  adb devices -l </dev/null 2>/dev/null | awk '
    NR > 1 && ($2 == "device" || $2 == "unauthorized" || $2 == "offline") {
      kind = ($0 ~ /usb:/) ? "usb" : "net"
      printf "%s\t%s\t%s\n", $1, $2, kind
    }
  '
}

device_state() {
  local serial="$1" line
  line="$(list_devices | awk -F '\t' -v s="$serial" '$1 == s { print $2; exit }')"
  [[ -n "$line" ]] || return 1
  printf '%s' "$line"
}

prop_of() {
  local serial="$1" tmp pid i out
  tmp="$(mktemp)"
  adb -s "$serial" shell getprop ro.serialno </dev/null >"$tmp" 2>/dev/null &
  pid=$!
  i=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ "$i" -ge 30 ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      rm -f "$tmp"
      return 1
    fi
    sleep 0.1
    i=$((i + 1))
  done
  wait "$pid" 2>/dev/null || true
  out="$(tr -d '[:space:]' <"$tmp" || true)"
  rm -f "$tmp"
  [[ -n "$out" ]] || return 1
  printf '%s' "$out"
}

adb_connect() {
  local ep="$1" tmp pid i
  tmp="$(mktemp)"
  adb connect "$ep" </dev/null >"$tmp" 2>&1 &
  pid=$!
  i=0
  while kill -0 "$pid" 2>/dev/null; do
    if [[ "$i" -ge 40 ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      rm -f "$tmp"
      return 1
    fi
    sleep 0.1
    i=$((i + 1))
  done
  wait "$pid" 2>/dev/null || true
  rm -f "$tmp"
  return 0
}

disconnect_if_wireless() {
  local serial="$1"
  case "$serial" in
    *:*|*_adb*)
      adb disconnect "$serial" </dev/null >/dev/null 2>&1 || true
      ;;
  esac
}

matches_expected() {
  local prop="$1"
  [[ -z "$EXPECTED" || "$prop" == "$EXPECTED" ]]
}

prefer_serial() {
  # stdin: verified serials, one per line. stdout: the one to use.
  local line usb="" tls="" classic="" other=""
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    if [[ -n "$EXPECTED" && "$line" == "$EXPECTED" ]]; then
      usb="$line"
    elif [[ "$line" == *_adb-tls-connect._tcp* ]]; then
      [[ -n "$tls" ]] || tls="$line"
    elif printf '%s\n' "$TLS_ENDPOINTS" | grep -F -x -q -- "$line"; then
      [[ -n "$tls" ]] || tls="$line"
    elif [[ "$line" == *:5555 ]]; then
      [[ -n "$classic" ]] || classic="$line"
    else
      [[ -n "$other" ]] || other="$line"
    fi
  done
  if [[ -n "$usb" ]]; then
    printf '%s' "$usb"
  elif [[ -n "$tls" ]]; then
    printf '%s' "$tls"
  elif [[ -n "$classic" ]]; then
    printf '%s' "$classic"
  else
    printf '%s' "$other"
  fi
}

count_lines() {
  awk 'NF && !seen[$0]++ { n++ } END { print n + 0 }'
}

remember_endpoint() {
  local serial="$1" host="" port="" name ep len best=0 envfile
  if [[ "$serial" == *.*.*.*:* ]]; then
    host="${serial%%:*}"
    port="${serial##*:}"
  else
    while IFS="$(printf '\t')" read -r name _ ep; do
      [[ -n "$name" && -n "$ep" ]] || continue
      case "$serial" in
        "$name"*)
          len="${#name}"
          if [[ "$len" -gt "$best" ]]; then
            best="$len"
            host="${ep%%:*}"
            port="${ep##*:}"
          fi
          ;;
      esac
    done <<EOF
$(list_mdns)
EOF
  fi
  if [[ -z "$host" ]]; then
    host="$(adb -s "$serial" shell ip -4 -o addr show wlan0 </dev/null 2>/dev/null | awk '{ print $4 }' | head -n 1 | cut -d/ -f1 | tr -d '[:space:]' || true)"
  fi
  [[ "$host" == *.*.*.* ]] || return 0
  envfile="$(env_file_for_cache || true)"
  [[ -n "$envfile" ]] || return 0
  RESOLVE_ENV_FILE="$envfile" \
  RESOLVE_HOST="$host" \
  RESOLVE_PORT="$port" \
  RESOLVE_SERIAL="$serial" \
  python3 - <<'PY'
import os
path = os.environ["RESOLVE_ENV_FILE"]
host = os.environ.get("RESOLVE_HOST", "")
port = os.environ.get("RESOLVE_PORT", "")
serial = os.environ.get("RESOLVE_SERIAL", "")
updates = {"ADB_WIRELESS_HOST": host, "DROIDCAM_WIFI_IP": host}
if port.isdigit():
    updates["ADB_WIRELESS_PORT"] = port
if serial and "\n" not in serial and "\r" not in serial:
    updates["ADB_WIRELESS_SERIAL"] = serial
with open(path, "r", encoding="utf-8") as handle:
    raw = handle.read()
if not any(line.startswith("ADB_WIRELESS_HOST=") for line in raw.splitlines()):
    raise SystemExit(0)
seen = set()
out = []
for line in raw.splitlines():
    key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else None
    if key in updates:
        out.append("%s=%s" % (key, updates[key]))
        seen.add(key)
    else:
        out.append(line)
for key, value in updates.items():
    if key not in seen:
        out.append("%s=%s" % (key, value))
text = "\n".join(out)
if raw.endswith("\n") or raw == "":
    text += "\n"
if text == raw:
    raise SystemExit(0)
tmp = path + ".resolve-tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    handle.write(text)
os.replace(tmp, path)
PY
  log "cached wireless address ${host}${port:+:$port} for DroidCam"
}

finish() {
  local serial="$1" how="$2"
  log "$how"
  remember_endpoint "$serial" || log "could not cache the wireless address"
  printf '%s\n' "$serial"
  exit 0
}

scan_port() {
  local port="$1" stamp now mtime age
  [[ "$port" =~ ^[0-9]+$ ]] || return 1
  stamp="/tmp/resolve-android-device.noscan.${port}"
  if [[ -f "$stamp" ]]; then
    now="$(date +%s)"
    mtime="$(stat -c %Y "$stamp" 2>/dev/null || stat -f %m "$stamp" 2>/dev/null || echo 0)"
    age=$((now - mtime))
    if [[ "$age" -lt 8 ]]; then
      return 0
    fi
  fi
  local found py_rc=0
  found="$(python3 - "$port" <<'PY'
import ipaddress
import os
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

port = int(sys.argv[1])
skip = re.compile(
    r"^(lo|docker|br-|veth|virbr|tailscale|wg|zt|tun|tap|cni|flannel|"
    r"podman|vboxnet|vmnet|xcl|awdl|llw|bridge|gif|stf|utun|ap[0-9]|anpi|usb)",
    re.I,
)

def rows():
    sysname = os.uname().sysname
    found = []
    if sysname == "Linux":
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[2] != "inet" or "/" not in parts[3]:
                continue
            ip, plen = parts[3].split("/", 1)
            found.append((parts[1], ip, int(plen)))
        return found
    out = subprocess.check_output(["ifconfig"], text=True, errors="replace")
    iface = None
    for line in out.splitlines():
        if line and not line[0].isspace():
            iface = line.split(":", 1)[0]
            continue
        if iface is None:
            continue
        hexmask = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)\s+netmask\s+0x([0-9a-fA-F]+)", line)
        dotted = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)\s+netmask\s+(\d+\.\d+\.\d+\.\d+)", line)
        if hexmask:
            plen = bin(int(hexmask.group(2), 16)).count("1")
            found.append((iface, hexmask.group(1), plen))
        elif dotted:
            octets = [int(part) for part in dotted.group(2).split(".")]
            plen = sum(bin(part).count("1") for part in octets)
            found.append((iface, dotted.group(1), plen))
    return found

def usable(iface, ip):
    if skip.match(iface):
        return False
    addr = ipaddress.ip_address(ip)
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return False
    if addr in ipaddress.ip_network("100.64.0.0/10"):
        return False
    return True

hosts = []
seen = set()
own = set()
for iface, ip, plen in rows():
    own.add(ip)
    if not usable(iface, ip):
        continue
    if plen < 24:
        plen = 24
    net = ipaddress.ip_network("%s/%s" % (ip, plen), strict=False)
    for host in net.hosts():
        text = str(host)
        if text in seen or text in own:
            continue
        seen.add(text)
        hosts.append(text)
        if len(hosts) > 1024:
            break

def open_port(ip):
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        sock.connect((ip, port))
        return ip
    except OSError:
        return None
    finally:
        sock.close()

if not hosts:
    raise SystemExit(0)
with ThreadPoolExecutor(max_workers=64) as pool:
    futures = [pool.submit(open_port, ip) for ip in hosts]
    hits = []
    for future in as_completed(futures):
        hit = future.result()
        if hit:
            hits.append(hit)
for hit in sorted(hits, key=lambda item: tuple(int(part) for part in item.split("."))):
    print(hit)
PY
)" || py_rc=$?
  if [[ "${py_rc:-0}" -ne 0 ]]; then
    log "subnet scan failed"
    return 1
  fi
  if [[ -z "$found" ]]; then
    : >"$stamp"
  else
    rm -f "$stamp"
    printf '%s\n' "$found"
  fi
}

list_mdns() {
  adb mdns services </dev/null 2>/dev/null | awk '
    NR > 1 && $2 ~ /^_adb/ && $3 ~ /:[0-9]+$/ {
      printf "%s\t%s\t%s\n", $1, $2, $3
    }
  '
}

collect_verified_online() {
  # prints verified wireless serials already in state "device"
  local serial state kind prop
  while IFS="$(printf '\t')" read -r serial state kind; do
    [[ -n "$serial" ]] || continue
    [[ "$kind" == "net" ]] || continue
    [[ "$state" == "device" ]] || continue
    case "$serial" in
      emulator-*) continue ;;
    esac
    prop="$(prop_of "$serial" || true)"
    [[ -n "$prop" ]] || continue
    if matches_expected "$prop"; then
      printf '%s\n' "$serial"
    fi
  done <<EOF
$(list_devices)
EOF
}

try_endpoint() {
  # Connect ep, then accept every verified wireless serial. Disconnect a
  # wireless serial only when this attempt is what brought it online and the
  # hardware serial does not match. A transport that was already online stays.
  local ep="$1" before line state kind prop was_online
  before="$(mktemp)"
  list_devices | awk -F '\t' '$2 == "device" { print $1 }' | sort >"$before"
  adb_connect "$ep" || true
  sleep 0.4
  while IFS="$(printf '\t')" read -r line state kind; do
    [[ -n "$line" ]] || continue
    [[ "$kind" == "net" ]] || continue
    case "$line" in
      emulator-*) continue ;;
    esac
    if grep -F -x -q "$line" "$before"; then
      was_online=1
    else
      was_online=0
    fi
    if [[ "$state" == "unauthorized" ]]; then
      if [[ "$was_online" -eq 0 ]]; then
        SAW_UNAUTH=1
      fi
      continue
    fi
    [[ "$state" == "device" ]] || continue
    prop="$(prop_of "$line" || true)"
    if [[ -n "$prop" ]] && matches_expected "$prop"; then
      VERIFIED_FILE_HITS="$(printf '%s\n%s\n' "$VERIFIED_FILE_HITS" "$line")"
      continue
    fi
    if [[ "$was_online" -eq 0 ]]; then
      disconnect_if_wireless "$line"
    fi
  done <<EOF
$(list_devices)
EOF
  rm -f "$before"
}

resolve() {
  command -v adb >/dev/null 2>&1 || die "adb is not on PATH"
  adb start-server </dev/null >/dev/null 2>&1 || die "adb server did not start"
  TLS_ENDPOINTS="$(list_mdns | awk -F '\t' '$2 ~ /_adb-tls-connect/ { print $3 }')"

  local serial state kind
  local usb_match="" usb_count=0 usb_only=""
  SAW_UNAUTH=0

  while IFS="$(printf '\t')" read -r serial state kind; do
    [[ -n "$serial" ]] || continue
    if [[ "$state" == "unauthorized" ]]; then
      if [[ "$kind" == "usb" && -n "$EXPECTED" && "$serial" == "$EXPECTED" ]]; then
        SAW_UNAUTH=1
      elif [[ "$kind" == "net" ]]; then
        case "$serial" in
          emulator-*) ;;
          *)
            if [[ -z "$EXPECTED" || "$serial" == *"$EXPECTED"* ]]; then
              SAW_UNAUTH=1
            fi
            ;;
        esac
      fi
    fi
    [[ "$kind" == "usb" && "$state" == "device" ]] || continue
    usb_count=$((usb_count + 1))
    usb_only="$serial"
    if [[ -n "$EXPECTED" && "$serial" == "$EXPECTED" ]]; then
      usb_match="$serial"
    fi
  done <<EOF
$(list_devices)
EOF

  if [[ -n "$usb_match" ]]; then
    finish "$usb_match" "USB ${usb_match} is online"
  fi
  if [[ -z "$EXPECTED" && "$usb_count" -eq 1 ]]; then
    finish "$usb_only" "USB ${usb_only} is the only wired device"
  fi
  if [[ -z "$EXPECTED" && "$usb_count" -gt 1 ]]; then
    refuse "several USB devices are attached and no ANDROID_USB_SERIAL is set"
  fi
  if [[ -n "$EXPECTED" && "$usb_count" -gt 1 ]]; then
    refuse "several USB devices are attached and none is ${EXPECTED}"
  fi

  local online
  online="$(collect_verified_online || true)"
  if [[ -n "$online" ]]; then
    local n chosen
    n="$(printf '%s\n' "$online" | count_lines)"
    if [[ -z "$EXPECTED" && "$n" -gt 1 ]]; then
      refuse "several wireless devices are online and no ANDROID_USB_SERIAL is set"
    fi
    chosen="$(printf '%s\n' "$online" | prefer_serial)"
    if [[ -n "$chosen" ]]; then
      local label="${EXPECTED:-one wireless device}"
      finish "$chosen" "wireless ${chosen} is already online (${label})"
    fi
  fi

  local name type ep score
  local preferred="" rest=""
  while IFS="$(printf '\t')" read -r name type ep; do
    [[ -n "$ep" ]] || continue
    score=5
    if [[ "$type" == *_adb-tls-connect* ]]; then
      score=1
    elif [[ "$type" == _adb._tcp* ]]; then
      score=3
    fi
    if [[ -n "$EXPECTED" && "$name" == *"$EXPECTED"* ]]; then
      score=$((score - 1))
    fi
    if [[ "$score" -le 2 ]]; then
      preferred="${preferred}${score} ${ep}
"
    else
      rest="${rest}${score} ${ep}
"
    fi
  done <<EOF
$(list_mdns)
EOF

  VERIFIED_FILE_HITS=""
  local ordered attempt_ep
  ordered="$(printf '%s%s\n' "$preferred" "$rest" | awk 'NF { print }' | sort -k1,1n | awk '{ print $2 }')"
  if [[ -n "$ordered" ]]; then
    while IFS= read -r attempt_ep; do
      [[ -n "$attempt_ep" ]] || continue
      log "mDNS connect ${attempt_ep}"
      try_endpoint "$attempt_ep"
    done <<EOF
$ordered
EOF
  fi

  if [[ -n "$VERIFIED_FILE_HITS" ]]; then
    local n chosen
    n="$(printf '%s\n' "$VERIFIED_FILE_HITS" | count_lines)"
    if [[ -z "$EXPECTED" && "$n" -gt 1 ]]; then
      refuse "several mDNS devices matched and no ANDROID_USB_SERIAL is set"
    fi
    chosen="$(printf '%s\n' "$VERIFIED_FILE_HITS" | prefer_serial)"
    [[ -n "$chosen" ]] && finish "$chosen" "mDNS attached ${chosen}"
  fi

  log "mDNS has no verified phone; scanning local subnet for TCP 5555"
  local hits hit
  hits="$(scan_port 5555 || true)"
  VERIFIED_FILE_HITS=""
  if [[ -n "$hits" ]]; then
    while IFS= read -r hit; do
      [[ -n "$hit" ]] || continue
      log "port 5555 open on ${hit}"
      try_endpoint "${hit}:5555"
    done <<EOF
$hits
EOF
  fi
  if [[ -n "$VERIFIED_FILE_HITS" ]]; then
    local n chosen
    n="$(printf '%s\n' "$VERIFIED_FILE_HITS" | count_lines)"
    if [[ -z "$EXPECTED" && "$n" -gt 1 ]]; then
      refuse "several hosts on port 5555 matched and no ANDROID_USB_SERIAL is set"
    fi
    chosen="$(printf '%s\n' "$VERIFIED_FILE_HITS" | prefer_serial)"
    [[ -n "$chosen" ]] && finish "$chosen" "subnet scan attached ${chosen}"
  fi

  if [[ "$SAW_UNAUTH" -eq 1 ]]; then
    refuse "phone ${EXPECTED:-} is unauthorized. Accept the debugging prompt on the phone"
  fi
  if [[ -n "$EXPECTED" ]]; then
    die "phone ${EXPECTED} was not found on USB, mDNS, or local port 5555"
  fi
  die "no adb device was found on USB, mDNS, or local port 5555"
}

lan_port() {
  local port="${1:-}"
  [[ "$port" =~ ^[0-9]+$ ]] || die "--lan-port needs a numeric port"
  # A direct lookup must not reuse the negative cache from a watcher loop.
  rm -f "/tmp/resolve-android-device.noscan.${port}"
  local hits n
  hits="$(scan_port "$port" || true)"
  n="$(printf '%s\n' "$hits" | count_lines)"
  if [[ "$n" -eq 1 ]]; then
    printf '%s\n' "$hits"
    exit 0
  fi
  if [[ "$n" -gt 1 ]]; then
    refuse "several LAN hosts are listening on port ${port}"
  fi
  die "no LAN host is listening on port ${port}"
}

if [[ "${1:-}" == "--lan-port" ]]; then
  lan_port "${2:-}"
fi
if [[ "$#" -gt 0 ]]; then
  die "usage: resolve_android_device.sh [--lan-port PORT]"
fi
resolve
