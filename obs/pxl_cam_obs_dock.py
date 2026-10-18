"""
OBS Studio Python Integration Script for Pxl Cam Studio.
Can be loaded in OBS via Tools -> Scripts to control the phone hardware ISP
and macOS Metal Studio pipeline directly from OBS.
"""
import urllib.request
import obspython as obs

BRIDGE_URL = "http://127.0.0.1:9877/api/control"


def _send(query: str) -> None:
    try:
        urllib.request.urlopen(f"{BRIDGE_URL}?{query}", timeout=0.5).read()
    except Exception:
        pass


def script_description() -> str:
    return (
        "<b>Pxl Cam Studio</b><br/>"
        "Direct hardware Camera2 + Metal Studio ISP controls &amp; 1080p minimize-safe stream.<br/>"
        "Custom Dock URL: <code>http://127.0.0.1:9877/dock</code><br/>"
        "1080p Direct Source URL: <code>http://127.0.0.1:9877/player</code>"
    )


def on_wand(_props, _prop):
    _send("wand=1")
    return True


def on_reset(_props, _prop):
    _send("reset=1")
    return True


def on_back_lens(_props, _prop):
    _send("lens=0")
    return True


def on_front_lens(_props, _prop):
    _send("lens=1")
    return True


def on_flash(_props, _prop):
    _send("flash=toggle")
    return True


def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_button(props, "btn_wand", "✨ Auto Studio Wand", on_wand)
    obs.obs_properties_add_button(props, "btn_back", "Back Lens (Rear)", on_back_lens)
    obs.obs_properties_add_button(props, "btn_front", "Front Lens (Selfie)", on_front_lens)
    obs.obs_properties_add_button(props, "btn_flash", "⚡ Toggle Flash Torch", on_flash)
    obs.obs_properties_add_button(props, "btn_reset", "↺ Reset Controls", on_reset)
    return props

