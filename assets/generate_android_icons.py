#!/usr/bin/env python3
"""
Generate Google Play Store & Android 15/16 Adaptive Icon sets (mipmap-mdpi..xxxhdpi + playstore-icon.png)
from the canonical Cyber Kitsune (#FF8400) Pxl Cam logo assets.
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RES = ROOT / "app" / "src" / "main" / "res"

LEGACY_DENSITIES = {
    "mipmap-mdpi": 48,
    "mipmap-hdpi": 72,
    "mipmap-xhdpi": 96,
    "mipmap-xxhdpi": 144,
    "mipmap-xxxhdpi": 192,
}

FOREGROUND_DENSITIES = {
    "mipmap-mdpi": 108,
    "mipmap-hdpi": 162,
    "mipmap-xhdpi": 216,
    "mipmap-xxhdpi": 324,
    "mipmap-xxxhdpi": 432,
}


def main() -> None:
    rounded_src = Image.open(ASSETS / "pxl_cam_logo_rounded.png").convert("RGBA")
    noborder_src = Image.open(ASSETS / "pxl_cam_logo_noborder.png").convert("RGBA")
    trans_src = Image.open(ASSETS / "pxl_cam_logo_transparent.png").convert("RGBA")

    # 1. Legacy launcher icons (rounded and square/noborder)
    for folder, px in LEGACY_DENSITIES.items():
        out_dir = RES / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        rounded_src.resize((px, px), Image.Resampling.LANCZOS).save(out_dir / "ic_launcher.png")
        rounded_src.resize((px, px), Image.Resampling.LANCZOS).save(out_dir / "ic_launcher_round.png")

    # 2. Adaptive icon foregrounds (108dp canvas with 62% safe-zone inset so no mask clips ears/bars)
    for folder, px in FOREGROUND_DENSITIES.items():
        out_dir = RES / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        inner = int(round(px * 0.62))
        fg_resized = trans_src.resize((inner, inner), Image.Resampling.LANCZOS)
        offset = (px - inner) // 2
        canvas.paste(fg_resized, (offset, offset), fg_resized)
        canvas.save(out_dir / "ic_launcher_foreground.png")

    # 3. In-app header/splash drawable
    drawable_dir = RES / "drawable"
    drawable_dir.mkdir(parents=True, exist_ok=True)
    rounded_src.resize((256, 256), Image.Resampling.LANCZOS).save(drawable_dir / "pxl_cam_logo.png")

    # 4. Google Play Store 512x512 hi-res icon (no alpha, pure #000000 background + #FF8400 Cyber Kitsune)
    play_graphics = ROOT / "app" / "src" / "main" / "play" / "listings" / "en-US" / "graphics" / "icon"
    play_graphics.mkdir(parents=True, exist_ok=True)
    noborder_src.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS).save(play_graphics / "playstore_icon_512.png")
    print("Generated Android adaptive icons and Play Store 512x512 icon.")


if __name__ == "__main__":
    main()

