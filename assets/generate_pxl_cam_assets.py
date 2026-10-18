#!/usr/bin/env python3
"""
Generate all Pxl Cam (pxl-cam) logo asset variants across all 4 character/color candidates,
matching the exact SVG + PNG border/transparent/scaled structure of shadowscraper & stash-viewer.

Variants generated per candidate (both .svg and .png at 1024x1024):
  - <stem>_noborder.{svg,png}
  - <stem>_noborder_scaled.{svg,png}
  - <stem>_transparent.{svg,png}
  - <stem>_scaled_transparent.{svg,png}
  - <stem>_square.{svg,png}
  - <stem>_square_transparent.{svg,png}
  - <stem>_rounded.{svg,png}
  - <stem>_rounded_transparent.{svg,png}

Also populates the canonical `pxl_cam_logo_*` files for the selected candidate (--active 1..4).
"""

import argparse
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASSETS_DIR = Path(__file__).resolve().parent
CANDIDATES_DIR = ASSETS_DIR / "candidates"
SHADOWSCRAPER_LOGOS_DIR = Path("/Users/aaryn/Projects/shadowscraper/assets/logos")

SIZE = 1024
SCALE_NB = 1.00          # standard no-border
SCALE_NB_SCALED = 1.18   # enlarged no-border (_noborder_scaled / _scaled_transparent)
SCALE_B = 0.88           # bordered: fits cleanly inside 24px inset border
BORDER_W = 14
BORDER_INSET = 24
CORNER_R = 90
BG_FUZZ = "8%"

CANDIDATES: Dict[int, Dict[str, str]] = {
    1: {
        "stem": "pxl_cam_logo_teal_ronin",
        "title": "Candidate 1 — Cyber Lens Ronin (Electric Mint Teal #00F5D4)",
        "source": "candidate_1_teal_lens_ronin_source.png",
        "color": "#00f5d4",
    },
    2: {
        "stem": "pxl_cam_logo_cobalt_owl",
        "title": "Candidate 2 — Mecha Night-Vision Owl (Electric Cobalt #2997FF)",
        "source": "candidate_2_cobalt_night_owl_source.png",
        "color": "#2997ff",
    },
    3: {
        "stem": "pxl_cam_logo_lime_cyborg",
        "title": "Candidate 3 — Cyclops Shutter Android (Laser Volt Lime #B8FF01)",
        "source": "candidate_3_lime_shutter_cyborg_source.png",
        "color": "#b8ff01",
    },
    4: {
        "stem": "pxl_cam_logo_coral_fox",
        "title": "Candidate 4 — Cyber Kitsune Director (Vibrant Cyber Orange #FF8400)",
        "source": "candidate_4_coral_cyber_fox_source.png",
        "color": "#ff8400",
    },
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ERROR running {' '.join(cmd)}:\n{result.stderr.strip()}")


def png_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def lock_transparent_png_color(png_path: Path, target_hex: str) -> None:
    """
    Ensure every visible pixel in a rendered transparent PNG has 100% pure,
    full-saturation target RGB (zero black premultiplication darkening from
    bilinear scaling) and crisp, high-opacity edges, tagged as explicit sRGB.
    """
    from PIL import Image, ImageCms
    import numpy as np

    tr, tg, tb = hex_to_rgb(target_hex)
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im, dtype=np.float32)

    alpha = arr[:, :, 3] / 255.0
    vis = alpha > 0.02
    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    out[vis, 0] = tr
    out[vis, 1] = tg
    out[vis, 2] = tb
    # Boost edge opacity so anti-aliased edges never look washed-out or faded
    boosted_a = np.where(vis, np.clip((alpha ** 0.65) * 255.0, 0.0, 255.0), 0.0)
    out[:, :, 3] = np.round(boosted_a).astype(np.uint8)

    srgb_profile = ImageCms.createProfile("sRGB")
    res_im = Image.fromarray(out, mode="RGBA")
    res_im.save(
        png_path,
        format="PNG",
        icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes(),
    )


def tag_opaque_png_srgb(png_path: Path) -> None:
    from PIL import Image, ImageCms

    srgb_profile = ImageCms.createProfile("sRGB")
    im = Image.open(png_path).convert("RGB")
    im.save(
        png_path,
        format="PNG",
        icc_profile=ImageCms.ImageCmsProfile(srgb_profile).tobytes(),
    )


def build_source_renders(source_png: Path, target_hex: str, tmp_dir: Path) -> Tuple[str, str]:
    from PIL import Image, ImageCms
    import numpy as np

    opaque_png = tmp_dir / f"{source_png.stem}_1024.png"
    transparent_png = tmp_dir / f"{source_png.stem}_1024_transparent.png"

    tr, tg, tb = hex_to_rgb(target_hex)
    src_im = Image.open(source_png).convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    arr = np.array(src_im, dtype=np.float32)

    # Extract pure foreground coverage mask from max channel intensity so
    # desaturated/shaded raster pixels are restored to 100% pure neon color
    # and anti-aliased edge pixels have zero black muddiness.
    peak = arr.max(axis=2)
    fg_samples = peak[peak > 28.0]
    ref_white = float(np.percentile(fg_samples, 75)) if len(fg_samples) > 0 else 240.0
    ref_white = max(ref_white, 160.0)

    coverage = np.clip((peak - 18.0) / (ref_white - 18.0), 0.0, 1.0)
    # Crisp gamma curve so interior artwork is 100% solid target color
    solid_cov = np.where(coverage > 0.02, np.clip(coverage ** 0.58, 0.0, 1.0), 0.0)

    # 1. Opaque render on pure #000000 black with 100% pure target color
    opaque_arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    opaque_arr[:, :, 0] = np.round(solid_cov * tr).astype(np.uint8)
    opaque_arr[:, :, 1] = np.round(solid_cov * tg).astype(np.uint8)
    opaque_arr[:, :, 2] = np.round(solid_cov * tb).astype(np.uint8)

    srgb_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(opaque_arr, mode="RGB").save(opaque_png, format="PNG", icc_profile=srgb_bytes)

    # 2. Transparent render: every visible pixel has exact pure RGB = (tr, tg, tb)
    # and crisp alpha coverage (zero black darkening on any foreground pixel).
    trans_arr = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    vis = solid_cov > 0.02
    trans_arr[vis, 0] = tr
    trans_arr[vis, 1] = tg
    trans_arr[vis, 2] = tb
    trans_arr[:, :, 3] = np.round(solid_cov * 255.0).astype(np.uint8)

    Image.fromarray(trans_arr, mode="RGBA").save(transparent_png, format="PNG", icc_profile=srgb_bytes)

    return png_data_uri(opaque_png), png_data_uri(transparent_png)


def make_borders(border_color: str) -> Tuple[str, str]:
    inset = BORDER_INSET
    w, h = SIZE - 2 * inset, SIZE - 2 * inset
    square_border = (
        f'  <rect x="{inset}" y="{inset}" width="{w}" height="{h}" '
        f'fill="none" stroke="{border_color}" stroke-width="{BORDER_W}"/>'
    )
    rounded_border = (
        f'  <rect x="{inset}" y="{inset}" width="{w}" height="{h}" '
        f'rx="{CORNER_R}" ry="{CORNER_R}" '
        f'fill="none" stroke="{border_color}" stroke-width="{BORDER_W}"/>'
    )
    return square_border, rounded_border


def make_svg(image_uri: str, transparent: bool, border: Optional[str], scale: float) -> str:
    c = SIZE / 2
    ft = f"translate({c},{c}) scale({scale}) translate(-{c},-{c})"

    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<svg xmlns="http://www.w3.org/2000/svg"',
        '     xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1"',
        f'     width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">',
    ]

    if not transparent:
        out.append(f'  <rect width="{SIZE}" height="{SIZE}" fill="#000000"/>')

    out.append(f'  <g transform="{ft}">')
    out.append(
        f'    <image x="0" y="0" width="{SIZE}" height="{SIZE}" '
        f'preserveAspectRatio="none" href="{image_uri}" xlink:href="{image_uri}"/>'
    )
    out.append('  </g>')

    if border:
        out.append(border)

    out.append('</svg>')
    return '\n'.join(out)


def generate_suite_for_candidate(
    cand_id: int,
    out_dir: Path,
    tmp_dir: Path,
) -> Dict[str, Path]:
    info = CANDIDATES[cand_id]
    stem = info["stem"]
    color = info["color"]
    source_path = CANDIDATES_DIR / info["source"]
    if not source_path.exists():
        raise SystemExit(f"Missing candidate source PNG: {source_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    opaque_uri, transparent_uri = build_source_renders(source_path, color, tmp_dir)
    square_border, rounded_border = make_borders(color)

    variant_specs = [
        ("noborder",            False, None,           SCALE_NB),
        ("noborder_scaled",     False, None,           SCALE_NB_SCALED),
        ("transparent",         True,  None,           SCALE_NB),
        ("scaled_transparent",  True,  None,           SCALE_NB_SCALED),
        ("square",              False, square_border,  SCALE_B),
        ("square_transparent",  True,  square_border,  SCALE_B),
        ("rounded",             False, rounded_border, SCALE_B),
        ("rounded_transparent", True,  rounded_border, SCALE_B),
    ]

    generated: Dict[str, Path] = {}
    print(f"==> Generating 16 assets (8 SVG + 8 PNG) for {info['title']}...")

    for suffix, is_transparent, border_svg, scale in variant_specs:
        full_stem = f"{stem}_{suffix}"
        svg_path = out_dir / f"{full_stem}.svg"
        png_path = out_dir / f"{full_stem}.png"
        image_uri = transparent_uri if is_transparent else opaque_uri

        svg_path.write_text(
            make_svg(image_uri, is_transparent, border_svg, scale),
            encoding="utf-8",
        )

        cmd = ["rsvg-convert", "-w", str(SIZE), "-h", str(SIZE)]
        if is_transparent:
            cmd += ["--background-color", "none"]
        cmd += [str(svg_path), "-o", str(png_path)]
        run(cmd)

        if is_transparent:
            lock_transparent_png_color(png_path, color)
        else:
            tag_opaque_png_srgb(png_path)

        # Also place directly in Scripts/pxl_cam/assets/ and shadowscraper/assets/logos/
        shutil.copy2(svg_path, ASSETS_DIR / svg_path.name)
        shutil.copy2(png_path, ASSETS_DIR / png_path.name)
        if SHADOWSCRAPER_LOGOS_DIR.is_dir():
            shutil.copy2(svg_path, SHADOWSCRAPER_LOGOS_DIR / svg_path.name)
            shutil.copy2(png_path, SHADOWSCRAPER_LOGOS_DIR / png_path.name)

        generated[f"{suffix}.svg"] = svg_path
        generated[f"{suffix}.png"] = png_path

    return generated


def activate_candidate(cand_id: int, generated: Dict[str, Path], copy_to_shadowscraper: bool = True) -> None:
    info = CANDIDATES[cand_id]
    print(f"\n==> Setting active Pxl Cam icon collection to {info['title']}")
    for key, src_path in generated.items():
        # key is e.g. "rounded.png", "rounded_transparent.svg"
        dest_name = f"pxl_cam_logo_{key}"
        dest_path = ASSETS_DIR / dest_name
        shutil.copy2(src_path, dest_path)
        if copy_to_shadowscraper and SHADOWSCRAPER_LOGOS_DIR.is_dir():
            shutil.copy2(src_path, SHADOWSCRAPER_LOGOS_DIR / dest_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pxl Cam logo asset collection.")
    parser.add_argument(
        "--active",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Which candidate (1..4) to activate as the default pxl_cam_logo_* set (default: 4 — Cyber Kitsune Fox).",
    )
    args = parser.parse_args()

    all_generated: Dict[int, Dict[str, Path]] = {}
    with tempfile.TemporaryDirectory(prefix="pxl_cam_assets_") as tmp:
        tmp_path = Path(tmp)
        for cid in (1, 2, 3, 4):
            cand_out_dir = CANDIDATES_DIR / f"candidate_{cid}"
            all_generated[cid] = generate_suite_for_candidate(cid, cand_out_dir, tmp_path)

    activate_candidate(args.active, all_generated[args.active])
    print("\nAll 4 candidate variant collections (64 files) + active pxl_cam_logo_* assets (16 files) generated successfully.")


if __name__ == "__main__":
    main()
