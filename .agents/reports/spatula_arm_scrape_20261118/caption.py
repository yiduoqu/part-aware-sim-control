# SPDX-License-Identifier: Apache-2.0
"""Caption a run's camera video the way the cable scenes' recordings are: a title, the materials, and the clock."""

import json
import sys
from pathlib import Path

import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # noqa: TID253

run, camera, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
TITLE = "tool use \u00b7 scrape a pan with a spatula"
PARTS = "head silicone E 1.5e6 Pa \u00b7 handle rigid \u00b7 pan rigid"
font = next(p for p in (r"C:\Windows\Fonts\verdanab.ttf", r"C:\Windows\Fonts\arialbd.ttf") if Path(p).exists())
rows = (
    json.loads((run / "diagnostics.json").read_text())
    if (run / "diagnostics.json").exists()
    else json.loads((run.parent / "diagnostics.json").read_text())
)


def label(image: Image.Image, text: str, at, size: int) -> None:
    """White bold ``text`` on a dark see-through box with its top-left corner at ``at``."""
    face = ImageFont.truetype(font, size)
    probe = ImageDraw.Draw(image)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=face)
    pad = round(0.45 * size)
    box = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (16, 16, 20, 175))
    image.alpha_composite(box, (at[0], at[1]))
    ImageDraw.Draw(image).text((at[0] + pad - left, at[1] + pad - top), text, font=face, fill=(255, 255, 255, 255))


writer = imageio.get_writer(run / name, fps=60, quality=8, macro_block_size=8)
written = 0
for index, frame in enumerate(iio.imiter(run / f"{camera}.mp4")):
    if index < start or index >= len(rows):
        continue
    image = Image.fromarray(frame).convert("RGBA")
    height = image.size[1]
    big, small = round(height * 0.036), round(height * 0.030)
    label(image, TITLE, (round(height * 0.04), round(height * 0.04)), big)
    label(image, PARTS, (round(height * 0.04), round(height * 0.04) + round(big * 1.9) + 4), small)
    clock = rows[index]["time_s"]
    stamp = f"t =  {int(clock // 3600):02d}:{int(clock // 60) % 60:02d}:{clock % 60:06.3f}"
    label(image, stamp, (round(height * 0.03), height - round(small * 1.9) - round(height * 0.02)), small)
    writer.append_data(np.asarray(image.convert("RGB")))
    if len(sys.argv) > 5 and index == int(sys.argv[5]):
        image.convert("RGB").save(run / f"check_{camera}.png")
    written += 1
writer.close()
print(camera, "frames", written, "font", font)
