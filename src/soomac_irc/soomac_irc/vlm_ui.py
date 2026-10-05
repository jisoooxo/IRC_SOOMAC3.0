"""VLM 입력을 바꾸지 않고 UI용 세 패널 JPEG만 만든다."""

from io import BytesIO

from PIL import Image, ImageDraw, ImageOps


def build_vlm_ui_jpeg(request: dict, panel_size=(480, 360)) -> bytes:
    panels = request.get("ui_panels", [])
    if len(panels) != 3:
        raise ValueError("VLM UI requires explicit reference/comparison/current panels")
    width, height = panel_size
    canvas = Image.new("RGB", (width * 3, height + 36), "#17202c")
    draw = ImageDraw.Draw(canvas)
    for index, panel in enumerate(panels):
        left = index * width
        draw.text((left + 12, 12), panel["label"], fill="white")
        source = panel.get("image")
        if source is None:
            draw.text((left + 12, 60), "NOT AVAILABLE / NOT REQUIRED", fill="#aab5c4")
            continue
        fitted = ImageOps.contain(source.convert("RGB"), (width - 8, height - 8))
        canvas.paste(fitted, (left + (width - fitted.width) // 2,
                             36 + (height - fitted.height) // 2))
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
