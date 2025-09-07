# app_openai.py
# Streamlit app using OpenAI Images API (gpt-image-1) with selectable outputs
# and a *consistent background* option using rembg for subject extraction.
#
# Setup:
#   pip install -r requirements.txt
#   export OPENAI_API_KEY=...
#   streamlit run app_openai.py

import os
import base64
import streamlit as st
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional

try:
    from openai import OpenAI
except Exception:
    st.error("Please `pip install openai>=1.0.0`")
    raise

# For consistent background compositing
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None

st.set_page_config(page_title="AI Dress Photoshoot (OpenAI)", page_icon="👗", layout="wide")
st.title("👗 AI Dress Photoshoot — OpenAI API (gpt-image-1)")
st.caption("Generate Front, Back, and Feature (close-up) with selectable outputs and optional consistent background.")

# Presets
PRESETS = {"Post (2160×2160)": (2160, 2160), "Story/Reel (1080×1920)": (1080, 1920)}
with st.sidebar:
    st.header("⚙️ Settings")
    preset = st.selectbox("Canvas preset", list(PRESETS.keys()), index=0)
    width, height = PRESETS[preset]
    creativity = st.slider("Creativity (1=literal, 10=creative)", 1, 10, 5, step=1)
    steps = st.slider("Detail level (steps)", 10, 40, 25, step=1)

    st.markdown("---")
    st.subheader("What to generate")
    gen_front = st.checkbox("Front", True)
    gen_back = st.checkbox("Back", True)
    gen_feature = st.checkbox("Feature (close-up)", True)

    st.markdown("---")
    st.subheader("Background")
    keep_bg = st.checkbox("Force consistent background for all outputs", False,
                          help="Extract the model + dress and place on a fixed background so every image matches exactly.")
    bg_mode = st.radio("Background type", ["Solid color", "Upload image"], horizontal=True, disabled=not keep_bg)
    bg_color = st.color_picker("Solid color", "#F3F4F6", disabled=not keep_bg or bg_mode!="Solid color")
    bg_image_file = st.file_uploader("Background image (applied to all shots)", type=["png","jpg","jpeg","webp"],
                                     disabled=not keep_bg or bg_mode!="Upload image")

st.markdown("### 1) (Optional) Upload a dress photo for reference (front)")
ref = st.file_uploader("Dress reference (optional, helps match fabric/pattern)", type=["png","jpg","jpeg","webp"])

def to_data_url(img_bytes: bytes, mime="image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def prompt_for(view: str) -> str:
    angle = "symmetrical front view, full body, facing camera" if view=="front" else \
            "back view, full body, facing away from camera" if view=="back" else \
            "front view, tight crop on upper torso and neckline"
    return (
        f"high-resolution studio fashion photo, neutral soft grey background, "
        f"professional lighting, a generic female catalog model wearing the same dress as the reference, {angle}. "
        f"Focus on accurate fabric color, pattern, and silhouette; realistic skin; natural pose; no logos, no jewelry. "
        f"Style intensity {creativity}/10; detail level {steps}/40."
    )

def generate_one(client: "OpenAI", prompt: str, size: Tuple[int,int], ref_bytes: bytes = None) -> bytes:
    w, h = size
    full_prompt = prompt
    if ref_bytes:
        data_url = to_data_url(ref_bytes, mime="image/png")
        full_prompt += (
            "\nReference garment attached as data URL; replicate its fabric color and pattern faithfully. "
            f"[reference_image: {data_url}]"
        )
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=full_prompt,
        size=f"{w}x{h}",
        n=1,
    )
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)

def ensure_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")

def build_bg_canvas(w: int, h: int) -> Image.Image:
    if not keep_bg:
        return None
    if bg_mode == "Solid color":
        return Image.new("RGB", (w, h), bg_color)
    else:
        if bg_image_file is None:
            return Image.new("RGB", (w, h), "#FFFFFF")
        bg = Image.open(bg_image_file).convert("RGB")
        # Scale background to fill canvas (center-crop)
        bw, bh = bg.size
        scale = max(w / bw, h / bh)
        new_size = (max(1, int(bw*scale)), max(1, int(bh*scale)))
        bg = bg.resize(new_size, Image.LANCZOS)
        # center-crop
        x = (bg.width - w)//2
        y = (bg.height - h)//2
        return bg.crop((x, y, x+w, y+h))

def composite_consistent_bg(img_bytes: bytes, w: int, h: int) -> bytes:
    """Use rembg to extract subject and place onto a fixed background/canvas."""
    if not keep_bg or rembg_remove is None:
        return img_bytes
    try:
        fg = Image.open(BytesIO(img_bytes)).convert("RGB")
        cut = rembg_remove(fg)  # returns RGBA bytes/array
        if isinstance(cut, (bytes, bytearray)):
            cut_img = Image.open(BytesIO(cut)).convert("RGBA")
        else:
            # rembg may return numpy array
            import numpy as np
            from PIL import Image as _Image
            cut_img = _Image.fromarray(cut).convert("RGBA")
        bg = build_bg_canvas(w, h)
        bg = bg.convert("RGBA")
        # scale the extracted subject to fit nicely within the canvas (90%)
        # compute bbox to get subject size
        alpha = cut_img.split()[-1]
        bbox = alpha.getbbox() or (0, 0, cut_img.width, cut_img.height)
        subject = cut_img.crop(bbox)
        # scale to fit
        usable_w, usable_h = int(w*0.9), int(h*0.9)
        scale = min(usable_w/subject.width, usable_h/subject.height)
        new_size = (max(1, int(subject.width*scale)), max(1, int(subject.height*scale)))
        subject = subject.resize(new_size, Image.LANCZOS)
        x = (w - subject.width)//2
        y = (h - subject.height)//2
        comp = bg.copy()
        comp.paste(subject, (x, y), subject)
        out = BytesIO()
        comp.convert("RGB").save(out, format="PNG")
        out.seek(0)
        return out.read()
    except Exception as e:
        # fallback: return original
        return img_bytes

go = st.button("🎨 Generate via OpenAI", use_container_width=True)

if go:
    if not (gen_front or gen_back or gen_feature):
        st.error("Please select at least one output (Front, Back, or Feature).")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OPENAI_API_KEY environment variable.")
        else:
            client = OpenAI(api_key=api_key)
            ref_bytes = ref.read() if ref is not None else None

            cols = st.columns(3, gap="large")
            col_map = {"front": cols[0], "back": cols[1], "feature": cols[2]}

            if gen_front:
                with st.spinner("Generating Front..."):
                    img_front = generate_one(client, prompt_for("front"), (width, height), ref_bytes)
                    img_front = composite_consistent_bg(img_front, width, height)
                with col_map["front"]:
                    st.subheader("Front")
                    st.image(img_front, use_column_width=True)
                    st.download_button("Download Front", data=img_front, file_name="front_openai.png")

            if gen_back:
                with st.spinner("Generating Back..."):
                    img_back  = generate_one(client, prompt_for("back"), (width, height), ref_bytes)
                    img_back  = composite_consistent_bg(img_back, width, height)
                with col_map["back"]:
                    st.subheader("Back")
                    st.image(img_back, use_column_width=True)
                    st.download_button("Download Back", data=img_back, file_name="back_openai.png")

            if gen_feature:
                with st.spinner("Generating Feature (close-up)..."):
                    img_feat  = generate_one(client, prompt_for("feature"), (width, height), ref_bytes)
                    img_feat  = composite_consistent_bg(img_feat, width, height)
                with col_map["feature"]:
                    st.subheader("Feature (Close-up)")
                    st.image(img_feat, use_column_width=True)
                    st.download_button("Download Feature", data=img_feat, file_name="feature_openai.png")

st.markdown("---")
if rembg_remove is None:
    st.warning("To enable *consistent background*, install `rembg`: `pip install rembg`. "
               "Without it, the app will still generate images but cannot enforce the exact same background.")
else:
    st.info("Consistent background is enabled with `rembg`. Tip: Use high-contrast backgrounds for best cutouts.")
