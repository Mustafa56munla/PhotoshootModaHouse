# app_openai.py
# Streamlit app using OpenAI Images API (gpt-image-1) with selectable outputs,
# valid-size enforcement, optional consistent background, and brand upscaling.
#
# Valid sizes for gpt-image-1:
#   - 1024x1024 (square)
#   - 1024x1536 (portrait)
#   - 1536x1024 (landscape)
#
# Deploy on Streamlit Cloud: add OPENAI_API_KEY in Secrets, set entrypoint to this file.

import os
import base64
import streamlit as st
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional

try:
    from openai import OpenAI
    from openai import BadRequestError
except Exception:
    st.error("Please `pip install openai>=1.0.0`")
    raise

# Optional background compositing for consistent look
try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None

st.set_page_config(page_title="AI Dress Photoshoot (OpenAI)", page_icon="👗", layout="wide")
st.title("👗 AI Dress Photoshoot — OpenAI API (gpt-image-1)")
st.caption("Generate Front, Back, and Feature (close-up) with selectable outputs, valid sizes, and optional consistent background.")

# Final brand canvases
BRAND_PRESETS = {"Post (2160×2160)": (2160, 2160), "Story/Reel (1080×1920)": (1080, 1920)}
# Allowed API sizes
API_SIZES = {"Square 1024": (1024, 1024), "Portrait 1024×1536": (1024, 1536), "Landscape 1536×1024": (1536, 1024)}

with st.sidebar:
    st.header("⚙️ Settings")
    brand_name = st.selectbox("Final canvas (brand size)", list(BRAND_PRESETS.keys()), index=0)
    brand_w, brand_h = BRAND_PRESETS[brand_name]
    api_size_name = st.selectbox("API generation size", list(API_SIZES.keys()), index=0)
    gen_w, gen_h = API_SIZES[api_size_name]

    creativity = st.slider("Creativity (1=literal, 10=creative)", 1, 10, 5, step=1)
    steps = st.slider("Detail level (steps)", 10, 40, 25, step=1)

    st.markdown("---")
    st.subheader("What to generate")
    gen_front = st.checkbox("Front", True)
    gen_back = st.checkbox("Back", True)
    gen_feature = st.checkbox("Feature (close-up)", True)

    st.markdown("---")
    st.subheader("Background")
    keep_bg = st.checkbox("Force consistent background for all outputs", False)
    bg_mode = st.radio("Background type", ["Solid color", "Upload image"], horizontal=True, disabled=not keep_bg)
    bg_color = st.color_picker("Solid color", "#F3F4F6", disabled=not keep_bg or bg_mode!="Solid color")
    bg_image_file = st.file_uploader("Background image (applied to all shots)", type=["png","jpg","jpeg","webp"],
                                    disabled=not keep_bg or bg_mode!="Upload image")

    st.markdown("---")
    upscale = st.checkbox("Upscale to brand canvas after generation", True)

st.markdown("### 1) (Optional) Upload a dress photo for reference (front)")
ref = st.file_uploader("Dress reference (optional — prompt will ask to match fabric/pattern)", type=["png","jpg","jpeg","webp"])

def prompt_for(view: str) -> str:
    angle = "symmetrical front view, full body, facing camera" if view=="front" else                 "back view, full body, facing away from camera" if view=="back" else                 "front view, tight crop on upper torso and neckline"
    base = (
        f"high-resolution studio fashion photo, neutral soft grey background, "
        f"professional lighting, a generic female catalog model wearing the same dress as the provided reference, {angle}. "
        f"Match the dress fabric color and pattern realistically; maintain natural body proportions; realistic skin; natural pose; no logos, no jewelry. "
        f"Style intensity {creativity}/10; detail level {steps}/40."
    )
    if ref is not None:
        base += " Use the uploaded reference image for exact fabric color and pattern."
    return base

def generate_one(client: "OpenAI", prompt: str, size: Tuple[int,int]) -> bytes:
    w, h = size
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=f"{w}x{h}",
        n=1,
    )
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)

def build_bg_canvas(w: int, h: int) -> Optional[Image.Image]:
    if not keep_bg:
        return None
    if bg_mode == "Solid color":
        return Image.new("RGB", (w, h), bg_color)
    else:
        if bg_image_file is None:
            return Image.new("RGB", (w, h), "#FFFFFF")
        bg = Image.open(bg_image_file).convert("RGB")
        bw, bh = bg.size
        scale = max(w / bw, h / bh)
        new_size = (max(1, int(bw*scale)), max(1, int(bh*scale)))
        bg = bg.resize(new_size, Image.LANCZOS)
        x = (bg.width - w)//2; y = (bg.height - h)//2
        return bg.crop((x, y, x+w, y+h))

def composite_consistent_bg(img_bytes: bytes, w: int, h: int) -> bytes:
    if not keep_bg or rembg_remove is None:
        return img_bytes
    try:
        fg = Image.open(BytesIO(img_bytes)).convert("RGB")
        cut = rembg_remove(fg)
        if isinstance(cut, (bytes, bytearray)):
            cut_img = Image.open(BytesIO(cut)).convert("RGBA")
        else:
            from PIL import Image as _Image
            import numpy as np
            cut_img = _Image.fromarray(cut).convert("RGBA")
        bg = build_bg_canvas(w, h).convert("RGBA")
        alpha = cut_img.split()[-1]
        bbox = alpha.getbbox() or (0, 0, cut_img.width, cut_img.height)
        subject = cut_img.crop(bbox)
        usable_w, usable_h = int(w*0.9), int(h*0.9)
        scale = min(usable_w/subject.width, usable_h/subject.height)
        subject = subject.resize((max(1,int(subject.width*scale)), max(1,int(subject.height*scale))), Image.LANCZOS)
        x = (w - subject.width)//2; y = (h - subject.height)//2
        comp = bg.copy()
        comp.paste(subject, (x, y), subject)
        out = BytesIO(); comp.convert("RGB").save(out, format="PNG"); out.seek(0)
        return out.read()
    except Exception:
        return img_bytes

def upscale_to_brand(img_bytes: bytes, brand_size: Tuple[int,int]) -> bytes:
    if not upscale:
        return img_bytes
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    target_w, target_h = brand_size
    canvas = Image.new("RGB", (target_w, target_h), "#FFFFFF")
    scale = min(target_w / img.width, target_h / img.height)
    new_w, new_h = max(1,int(img.width*scale)), max(1,int(img.height*scale))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    x = (target_w - new_w)//2; y = (target_h - new_h)//2
    canvas.paste(img_resized, (x, y))
    out = BytesIO(); canvas.save(out, format="PNG"); out.seek(0)
    return out.read()

go = st.button("🎨 Generate via OpenAI", use_container_width=True)

if go:
    if not (gen_front or gen_back or gen_feature):
        st.error("Please select at least one output (Front, Back, or Feature).")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OPENAI_API_KEY environment variable (set it in Streamlit secrets).")
        else:
            client = OpenAI(api_key=api_key)
            cols = st.columns(3, gap="large")
            col_map = {"front": cols[0], "back": cols[1], "feature": cols[2]}

            def run_one(tag: str):
                pr = prompt_for(tag)
                raw = generate_one(client, pr, (gen_w, gen_h))
                raw = composite_consistent_bg(raw, gen_w, gen_h)
                final = upscale_to_brand(raw, (brand_w, brand_h))
                return final

            try:
                if gen_front:
                    with st.spinner("Generating Front..."):
                        img_front = run_one("front")
                    with col_map["front"]:
                        st.subheader("Front")
                        st.image(img_front, use_column_width=True)
                        st.download_button("Download Front", data=img_front, file_name="front_openai.png")
                if gen_back:
                    with st.spinner("Generating Back..."):
                        img_back  = run_one("back")
                    with col_map["back"]:
                        st.subheader("Back")
                        st.image(img_back, use_column_width=True)
                        st.download_button("Download Back", data=img_back, file_name="back_openai.png")
                if gen_feature:
                    with st.spinner("Generating Feature (close-up)..."):
                        img_feat  = run_one("feature")
                    with col_map["feature"]:
                        st.subheader("Feature (Close-up)")
                        st.image(img_feat, use_column_width=True)
                        st.download_button("Download Feature", data=img_feat, file_name="feature_openai.png")
            except BadRequestError as bre:
                st.error("OpenAI BadRequestError. Try a different API size, shorten prompt, or retry.")
                st.exception(bre)
            except Exception as e:
                st.error("Unexpected error during generation.")
                st.exception(e)

st.markdown("---")
if rembg_remove is None:
    st.warning("To enable *consistent background*, install `rembg`: it's already in requirements for Streamlit Cloud.")
else:
    st.info("Consistent background enabled via `rembg`.")
