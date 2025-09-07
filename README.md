# AI Dress Photoshoot — OpenAI (Streamlit Cloud Ready)

Generate **Front / Back / Feature** shots with OpenAI `gpt-image-1`.
- Valid API sizes only (1024×1024, 1024×1536, 1536×1024), then optional upscale to **2160×2160** or **1080×1920**.
- Optional **consistent background** (solid color or uploaded image) via automatic cutout.

## Deploy on Streamlit Cloud
1) Create a GitHub repo and upload `app_openai.py` and `requirements.txt`.
2) In Streamlit Cloud → **Secrets**:
   ```
   OPENAI_API_KEY = "sk-..."
   ```
3) Set the entrypoint to `app_openai.py` and deploy.
