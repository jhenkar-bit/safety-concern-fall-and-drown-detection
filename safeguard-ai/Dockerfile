# SafeGuard AI - CPU image (runs the Streamlit app by default)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

# CPU-only PyTorch first: far smaller than the default CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install lapx

# Bake the YOLO models into the image so nothing downloads at run time
RUN python -c "from ultralytics import YOLO; [YOLO(m) for m in ('yolo11s-pose.pt','yolo11n-pose.pt','yolo11n.pt')]"

COPY . .
EXPOSE 8501 8000
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
