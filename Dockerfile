# Imagem com PyTorch + CUDA 12.4 + cuDNN 9 (Python 3.11). O PyTorch já vem na imagem.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Dependências do experimento (sem o torch, para não trocar a versão com CUDA da imagem)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

# Uso fora do VS Code: abre o Jupyter Lab. O token de acesso aparece no log do container.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
