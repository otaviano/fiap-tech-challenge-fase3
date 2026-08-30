# Imagem de serviço do assistente. O fine-tuning NÃO roda aqui: o container
# serve a API, e a LLM customizada é servida à parte (llama.cpp/Ollama/vLLM),
# apontada por LLM_BASE_URL.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/
COPY data/processed/ ./data/processed/
RUN pip install --no-cache-dir -e .

# O prontuário é reconstruído do seed determinístico na imagem.
RUN med-assist ehr seed

ENV LLM_BASE_URL=http://host.docker.internal:8080/v1 \
    LLM_MODEL=med-assist \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "med_assist.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
