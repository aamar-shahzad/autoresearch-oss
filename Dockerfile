FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY autoresearch_oss ./autoresearch_oss
COPY examples ./examples
RUN pip install --no-cache-dir -e ".[dev]"
# GPU/LLM image: build FROM a CUDA PyTorch base and `pip install -e ".[llm]"` instead.
ENTRYPOINT ["python", "-m", "autoresearch_oss"]
CMD ["quickstart"]
