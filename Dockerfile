FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml pyproject.toml

RUN pip install --no-cache-dir --upgrade pip build setuptools wheel

RUN pip install --no-cache-dir . streamlit huggingface-hub

COPY . /app

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
