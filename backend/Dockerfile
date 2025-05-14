FROM python:3.12.10-slim

ENV POETRY_VERSION=1.8.2

WORKDIR /app

RUN pip3 install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./

COPY src ./src
COPY model ./model

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

EXPOSE 80

# Запуск FastAPI через python
CMD ["python3", "src/summarizer/main.py"]