#FROM agrigorev/zoomcamp-model:2025
FROM python:3.13.5-slim-bookworm

WORKDIR /code

#RUN pip install uv

# Copy the 'uv' and 'uvx' executables from the latest uv image into /bin/ in this image
# 'uv' is a fast Python package installer and environment manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked


COPY diabetes_model.bin .
COPY predict.py .
EXPOSE 9696

ENTRYPOINT  ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]