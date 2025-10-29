FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

RUN pip install uv

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked


COPY pipeline_v1.bin .
COPY predictapp.py .
EXPOSE 9696

ENTRYPOINT  ["uvicorn", "predictapp:app", "--host", "0.0.0.0", "--port", "9696"]
