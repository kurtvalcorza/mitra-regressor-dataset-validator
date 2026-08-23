# DIMER validator — Mitra tabular regression
# CPU-only: validation just parses CSVs. Keep it small and fast.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY validator.py validate.py contract_hardening.py ./
CMD ["python", "validate.py"]
