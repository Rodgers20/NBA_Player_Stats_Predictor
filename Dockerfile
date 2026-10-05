FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN NEXT_EXPORT=1 NEXT_PUBLIC_DATA_MODE=api npm run build

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt
COPY . .
COPY --from=frontend /frontend/out /app/frontend/out
ENV SERVE_FRONTEND=1
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
