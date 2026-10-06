# INVISIBLE³D API

## Local development

From the repository root:

    python -m uvicorn api.main:app --reload

Open:

    http://127.0.0.1:8000/docs

Health check:

    http://127.0.0.1:8000/health

## API flow

1. POST /simulate
2. Poll GET /jobs/{job_id}
3. POST /reconstruct/ewald or /reconstruct/gradient
4. Poll GET /jobs/{job_id}
5. GET /jobs/{job_id}/results

The first deployment release uses synthetic data and local job storage. For production-scale datasets, replace the local runtime store with object storage and a persistent job queue.
