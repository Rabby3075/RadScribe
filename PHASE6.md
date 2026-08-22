# Phase 6 - Local Deploy

Phase 6 turns RadScribe from a set of tools into a small app someone can open and use. This phase does not add new model logic. It wraps the Phase 4 agent with a FastAPI backend, adds one upload page, and packages the project for local Docker use.

## What Was Built

The app has two surfaces:

- FastAPI backend in `src/api/`
- one upload page in `web/index.html`

The backend exposes:

- `GET /health`
- `GET /`
- `POST /analyze`

`POST /analyze` accepts an uploaded image, writes it to a temporary file, calls `run_agent(image_path)`, and returns a structured response with:

- report text
- main findings
- borderline findings
- vision probabilities
- retrieval score
- critic result
- trace path
- visible disclaimer

## Local Run

Start the API:

```bash
python -m uvicorn src.api.main:app --reload
```

Open the UI:

```text
http://127.0.0.1:8000/
```

Health check:

```text
http://127.0.0.1:8000/health
```

Command-line upload test:

```bash
curl.exe -F "image=@data/processed/images_224/797_IM-2332-1001.dcm.png" http://127.0.0.1:8000/analyze
```

Junk upload test:

```bash
curl.exe -F "image=@some.txt" http://127.0.0.1:8000/analyze
```

A bad upload should return a normal refused response, not a server crash.

## UI

The UI is intentionally small:

- upload box
- visible safety disclaimer
- report display
- status badge
- retrieval support bar
- vision probability chart
- main / borderline / low labels
- trace path

The disclaimer is shown before any output and also remains inside the final report text.

## Docker

Build:

```bash
docker build -t radscribe:local .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 radscribe:local
```

Docker Compose:

```bash
docker compose up --build
```

The OpenAI key is passed at runtime through `.env` or an environment variable. It is not copied into the image.

The image includes:

- `src/`
- `web/`
- `data/kb/`
- `outputs/vision/finetuned_densenet121_best.pt`

The image does not include the full raw image dataset, notebooks, old traces, or `.env`.

## Safety Behavior

The app keeps the same Phase 4 safety behavior:

- unreadable files are refused
- obvious colorful non-X-ray images are refused by a simple radiograph-style check
- no confident finding leads to no disease-specific draft
- every output includes the educational disclaimer
- drafted reports are checked by the critic against retrieved evidence

The current radiograph-style check is basic. It can catch obvious bad inputs like colorful logos, but it is not a real chest-X-ray detector. A black-and-white non-X-ray image may still pass. A production version would need a dedicated chest-X-ray / out-of-domain classifier.

## Docker Check

Docker was installed on the local machine, but the first build check could not run because Docker Desktop was not running:

```text
dockerDesktopLinuxEngine: The system cannot find the file specified
```

Next check after starting Docker Desktop:

```bash
docker build -t radscribe:local .
docker run --env-file .env -p 8000:8000 radscribe:local
```

## Limits

This is a local demo deployment, not a production system.

Known limits:

- no login or user accounts
- no database
- no rate limiting
- no HIPAA-grade security controls
- no DICOM upload path in the UI
- no robust out-of-domain image detector
- generic 500 errors are returned for unexpected failures, with details kept in server logs
- cloud deployment was intentionally skipped for cost and scope control

For production, RadScribe would need stronger input validation, DICOM and PHI handling, authentication, audit logs, rate limits, monitoring, and external clinical validation.

## Conclusion

Phase 6 keeps the deployment small on purpose. The useful result is that the evaluated RadScribe agent now has a FastAPI service, a browser UI, visible safety framing, and a Docker path for local demo use.
