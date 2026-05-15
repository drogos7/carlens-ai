# CarLens AI

AI-powered car diagnostic assistant: scan OBD scanner screens or enter fault codes / VINs, get structured repair guidance.

**Repository:** [github.com/drogos7/carlens-ai](https://github.com/drogos7/carlens-ai)  
**Author:** [@drogos7](https://github.com/drogos7)

## Features

- **Photo scan** — local OCR (RapidOCR) reads DTCs from diagnostic displays
- **VIN decode** — vehicle lookup via local DB + NHTSA vPIC API
- **Brand libraries** — knowledge organized per manufacturer (`mercedes-benz`, `bmw`, …)
- **BMW hex codes** — manufacturer-specific `E` codes (e.g. iDrive fault lists)
- **Premium web UI** — Expo / React Native desktop workspace

## Project structure

```
app/          Expo frontend (TypeScript)
backend/      FastAPI API + SQLite knowledge DB
scripts/      Dev helpers (ingest, run)
```

### Knowledge base (by brand)

```
backend/knowledge/brands/
  mercedes-benz/
    brand.json
    codes/*.json
    vins.json
    sources.json
  bmw/
    brand.json
    codes/istapopular.json
    vins.json
    sources.json
```

See [backend/knowledge/BRANDS.md](backend/knowledge/BRANDS.md) for adding new brands or codes.

## Quick start

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn.exe main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```powershell
cd app
npm install
npx expo start
```

Set `EXPO_PUBLIC_API_BASE_URL` in `app/.env` (e.g. `http://127.0.0.1:8000`).

## API highlights

| Endpoint | Description |
|----------|-------------|
| `GET /brands` | List registered vehicle brands |
| `POST /scan-error-local` | OCR image → DTC / VIN result |
| `GET /lookup?q=` | Manual code or VIN lookup |
| `GET /codes/{code}?brand=` | Repair guide for a code |

## Ingest forum / manual sources

```powershell
cd backend
.\.venv\Scripts\python.exe knowledge\ingest_sources.py mercedes-benz
.\.venv\Scripts\python.exe knowledge\ingest_sources.py bmw
```

Respects `robots.txt`. See [backend/knowledge/INGEST.md](backend/knowledge/INGEST.md).

## Tech stack

- **Backend:** Python 3, FastAPI, SQLite, RapidOCR, optional OpenAI/Anthropic vision
- **Frontend:** Expo, React Native, TypeScript

## License

MIT — see repository for details.
