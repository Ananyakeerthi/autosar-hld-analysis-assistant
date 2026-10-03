# AUTOSAR HLD Document Analysis Assistant

Streamlit app + local Ollama (Llama 3.2) RAG pipeline for AUTOSAR HLD document
Q&A, architecture extraction, dependency mapping, completeness checking, and
revision comparison.

## Prerequisites

- Docker Desktop installed and running
- [Ollama](https://ollama.com) installed **on your host machine** (not in Docker)
  with the model pulled:
  ```
  ollama pull llama3.2
  ```
  Ollama must be running (`ollama serve`, or just have the Ollama app open)
  **before** you start the container — the app connects out to it.

## Run with Docker Compose (recommended)

From the folder containing `Dockerfile`, `docker-compose.yml`, `app.py`, and
`requirements.txt`:

```powershell
docker compose up --build
```

Then open: http://localhost:8501

To stop:
```powershell
docker compose down
```

Exported reports land in a local `hld_analysis_exports/` folder next to
`docker-compose.yml` — they persist even if the container is removed.

## Run with plain Docker (no compose)

```powershell
docker build -t autosar-hld-assistant .

docker run -p 8501:8501 ^
  -e OLLAMA_HOST=http://host.docker.internal:11434 ^
  -v %cd%\hld_analysis_exports:/app/hld_analysis_exports ^
  --add-host=host.docker.internal:host-gateway ^
  autosar-hld-assistant
```

(On PowerShell, replace the `^` line continuations with a single line, or
use `` ` `` instead of `^`.)

## Run without Docker at all (plain Python)

```powershell
pip install -r requirements.txt
streamlit run app.py
```

This uses `localhost:11434` by default, which works directly since Ollama
and the app are both on your host machine with no container in between.

## Troubleshooting

- **Sidebar shows "Ollama not reachable"** — make sure `ollama serve` is
  running on your host, and that you pulled `llama3.2`. If using Docker,
  double-check `OLLAMA_HOST` is set to `http://host.docker.internal:11434`,
  not `localhost` (localhost inside a container refers to the container
  itself, not your host machine).
- **500 error from Ollama** — restart Ollama (`taskkill /IM ollama.exe /F`
  then `ollama serve` again, or reopen the Ollama desktop app).
- **Slow responses** — expected with a local CPU-only model; larger prompts
  (e.g. Revision Comparison with `max_changed_pages` set high) take longer.
