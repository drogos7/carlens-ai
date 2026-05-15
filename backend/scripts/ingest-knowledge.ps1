# Run forum / OBD knowledge ingestion (respects robots.txt)
Set-Location $PSScriptRoot\..
.\.venv\Scripts\python.exe knowledge\ingest_sources.py mercedes-benz
