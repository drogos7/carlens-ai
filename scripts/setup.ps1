# One-shot setup: portable Node (no admin), Python venv + pip, npm in app/.
# Run from repo root:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Tools = Join-Path $Root 'tools'
$NodeDir = Join-Path $Tools 'node'
$Zip = Join-Path $Tools 'node-win-x64.zip'
$NodeVersion = '22.14.0'
$NodeDist = "node-v$NodeVersion-win-x64"

New-Item -ItemType Directory -Force -Path $Tools | Out-Null

if (-not (Test-Path (Join-Path $NodeDir 'node.exe'))) {
  Write-Host 'Downloading portable Node.js...'
  Invoke-WebRequest -Uri "https://nodejs.org/dist/v$NodeVersion/$NodeDist.zip" -OutFile $Zip
  if (Test-Path $NodeDir) { Remove-Item -Recurse -Force $NodeDir }
  Expand-Archive -Path $Zip -DestinationPath $Tools -Force
  Rename-Item (Join-Path $Tools $NodeDist) $NodeDir
}

$env:PATH = "$NodeDir;$env:PATH"
Write-Host "Node: $(node -v)  npm: $(npm -v)"

# Backend
Push-Location (Join-Path $Root 'backend')
if (-not (Test-Path '.venv')) {
  if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv .venv 2>$null; if (-not (Test-Path '.venv')) { py -3.10 -m venv .venv } }
  else { python -m venv .venv }
}
& .\.venv\Scripts\python.exe -m pip install -q --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
Pop-Location

# App
Push-Location (Join-Path $Root 'app')
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
npm install
npx --yes expo install --fix
Pop-Location

Write-Host 'Done. Start backend:  cd backend; .\.venv\Scripts\activate; uvicorn main:app --reload --host 0.0.0.0 --port 8000'
Write-Host 'Start app:        cd app; npx expo start   (use tools\node first if npm not on PATH)'
