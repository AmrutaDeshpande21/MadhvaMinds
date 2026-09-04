# Script to fully start the MadhvaMinds Incident Intelligence Platform

Write-Host "Starting Docker services (Postgres, Redis)..." -ForegroundColor Cyan
docker-compose up -d

Write-Host "Starting FastAPI Backend..." -ForegroundColor Cyan
# Start backend in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit -Command `"cd backend; if (-Not (Test-Path '.venv')) { python -m venv .venv }; .\.venv\Scripts\Activate.ps1; pip install -r ../requirements.txt; uvicorn main:app --host 0.0.0.0 --port 8000 --reload`""

Write-Host "Starting Next.js Frontend..." -ForegroundColor Cyan
# Start frontend in a new PowerShell window
Start-Process powershell -ArgumentList "-NoExit -Command `"cd frontend; npm install; npm run dev`""

Write-Host "All services are starting up!" -ForegroundColor Green
Write-Host "Frontend will be available at: http://localhost:3000" -ForegroundColor Yellow
Write-Host "Backend API will be available at: http://localhost:8000" -ForegroundColor Yellow
