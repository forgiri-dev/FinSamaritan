# Script to restart the FastAPI server
# This script helps avoid PowerShell execution policy issues

Write-Host "Starting FinSamaritan Backend Server..." -ForegroundColor Green
Write-Host ""

# Change to backend directory
Set-Location $PSScriptRoot

# Activate virtual environment (if using .venv in project root)
if (Test-Path "..\.venv\Scripts\python.exe") {
    Write-Host "Using virtual environment: ..\.venv" -ForegroundColor Yellow
    & "..\.venv\Scripts\python.exe" main.py
} elseif (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "Using virtual environment: .venv" -ForegroundColor Yellow
    & ".venv\Scripts\python.exe" main.py
} else {
    Write-Host "No virtual environment found. Using system Python..." -ForegroundColor Yellow
    python main.py
}

