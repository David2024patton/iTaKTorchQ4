$ErrorActionPreference = "Stop"
cd "E:\.agent\iTaK Eco\Torch"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   iTaK Torch - Vulkan Docker Build Script   " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "This script runs in a separate window so the AI agent doesn't freeze."

Write-Host "`n--- Step 1: Building Linux Go Binary ---" -ForegroundColor Yellow
$env:GOOS="linux"
$env:GOARCH="amd64"
$env:CGO_ENABLED="0"
go build -o torch_linux_amd64 ./cmd/torch/
Write-Host "Linux Go Binary Built Successfully." -ForegroundColor Green

Write-Host "`n--- Step 2: Compiling llama.cpp Vulkan (Docker Multi-stage Build) ---" -ForegroundColor Yellow
Write-Host "This will take 5-10 minutes to download and compile everything. Please wait..." -ForegroundColor Magenta
docker build -t itaktorch .
Write-Host "Docker Image Built Successfully." -ForegroundColor Green

Write-Host "`n--- Step 3: Starting Vulkan Docker Container ---" -ForegroundColor Yellow
docker stop itaktorch-live 2>$null
docker rm itaktorch-live 2>$null
docker run -d --name itaktorch-live --gpus all -p 39271:39271 -v "E:/.agent/iTaK Eco/Torch/models:/models" itaktorch
Write-Host "Vulkan Container Started." -ForegroundColor Green

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "All done! You can close this window now." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
