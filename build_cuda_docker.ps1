$ErrorActionPreference = "Stop"
cd "E:\.agent\iTaK Eco\Torch"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   iTaK Torch - CUDA Docker Build Script    " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

Write-Host "`n--- Step 1: Building Linux Go Binary ---" -ForegroundColor Yellow
$env:GOOS="linux"
$env:GOARCH="amd64"
$env:CGO_ENABLED="0"
go build -o torch_linux_amd64 ./cmd/torch/
Write-Host "Linux Go Binary Built Successfully." -ForegroundColor Green

Write-Host "`n--- Step 2: Building CUDA Docker Image ---" -ForegroundColor Yellow
docker build -t itaktorch-cuda -f Dockerfile.cuda .
Write-Host "Docker Image Built Successfully." -ForegroundColor Green

Write-Host "`n--- Step 3: Starting CUDA Docker Container ---" -ForegroundColor Yellow
docker stop itaktorch-cuda-live 2>$null
docker rm itaktorch-cuda-live 2>$null
docker run -d --name itaktorch-cuda-live --gpus all -p 39272:39272 -v "E:/.agent/iTaK Eco/Torch/models:/models" itaktorch-cuda
Write-Host "CUDA Container Started." -ForegroundColor Green

Write-Host "`n=============================================" -ForegroundColor Cyan
Write-Host "All done! CUDA container on port 39272." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
