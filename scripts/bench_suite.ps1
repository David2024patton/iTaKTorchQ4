# Torch Resource-Monitored Benchmark
# Captures GPU/CPU/RAM before and after each test

param(
    [string]$Model = "e:\.agent\GOAgent\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$OutDir = "e:\.agent\iTaK Eco\Torch\scripts\bench_results"
)

$env:GOOS = ''
$env:GOARCH = ''
$TorchRoot = "e:\.agent\iTaK Eco\Torch"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Get-Resources {
    $mem = Get-WmiObject Win32_OperatingSystem
    $memUsedGB = [math]::Round(($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / 1MB, 2)
    $memTotalGB = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 2)

    $gpuLine = ""
    try {
        $gpuLine = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>$null
    } catch {}

    return @{
        RAM = "${memUsedGB}GB / ${memTotalGB}GB"
        GPU = if ($gpuLine) { $gpuLine.Trim() } else { "N/A" }
    }
}

function Invoke-Bench {
    param([string]$Name, [string]$Lib, [int]$GpuLayers = -1, [int]$NCtx = 4096)

    Write-Host "`n=== $Name ===" -ForegroundColor Cyan

    $pre = Get-Resources
    Write-Host "  [BEFORE] RAM: $($pre.RAM) | GPU: $($pre.GPU)" -ForegroundColor DarkGray

    $env:ITAK_TORCH_LIB = $Lib
    $env:YZMA_BENCHMARK_MODEL = $Model

    $args = @("test", "./pkg/torch/llama/", "-run='^$'", "-bench=BenchmarkInference",
              "-benchtime=30s", "-count=1", "-timeout=300s", "-nctx=$NCtx", "-ngpulayers=$GpuLayers")

    # Start a resource sampling loop in a separate process that writes to a file
    $monFile = Join-Path $OutDir "mon_${Name}.csv"
    "timestamp,ram_used_gb,gpu_util,gpu_mem_used,gpu_mem_total,gpu_temp" | Set-Content $monFile
    $monProc = Start-Process powershell -ArgumentList "-NoProfile -Command `"while (`$true) { `$m = Get-WmiObject Win32_OperatingSystem; `$ru = [math]::Round((`$m.TotalVisibleMemorySize - `$m.FreePhysicalMemory) / 1MB, 2); try { `$g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>`$null); if (`$g) { `$p = `$g.Split(',').Trim(); Add-Content '$monFile' `\"```$(Get-Date -f s),`$ru,`$(`$p[0]),`$(`$p[1]),`$(`$p[2]),`$(`$p[3])`\" } else { Add-Content '$monFile' `\"```$(Get-Date -f s),`$ru,0,0,0,0`\" } } catch { Add-Content '$monFile' `\"```$(Get-Date -f s),`$ru,0,0,0,0`\" }; Start-Sleep 3 }`"" -PassThru -WindowStyle Hidden

    $result = & go @args 2>&1 | Out-String

    # Stop monitor
    if ($monProc -and !$monProc.HasExited) {
        Stop-Process $monProc -Force -ErrorAction SilentlyContinue
    }

    $post = Get-Resources
    Write-Host "  [AFTER] RAM: $($post.RAM) | GPU: $($post.GPU)" -ForegroundColor DarkGray

    # Parse monitor CSV
    $monData = Import-Csv $monFile -ErrorAction SilentlyContinue
    $avgCPU = "N/A"; $avgRAM = "N/A"; $peakRAM = "N/A"; $avgGPU = "N/A"; $peakVRAM = "N/A"; $avgTemp = "N/A"; $samples = 0
    if ($monData -and $monData.Count -gt 0) {
        $samples = $monData.Count
        $avgRAM = [math]::Round(($monData | ForEach-Object { [double]$_.ram_used_gb } | Measure-Object -Average).Average, 2)
        $peakRAM = [math]::Round(($monData | ForEach-Object { [double]$_.ram_used_gb } | Measure-Object -Maximum).Maximum, 2)
        $avgGPU = [math]::Round(($monData | ForEach-Object { [double]$_.gpu_util } | Measure-Object -Average).Average, 1)
        $peakVRAM = ($monData | ForEach-Object { [int]$_.gpu_mem_used } | Measure-Object -Maximum).Maximum
        $avgTemp = [math]::Round(($monData | ForEach-Object { [double]$_.gpu_temp } | Measure-Object -Average).Average, 0)
    }

    $outFile = Join-Path $OutDir "beast_${Name}.txt"
    @"
=== TORCH BENCHMARK: $Name ===
Machine: Beast (Windows)
CPU: Intel i9-14900K (24C/32T)
GPU: NVIDIA RTX 4070 Ti SUPER (16GB)
iGPU: Intel UHD 770
RAM: 64GB DDR5
Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (~4.9GB)
Library: $Lib
GPU Layers: $GpuLayers
Context: $NCtx
Date: $Timestamp

--- BENCHMARK OUTPUT ---
$result

--- RESOURCES BEFORE ---
RAM: $($pre.RAM)
GPU: $($pre.GPU)

--- RESOURCES AFTER ---
RAM: $($post.RAM)
GPU: $($post.GPU)

--- RESOURCES DURING (sampled every 3s) ---
Avg RAM Used:     ${avgRAM} GB
Peak RAM Used:    ${peakRAM} GB
Avg GPU Util:     ${avgGPU}%
Peak VRAM Used:   ${peakVRAM} MB
Avg GPU Temp:     ${avgTemp}C
Samples:          $samples
"@ | Out-File -Encoding utf8 $outFile

    Write-Host $result -ForegroundColor Yellow
    Write-Host "  Resources: RAM avg=${avgRAM}GB peak=${peakRAM}GB | GPU avg=${avgGPU}% | VRAM peak=${peakVRAM}MB | Temp=${avgTemp}C ($samples samples)" -ForegroundColor Green
}

Write-Host "`n=== TORCH BENCHMARK SUITE - BEAST ===" -ForegroundColor White
Write-Host "Model: Llama 3.1 8B Q4_K_M (~4.9GB)`n" -ForegroundColor Gray

Invoke-Bench -Name "cpu_only"          -Lib "$TorchRoot\lib\windows_amd64"      -GpuLayers 0
Invoke-Bench -Name "gpu_full"          -Lib "$TorchRoot\lib\windows_amd64_cuda" -GpuLayers 99
Invoke-Bench -Name "gpu_split_50"      -Lib "$TorchRoot\lib\windows_amd64_cuda" -GpuLayers 16
Invoke-Bench -Name "gpu_split_25"      -Lib "$TorchRoot\lib\windows_amd64_cuda" -GpuLayers 8
Invoke-Bench -Name "cuda_cpu_fallback" -Lib "$TorchRoot\lib\windows_amd64_cuda" -GpuLayers 0

Write-Host "`n=== ALL BEAST BENCHMARKS COMPLETE ===" -ForegroundColor Green
Write-Host "Results in: $OutDir" -ForegroundColor Green
