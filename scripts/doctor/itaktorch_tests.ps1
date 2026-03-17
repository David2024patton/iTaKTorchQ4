# iTaK Torch Unit Test Runner (Doctor Agent)
# Purpose: Run all iTaK Torch unit tests. Requires ITAK_TORCH_LIB set to the llama.cpp lib directory.
# Uses 'go test' which only creates in-process HTTP servers (no firewall popup).
# Exit code 0 = all tests pass, non-zero = failure.

param(
    [string]$LibDir = (Join-Path $env:USERPROFILE ".itaktorch\lib")
)

Write-Host "iTaK Torch Unit Test Runner" -ForegroundColor Cyan
Write-Host "Lib dir: $LibDir"
Write-Host "---"

# Set env for llama.cpp DLL loading.
$env:ITAK_TORCH_LIB = $LibDir
$env:PATH = "$LibDir;$env:PATH"

# Find iTaK Agent root (assume we're running from scripts/doctor/).
$itakagentRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $itakagentRoot "go.mod"))) {
    $itakagentRoot = "e:\.agent\iTaK Agent"
}

Write-Host "iTaK Agent root: $itakagentRoot"
Write-Host "---"

Push-Location $itakagentRoot
try {
    $output = go test ./pkg/torch/... -count=1 -short 2>&1
    $exitCode = $LASTEXITCODE
    
    foreach ($line in $output) {
        if ($line -match "^ok") {
            Write-Host "[PASS] $line" -ForegroundColor Green
        }
        elseif ($line -match "FAIL") {
            Write-Host "[FAIL] $line" -ForegroundColor Red
        }
        else {
            Write-Host "       $line"
        }
    }
    
    Write-Host "---"
    if ($exitCode -eq 0) {
        Write-Host "All tests passed." -ForegroundColor Green
    }
    else {
        Write-Host "Some tests failed." -ForegroundColor Red
    }
    
    exit $exitCode
}
finally {
    Pop-Location
}
