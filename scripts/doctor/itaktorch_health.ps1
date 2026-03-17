# iTaK Torch Health Check Script (Doctor Agent)
# Purpose: Verify iTaK Torch server is functional without triggering Windows Firewall.
# Uses 127.0.0.1 (localhost only) - no firewall popup.
# Exit code 0 = all checks pass, non-zero = failure.

param(
    [int]$Port = 11434,
    [int]$TimeoutSec = 5
)

$baseUrl = "http://127.0.0.1:$Port"
$passed = 0
$failed = 0

function Test-Endpoint {
    param([string]$Name, [string]$Url, [string]$Method = "GET", [string]$Body = $null)
    
    try {
        $params = @{
            Uri         = $Url
            Method      = $Method
            TimeoutSec  = $TimeoutSec
            ErrorAction = "Stop"
        }
        if ($Body) {
            $params["Body"] = $Body
            $params["ContentType"] = "application/json"
        }
        
        $response = Invoke-RestMethod @params
        Write-Host "[PASS] $Name" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[FAIL] $Name - $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

Write-Host "iTaK Torch Health Check - $baseUrl" -ForegroundColor Cyan
Write-Host "---"

# Test 1: Health endpoint
if (Test-Endpoint "GET /health" "$baseUrl/health") {
    $passed++
}
else { $failed++ }

# Test 2: Models endpoint
if (Test-Endpoint "GET /v1/models" "$baseUrl/v1/models") {
    $passed++
}
else { $failed++ }

# Test 3: Chat completions
$chatBody = '{"model":"test","messages":[{"role":"user","content":"ping"}]}'
if (Test-Endpoint "POST /v1/chat/completions" "$baseUrl/v1/chat/completions" "POST" $chatBody) {
    $passed++
}
else { $failed++ }

Write-Host "---"
Write-Host "Results: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })

exit $failed
