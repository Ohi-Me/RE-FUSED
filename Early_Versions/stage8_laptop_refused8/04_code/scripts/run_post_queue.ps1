# Post-training queue (detached): waits for the O2 queues, then O2 evaluation/selection -> O4 -> O5 rule arms -> O5 PPO;
# waits for the O3 queue, then O3 evaluation. Each step logs to 12_logs\post_queue.log.
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"
$log = "12_logs\post_queue.log"
function Has-Marker($file, $marker) {
    if (-not (Test-Path $file)) { return $false }
    $raw = [System.IO.File]::ReadAllText((Resolve-Path $file)) -replace "`0", ""
    return $raw -match $marker
}
"post queue waiting for O2 queues $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
while (-not ((Has-Marker "12_logs\o2_gpu_queue.log" "GPU_QUEUE_DONE") -and (Has-Marker "12_logs\o2_cpu_queue.log" "CPU_QUEUE_DONE"))) { Start-Sleep -Seconds 300 }
"O2 evaluation start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o2\o2_06_evaluate.py *>> $log
"O4 start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o4\o4_01_assessment.py *>> $log
"O5 rule arms start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o5\o5_01_scheduling.py *>> $log
"O5 PPO start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o5\o5_02_ppo.py *>> $log
"POST_O2_DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
while (-not (Has-Marker "12_logs\o3_queue.log" "O3_QUEUE_DONE")) { Start-Sleep -Seconds 300 }
"O3 evaluation start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o3\o3_02_evaluate.py *>> $log
"POST_O3_DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
