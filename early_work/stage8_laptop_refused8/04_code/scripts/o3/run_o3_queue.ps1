# O3 GPU queue (resumable): waits for the O2 GPU queue, then trains LA-MCAG main variants (5 seeds) and ablations (3 seeds).
Set-Location (Join-Path $PSScriptRoot "..\..\..")
$env:PYTHONIOENCODING = "utf-8"
$log = "12_logs\o3_queue.log"
"O3 queue waiting for O2 GPU queue $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
while (-not (Select-String -Path "12_logs\o2_gpu_queue.log" -Pattern "GPU_QUEUE_DONE" -Quiet -Encoding utf8)) {
    $raw = [System.IO.File]::ReadAllText((Resolve-Path "12_logs\o2_gpu_queue.log")) -replace "`0", ""
    if ($raw -match "GPU_QUEUE_DONE") { break }
    Start-Sleep -Seconds 120
}
"O3 queue start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o3\o3_01_lamcag.py T1,T2,T3,T4,T5 mcag_instance,fusion_fixed,mcag_fixed,mcag_regime 5 *>> $log
py -3.10 -u 04_code\scripts\o3\o3_01_lamcag.py T1,T2,T5 null_context,permuted_context,market_only 3 *>> $log
py -3.10 -u 04_code\scripts\o3\o3_01_lamcag.py T1,T2,T3 no_carbon 3 *>> $log
"O3_QUEUE_DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
