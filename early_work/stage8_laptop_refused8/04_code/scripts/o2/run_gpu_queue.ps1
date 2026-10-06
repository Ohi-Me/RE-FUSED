# O2 GPU queue (serial, resumable): PatchTST -> BiLSTM -> Chronos-Bolt -> TFT. Each step skips completed outputs.
Set-Location (Join-Path $PSScriptRoot "..\..\..")
$env:PYTHONIOENCODING = "utf-8"
$log = "12_logs\o2_gpu_queue.log"
"GPU queue start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
py -3.10 -u 04_code\scripts\o2\o2_03_nf.py patchtst T1,T2,T3,T4,T5 5 *>> $log
py -3.10 -u 04_code\scripts\o2\o2_04_bilstm.py T1,T2,T3,T4,T5 5 *>> $log
py -3.10 -u 04_code\scripts\o2\o2_05_chronos.py T1,T2,T3,T4,T5 *>> $log
py -3.10 -u 04_code\scripts\o2\o2_03_nf.py tft T1,T2,T3,T4,T5 5 *>> $log
"GPU_QUEUE_DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
