# RE-FUSED-8 remaining development pipeline, strictly sequential (one job at a time: parallel jobs starved each other).
# Every step is resumable and skips completed outputs.
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"
$log = "12_logs\sequential.log"
function Step($name, $cmd) {
    "=== $name start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
    Invoke-Expression "$cmd *>> $log"
    "=== $name end $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
}
Step "lgbm"        "py -3.10 -u 04_code\scripts\o2\o2_02_lgbm.py T1 T2 T3 T4 T5"
Step "tft"         "py -3.10 -u 04_code\scripts\o2\o2_03_nf.py tft T1,T2,T3,T4,T5 5"
Step "o2_evaluate" "py -3.10 -u 04_code\scripts\o2\o2_06_evaluate.py"
Step "o4"          "py -3.10 -u 04_code\scripts\o4\o4_01_assessment.py"
Step "o5_rules"    "py -3.10 -u 04_code\scripts\o5\o5_01_scheduling.py"
Step "o5_ppo"      "py -3.10 -u 04_code\scripts\o5\o5_02_ppo.py"
Step "o3_main"     "py -3.10 -u 04_code\scripts\o3\o3_01_lamcag.py T5,T4,T3,T2,T1 mcag_instance,fusion_fixed,mcag_fixed,mcag_regime 5"
Step "o3_ablation" "py -3.10 -u 04_code\scripts\o3\o3_01_lamcag.py T1,T2,T5 null_context,permuted_context,market_only 3"
Step "o3_ablation2" "py -3.10 -u 04_code\scripts\o3\o3_01_lamcag.py T1,T2,T3 no_carbon 3"
Step "o3_evaluate" "py -3.10 -u 04_code\scripts\o3\o3_02_evaluate.py"
"ALL_SEQUENTIAL_DONE $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $log
