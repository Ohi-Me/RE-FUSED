# Environment (captured 13 Sep 2026)

| Item | Value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| Python | 3.10.11 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| sklearn | 1.7.2 |
| torch | 2.5.1+cu121 |
| lightgbm | 4.7.0 |
| neuralforecast | 3.2.2 |
| chronos-forecasting | 2.3.2 |
| transformers | 5.3.0 |
| pyarrow | 25.0.1 |
| properscoring | 0.1 |
| CUDA available | True |
| GPU | NVIDIA GeForce GTX 1660 Ti with Max-Q Design |
| CUDA (torch build) | 12.1 |
| LaTeX | pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026) |

Install: `py -3.10 -m pip install -r 13_env/requirements-lock.txt -c 13_env/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu121`.
Warning: installing neuralforecast or chronos-forecasting without the constraints file can replace the CUDA build of torch with a CPU build.
