# logs

| Folder / file | What it holds |
|---|---|
| `stages/` | Output of every pipeline stage (`<stage>.log`) |
| `done/` | One record per finished stage: time, PBS job, GPU (MIG) instance, return code |
| `pbs/` | Cluster job scripts and their output |
| `autopilot/` | Post-run reporting and integrity jobs |
| `local_validation/` | Early runs on the laptop, before the move to the H100 |
| `freeze*.json` | The freezing of the pre-registration |
