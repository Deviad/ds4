#!/bin/bash
exec bash -lc "set -euo pipefail
cd \"/Users/spotted/projects/ds4-finetuning\"
unset SSLKEYLOGFILE
A4_PHASE_A_OUTPUT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a\"
A4_PHASE_A_START_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/phase-a-start.safetensors\"
A4_PHASE_A_STEP1_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000001_adapters.safetensors\"
A4_PHASE_A_STEP2_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000002_adapters.safetensors\"
A4_PHASE_A_FINAL_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/adapters.safetensors\"
A4_PHASE_A_CONFIG=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/adapter_config.json\"
A4_PHASE_A_LOG=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-a-log.txt\"
A4_PHASE_A_REPORT=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-a-report.json\"
A4_PHASE_A_OK=\"/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-a-ok\"
A4_PHASE_A_FAIL=\"/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-a-fail\"
A4_PHASE_B_OUTPUT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b\"
A4_PHASE_B_START_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/resume-start.safetensors\"
A4_PHASE_B_STEP1_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/0000001_adapters.safetensors\"
A4_PHASE_B_FINAL_CHECKPOINT=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/adapters.safetensors\"
A4_PHASE_B_CONFIG=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-b/adapter_config.json\"
A4_PHASE_B_RESUME=\"/Volumes/Data NVME/mlx-ft/ds4/adapters-segmented-pilot-attempt-4-phase-a/0000002_adapters.safetensors\"
A4_PHASE_B_LOG=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-b-log.txt\"
A4_PHASE_B_REPORT=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-b-report.json\"
A4_PHASE_B_OK=\"/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-b-ok\"
A4_PHASE_B_FAIL=\"/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-phase-b-fail\"
A4_FINAL_REPORT=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/pilot-report.json\"
A4_FINAL_OK=\"/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-ok\"
A4_FINAL_FAIL=\"/Volumes/Data NVME/mlx-ft/ds4/.ds4-segmented-pilot-attempt-4-fail\"
A4_PHASE_A_AUTHORIZATION=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-a-authorization.json\"
A4_PHASE_B_AUTHORIZATION=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-b-authorization.json\"
PYTHON=\"/Volumes/Data NVME/mlx-ft/ds4/.venv/bin/python\"
PILOT_SCRIPT=\"/Users/spotted/projects/ds4-finetuning/scripts/ds4_segmented_pilot.py\"
MODEL=\"/Volumes/Data NVME/mlx-ft/ds4/model-4bit\"
DATA=\"/Volumes/Data NVME/datasets/anthropomorphic-frankenmerge/mlx-4096-smoke\"
CONFIG=\"/Volumes/Data NVME/mlx-ft/ds4/lora-config.json\"
LOG=\"/Users/spotted/projects/ds4-finetuning/agent-output/cmux-14-5-attempt-4/phase-a-log.txt\"
AUTHORIZATION_JSON=\"{\\"attempt2_runtime_manifest_sha256\\":\\"2cc1d2359017cce2130a9428fd202e8c2f35b5d5810030c95aaac8b464fd3d88\\",\\"attempt3_historical_evidence_sha256\\":\\"dc576e1ca032c3246114801fd502cf6e24cdc8916f4292bfdcb63b4f32df8442\\",\\"canonical_command_sha256\\":\\"c4d22b662b6e2cbd585db383675bda39da0c974b5d396e5ced57de0bdffae829\\",\\"catalog_source_sha256\\":\\"015f7e9c94555c9b3fbf24f3e29d81a723e373e76dba643641554310641e761e\\",\\"phase\\":\\"phase-a\\",\\"phase_authorization_sha256\\":\\"b9063fb2e10e8f5b99816390e642772bca82427a0ae91cac040db8246e09796b\\",\\"pilot_source_sha256\\":\\"021a4c2979bd7f801b9520212b51052ea67abe5d374485d741f2fe484ff374ff\\",\\"pre_write_verification_sha256\\":\\"17bf86648c85837117fbf7e45c4deab06100edaee4fa93feae4acc63d93f526c\\",\\"protected_files_manifest_sha256\\":\\"207b43da07d146819d9cd83a5f2711ae4fdaba4de5415f6327edd8add1ae0115\\",\\"revision\\":\\"07fe6c8e13651892e05014b40e78210f0d306462\\",\\"trusted_identity_sha256\\":\\"3f23e879163cbb99621547061de69befbc42bb25f110dad5514ee52cfb370c93\\"}\"
PYTHONUNBUFFERED=1 PYTHONPATH=\"/Users/spotted/projects/ds4-finetuning/python-envs/mlx/src:/Users/spotted/projects/ds4-finetuning/vendor/mlx-lm\" \"$PYTHON\" \"$PILOT_SCRIPT\" --attempt 4 --phase phase-a --launch-check-only --log-path \"$LOG\" --authorization-json \"$AUTHORIZATION_JSON\" --model \"$MODEL\" --data \"$DATA\" --adapter-path \"${A4_PHASE_A_OUTPUT}\" --config \"$CONFIG\"
mkdir -p \"$(dirname \"$LOG\")\"
set -o noclobber; exec 3>\"$LOG\"; set +o noclobber
printf 'attempt-4 namespace phase=phase-a output=%s start=%s final=%s config=%s log=%s\n' \"$A4_PHASE_A_OUTPUT\" \"$A4_PHASE_A_START_CHECKPOINT\" \"$A4_PHASE_A_FINAL_CHECKPOINT\" \"$A4_PHASE_A_CONFIG\" \"$LOG\" >&3
set +e
PYTHONUNBUFFERED=1 PYTHONPATH=\"/Users/spotted/projects/ds4-finetuning/python-envs/mlx/src:/Users/spotted/projects/ds4-finetuning/vendor/mlx-lm\" \"$PYTHON\" \"$PILOT_SCRIPT\" --attempt 4 --phase phase-a --log-path \"$LOG\" --log-fd 3 --authorization-json \"$AUTHORIZATION_JSON\" --model \"$MODEL\" --data \"$DATA\" --adapter-path \"${A4_PHASE_A_OUTPUT}\" --config \"$CONFIG\" --train --fine-tune-type lora --num-layers 16 --iters 2 --batch-size 1 --learning-rate 1e-5 --max-seq-length 4096 --mask-prompt --grad-checkpoint --grad-accumulation-steps 1 --seed 0 --optimizer adam --val-batches 25 --steps-per-report 1 --steps-per-eval 2 --save-every 1 --segment-size 1 2>&1 | tee /dev/fd/3
status=${PIPESTATUS[0]}
exec 3>&-
exit \"$status\""
