#!/bin/bash
# Entrypoint of rt-avatar under kiri-sched (stack /box/config/kiri/stacks/avatar).
# Runs the resident LongCat-Video-Avatar render server exactly as avatar-serve.pre-sched did, plus a
# readiness endpoint for the scheduler: http://127.0.0.1:${AVATAR_HEALTH_PORT:-8190}/health answers
# 200 only after the server printed "[SERVE] ready" (404 before), and stops when the server stops.
unset LD_LIBRARY_PATH
. /home/rtx3090/llm/avatar/venv/bin/activate
cd /home/rtx3090/llm/engines/LongCat-Video || exit 1
H=$(mktemp -d /tmp/avatar-health.XXXX)
python3 -m http.server "${AVATAR_HEALTH_PORT:-8190}" --bind 127.0.0.1 --directory "$H" >/dev/null 2>&1 &
hp=$!
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 \
  --checkpoint_dir=/models/video/LongCat-Video-Avatar-1.5 --model_type avatar-v1.5 --use_distill --use_int8 \
  --mix_steps 0 --lora_path lora/dmd_lora.safetensors --stage_1=ai2v --resolution 480p 2>&1 \
  | awk -v f="$H/health" '{ print; fflush() } /\[SERVE\] ready/ && !s { s = 1; print "ok" > f; close(f) }'
rc=${PIPESTATUS[0]}
kill "$hp" 2>/dev/null
exit "$rc"
