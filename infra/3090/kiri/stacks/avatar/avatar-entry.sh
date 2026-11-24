#!/bin/bash
# Entrypoint of rt-avatar under kiri-sched (stack /box/config/kiri/stacks/avatar).
# Runs the resident LongCat-Video-Avatar render server exactly as avatar-serve.pre-sched did, plus a
# readiness endpoint for the scheduler: http://127.0.0.1:${AVATAR_HEALTH_PORT:-8190}/health answers
# 200 only after the server printed "[SERVE] ready" (404 before), and stops when the server stops.
# Output passes through tee unchanged (no line buffering, so partial lines and tqdm's \r show at once).
# The ready line is watched by a bash read loop, not awk: the image's awk is mawk, which reads a pipe in
# large blocks, so a ready line followed by silence was never seen (the 11:18 and 11:30 failures).
# Optional extra environment for diagnosis (e.g. NCCL_DEBUG=INFO): ~/llm/avatar/avatar-entry.env
[ -f /home/rtx3090/llm/avatar/avatar-entry.env ] && { set -a; . /home/rtx3090/llm/avatar/avatar-entry.env; set +a; }
unset LD_LIBRARY_PATH
. /home/rtx3090/llm/avatar/venv/bin/activate
cd /home/rtx3090/llm/engines/LongCat-Video || exit 1
H=$(mktemp -d /tmp/avatar-health.XXXX)
python3 -m http.server "${AVATAR_HEALTH_PORT:-8190}" --bind 127.0.0.1 --directory "$H" >/dev/null 2>&1 &
hp=$!
set -o pipefail
# In the background so a docker stop (SIGTERM to this shell) reaches torchrun at once instead of
# waiting for the kill after the grace period; torchrun passes it on to its workers.
trap 'trap - TERM INT; kill -TERM 0' TERM INT
torchrun --nproc_per_node=2 run_demo_avatar_single_audio_to_video.py --context_parallel_size=2 \
  --checkpoint_dir=/models/video/LongCat-Video-Avatar-1.5 --model_type avatar-v1.5 --use_distill --use_int8 \
  --mix_steps 0 --lora_path lora/dmd_lora.safetensors --stage_1=ai2v --resolution 480p 2>&1 \
  | tee >(while IFS= read -r l; do case $l in *"[SERVE] ready"*) echo ok > "$H/health" ;; esac; done) &
wait $!
rc=$?
kill "$hp" 2>/dev/null
exit "$rc"
