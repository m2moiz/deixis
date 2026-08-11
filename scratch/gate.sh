set -e
cd "$(git rev-parse --show-toplevel)"
rm -f scratch/gate_full.json scratch/gate_resumed.json scratch/gate_resumed.json.ckpt
echo "=== 1. uninterrupted baseline ==="
uv run python -m jaano.suno scratch/meeting.wav -o scratch/gate_full.json --no-resume 2>&1 | tail -1

echo "=== 2. interrupted run (kill at ~45s) ==="
uv run python -m jaano.suno scratch/meeting.wav -o scratch/gate_resumed.json --no-resume >scratch/gate_int.log 2>&1 &
PID=$!
sleep 45
kill -9 $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "killed. checkpoint:"; ls -la scratch/gate_resumed.json.ckpt 2>/dev/null | awk '{print $5, $9}' || echo "NO CHECKPOINT WRITTEN"

echo "=== 3. resume ==="
uv run python -m jaano.suno scratch/meeting.wav -o scratch/gate_resumed.json 2>&1 | tail -2

echo "=== 4. compare ==="
shasum -a 256 scratch/gate_full.json scratch/gate_resumed.json
python3 -c "
import json
a=json.load(open('scratch/gate_full.json')); b=json.load(open('scratch/gate_resumed.json'))
print('text identical     :', a['text']==b['text'])
print('sentences identical:', a['sentences']==b['sentences'])
print('sentence count     :', len(a['sentences']), '/', len(b['sentences']))
"
