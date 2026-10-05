# supervisionvidæo

**Produce and verify in one contract.** A local video generator that refuses to
emit a render it has not measured.

```python
from supervisionvidaeo import produce, SupervisionRefused

try:
    receipt = produce({"palette": "cyan", "motion": 0.03, "duration": 3})
    print(receipt.sha256)        # proof: spec ⊕ supervision
    print(receipt.supervision)   # calidad, luminancia, contraste, movimiento
except SupervisionRefused as e:
    print("refused:", e.reason)  # the render never became a path
```

Most generative pipelines hand you an output and hope it looks right. This one
supervises every render and **raises instead of returning a broken file**. The
producer and the verifier are not two libraries you wire together — they are one
loop:

```
SPEC → PRODUCE → SUPERVISE → (PASS: emit receipt) | (FAIL: refuse)
```

---

## Why this exists

Every cloud video backend (`fal`, `openrouter`, `xai`, `deepinfra`) returns a
URL and trusts that the result is usable. None of them verify. This package is
the opposite bet: **a render that fails its gate is not a product.**

Two properties fall out of that:

- **Deterministic.** The verifier is statistics + geometry, not machine
  learning. Every verdict traces to a measurable threshold, so it is auditable —
  "luminance 67.98, contrast 23.78, motion 4.61 → PASS" beats "probably fine."
- **Honest.** `produce()` returns a `Receipt` or raises. There is no code path
  that returns a path to a black video.

---

## Install

```bash
pip install -e .
```

Requires Python 3.10+, NumPy, and OpenCV (headless). **No ML models, no cloud,
no API keys.**

---

## The contract

### `produce(raw_spec, out_path=None, *, producer=None, sample_every=20, max_attempts=1) → Receipt`

Coerces anything into a bounded `SceneSpec`, produces, supervises, and returns a
`Receipt` **only on PASS**. Raises `SupervisionRefused` otherwise.

```python
receipt = produce({
    "title": "a calm cyan field",
    "palette": "cyan",     # gold | cyan | neon | amber | mono
    "motion": 0.03,        # clamped to 0.01–0.08
    "duration": 3,         # clamped to 2–6
    "particles": 6000,     # clamped to 2000–20000
})
```

Bad input never raises — it clamps. `SceneSpec.bounded()` is total.

### `verify(mp4_path) → dict`

Supervise an existing video. The gate, without the producer.

```python
verify("clip.mp4")
# {'calidad': 'PASS', 'luminancia_media': 67.98, 'frames_negros': 0,
#  'receipt': '060884c2...', ...}
```

### `toolchain_status() → dict`

Report whether this package can actually produce — never assume.

---

## The receipt

A `Receipt` binds the producer's input to the verifier's output:

```
sha256 = H(spec_fingerprint ⊕ supervision_receipt ⊕ frames ⊕ size_bytes)
```

Same scene → same proof. Change the spec or the render and the proof changes.
This is the artifact a downstream consumer trusts:

```json
{
  "spec": { "palette": "cyan", "motion": 0.03, "duration": 3, ... },
  "spec_fingerprint": "3136ba04...",
  "output": "proof.mp4",
  "frames": 90,
  "size_bytes": 4015984,
  "supervision": { "calidad": "PASS", "luminancia": 67.98, "contraste": 23.78,
                   "movimiento": 4.606, "frames_negros": 0 },
  "supervision_receipt": "060884c2...",
  "sha256": "6a641151...",
  "produced_at": "2026-10-05T05:09:32Z"
}
```

---

## The verifier

Deterministic computer vision — no deep learning. Detects:

| Check | Threshold | Meaning |
|---|---|---|
| black frames | luminance < 12 | render failed to draw |
| blown out | luminance > 235 | over-exposed |
| flat | contrast < 8 | no detail |
| static | motion < 0.5 | nothing moved |
| QR per frame | OpenCV detect | optional provenance |

63 tests cover the vision modules; 19 cover the produce/verify contract.

---

## The producer

The default producer is the local [vidæo](https://github.com/MYaelMendez)
toolchain (CuPy on the operator's GPU). It is **injectable** — pass any callable
`(spec, out_path) -> frame_count` to run the loop without a GPU:

```python
def my_producer(spec, out):
    render_frames(spec, out)
    return spec.duration * spec.fps

receipt = produce({"palette": "gold"}, "out.mp4", producer=my_producer)
```

This is why the refusal tests run anywhere: a producer that emits black is
refused on any machine.

---

## What this is not

- **Not photorealistic generation.** It renders procedural particle fields and
  deterministic surfaces, not diffusion-model output. Use a cloud backend for
  cinematic video.
- **Not image-to-video.** Text-to-video only, and it says so instead of failing
  silently.
- **Not ML.** The verifier is deterministic and auditable by design.

---

## Philosophy

> When you tell the truth you don't have to remember anything.

A generator that verifies its own output is a generator you can build on. This
package is the smallest honest version of that idea: one loop, one contract, one
proof — or an exception.

**#opensourceware** — the tool is open, the secrets are not.

## License

MIT.
