# Review: Web Audio, transport and A/V sync

Status: **DONE** — 2026-08-11. Static + derived review of `app/static/app.js`
against the Web Audio and HTML media specs, `index.html`, `style.css`, and the
artefact contract in `app/main.py`.

Reproduce the structural half with `python scripts/check_js_syntax.py
app/static/app.js` and the transport arithmetic (W-14, W-15) with `python
scripts/check_transport.py` — stdlib only, no Node, no npm, no browser, so both
run on the demo machine. The remaining numeric claims are arithmetic on spec
constants and are shown inline per finding.

**Scope.** Not "does the DSP mute the interferer" (that is `REDTEAM_SILENCE.md`)
and not "do the modules agree about indices" (that is `REVIEW_CONTRACTS.md`), but
"does the browser deliver what the backend produced, on time, and does the click
that switches speakers do exactly what it claims". The failure class here is
**time**: everything is correct at rest, and wrong under a race, a repeat, a
seek, or a tab switch. None of it raises where you can see it — the console is
behind the projector.

**Verdict: fifteen findings, all fixed.** Three are high severity and all three
are reachable at a presentation, by the two most likely things a presenter does:
switch speakers quickly, and alt-tab away and back.

The last two (W-14, W-15) were found by *running* the app, after this pass had
declared it clean, and both were visible on the very first play-through. They are
kept in this document rather than filed separately because they belong to the
same failure class — time — and because the gap is worth stating plainly: a
static review reads what the code does against the spec, and neither of these
defects contradicts the spec. They contradict the *clock the spec hands you*.

---

## Summary

| # | finding | severity | status |
|:--|:--------|:---------|:-------|
| W-1 | `loop()` chains multiply — one per tab-away, each fighting the others for `playbackRate` | **high** | fixed |
| W-2 | Hard reseek re-fires while the seek is still running → seek storm | **high** | fixed |
| W-3 | A fast second click throws mid-`forEach`, leaving gains partly applied | **high** | fixed |
| W-4 | `position()` fabricates 80 ms of drift on every seek and resume | medium | fixed |
| W-5 | 5 ms scheduling lead is shorter than the 8 ms render quantum → truncated crossfade | medium | fixed |
| W-6 | Bit-exact playback is silently rate-dependent and unreported | medium | fixed |
| W-7 | `hitTest()` reads the audio clock, `draw()` the video clock — up to 6 frames apart | medium | fixed |
| W-8 | Nothing repaints while paused; first paint lands in a 150 px box | medium | fixed |
| W-9 | Four async failure paths hang the progress bar with no message | medium | fixed |
| W-10 | `init()` re-entry leaks the previous audio graph, still connected to destination | medium | fixed |
| W-11 | `levels()` allocates 480 KiB/s; `meters()` cannot be stopped | low | fixed |
| W-12 | Canvas backing store sized in CSS pixels — soft strokes on the demo laptop | low | fixed |
| W-13 | Keyboard handler steals browser and OS shortcuts; held Space re-toggles | low | fixed |
| W-14 | Deadband is exactly half a frame period, so a **perfectly synced** clip reads yellow on 50% of ticks | medium | fixed |
| W-15 | `onended` guard reads the output clock, so the play/pause button never resets at end of clip | medium | fixed |

W-14 and W-15 are later additions: they came from the user's Stage-1 test run,
not from this static pass, and both are recorded here with the same evidence
standard as the rest. That they were *reported* rather than *derived* is the
useful part — see the note at the end of each on why reading the code was never
going to surface them.

Plus one hardening change that is **not** a live defect, and one deliberate
non-change, both recorded below so neither is mistaken for an oversight later.

---

## W-1 — the corrector multiplied every time the tab lost focus

**High.** Reachable by alt-tabbing during playback, which is what a presenter
does to reach their slides. The trigger and the damage coincide.

`transport.loop()` had two callers:

```js
// transport.toggle()
await el.video.play().catch(() => {});
this.loop();

// visibilitychange handler
if (!document.hidden && engine.playing) { ...; transport.loop(); }
```

and a chain's only exit was:

```js
const tick = () => { if (!engine.playing) return; ... requestAnimationFrame(tick); };
```

`engine.playing` is `true` for exactly as long as the problem lasts. So the
visibilitychange chain did not *replace* the running one, it **joined** it —
permanently. Every tab-away-and-back while playing left one more chain running
for the life of the page:

| tab switches while playing | live chains | `playbackRate` writers | canvas redraws / frame |
|:---|:---|:---|:---|
| 0 | 1 | 1 | 1 |
| 1 | 2 | 2 | 2 |
| 5 | 6 | 6 | 6 |

Extra redraws are merely wasteful. The controller is not: `correct()` is a
P-controller writing a shared actuator, and N copies do not average — each reads
the drift *after* the others have already acted on it and applies its own full
correction on top. Worse, each can independently trip the `HARD_SEEK` branch, and
a reseek invalidates the `mediaTime` the other N−1 callbacks are mid-flight
reasoning about.

And the moment this fires hardest is the moment it was created. rAF and rVFC do
not run in a hidden tab while the `AudioContext` keeps going, so on return the
drift is seconds — comfortably past `HARD_SEEK` — so the very first tick after
the chain doubles is a reseek, from two chains at once.

**Fix.** A generation token. `loop()` takes a ticket, and a chain retires itself
on its next tick if a newer one exists:

```js
const me = ++this.gen;
const alive = () => engine.playing && me === this.gen;
```

Both the rVFC chain and the rAF fallback exit on `!alive()`. `pause()`,
`onEnded()` and `load()` also bump `gen`, so a chain cannot outlive the job it
was started for. It is now safe for any number of callers to ask for a loop, as
often as they like — which is the property the visibilitychange handler needed
all along.

**The same race, one level up.** `toggle()` is `async` and awaits
`engine.play()` and `video.play()`. A second Space press lands mid-`await`, while
`engine.playing` is still `false`, so it takes the "not playing" branch too and
calls `engine.play()` again — whose `stopSource()` kills the source the first
call just started. Fixed with a `busy` re-entrancy guard around the whole body.

---

## W-2 — one large drift became a seek that never converged

**High.** Same trigger as W-1: returning to the tab.

```js
if (Math.abs(drift) > HARD_SEEK) {
  el.video.currentTime = audioPos;      // fired again on the very next frame
  el.video.playbackRate = 1;
}
```

Assigning `currentTime` starts a seek; it does not complete one. In the rVFC
path — the primary path in Chrome — `correct()` is driven by `meta.mediaTime`,
which is the timestamp of the frame **actually presented**. That keeps reporting
the pre-seek frame until the seek finishes and a new frame is composited. So
every callback in between computes the same out-of-range drift and issues the
same seek, each one cancelling the one before it. The condition that would end
the loop is precisely the thing the loop keeps restarting.

Compounding: with W-1 in play there are N chains doing this to each other.

**Fix.** Consult the element's own state, which is what it is for:

```js
if (!el.video.seeking) { el.video.currentTime = audioPos; el.video.playbackRate = 1; }
```

Note the rAF *fallback* path was never exposed to this, because it reads
`el.video.currentTime`, which updates synchronously to the seek target. That
asymmetry is exactly why the bug is invisible in a fallback-path test.

---

## W-3 — the second of two fast clicks threw part-way through the switch

**High.** Reachable by double-clicking two faces, or pressing `1` then `2`, within
the 25 ms crossfade.

The Web Audio spec is explicit: scheduling any automation event **inside** the
time interval of a running `setValueCurveAtTime` MUST throw `NotSupportedError`
— and `cancelScheduledValues(t)` only removes events at or after `t`, so a curve
that *started* before `t` is still active and still fatal.

That is exactly the shape of a fast second click:

```
click 1 at t1     -> setValueCurveAtTime(curve, t1, 0.025)      on every gain
click 2 at t2     -> cancelScheduledValues(t2)   [curve started at t1 < t2: SURVIVES]
                  -> setValueAtTime(v, t2)       [t2 is inside [t1, t1+25ms): THROWS]
```

The old code had no `catch`, so the exception propagated out of `select()` from
inside the `forEach`. Two consequences, both silent:

- **Likely case:** it throws on the first gain, so *no* gain moves and
  `ui.markActive(k)` — which sits after the loop — never runs. The click does
  nothing at all. To a presenter, speaker switching has simply stopped working.
- **Worst case:** it throws after some gains have been processed, leaving the
  incoming gain scheduled up to 1 with the outgoing one never taken down. **Both
  speakers audible simultaneously** — the precise failure this project exists to
  prevent, produced by the UI rather than the DSP.

**Fix.** Wrap the automation in `try`, and on any throw abandon the crossfade to
preserve the invariant:

```js
hardSet(k) {
  const now = this.ctx.currentTime;
  this.gains.forEach((g, i) => {
    const target = (i === k) ? 1 : 0;
    try { g.gain.cancelScheduledValues(0); } catch (_) {}
    try { g.gain.setValueAtTime(target, now); } catch (_) { g.gain.value = target; }
  });
}
```

`cancelScheduledValues(0)` clears the **entire** timeline, not a scoped window:
a partially-applied event left behind would fire later and re-open the channel
being closed. The final `g.gain.value = target` is a third fallback for the case
where even `setValueAtTime` refuses.

The trade is deliberate and stated: a 25 ms click is audible; two speakers at
once is a failed demo. `ui.markActive(k)` also moved outside the `try` so the
label can never disagree with the audio.

---

## W-4 — the transport invented 80 ms of drift at every seek

**Medium.** Silent: the video visibly hunts for a few seconds after each seek.

```js
// old
return Math.min(this.duration, Math.max(0, this.anchorOffset + (now - this.anchorCtxTime)));
```

`play()` schedules the source 80 ms in the future (`t0 = currentTime + 0.08`) and
anchors to `t0`, so for those 80 ms `now - anchorCtxTime` is **negative**. The
outer `Math.max(0, …)` clamps the *sum*, which rescues the case `anchorOffset ===
0` and nothing else. Meanwhile `seek()`/`toggle()` set `video.currentTime`
immediately, so the video is already at the target.

Derived over the cushion (drift = video − reported audio; `rate` is the
corrector's response):

```
case                              t   video      old      new  drift_old  drift_new  rate_old
play from 0                    0.00    0.00     0.00     0.00         0ms        0ms     1.000
play from 0                    0.08    0.08     0.00     0.00        80ms       80ms     0.980

resume from 30 s               0.00   30.00    29.92    30.00        80ms        0ms     0.980
resume from 30 s               0.04   30.04    29.96    30.00        80ms       40ms     0.980
seek to 30 s                   0.00   30.00    29.92    30.00        80ms        0ms     0.980
```

At offset 0 the clamp masks it — which is why it survived: the first thing anyone
tests is pressing Play on a fresh clip. Everywhere else the reported position sat
a constant 80 ms behind the anchor, 4× the 20 ms deadband, so the corrector
applied its **full −2 % authority** and held it. Recovering 80 ms at 2 % takes
4 s, and every arrow-key scrub restarts the penalty.

**Fix.** Clamp the elapsed term, not the sum:

```js
const elapsed = Math.max(0, now - this.anchorCtxTime);
return Math.min(this.duration, Math.max(0, this.anchorOffset + elapsed));
```

The residual drift under "play from 0" is not a bug: the audio genuinely has not
started, the video genuinely has, and braking is the right response.

---

## W-5 — the crossfade was scheduled inside the block already being rendered

**Medium.** Audible as a click on roughly a third of switches.

The Web Audio render quantum is a fixed **128 frames**. That is a frame count,
not a duration, so it scales inversely with the context rate — and this app
deliberately runs a 16 kHz context (see W-6):

```
  16000 Hz: 128 frames =  8.000 ms  |  5 ms lead = 0.62 quanta  |  20 ms lead = 2.50 quanta
  44100 Hz: 128 frames =  2.902 ms  |  5 ms lead = 1.72 quanta  |  20 ms lead = 6.89 quanta
  48000 Hz: 128 frames =  2.667 ms  |  5 ms lead = 1.88 quanta  |  20 ms lead = 7.50 quanta
```

`ctx.currentTime` advances one quantum at a time, so `currentTime + 0.005` at
16 kHz names an instant **inside the block being rendered right now** — 0.62 of a
quantum is less than one. The curve's head is truncated to the block boundary,
losing up to 8 ms of a 25 ms fade:

```
  XFADE 25 ms = 3.125 quanta; truncating one quantum loses 32% of the fade
```

A crossfade that starts 32 % of the way in is a step of `sin(0.32·π/2) ≈ 0.48` —
a −6 dB discontinuity, i.e. a click. This is the failure that the 16 kHz decision
*causes*: at 44.1/48 kHz the same 5 ms lead clears a quantum and the bug does not
exist. Tuned on a default-rate context, shipped on a 16 kHz one.

**Fix.** `SCHED_LEAD = 0.020` — 2.5 quanta at 16 kHz, ≥ 1 quantum at every rate
in use, and still far below the ~100 ms at which a switch stops feeling instant.

---

## W-6 — "bit-exact zeros" was rate-dependent, and the page did not say so

**Medium.** Not a crash: a silent downgrade of the project's headline claim.

`decodeAudioData` resamples to the context rate. The 16 kHz context is therefore
not a memory optimisation — it is what makes the decode a **copy** instead of a
filter:

```
  16 kHz buffer: 7.3 MiB ; 48 kHz buffer: 22.0 MiB (x3)      [60 s, 2 ch, f32]
```

The 3× is the incidental part. The load-bearing part is that a resampler is a
windowed sinc: it forms each output sample from a weighted span of inputs, so it
**drags energy from the speech on either side of a gated pause straight into the
middle of it**. `REDTEAM_SILENCE.md` measures exact zeros server-side; if the
browser resamples, those samples are no longer zero in the buffer actually
played, and the residual sits exactly where the gate was working hardest — in the
pauses, which is where the original "ghostly whispers" complaint lives.

The old code passed `{ sampleRate: 16000 }` and assumed it took. Not every
implementation honours an explicit rate, and the constructor can throw — which
would have taken the whole player down.

**Fix.** Fall back rather than fail, and *report* the degraded path instead of
quietly weakening the claim:

```js
try { this.ctx = new Ctor({ sampleRate: 16000 }); }
catch (_) { this.ctx = new Ctor(); this.resampled = this.ctx.sampleRate !== 16000; }
```

The diagnostics panel now carries `context_rate` and `bit_exact_playback`, so
"the zeros are exact" is a checkable statement on the machine in the room rather
than an inherited assumption. Running at the device rate is far better than not
running; being unable to tell which happened is not.

---

## W-7 — the click was matched against a frame that was not on screen

**Medium.** Wrong-speaker or dead click, silently.

```js
draw(mediaTime)  -> idx = Math.round(mediaTime * fps)          // VIDEO clock
hitTest(ev)      -> idx = Math.round(engine.position() * fps)  // AUDIO clock
```

The whole transport exists because these two clocks differ. They are only
*guaranteed* within `HARD_SEEK`:

```
  HARD_SEEK 250 ms at 25 fps = 6.25 frames of head movement
```

Six frames is enough for a turning head to leave its old box. So the user clicks
the green rectangle they can see, `hitTest` matches against a rectangle from a
different frame, and the click either misses (nothing happens) or lands in the
*other* speaker's box from that frame — selecting the wrong person, which reads
as the matcher having failed.

**Fix.** `draw()` records what it actually drew (`this.lastIdx = idx`) and
`hitTest()` reuses it. The click is now resolved against the pixels on screen by
construction, whatever the clocks are doing.

---

## W-8 — the overlay only ever repainted during playback

**Medium.** Three symptoms, one cause.

`draw()` was called from exactly two places: the correction loop, and once from
`load()`. The loop only runs while playing. So:

1. **Scrubbing while paused.** `ArrowLeft`/`ArrowRight` → `seek()` moved the
   video, but boxes and clock stayed on the frame we left. Combined with W-7 the
   overlay was doubly stale.
2. **Resize.** No `resize` handler at all. The canvas is stretched by CSS, so
   geometry survives, but the backing store keeps the old pixel size and the
   drawing is scaled — a window resize during a demo visibly softens the boxes.
3. **The first paint.** `load()` calls `ui.draw(0)` immediately after setting
   `video.src`. Before metadata arrives a `<video>` has **no intrinsic aspect
   ratio**, so the shipped `#video { width:100%; height:auto }` resolves to the
   CSS default height of **150 px**. The first paint lands in a 150 px-tall box,
   and while paused nothing ever repaints it.

**Fix.** A `repaint()` that redraws at the current position without advancing
anything, wired to the three moments the geometry can change while the loop is
not running:

```js
el.video.addEventListener('loadedmetadata', () => ui.repaint());
el.video.addEventListener('seeked', () => { if (!engine.playing) ui.repaint(); });
window.addEventListener('resize', () => ui.repaint());
```

plus an explicit repaint in `seek()`'s paused branch. `loadedmetadata` is the one
that fixes symptom 3: it fires exactly when the element learns its ratio and
resizes itself.

---

## W-9 — every async failure looked identical to a slow model load

**Medium.** Same symptom class as `REVIEW_CONTRACTS` F-1, from four new places.

Four paths could reject with nothing catching them, leaving the progress panel up
and the page dead:

| path | what rejects | old symptom |
|:-----|:-------------|:------------|
| `upload()` | `fetch('/api/jobs')` — server down | panel at 0 %, no message |
| `listen()` | `JSON.parse(e.data)` — malformed SSE | bar frozen mid-run |
| `listen()` → `load()` | four fetches + a decode, inside an event handler | bar at **100 %**, console-only rejection |
| `boot()` | `/?job=demo` opened cold | blank page, upload panel hidden |

The third is the worst: an unhandled rejection inside an `onmessage` handler is
console-only, and the bar reads 100 % — the job *did* finish. The fourth is the
presentation path.

**Fix.** One `toUpload(message)` helper that shows the error **and** restores the
panel state, used by all four. Each open-coding its own recovery is how one of
them ends up hiding a panel it never unhides.

---

## W-10 — retrying a job left the previous one playing underneath

**Medium.** Reachable by the error-then-retry path, i.e. immediately after any
W-9 failure.

`init()` guarded the `AudioContext` (`if (!this.ctx)`) but not the graph. On a
second call it built a fresh splitter, gains, analysers and master and connected
them to `destination` — while the previous ones were **still connected**, still
holding a running `AudioBufferSourceNode`. The old audio kept playing, on gain
nodes no longer referenced by `engine.gains`, so no click could reach them and
`hardSet()` could not silence them either. Two jobs, audible at once, one of them
uncontrollable.

**Fix.** `engine.reset()` — stop the source, disconnect every node, drop the
references, and reset the transport fields — called at the top of `init()` after
the context is secured. Every `disconnect()` is individually guarded, because a
node that was never connected throws and a *half*-completed teardown is worse
than none: it leaves audible nodes with no handle left to silence them.

---

## W-11 — half a megabyte of garbage per second, and a loop that could not be stopped

**Low**, but it lands as a dropped frame at an unpredictable moment.

```js
// old: a fresh buffer per analyser, per frame
return this.analysers.map((a) => { const buf = new Float32Array(a.fftSize); ... });
```

```
  2 analysers x Float32Array(1024) = 8192 B/frame
  at 60 fps = 480.0 KiB/s = 0.469 MiB/s
  over a 3-minute demo = 84.4 MiB allocated
```

None of it survives the frame, so it is pure minor-GC pressure — and a minor GC
during playback is a dropped frame in the overlay, on the one screen anyone is
looking at.

`meters()` had the second half of W-1's disease with none of its excuse: it
recursed unconditionally, with no exit at all. Loading a second job started a
second permanent loop; both then walked the DOM and read every analyser forever,
including while the player panel was hidden.

**Fix.** Buffers hoisted into `this._lvlBuf`, rebuilt only when the analyser
count changes. `meters()` gets the same generation token as `transport.loop()`
and additionally exits when `el.panelPlayer.hidden`.

The **channel-indexed** meter lookup from `REVIEW_CONTRACTS` F-5 is preserved
verbatim through this rewrite — `levels[]` is indexed by channel and
`el.speakers.children` by track, and they diverge the moment a face is detected
without a stem. Re-verified after the edit; the comment explaining it is retained
so the next rewrite does not undo it.

---

## W-12 — the overlay was drawn at 1× on a 2× panel

**Low**, cosmetic, on the one screen that matters.

`cv.width = w` sizes the backing store in **CSS** pixels while CSS stretches the
canvas to the element box. On a HiDPI laptop the compositor upscales, and the
2 px box strokes and 12 px labels come out visibly soft.

**Fix.** Size in device pixels and scale the context once, so all drawing code
keeps working in CSS-pixel coordinates:

```js
const dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
g.setTransform(dpr, 0, 0, dpr, 0, 0);
```

Capped at 2×: beyond that the canvas costs memory for no visible gain. The
`if` matters — assigning `canvas.width` clears the canvas even when the value is
unchanged, so an unconditional assignment would blank the overlay every frame.

The box destructure inside the draw loop was renamed `bw2`/`bh2`; without it the
new backing-store variables would be shadowed and every box drawn at canvas size.
`scripts/check_js_syntax.py` asserts the rename stayed complete.

---

## W-13 — the player ate browser and OS shortcuts

**Low**, but it fires during a demo, not during testing.

The `keydown` handler checked no modifiers. `Ctrl+1` (switch to browser tab 1)
also selected speaker 1; `Alt+←` (back) also scrubbed 5 s. And `Space` had no
`e.repeat` guard, so *holding* it auto-repeated play/pause dozens of times a
second — which, combined with W-1's missing `busy` guard, is a reliable way to
tear the audio graph apart.

**Fix.** Return early on `ctrlKey || metaKey || altKey`; `Space` still
`preventDefault()`s (so the page never scrolls) but only toggles when
`!e.repeat`.

---

## W-14 — a perfectly synced clip reported itself out of sync

**Medium.** Reported as *"not always green in sync. it becomes yellow most of the
times"*. Cosmetic-looking, and it was not: the same value steered `playbackRate`.

`requestVideoFrameCallback` hands you the `mediaTime` of the frame **actually on
screen**, and a frame is on screen at a frame boundary or not at all. So
`mediaTime` is the true media position *floored* to `1/fps`. Two clocks in
perfect agreement therefore report a difference that sweeps the whole frame
period as playback advances between boundaries:

```
fps 25 -> frame period 40 ms
mediaTime - audioPos  sweeps -40..0 ms, uniformly, on a clip that is exactly in sync
```

`DEADBAND` was `0.020` — **exactly half of that**. So the readout tripped on half
of every sweep and wrote a correction on half of every sweep. Measured over
20001 ticks of a synced clip:

```
old fixed 20 ms band, no debias:  warn 50.0%   rate!=1 50.0%
```

Half the yellow was the readout being honest about a number that meant nothing.

### The fix that does not work

Widening the band to a full frame period makes the yellow go away, which is why
it is tempting. `check_transport.py` rejects it: **quantisation is one-sided**
(`mediaTime` is floored, so it can only lag) while a deadband is symmetric. A
±40 ms band spends its entire allowance excusing a lag that never exceeds 40 ms,
and buys nothing in the other direction:

```
widened band (40 ms), no debias:  warn 0.0%  on a synced clip     <- looks fixed
  ... but a real +35 ms desync is flagged on 0% of ticks          <- it is blind
```

### The fix

Half a frame period is the *expected value* of a uniform sweep, so it is a known
offset rather than error. `transport.quantBias()` returns it and `correct()`
subtracts it before anything judges or displays the residual — then the band
stays at 20 ms, tight in **both** directions:

```
debias + 20 ms band:  warn 0.0%   rate!=1 0.0%
```

Detection floor is `DEADBAND + 1/(2·fps)`: 40.8 ms at 24 fps, 40.0 at 25, 36.7 at
30, 28.3 at 60 — all under the ~100 ms threshold at which lip-sync error becomes
visible, so nothing a viewer could notice is masked. A real desync of the floor
plus 10 ms is flagged on 100% of ticks, both signs, at all four rates.

The displayed number was fixed with the same edit, not separately: showing the
raw drift would have a synced clip reading a steady `-40 ms`, which is worse than
the yellow — it is a wrong number in a confident colour.

While in there: the P-controller steers on the **excess over the deadband**, not
the raw drift. Steering on the raw value commands `KP × DEADBAND` = 1% at the
very instant it starts caring — a rate step out of nowhere that over-corrects
back inside the band and limit-cycles. The excess form goes to zero with the
overshoot, verified across overshoots of 1e-3 down to 1e-6.

### Why the static pass missed it

Nothing here contradicts a spec. `DEADBAND = 0.020` is defensible in isolation,
`mediaTime` is used exactly as documented, and the arithmetic is correct. The
defect only exists in the *relationship* between a constant chosen in seconds and
a frame rate discovered at runtime — and it is invisible until you know the clip
is 25 fps. Reading the code cannot tell you that; playing the clip can.

---

## W-15 — the play button never reset at the end of a clip

**Medium.** Reported as *"after playing the video the pause button keeps stuck at
pause even when the video has ended… it should automatically become play"*.

Web Audio hands you two clocks, and they are not interchangeable:

- `ctx.currentTime` — the **rendering** clock. This is what `src.start(t)` is
  scheduled against.
- `getOutputTimestamp().contextTime` — the **output** clock: the frame leaving
  the speakers. It lags the rendering clock by `baseLatency + outputLatency`,
  routinely **100–200 ms** under shared-mode WASAPI on Windows.

`engine.position()` is built on the output clock, correctly — it is what the
overlay and the time display should follow. The `onended` guard then reused it:

```js
src.onended = () => { if (… && this.position() >= this.duration - 0.05) transport.onEnded(); };
```

`onended` is a **one-shot callback, not a polled condition**, and that is the
whole defect. It is delivered once, at the instant the buffer drains, and the
guard body runs at that single instant — when `position()` is still
`out_latency` short of `duration`. The guard was false, `onEnded()` never ran,
and nothing ever re-tested it. `position()` does go on to reach `duration` (it
clamps there) a beat later, but by then the only callback that would have read it
is gone.

So the button sat on "Pause" after every clip and had to be pressed twice.

The 50 ms slack is why this was not caught earlier — it is not uniformly broken,
it is broken above a threshold:

```
latency   5 ms -> fires        latency  60 ms -> SILENTLY FAILS
latency  20 ms -> fires        latency 100 ms -> SILENTLY FAILS
latency  40 ms -> fires        latency 200 ms -> SILENTLY FAILS
```

A machine with a small output buffer works. The presentation laptop does not.

### Fix

`play()` records the context time at which the buffer runs out on its own, and
the guard compares the **rendering** clock against it:

```js
const endsAt = t0 + (this.duration - startAt);
src.onended = () => {
  if (this.source === src && this.playing
      && this.ctx.currentTime >= endsAt - 0.05) transport.onEnded();
};
```

Exact at any buffer depth, verified across output latencies of 5/50/200 ms and
start offsets of 0.0/3.5/9.0 s. The guard still has to reject a *stale* callback —
Chrome delivers a stopped source's `onended` asynchronously, after a seek has
already started the next one — so both halves are checked: it tolerates up to 2 s
of delivery jitter past the scheduled end, and rejects deliveries at 0.5/2/6 s
into a 9.2 s clip. The `source === src` identity check is kept as well; the time
guard no longer depends on it.

---

## Not a defect

**`setTracks` builds buttons with `innerHTML` interpolating `t.label`.** This is
the shape of an XSS bug, but `label` is not user data: `FaceTrack.label()` in
`app/vision.py:65` is

```python
return f"Speaker {chr(ord('A') + self.track_id)}"
```

— fully server-constructed from an integer, with no path from the uploaded file
to the string. Recorded here so it is not "fixed" later, and so that the day
someone makes labels user-editable, the reason this was safe is written down.

**`draw()` and `hitTest()` use different geometry APIs.** `draw()` uses
`clientWidth`/`clientHeight`, `hitTest()` uses `getBoundingClientRect()`. I
initially took this for a letterboxing bug — normalised box coordinates are
relative to the *video content*, and a `<video>` letterboxes its content inside a
box of the wrong ratio. The shipped CSS rules it out:

```css
#video   { width: 100%; height: auto; display: block; }
#overlay { position: absolute; inset: 0; width: 100%; height: 100%; }
```

With `height: auto` the element box *takes* the intrinsic ratio, so content fills
it exactly, there are no bars, and the two APIs agree. The claim was dropped
rather than reported. (What is left of the real problem — the pre-metadata case,
where there is no intrinsic ratio yet — is W-8, symptom 3.)

**The stale-`onended` identity check is hardening, not a fix.** `play()` now
verifies `this.source === src` before treating an `ended` event as end-of-clip,
guarding against a stopped source's event arriving after its replacement is
running. Stated honestly: this is **not currently reachable**, because
`stopSource()` already nulls the handler *before* calling `stop()`:

```js
try { this.source.onended = null; this.source.stop(); } catch (_) {}
```

It is kept because the guarantee currently rests on one line in a different
method, and the symptom if that line ever moves — playback ending a beat after
every backward seek — is obscure enough to cost an afternoon.

---

## Deliberately out of scope

**No `max-height` on `#video`.** A tall portrait clip overflows the viewport, and
the obvious fix is `max-height: 70vh`. That is declined on purpose: it makes the
element box stop matching the intrinsic ratio, which reintroduces exactly the
letterboxing that the "Not a defect" note above establishes does not currently
exist — and `draw()`/`hitTest()` do not model bars. Doing it properly means
computing the content rect from `videoWidth`/`videoHeight` in both functions. It
is a real improvement, it is not a bug fix, and it should not be smuggled in as
one.

---

## Verification

No Node.js on the target machine, so the structural check is stdlib Python. It
strips comments and string/template bodies (keeping `${…}` placeholders, which
hold real code), then checks bracket balance and — the thing that actually
matters after sixteen hand edits to a file made of three large object literals —
that each literal still closes before the next top-level declaration, and that
every method landed inside the right one:

```
$ python scripts/check_js_syntax.py app/static/app.js
structural check -- app\static\app.js  (32418 chars, 773 lines)
  PASS  every (), [] and {} balances
  PASS  `engine` literal spans lines 63-269, self-contained
  PASS  `transport` literal spans lines 275-401, self-contained
  PASS  `ui` literal spans lines 419-568, self-contained
  PASS  `engine.init` / `reset` / `position` / `play` / `pause` / `stopSource`
        / `select` / `hardSet` / `levels` defined inside `engine`
  PASS  `transport.toggle` / `onEnded` / `seek` / `loop` / `correct` inside `transport`
  PASS  `ui.setTracks` / `markActive` / `draw` / `repaint` / `hitTest`
        / `updateTime` / `meters` inside `ui`
  PASS  DPR backing-store vars are not shadowed by the box destructure

structure OK
```

This is a structural check, not a parser: it would not catch a misspelled
identifier. It catches the failure mode this review could actually introduce.
`transport.quantBias` is on its method list deliberately — `check_transport.py`
mirrors it in Python, and a rename here would leave that mirror testing a
function that no longer exists, silently.

W-14 and W-15 are arithmetic, so they are checked as arithmetic rather than
asserted in prose. `scripts/check_transport.py` reproduces both defects from the
same constants the browser uses, then pins the fixes — 40 assertions, no browser
and no Node:

```
$ python scripts/check_transport.py
clip fps 25, frame period 40 ms, deadband 20 ms, quantBias 20 ms

  W-14 old fixed 20 ms band, no debias:  warn  50.0%   rate!=1  50.0%
  PASS  old behaviour reproduces the report (yellow most of the time)
  PASS  the tempting 'widen the band' alternative goes blind to +35 ms real error
  W-14 fix   debias + 20 ms band:            warn   0.0%   rate!=1   0.0%
  PASS  a real ±50 ms desync is flagged  fps=25  -- flagged on 100% of ticks
  PASS  detection floor stays under the ~100 ms visibility threshold  fps=60  -- 28.3 ms
  PASS  correction is continuous at the band edge (no step)

  W-15 onended guard, evaluated once per delivery
  PASS  legacy guard happens to work at low output latency 5 ms
  PASS  legacy guard SILENTLY FAILS above the slack, latency 200 ms
  PASS  new guard fires at the natural end  start=9.0s latency=200ms
  PASS  new guard tolerates 2000 ms of delivery jitter
  PASS  new guard rejects a stale callback delivered at 6s

all checks passed
```

Both checks assert the **old** behaviour too, not just the new. A check that only
confirms the fix cannot tell you whether it was ever needed, and for W-15 the old
guard passes below ~50 ms of output latency — so a test written on a low-latency
machine would have agreed with the broken code.

The backend suite is unchanged — this review touched frontend JS only, so
anything but a clean run would be a regression:

```
$ PYTHONPATH=. python scripts/check_fusion.py
...
all checks passed
```

The equal-power curve endpoints were re-derived rather than trusted, because the
existing comment cites a specific constant:

```
DOWN[-1] in double:       6.123233995736766e-17
DOWN[-1] as Float32Array: 6.123234262925839e-17   -> -324 dBFS
power sum up^2+down^2:    min 0.999999943  max 1.000000059
```

Confirmed: the curve is equal-power to within 6e-8, and its tail is −324 dB, not
zero. That is inaudible but it is **not silence**, and this project's claim is
silence — so the `setValueAtTime(0, t + XFADE + 0.001)` that pins it to a
bit-exact zero is load-bearing and must not be removed as redundant.

---

## Files touched

| file | change |
|:-----|:-------|
| `app/static/app.js` | W-1..W-13; new `engine.reset` / `engine.hardSet` / `ui.repaint` / `toUpload`; `SCHED_LEAD`, `MAX_DPR` |
| `app/static/app.js` | W-14: `DEADBAND` re-documented, new `transport.quantBias`, `correct()` debiases then steers on the excess. W-15: `play()` records `endsAt`; `onended` reads the rendering clock |
| `scripts/check_js_syntax.py` | **new** — Node-free structural check for the frontend; `transport.quantBias` pinned for W-14 |
| `scripts/check_transport.py` | **new** — reproduces and pins W-14 and W-15 (40 assertions) |
| `docs/REVIEW_WEBAUDIO.md` | **new** — this document |

No backend file was modified by this review.
