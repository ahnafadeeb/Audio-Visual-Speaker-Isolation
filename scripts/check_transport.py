"""Simulate the A/V transport fixes that the user's Stage-1 report exposed.

Two defects, both reproduced from first principles with no browser involved:

W-14 / drift readout  "not always green in sync. it becomes yellow most of the
    times".  A video frame is on screen at a frame boundary or not at all, so
    rVFC's mediaTime is the true position FLOORED to 1/fps.  With the clocks in
    perfect agreement the raw mediaTime - audioPos sweeps the whole frame
    period, and the old fixed 20 ms deadband (exactly half a frame at 25 fps)
    tripped on 50% of it.  The shipped fix debiases by half a frame period
    first (the expected value of that sweep), then judges the residual against
    the same 20 ms band -- which stays tight enough to flag real desync in
    BOTH directions, unlike the tempting "widen the band" alternative that
    spends its whole allowance on the one-sided quantisation.

W-15 / end-of-clip button  "after playing the video the pause button keeps stuck
    at pause even when the video has ended".  The onended guard tested
    position() >= duration - 0.05, but position() reads
    getOutputTimestamp().contextTime -- the frame leaving the SPEAKERS, which
    lags ctx.currentTime (the rendering clock src.start() is scheduled
    against) by baseLatency + outputLatency, routinely 100-200 ms under
    shared-mode WASAPI on Windows.

    onended is a ONE-SHOT callback, not a polled condition, and that is the
    whole defect.  It is delivered once, at the instant the buffer drains, and
    the guard body runs at that single instant -- when position() is still
    out_latency short of duration.  The guard was false, onEnded() never ran,
    and nothing ever re-tested it.  position() does go on to reach duration (it
    clamps there), but by then the only callback that would have read it is
    long gone.  The fix compares ctx.currentTime against the context time at
    which the buffer runs out on its own: exact regardless of output buffer
    depth.  Checked below as a one-shot predicate per delivery, not a sweep.

Run::

    .venv/Scripts/python.exe scripts/check_transport.py
"""

from __future__ import annotations

DEADBAND = 0.020
MAX_RATE = 0.02
KP = 0.5

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def quant_bias(fps: float) -> float:
    """Mirror of transport.quantBias() in app/static/app.js."""
    return 0.5 / (fps if fps > 0 else 25.0)


def drift_sweep(fps: float, *, band: float = DEADBAND, debias: bool = True,
                true_drift: float = 0.0, n: int = 20001):
    """Fraction of ticks reading 'warn', and fraction writing playbackRate.

    ``true_drift`` is genuine A/V error in seconds ON TOP of quantisation;
    positive means the video really is ahead.  With ``true_drift=0`` the clip
    is perfectly synced and any warn is a false alarm.
    """
    frame = 1.0 / fps
    bias = quant_bias(fps) if debias else 0.0
    warn = rate_off = 0
    for i in range(n):
        audio = i * (10.0 / n)                              # 10 s of playback
        media = ((audio + true_drift) // frame) * frame     # floored to a boundary
        drift = (media - audio) + bias
        if abs(drift) > band:
            warn += 1
            excess = drift - (1 if drift > 0 else -1) * band
            if max(-MAX_RATE, min(MAX_RATE, -KP * excess)) != 0.0:
                rate_off += 1
    return warn / n, rate_off / n


def ended_guard(*, out_latency: float, start_at: float, duration: float,
                cushion: float = 0.08, guard: str, slack: float = 0.05,
                delivered_at: float | None = None) -> bool:
    """Evaluate ONE onended delivery.  True if transport.onEnded() would run.

    A one-shot predicate, not a sweep: the browser delivers onended once and
    the guard body runs at that single instant.  Modelling it as a polled
    condition is the mistake that hides the bug -- a polled legacy guard
    eventually passes, because position() climbs to duration a beat later.

    ``delivered_at`` is the rendering-clock time of the delivery: ``endsAt``
    (the default) for a buffer that drained on its own, later by the delivery
    jitter browsers add, earlier for a stale callback racing a seek or a
    channel switch -- which the guard must reject.

    The OUTPUT clock (getOutputTimestamp().contextTime) lags the rendering
    clock by ``out_latency``.  ``position()`` is built on it, clamps the
    elapsed term at 0 and the total at ``duration``, mirroring app.js.
    """
    t0 = 0.0
    ends_at = t0 + cushion + (duration - start_at)
    t = ends_at if delivered_at is None else delivered_at
    if guard == "new":
        return t >= ends_at - slack
    out_clock = t - out_latency                       # getOutputTimestamp()
    pos = min(duration, max(0.0, start_at + max(0.0, out_clock - (t0 + cushion))))
    return pos >= duration - slack


def main() -> None:
    print("A/V transport -- drift readout and end-of-clip button\n")

    fps = 25.0
    print(f"clip fps {fps:g}, frame period {1000 / fps:.0f} ms, "
          f"deadband {DEADBAND * 1000:.0f} ms, quantBias {quant_bias(fps) * 1000:.0f} ms")

    # --- W-14: drift -------------------------------------------------------- #
    w_old, r_old = drift_sweep(fps, debias=False)
    w_new, r_new = drift_sweep(fps)
    print(f"\n  W-14 old fixed 20 ms band, no debias:  warn {w_old:6.1%}   rate!=1 {r_old:6.1%}")
    check("old behaviour reproduces the report (yellow most of the time)",
          w_old > 0.4, f"warn on {w_old:.1%} of ticks with the clocks in perfect agreement")

    wide = DEADBAND + 0.5 / fps                      # the tempting wrong fix
    w_wide, _ = drift_sweep(fps, band=wide, debias=False)
    w_blind, _ = drift_sweep(fps, band=wide, debias=False, true_drift=0.035)
    check("the tempting 'widen the band' alternative goes blind to +35 ms real error",
          w_wide == 0.0 and w_blind < 1.0,
          f"real +35 ms desync flagged on only {w_blind:.0%} of ticks")

    print(f"  W-14 fix   debias + 20 ms band:            warn {w_new:6.1%}   rate!=1 {r_new:6.1%}")
    check("the fix reads green on a synced clip", w_new == 0.0, f"warn {w_new:.1%}")
    check("the fix stops writing playbackRate on a synced clip",
          r_new == 0.0, f"rate!=1 on {r_new:.1%} of ticks")
    for fps_i in (24.0, 25.0, 30.0, 60.0):
        detect = DEADBAND + quant_bias(fps_i)
        for sign in (+1, -1):
            err = sign * (detect + 0.010)
            w_bad, _ = drift_sweep(fps_i, true_drift=err)
            check(f"a real {err * 1000:+.0f} ms desync is flagged  fps={fps_i:g}",
                  w_bad == 1.0, f"flagged on {w_bad:.0%} of ticks")
        check(f"detection floor stays under the ~100 ms visibility threshold  fps={fps_i:g}",
              detect < 0.100, f"{detect * 1000:.1f} ms")

    # The P-controller steers on the EXCESS over the deadband, not the raw
    # drift.  Steering on the raw value commands KP*DEADBAND at the very
    # instant it stops caring -- a 1% rate step out of nowhere, which then
    # over-corrects back into the band and limit-cycles.
    step = KP * DEADBAND
    worst = 0.0
    for over in (1e-3, 1e-4, 1e-5, 1e-6):
        d = DEADBAND + over
        adj_excess = max(-MAX_RATE, min(MAX_RATE, -KP * (d - DEADBAND)))
        adj_raw = max(-MAX_RATE, min(MAX_RATE, -KP * d))
        worst = max(worst, abs(adj_excess))
        if not (abs(adj_excess) <= KP * over * 1.000001
                and abs(adj_raw) >= step * 0.99):
            worst = float("inf")
    check("correction is continuous at the band edge (no step)",
          worst <= KP * 1e-3 * 1.000001,
          f"excess-form -> {worst * 100:.5f}% as overshoot -> 0, "
          f"vs a fixed {step * 100:.2f}% step in the raw form")

    # --- W-15: end-of-clip button ------------------------------------------ #
    print("\n  W-15 onended guard, evaluated once per delivery")
    DUR, SLACK = 9.218, 0.05

    # The legacy guard is not uniformly broken -- it is broken above a
    # threshold, which is exactly why it survived development.  position()
    # lags the true end by the output latency, so at the single instant
    # onended is delivered it reads `duration - out_latency`; the guard's own
    # 50 ms slack absorbs that only while out_latency <= slack.  A dev machine
    # on WASAPI exclusive mode or a small buffer sits under it and the button
    # works; shared-mode Windows output is routinely 100-200 ms and it never
    # fires again.
    for lat in (0.005, 0.02, 0.04):
        ok = ended_guard(out_latency=lat, start_at=0.0, duration=DUR, guard="old")
        check(f"legacy guard happens to work at low output latency {lat * 1000:.0f} ms",
              ok, "fires -- this is why the defect looked machine-specific")
    for lat in (0.06, 0.10, 0.15, 0.20):
        ok = ended_guard(out_latency=lat, start_at=0.0, duration=DUR, guard="old")
        check(f"legacy guard SILENTLY FAILS above the slack, latency {lat * 1000:.0f} ms",
              not ok, f"position() reads {DUR - lat:.3f}s of {DUR}s at the one "
                      f"instant it is asked -- button stays stuck on 'Pause'")

    # The new guard reads the rendering clock, which src.start() was scheduled
    # against, so it is exact at any buffer depth and any seek offset.
    for start_at in (0.0, 3.5, 9.0):
        for lat in (0.005, 0.05, 0.20):
            ok = ended_guard(out_latency=lat, start_at=start_at, duration=DUR,
                             guard="new")
            check(f"new guard fires at the natural end  start={start_at}s "
                  f"latency={lat * 1000:.0f}ms", ok)

    # Browsers deliver onended on a task queue, so it can land late; it must
    # still fire.  A stale callback from a seek lands far early; the time guard
    # must reject it on its own, without leaning on the `source === src` check.
    ends_at = 0.08 + DUR
    for late in (0.0, 0.05, 0.5, 2.0):
        check(f"new guard tolerates {late * 1000:.0f} ms of delivery jitter",
              ended_guard(out_latency=0.20, start_at=0.0, duration=DUR,
                          guard="new", delivered_at=ends_at + late))
    for at in (0.5, 2.0, 6.0):
        check(f"new guard rejects a stale callback delivered at {at:g}s",
              not ended_guard(out_latency=0.20, start_at=0.0, duration=DUR,
                              guard="new", delivered_at=at))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
