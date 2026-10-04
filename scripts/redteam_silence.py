"""Adversarial attacks on the hard-mute claim.

`gating_knee.py` shows the gate works on signals built to resemble speech.  This
script builds signals designed to BREAK it.  Every attack targets a specific
assumption in `GateConfig`, and a failure here is a real defect, not a tuning
preference.

The claim under attack, stated precisely:

    The unselected speaker is digitally silent, and the selected speaker is
    not damaged.

Both halves matter.  A gate that achieves 100% silence by cutting the target's
consonants has not succeeded, it has moved the failure somewhere less
measurable.  So every attack reports BOTH numbers.

Run::

    PYTHONPATH=. python scripts/redteam_silence.py
"""

from __future__ import annotations

import numpy as np

from app import dsp
from app.config import CONFIG

SR = CONFIG.audio.sample_rate
G = CONFIG.gate
FPS = CONFIG.vision.target_fps
RNG = np.random.default_rng(7)

FINDINGS: list[tuple[str, str]] = []


def report(name: str, ok: bool, detail: str, note: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<38} {detail}")
    if note:
        print(f"        {note}")
    if not ok:
        FINDINGS.append((name, detail))


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #

def vowel(dur_s: float, f0: float, amp: float = 1.0) -> np.ndarray:
    """Voiced segment: harmonic stack, -6 dB/octave."""
    t = np.arange(int(dur_s * SR)) / SR
    sig = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + RNG.uniform(0, 6.28))
              for k in range(1, 14))
    return (amp * sig / 3.0).astype(np.float32)


def fricative(dur_s: float, amp: float, hp: float = 4000.0) -> np.ndarray:
    """Unvoiced consonant: high-passed noise.  These are the quiet, wideband
    sounds a level-threshold gate destroys first."""
    n = int(dur_s * SR)
    x = RNG.normal(size=n)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[freqs < hp] *= 0.05
    out = np.fft.irfft(spec, n=n)
    return (amp * out / (np.abs(out).max() + 1e-12)).astype(np.float32)


def lips_for(spans, total_s: float, active_jitter: float = 0.055) -> list[float]:
    """Lip signal that opens during `spans`.  Deliberately imperfect: resting
    aperture is nonzero and jitters, as a real tracker's does."""
    t = np.arange(int(total_s * FPS)) / FPS
    lip = 0.020 + 0.002 * RNG.normal(size=t.size)
    for s, e in spans:
        m = (t >= s) & (t < e)
        loc = t[m] - s
        lip[m] += active_jitter * (0.5 + 0.5 * np.sin(2 * np.pi * 4.6 * loc)) ** 2
    return lip.tolist()


def kept_pct(y: np.ndarray, spans) -> float:
    """Percent of samples surviving inside `spans`."""
    m = np.zeros(y.size, dtype=bool)
    for s, e in spans:
        m[int(s * SR):min(int(e * SR), y.size)] = True
    return 100.0 * np.count_nonzero(y[m] != 0.0) / max(m.sum(), 1)


def gate1(x: np.ndarray, lip=None, cfg=None) -> np.ndarray:
    """Gate a single stem, optionally with vision."""
    cfg = cfg if cfg is not None else G     # read G late -- --open/--close rebind it
    out = dsp.apply_gate(np.stack([x, np.zeros_like(x)]), sample_rate=SR, cfg=cfg,
                         lips=[lip, None] if lip is not None else None,
                         video_fps=FPS)
    return out[0]


# --------------------------------------------------------------------------- #
# Attack 1 -- sibilants vs a global reference
# --------------------------------------------------------------------------- #

def attack_sibilants() -> None:
    print("\n[1] quiet consonants vs open_db relative to a GLOBAL percentile")
    print("    /s/ and /f/ sit 20-30 dB below vowel peaks. open_db is -30 dB")
    print("    relative to the stem's 95th-percentile frame. Do they survive?")

    for drop_db in (-20, -25, -30, -35):
        amp = 10 ** (drop_db / 20.0)
        # "SASS": loud vowel, quiet fricative, loud vowel, quiet fricative.
        parts = [vowel(0.45, 120), fricative(0.28, amp),
                 vowel(0.45, 120), fricative(0.28, amp)]
        x = np.concatenate(parts)
        edges = np.cumsum([0] + [p.size for p in parts]) / SR
        fric_spans = [(edges[1], edges[2]), (edges[3], edges[4])]

        y = gate1(x)
        k = kept_pct(y, fric_spans)
        report(f"fricative at {drop_db} dB survives", k > 80.0, f"kept {k:5.1f}%")


# --------------------------------------------------------------------------- #
# Attack 2 -- the global reference itself
# --------------------------------------------------------------------------- #

def attack_dynamic_range() -> None:
    print("\n[2] ref_percentile=95 is GLOBAL over the clip")
    print("    One shout raises the reference for the whole clip, so quiet")
    print("    speech elsewhere is measured against the shout. Sweep the")
    print("    quiet level to find where that actually bites.")
    print("    (measured on the speech spans only -- an earlier version of this")
    print("     attack included a deliberate silent gap and blamed the gate)")

    gap = np.zeros(int(0.25 * SR), np.float32)
    pad = np.zeros(int(0.5 * SR), np.float32)
    shout = vowel(0.8, 115, 1.0)

    # Speech spans only -- the gap between them is meant to be silent.
    spans = [(0.0, 0.4), (0.65, 1.05)]
    total = (0.4 + 0.25 + 0.4 + 0.5 + 0.8 + 0.5)

    for quiet_db in (-18, -22, -26, -28, -30, -34, -40):
        amp = 10 ** (quiet_db / 20.0)
        quiet = np.concatenate([vowel(0.4, 115, amp), gap, vowel(0.4, 115, amp)])
        with_shout = np.concatenate([quiet, pad, shout, pad])

        k_alone = kept_pct(gate1(np.concatenate([quiet, pad])), spans)
        k_after = kept_pct(gate1(with_shout), spans)
        # Same clip, but vision agrees perfectly that the mouth is moving.
        k_seen = kept_pct(gate1(with_shout, lips_for(spans, total)), spans)

        headroom = quiet_db - G.open_db      # margin before the threshold bites
        report(f"quiet speech at {quiet_db} dB survives a shout",
               k_after > 90.0,
               f"alone {k_alone:5.1f}% -> shout {k_after:5.1f}% -> "
               f"+vision {k_seen:5.1f}%  (headroom {headroom:+.0f} dB)")

    print("\n    The shout sets ref, so a quiet passage's margin is exactly")
    print("    (quiet_db - open_db) -- the knee lands where the arithmetic says.")
    print("    Note the +vision column: it never rescues anything. The fusion is")
    print("    ONE-SIDED by construction -- visual_veto_db only shifts thresholds")
    print("    UP. Vision can add suppression; it can never restore sensitivity.")


# --------------------------------------------------------------------------- #
# Attack 2b -- is the obvious fix worse than the disease?
# --------------------------------------------------------------------------- #

def attack_local_reference() -> None:
    print("\n[2b] would a LOCAL (windowed) reference fix [2]?")
    print("    That is the textbook answer to a global percentile. Measure what")
    print("    it costs, on the one span the whole project exists to protect:")
    print("    a stretch containing nothing but the interferer's residual.")

    loud = vowel(1.5, 115, 1.0)
    for resid_db in (-35, -45):
        resid = vowel(1.2, 200, 10 ** (resid_db / 20.0))
        clip = np.concatenate([loud, np.zeros(int(0.3 * SR), np.float32), resid])
        r0 = (1.5 + 0.3)

        # Global reference: the loud passage sets ref for the whole clip.
        k_global = kept_pct(gate1(clip), [(r0, r0 + 1.2)])
        # A window containing only the residual IS a local reference -- the
        # residual becomes its own 95th percentile, so rel ~ 0 dB and the gate
        # cannot tell it from speech.
        k_local = kept_pct(gate1(resid), [(0.0, 1.2)])

        report(f"global ref beats local ref at {resid_db} dB",
               k_global < 1.0 < k_local,
               f"global {k_global:5.1f}% leaks / local {k_local:5.1f}% leaks")

    print("\n    So [2] is not a bug to fix, it is the cost of the design that")
    print("    makes Objective A possible at all. A windowed reference re-opens")
    print("    the gate on pure residual -- it trades a quiet-speech limit for")
    print("    the exact ghost whisper this project exists to remove.")


# --------------------------------------------------------------------------- #
# Attack 2c -- does the fusion buy headroom to lower open_db?
# --------------------------------------------------------------------------- #

def attack_threshold_headroom() -> None:
    print("\n[2c] the fusion should make a LOWER open_db affordable")
    print("    Pre-A4b, open_db was the only thing suppressing residual, so it")
    print("    could not go low. Now the visual veto supplies suppression where")
    print("    the mouth is still -- so open_db may be free to drop, buying back")
    print("    the dynamic range [2] costs. Four numbers must hold TOGETHER:")
    print("      quiet%  quiet speech (-34 dB) surviving a shout   (want ~100)")
    print("      kept%   A's own speech surviving                  (want ~100)")
    print("      AV0%    exact silence on a residual-only span, WITH vision")
    print("      Ac0%    the same span with NO vision -- the fallback path every")
    print("              unmatched face takes. This is the one that can regress.")

    import dataclasses

    # (a) the dynamic-range case from [2], at a level that currently fails
    amp = 10 ** (-34 / 20.0)
    gap = np.zeros(int(0.25 * SR), np.float32)
    pad = np.zeros(int(0.5 * SR), np.float32)
    quiet_clip = np.concatenate([vowel(0.4, 115, amp), gap, vowel(0.4, 115, amp),
                                 pad, vowel(0.8, 115, 1.0), pad])
    quiet_spans = [(0.0, 0.4), (0.65, 1.05)]
    quiet_lips = lips_for(quiet_spans, quiet_clip.size / SR)

    # (b) the Objective A case: A's channel during B's turn, swept over the
    # separator quality range gating_knee.py covers.
    total = 4.0
    a_spans, b_spans = [(0.0, 1.5), (2.5, 4.0)], [(1.7, 2.3)]
    A = np.zeros(int(total * SR), np.float32)
    B = np.zeros(int(total * SR), np.float32)
    for s, e in a_spans:
        seg = vowel(e - s, 115)
        A[int(s * SR):int(s * SR) + seg.size] = seg
    for s, e in b_spans:
        seg = vowel(e - s, 200)
        B[int(s * SR):int(s * SR) + seg.size] = seg
    a_lips, b_lips = lips_for(a_spans, total), lips_for(b_spans, total)
    qs, qe = int(1.8 * SR), int(2.2 * SR)

    raws = {}
    for leak_db in (-12, -9, -6, -3):
        leak = 10 ** (leak_db / 20.0)
        raws[leak_db] = dsp.wiener_separate(
            np.stack([A + leak * B, B + leak * A]).astype(np.float32),
            sample_rate=SR, nfft=CONFIG.audio.nfft, hop=CONFIG.audio.hop,
            exponent=G.mask_exponent, floor=G.mask_floor,
            cepstral_order=G.cepstral_smooth_order)

    def zeros_pct(out) -> float:
        seg = out[0][qs:qe]
        return 100.0 * np.count_nonzero(seg == 0.0) / seg.size

    # The column that kills the apparent free lunch.  Every AV0/Ac0 above is
    # measured on a residual that happens to sit far below open_db, so lowering
    # open_db looks costless.  This one puts a residual RIGHT at the threshold's
    # new position -- which is where the trade is actually paid.
    loud = vowel(1.5, 115, 1.0)
    probe = np.concatenate([loud, np.zeros(int(0.3 * SR), np.float32),
                            vowel(1.2, 200, 10 ** (-35 / 20.0))])
    probe_span = [(1.8, 3.0)]

    hdr = "".join(f"{f'AV0@{d}':>9}{f'Ac0@{d}':>9}" for d in raws)
    print(f"\n{'open/close':>13}{'quiet%':>8}{'kept%':>7}{'r-35%':>7}{hdr}")
    print("    " + "-" * (35 + 18 * len(raws)))

    for open_db, close_db in ((-30, -40), (-34, -44), (-36, -46), (-38, -48),
                              (-40, -50), (-45, -55)):
        cfg = dataclasses.replace(G, open_db=float(open_db), close_db=float(close_db))
        acou = dataclasses.replace(cfg, visual_fusion=False)

        q = kept_pct(gate1(quiet_clip, quiet_lips, cfg), quiet_spans)
        r = kept_pct(gate1(probe, None, cfg), probe_span)
        row, kept = "", 0.0
        for leak_db, raw in raws.items():
            av = dsp.apply_gate(raw, sample_rate=SR, cfg=cfg,
                                lips=[a_lips, b_lips], video_fps=FPS)
            ac = dsp.apply_gate(raw, sample_rate=SR, cfg=acou)
            row += f"{zeros_pct(av):>9.1f}{zeros_pct(ac):>9.1f}"
            kept = kept_pct(av[0], a_spans)      # worst (highest) leak wins
        print(f"{f'{open_db}/{close_db}':>13}{q:>8.1f}{kept:>7.1f}{r:>7.1f}{row}")

    print("\n    quiet% and r-35% are the SAME AXIS read from opposite ends, and")
    print("    they flip within one row of each other: -36 buys quiet speech at")
    print("    -34 dB and pays for it by passing a residual at -35 dB. The trade")
    print("    is conserved to within the 2 dB step of this sweep. That is the")
    print("    doc's 'no operating point where both hold', measured rather than")
    print("    argued -- and the reason open_db stays at -30 with margin.")
    print("    The AV0 columns alone would have said -36 was free; they are")
    print("    measured on residuals far below the threshold. Do not read them")
    print("    without r-35%.")


# --------------------------------------------------------------------------- #
# Attack 3 -- onsets and the lookahead
# --------------------------------------------------------------------------- #

def attack_onsets() -> None:
    print("\n[3] min_on_ms=60 + lookahead_ms=20 vs plosive onsets")
    print("    A word starting with /p/ or /t/ has a sharp attack. Does the")
    print("    gate's dwell requirement eat the first consonant?")

    for burst_ms in (10, 20, 40, 80):
        n = int(burst_ms / 1000 * SR)
        burst = fricative(burst_ms / 1000, 0.6, hp=1500)
        x = np.concatenate([np.zeros(int(0.4 * SR), np.float32), burst,
                            vowel(0.5, 120), np.zeros(int(0.3 * SR), np.float32)])
        start = 0.4
        y = gate1(x)
        k = kept_pct(y, [(start, start + burst_ms / 1000)])
        report(f"{burst_ms:>3} ms plosive burst survives", k > 50.0, f"kept {k:5.1f}%")


# --------------------------------------------------------------------------- #
# Attack 4 -- backchannels
# --------------------------------------------------------------------------- #

def attack_backchannel() -> None:
    print("\n[4] conversational backchannels ('mhm', 'yeah') vs min_on_ms")
    print("    Short, quiet, and the listener's mouth barely moves.")

    for dur_ms, amp_db in ((150, -12), (250, -12), (150, -20), (400, -18)):
        amp = 10 ** (amp_db / 20.0)
        x = np.concatenate([vowel(1.2, 118), np.zeros(int(0.6 * SR), np.float32),
                            vowel(dur_ms / 1000, 118, amp),
                            np.zeros(int(0.6 * SR), np.float32)])
        t0 = 1.2 + 0.6
        k = kept_pct(gate1(x), [(t0, t0 + dur_ms / 1000)])
        report(f"{dur_ms:>3} ms backchannel at {amp_db} dB", k > 50.0,
               f"kept {k:5.1f}%")


# --------------------------------------------------------------------------- #
# Attack 5 -- vision disagrees with real speech
# --------------------------------------------------------------------------- #

def attack_vision_disagrees() -> None:
    print("\n[5] the veto vs speech the mouth does not advertise")
    print("    Mumbling, a hand over the mouth, extreme head pose, a beard.")
    print("    Vision says still; audio says speech. The veto must not win.")

    total = 3.0
    speech_spans = [(0.3, 1.2), (1.8, 2.7)]
    x = np.zeros(int(total * SR), np.float32)
    for s, e in speech_spans:
        seg = vowel(e - s, 120)
        x[int(s * SR):int(s * SR) + seg.size] = seg

    cases = {
        "vision agrees":        lips_for(speech_spans, total),
        "half-amplitude mouth": lips_for(speech_spans, total, active_jitter=0.008),
        "mouth barely moves":   lips_for(speech_spans, total, active_jitter=0.002),
        "vision lags 200 ms":   lips_for([(s + 0.2, e + 0.2) for s, e in speech_spans], total),
        "vision leads 200 ms":  lips_for([(max(0, s - 0.2), e - 0.2) for s, e in speech_spans], total),
    }
    for name, lip in cases.items():
        k = kept_pct(gate1(x, lip), speech_spans)
        report(name, k > 80.0, f"kept {k:5.1f}%")


# --------------------------------------------------------------------------- #
# Attack 6 -- can the veto be starved into muting everything?
# --------------------------------------------------------------------------- #

def attack_total_mute() -> None:
    print("\n[6] can any vision input mute a genuinely speaking stem?")
    print("    The worst outcome: a speaker selected and silent.")

    total = 3.0
    spans = [(0.2, 2.8)]
    x = np.zeros(int(total * SR), np.float32)
    seg = vowel(2.6, 120)
    x[int(0.2 * SR):int(0.2 * SR) + seg.size] = seg

    n_lip = int(total * FPS)
    hostile = {
        "constant zero lip":   [0.0] * n_lip,
        "constant nonzero":    [0.03] * n_lip,
        "all NaN":             [float("nan")] * n_lip,
        "single spike":        [0.02] * (n_lip - 1) + [0.9],
        "negative values":     [-0.05] * n_lip,
        "huge values":         [1e6] * n_lip,
        "one frame only":      [0.02],
        "empty":               [],
    }
    for name, lip in hostile.items():
        try:
            k = kept_pct(gate1(x, lip), spans)
        except Exception as exc:
            report(name, False, f"RAISED {type(exc).__name__}: {exc}")
            continue
        report(name, k > 80.0, f"kept {k:5.1f}%",
               "" if k > 80.0 else "hostile vision muted a speaking stem")


# --------------------------------------------------------------------------- #
# Attack 7 -- does the silence guarantee actually hold?
# --------------------------------------------------------------------------- #

def attack_silence_is_exact() -> None:
    print("\n[7] is the silence EXACT, or merely quiet?")
    print("    A -60 dB floor is still an audible ghost on headphones.")

    total = 4.0
    a_spans, b_spans = [(0.0, 1.5), (2.5, 4.0)], [(1.7, 2.3)]
    A = np.zeros(int(total * SR), np.float32)
    B = np.zeros(int(total * SR), np.float32)
    for s, e in a_spans:
        seg = vowel(e - s, 115)
        A[int(s * SR):int(s * SR) + seg.size] = seg
    for s, e in b_spans:
        seg = vowel(e - s, 200)
        B[int(s * SR):int(s * SR) + seg.size] = seg

    leak = 10 ** (-6 / 20.0)
    raw = dsp.wiener_separate(
        np.stack([A + leak * B, B + leak * A]).astype(np.float32), sample_rate=SR,
        nfft=CONFIG.audio.nfft, hop=CONFIG.audio.hop,
        exponent=G.mask_exponent, floor=G.mask_floor,
        cepstral_order=G.cepstral_smooth_order)

    out = dsp.apply_gate(raw, sample_rate=SR, cfg=G,
                         lips=[lips_for(a_spans, total), lips_for(b_spans, total)],
                         video_fps=FPS)

    s, e = int(1.8 * SR), int(2.2 * SR)
    seg = out[0][s:e]
    zeros = 100.0 * np.count_nonzero(seg == 0.0) / seg.size
    report("ch0 is exactly zero during B's turn", zeros > 99.0, f"{zeros:5.1f}% exact")

    st = dsp.silence_stats(seg)
    report("no residual floor at all", st["peak"] == 0.0,
           f"peak {st['peak']:.3e}",
           "" if st["peak"] == 0.0 else "nonzero peak = audible ghost whisper")

    # The ramp must reach exact zero, not asymptote to it.
    env = dsp.gate_mask(A, sample_rate=SR,
                        open_db=G.open_db, close_db=G.close_db)
    report("gate envelope reaches exact 0.0", bool((env == 0.0).any()),
           f"min {env.min():.3e}")
    report("gate envelope reaches exact 1.0", bool((env == 1.0).any()),
           f"max {env.max():.6f}")


# --------------------------------------------------------------------------- #
# Attack 8 -- reverb tails
# --------------------------------------------------------------------------- #

def attack_reverb() -> None:
    print("\n[8] reverb tails -- whamr16k input is reverberant")
    print("    A decaying tail crosses close_db slowly. Does the gate chatter?")

    dry = np.concatenate([vowel(0.5, 120), np.zeros(int(1.5 * SR), np.float32),
                          vowel(0.5, 120), np.zeros(int(1.5 * SR), np.float32)])
    for rt60 in (0.3, 0.6, 1.0):
        n = int(rt60 * SR)
        ir = (RNG.normal(size=n) * np.exp(-6.9 * np.arange(n) / n)).astype(np.float32)
        ir[0] = 1.0
        wet = np.convolve(dry, ir, mode="full")[:dry.size].astype(np.float32)
        wet /= np.abs(wet).max() + 1e-12

        env = dsp.gate_mask(wet, sample_rate=SR,
                            open_db=G.open_db, close_db=G.close_db)
        # Count open->close transitions; a clean gate has one per utterance.
        binary = (env > 0.5).astype(np.int8)
        transitions = int(np.count_nonzero(np.diff(binary) < 0))
        report(f"RT60 {rt60}s: no gate chatter", transitions <= 3,
               f"{transitions} closings for 2 utterances")


def main() -> None:
    import argparse
    import dataclasses

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", dest="open_db", type=float, default=None,
                    help="override gate.open_db (to evaluate a candidate)")
    ap.add_argument("--close", dest="close_db", type=float, default=None,
                    help="override gate.close_db")
    args = ap.parse_args()

    global G
    if args.open_db is not None or args.close_db is not None:
        G = dataclasses.replace(
            G,
            open_db=args.open_db if args.open_db is not None else G.open_db,
            close_db=args.close_db if args.close_db is not None else G.close_db)

    print("=" * 72)
    print("RED TEAM: attacking the hard-mute claim")
    print("=" * 72)
    print(f"gate: open={G.open_db} close={G.close_db} ref_pct={G.ref_percentile} "
          f"min_on={G.min_on_ms} veto={G.visual_veto_db} hold={G.visual_hold_ms}")

    attack_sibilants()
    attack_dynamic_range()
    attack_local_reference()
    attack_threshold_headroom()
    attack_onsets()
    attack_backchannel()
    attack_vision_disagrees()
    attack_total_mute()
    attack_silence_is_exact()
    attack_reverb()

    print("\n" + "=" * 72)
    if FINDINGS:
        print(f"{len(FINDINGS)} FINDING(S):")
        for name, detail in FINDINGS:
            print(f"  - {name}: {detail}")
    else:
        print("no findings -- every attack was survived")
    print("=" * 72)


if __name__ == "__main__":
    main()
