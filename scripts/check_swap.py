"""Simulate the user-overridable stem-to-face binding, with no browser involved.

Why this exists.  `docs/DIAG_MATCHER.md` closes with a measurement, not an
opinion: on the test clip the audio-visual matcher is **at chance** (p = 0.713,
and the two decodes of the same file return opposite pairings).  Nothing in the
audio or the visual feature repairs that -- fifteen features all invert, and the
one time scale that gets it right fails a circular-shift null.  So the pipeline
cannot be made to know which voice belongs to which face on clips like this one,
and the only honest fix is to let the person who can hear it say so.

That makes ``ui.nextPairing()`` load-bearing for the exact symptom the field
report opened with -- *"selecting Male played Female audio"* -- and it is worth
more than a glance, because it lives at the junction of the three index spaces
``app/channels.py`` exists to keep apart:

    track    a detected face, 0..n_tracks-1, the order the buttons are drawn in
    stem     a separated source, 0..n_sources-1
    channel  a column of stems_demo.wav, which is a stem index by construction

``t.channel`` is a *stem value stored against a track*.  Permuting it is
therefore a relabelling in track space, and every path that reads it -- the
buttons, the canvas boxes, the hit test, the meters, the digit shortcuts -- has
to agree about which space it is in.  Each check below fails loudly if one of
them slips back into channel space.

One warning worth carrying, because it has produced a wrong conclusion twice:
``meta.assignment`` is track -> **stem** and ``tracks[].channel`` is track ->
**channel**, and ``plan_channels`` renumbers between them.  On this clip they
read ``[1, 0]`` and ``[0, 1]`` -- the same decision, printed as opposite
permutations.  The numbers in the fixtures below are all in CHANNEL space, which
is the only space this control touches, and their provenance is
``scripts/diag_chain.py``, which measures every hop instead of asserting one.

Mirrors ``app/static/app.js``: ``nextPermutation()``, ``ui.nextPairing()``,
``ui.selectTrack()``, ``ui.overridden()`` and the gain invariant that
``engine.select()`` maintains.  ``scripts/check_js_syntax.py`` asserts the JS
still has each of those, so this mirror cannot quietly come loose from the code
it claims to test.

Run::

    .venv/Scripts/python.exe scripts/check_swap.py
"""

from __future__ import annotations

import itertools
import random

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Mirrors of app/static/app.js
# ---------------------------------------------------------------------------

def next_permutation(a: list[int]) -> bool:
    """Port of nextPermutation() -- in place, returns False on the wrap."""
    i = len(a) - 2
    while i >= 0 and a[i] >= a[i + 1]:
        i -= 1
    if i < 0:
        a.reverse()
        return False
    j = len(a) - 1
    while a[j] <= a[i]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1:] = reversed(a[i + 1:])
    return True


def cyclic_rotate(a: list[int]) -> None:
    """The design this replaced, kept only to demonstrate why it was wrong."""
    a[:] = a[1:] + a[:1]


class Ui:
    """The parts of `ui` and `engine` that the pairing control touches."""

    def __init__(self, channels: list[int | None], n_channels: int | None = None):
        # `channels[i]` is the stem bound to track i; None means "face detected,
        # no stem assigned" -- a real state the matcher produces, and one that
        # must survive every permutation untouched.
        self.tracks = [{"label": f"Speaker {chr(65 + i)}", "channel": c}
                       for i, c in enumerate(channels)]
        self.base = list(channels)
        self.steps = 0
        n = n_channels if n_channels is not None else sum(
            1 for c in channels if c is not None)
        # engine.gains: exactly one at 1.0, every other at bit-exact 0.0.
        self.gains = [0.0] * n
        self.active: int | None = None
        first = next((c for c in channels if c is not None), None)
        if first is not None:
            self.select_track(next(i for i, c in enumerate(channels)
                                   if c is not None))

    # -- engine.select ------------------------------------------------------
    def select(self, k: int | None) -> None:
        if k is None or not (0 <= k < len(self.gains)):
            return                      # refuse; leave the current speaker up
        self.active = k
        for i in range(len(self.gains)):
            self.gains[i] = 1.0 if i == k else 0.0

    def select_track(self, i: int) -> None:
        if not (0 <= i < len(self.tracks)):
            return
        self.select(self.tracks[i]["channel"])

    # -- ui.nextPairing -----------------------------------------------------
    def next_pairing(self) -> bool:
        idx = [i for i, t in enumerate(self.tracks) if t["channel"] is not None]
        if len(idx) < 2:
            return False
        sel = next((i for i in idx if self.tracks[i]["channel"] == self.active),
                   None)
        chans = [self.tracks[i]["channel"] for i in idx]
        next_permutation(chans)
        for k, i in enumerate(idx):
            self.tracks[i]["channel"] = chans[k]
        self.steps += 1
        if sel is not None:
            self.select_track(sel)
        return True

    def overridden(self) -> bool:
        return any(t["channel"] != self.base[i]
                   for i, t in enumerate(self.tracks))

    # -- what the rest of the UI reads -------------------------------------
    def binding(self) -> list[int | None]:
        return [t["channel"] for t in self.tracks]

    def meters(self, levels: list[float]) -> list[float]:
        """ui.meters(): one bar per TRACK, indexed by dataset.channel."""
        return [0.0 if t["channel"] is None else levels[t["channel"]]
                for t in self.tracks]

    def audible_track(self) -> int | None:
        """Which FACE the user is hearing, read back through the binding."""
        return next((i for i, t in enumerate(self.tracks)
                     if t["channel"] == self.active), None)


# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(20260811)

    print("=" * 78)
    print("stem->face pairing override -- simulation of ui.nextPairing()")
    print("=" * 78)

    # -- 1. the permutation walk itself ------------------------------------
    print("\n1. nextPermutation() is a single cycle over all n! orders")
    ok_lex = True
    for n in range(2, 7):
        want = [list(p) for p in itertools.permutations(range(n))]
        a, got = list(range(n)), [list(range(n))]
        for _ in range(len(want) - 1):
            next_permutation(a)
            got.append(list(a))
        ok_lex &= got == want
    check("matches itertools lexicographic order for n = 2..6", ok_lex)

    # Started anywhere, not just at the identity: the pipeline's own binding is
    # the starting point, and it need not be sorted -- an unmatched face or a
    # partial claim leaves holes, and one press from this clip's shipped [0, 1]
    # is what the user has to do.  If the walk were not a single cycle, some
    # pairings would be unreachable from where the user actually starts.
    ok_cycle, ok_cover = True, True
    for n in range(2, 6):
        fact = 1
        for k in range(2, n + 1):
            fact *= k
        for start in itertools.permutations(range(n)):
            a, seen = list(start), {tuple(start)}
            for _ in range(fact - 1):
                next_permutation(a)
                seen.add(tuple(a))
            ok_cover &= len(seen) == fact
            next_permutation(a)
            ok_cycle &= tuple(a) == tuple(start)
    check("from ANY starting pairing, n! steps return to it", ok_cycle)
    check("and the walk visits every one of the n! pairings", ok_cover)

    # The reason the design changed mid-implementation.  A cyclic rotation is
    # shorter and identical for two speakers, so the gap is invisible on the
    # clip in hand and shows up only on a three-speaker one.
    reach_cyc, reach_lex = set(), set()
    a = [0, 1, 2]
    for _ in range(6):
        cyclic_rotate(a)
        reach_cyc.add(tuple(a))
    a = [0, 1, 2]
    for _ in range(6):
        next_permutation(a)
        reach_lex.add(tuple(a))
    transpositions = {(1, 0, 2), (0, 2, 1), (2, 1, 0)}
    check("cyclic rotation reaches only 3 of the 6 three-face pairings",
          len(reach_cyc) == 3, f"{sorted(reach_cyc)}")
    check("...and NONE of the 3 transpositions, so swapping two of three faces "
          "would be unfixable", not (reach_cyc & transpositions))
    check("permutation stepping reaches all 6, transpositions included",
          len(reach_lex) == 6 and transpositions <= reach_lex)

    # -- 2. the reported symptom -------------------------------------------
    print("\n2. the field report: one press fixes an inverted 2-speaker pairing")
    # Ground truth, measured by scripts/diag_chain.py on runs/pair_v2 with no
    # hardcoded constant, and stated carefully because getting the index space
    # wrong here is exactly what produced two wrong conclusions before it:
    #
    #   separator      stem 0 = the man (f0 149.3 Hz), stem 1 = the woman (225.5)
    #   matcher        assignment [1, 0], i.e. track 0 <- stem 1  -- INVERTED
    #   plan_channels  renumbers, so the export ships channel_of [0, 1]
    #   export         channel 0 = the woman (213.9 Hz), channel 1 = the man (147.4)
    #   faces          track 0 is the man (centroid x 0.297), track 1 the woman
    #
    # So the BROWSER receives [0, 1] -- not [1, 0].  `meta.assignment` is [1, 0]
    # in *stem* space and looks like the same permutation, which is the trap.
    # For the man (track 0) to hear the man he must own channel 1, so the correct
    # binding in CHANNEL space -- the only space this control operates in -- is
    # [1, 0].
    shipped, truth = [0, 1], [1, 0]
    ui = Ui(list(shipped))
    check("as shipped, the man's button is bound to the woman's channel",
          ui.binding() == shipped and ui.binding() != truth)
    ui.next_pairing()
    check("one press lands on the correct pairing", ui.binding() == truth,
          f"{ui.binding()}")
    check("and reports itself as overridden", ui.overridden())
    ui.next_pairing()
    check("a second press returns to the pipeline's pairing",
          ui.binding() == shipped and not ui.overridden())

    # -- 3. the selection follows the FACE ---------------------------------
    print("\n3. the same face stays selected; the audio behind it changes")
    ui = Ui([1, 0])
    ui.select_track(0)                      # click the man
    before_face, before_ch = ui.audible_track(), ui.active
    ui.next_pairing()
    check("still hearing the same face", ui.audible_track() == before_face == 0)
    check("but a different stem", ui.active != before_ch,
          f"{before_ch} -> {ui.active}")
    # Carrying the selection as a channel instead would keep the same audio and
    # move the highlight to the other face -- exactly backwards.
    check("the highlight did not jump to the other face",
          ui.audible_track() == 0)

    ok_face, ok_change = True, True
    for n in (2, 3, 4):
        for start in itertools.permutations(range(n)):
            for face in range(n):
                u = Ui(list(start))
                u.select_track(face)
                u.next_pairing()
                ok_face &= u.audible_track() == face
                ok_change &= u.active is not None
    check("holds for every face of every 2/3/4-speaker pairing", ok_face)
    check("and something is always audible afterwards", ok_change)

    # -- 4. invariants that would break the audio --------------------------
    print("\n4. invariants: a permutation, and exactly one channel audible")
    ok_perm, ok_gain, ok_none = True, True, True
    for n in (2, 3, 4):
        for start in itertools.permutations(range(n)):
            u = Ui(list(start))
            for _ in range(3 * n + 2):
                u.next_pairing()
                b = [c for c in u.binding() if c is not None]
                ok_perm &= sorted(b) == list(range(n))          # no dup, no loss
                ok_gain &= (sum(1 for g in u.gains if g == 1.0) == 1
                            and all(g == 0.0 for g in u.gains if g != 1.0))
                ok_none &= all(c is not None for c in u.binding())
    check("the binding stays a permutation -- no stem duplicated or lost",
          ok_perm)
    check("exactly one gain at 1.0, every other at bit-exact 0.0", ok_gain)
    check("no track ever loses its stem", ok_none)

    # A duplicated stem is the specific failure worth naming: two faces bound to
    # one channel leaves the other channel unreachable, so one speaker can never
    # be heard again -- and the meters would show two identical bars, which is
    # what makes it recognisable in the demo.
    ui = Ui([0, 1])
    ui.next_pairing()
    m = ui.meters([0.4, 0.0])               # channel 0 loud, channel 1 silent
    check("meters follow the binding, not the button position", m == [0.0, 0.4],
          f"{m}")

    # -- 5. unmatched faces ------------------------------------------------
    print("\n5. faces with no stem are left alone")
    ui = Ui([0, None, 1], n_channels=2)
    seq = []
    for _ in range(4):
        ui.next_pairing()
        seq.append(ui.binding())
    check("the unmatched track never receives a stem",
          all(b[1] is None for b in seq), f"{seq}")
    check("the two matched tracks still permute between themselves",
          [0, None, 1] in seq and [1, None, 0] in seq, f"{seq}")

    print("\n6. the control refuses when there is nothing to permute")
    for label, chans in (("one face, one stem", [0]),
                         ("one matched + one unmatched", [0, None]),
                         ("no stems at all", [None, None])):
        u = Ui(chans, n_channels=1)
        before = u.binding()
        moved = u.next_pairing()
        check(f"{label}: no-op", moved is False and u.binding() == before
              and u.steps == 0)

    # -- 7. the digit shortcuts --------------------------------------------
    print("\n7. digit shortcuts are stable across an override")
    # Digit d selects the d-th BUTTON.  In channel space it would select the
    # d-th column of the wav, which is a different face before and after a press
    # -- i.e. the shortcut would change meaning under the user's hands.
    ui = Ui([0, 1])
    ui.select_track(0)                       # what pressing "1" does
    face_before = ui.audible_track()
    ui.next_pairing()
    ui.select_track(0)                       # pressing "1" again
    check("\"1\" selects the same face before and after a press",
          ui.audible_track() == face_before == 0)

    # The old channel-indexed behaviour, stated precisely.  It is correct exactly
    # while channel_of is the identity -- which it IS as this clip ships, so the
    # earlier claim that it "picked the wrong face on this clip" was wrong and is
    # corrected here.  The bug is not hypothetical though: on this clip the
    # pairing is inverted, so the user MUST press swap, and the moment they do,
    # channel_of stops being the identity and the shortcut starts picking the
    # wrong face.  It is a latent bug that this clip's own fix activates.
    ui2 = Ui([0, 1])
    ui2.select(0)                            # engine.select(d - 1), the old code
    check("channel-indexed shortcut is correct while the binding is the identity",
          ui2.audible_track() == 0)
    ui2.next_pairing()                       # the press this clip requires
    ui2.select(0)                            # the old code again, same keypress
    check("...and picks the WRONG face as soon as the user swaps",
          ui2.audible_track() == 1, f"binding {ui2.binding()}, heard track "
                                    f"{ui2.audible_track()}")
    # An unmatched face reaches the same failure without any keypress at all:
    # channel_of is [0, None, 1], so "3" must select track 2 (channel 1), while
    # engine.select(2) addresses a column the wav does not have.
    ui3 = Ui([0, None, 1], n_channels=2)
    ui3.select_track(2)
    check("with an unmatched face, \"3\" still selects the third BUTTON",
          ui3.audible_track() == 2 and ui3.active == 1)
    before = ui3.active
    ui3.select(2)                            # engine.select(d - 1) for d = 3
    check("...where the channel-indexed shortcut addresses a nonexistent column",
          ui3.active == before, "refused, leaving the previous speaker up")

    # -- 8. a long random walk ---------------------------------------------
    print("\n8. 4000 random presses hold every invariant at once")
    ok = True
    for _ in range(400):
        n = rng.randint(2, 4)
        chans: list[int | None] = list(rng.sample(range(n), n))
        holes = rng.randint(0, 1)
        for _ in range(holes):
            chans.insert(rng.randrange(len(chans) + 1), None)
        u = Ui(chans, n_channels=n)
        for _ in range(10):
            if rng.random() < 0.3:
                u.select_track(rng.randrange(len(u.tracks)))
            u.next_pairing()
            b = [c for c in u.binding() if c is not None]
            ok &= sorted(b) == list(range(n))
            ok &= sum(1 for g in u.gains if g == 1.0) == 1
            ok &= u.audible_track() is not None
            ok &= u.overridden() == (u.binding() != u.base)
    check("permutation, single audible channel, and an accurate override flag",
          ok)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + "; ".join(FAILURES))
        raise SystemExit(1)
    print("all pairing-override checks passed")


if __name__ == "__main__":
    main()
