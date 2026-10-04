"""Index arithmetic between the three spaces that meet in the pipeline.

Pure Python, no imports beyond typing -- so it can be tested without torch,
mediapipe, ffmpeg, or a GPU, and so nothing here can fail for an environmental
reason.  That matters because these functions fail *silently* when they are
wrong: they play the wrong person's voice rather than raising.

The three spaces are NOT interchangeable:

  * **track index** -- position in ``tracks``; what the browser draws boxes for.
  * **stem index** -- row of the separator's output, before reordering.
  * **channel index** -- row of the exported WAV, after reordering; what the UI
    actually plays.

``assignment`` is the bridge and it is asymmetric: **indexed by track, holding
stem values**, with ``-1`` for a track the matcher could not claim.  Nearly
every bug in this area is a value from one space used as an index into another.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["claims", "greedy_one_to_one", "lips_by_stem", "plan_channels"]


def greedy_one_to_one(score: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Greedy one-to-one assignment maximising total score.  No scipy.

    Repeatedly takes the highest-scoring free (row, column) pair.  Returns
    ``(row, col)`` pairs, at most ``min(n_rows, n_cols)`` of them, with every
    row and every column used at most once.

    This exists because the natural one-liner -- ``np.argmax(score, axis=1)``
    -- is **not** an assignment.  It picks each row's favourite independently,
    so two rows routinely name the same column: on random square matrices that
    happens in 72% of cases (measured, n in 2..4).  Both scipy fallbacks in this
    codebase used it, and a collision is silently destructive in each:

    * ``matching`` -- two faces claim one stem, and the pipeline then has to
      guess which face owns the channel (see :func:`claims`).
    * ``separation._align_permutation`` -- ``cur[order]`` duplicates one
      source across two channels and drops the other entirely, so one speaker
      is heard on both buttons and the other is gone.

    Greedy is not always optimal (it matches the Hungarian optimum on 83% of
    random 2x2 matrices, 54% at 4x4), but it is *always* a valid one-to-one
    assignment, which argmax is not.  It only runs when scipy is missing, and
    scipy is a hard requirement -- ``dsp`` imports it at module scope -- so in
    practice this is a guardrail, not a working code path.  It is here so the
    guardrail cannot itself be the bug.

    Pure Python and dependency-free, matching the rest of this module: the
    matrices are at most 4x4, so O(n^2 log n) sorting costs nothing.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, row in enumerate(score):
        for j, v in enumerate(row):
            pairs.append((float(v), i, j))
    # Sort by descending score; ties break on (row, col) so the result is
    # deterministic rather than dependent on numpy's iteration order.
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    used_r: set[int] = set()
    used_c: set[int] = set()
    out: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out.append((i, j))
    return sorted(out)


def claims(assignment: Sequence[int], n_tracks: int, n_stems: int) -> list[tuple[int, int]]:
    """Resolve ``assignment`` into ``(track, stem)`` pairs, one stem at most once.

    **This is the single arbiter of a duplicated stem claim**, and it exists
    because the two consumers used to disagree.  ``lips_by_stem`` assigned into
    a list in track order, so the LAST claimant won; ``plan_channels`` deduped
    with a ``seen`` set, so the FIRST claimant won.  With ``assignment=[0, 0]``
    the result was that channel 0 was *gated with track 1's lips* while the UI
    labelled and drew it as track 0 -- a wrong-voice failure that raises
    nothing, and that the existing permutation check could not see because
    ``order`` stayed a valid permutation the whole time.

    First claimant wins, matching the ``plan_channels`` behaviour that the
    exported channel order already depended on.  Later claimants are dropped;
    their tracks report ``channel_of is None`` and the UI marks them unreliable.

    Duplicates are not hypothetical: the Hungarian solver is one-to-one, but
    ``matching.match_stems_to_tracks`` falls back to ``np.argmax(score, axis=1)``
    when scipy is unavailable, and argmax has no such constraint.

    Out-of-range and ``-1`` entries are filtered here too, so no caller has to
    re-derive the validity rule.
    """
    seen: set[int] = set()
    out: list[tuple[int, int]] = []
    for track_i in range(n_tracks):
        if track_i >= len(assignment):
            break
        stem_j = assignment[track_i]
        if 0 <= stem_j < n_stems and stem_j not in seen:
            seen.add(stem_j)
            out.append((track_i, stem_j))
    return out


def lips_by_stem(assignment: Sequence[int], tracks: Sequence[Any],
                 n_stems: int) -> list[Any]:
    """Re-index lip signals from track-space into stem-space.

    Returns a list of length ``n_stems``; entry ``j`` is the lip signal of the
    track matched to stem ``j``, or ``None`` if no track claimed it.

    ``None`` is load-bearing: it makes the gate fall back to pure-acoustic for
    that stem.  It must never be confused with an all-zero lip signal, which
    would read as "mouth perfectly still" and mute the stem completely.

    Shares :func:`claims` with :func:`plan_channels` so that the lips a stem is
    gated with always belong to the track the UI labels it as.
    """
    lips: list[Any] = [None] * n_stems
    for track_i, stem_j in claims(assignment, len(tracks), n_stems):
        lips[stem_j] = tracks[track_i].lip
    return lips


def plan_channels(assignment: Sequence[int], n_tracks: int, n_stems: int
                  ) -> tuple[list[int], dict[int, int | None]]:
    """Decide the export channel order and each track's channel.

    Returns ``(order, channel_of)``:

    * ``order`` is a permutation of **all** ``n_stems`` stem indices -- matched
      stems first in track order, then the unclaimed ones.  It is always a full
      permutation, so ``stems[order]`` never loses or duplicates audio.
    * ``channel_of[track]`` is that track's WAV channel, or ``None``.

    Why this is not a one-liner: ``assignment`` holds ``-1`` for any track the
    matcher could not claim, which happens whenever there are more faces than
    stems -- a three-person clip, a poster on the wall, a reflection in a
    window.  Building the order from a naive filter shifts every later track
    onto its neighbour's audio and hands the last one a channel that does not
    exist in the WAV, so clicking that face mutes everything.  Unmatched tracks
    carry ``None`` all the way to the browser instead, and unclaimed stems stay
    playable, just unlabelled.
    """
    # Duplicate/invalid claims are resolved by `claims`, which lips_by_stem
    # shares, so a stem's channel and the lips it was gated with can never
    # disagree about which track owns it.
    matched = claims(assignment, n_tracks, n_stems)
    seen = {j for _, j in matched}

    order = [j for _, j in matched]
    order += [j for j in range(n_stems) if j not in seen]

    channel_of: dict[int, int | None] = {i: None for i in range(n_tracks)}
    for pos, (track_i, _) in enumerate(matched):
        channel_of[track_i] = pos
    return order, channel_of
