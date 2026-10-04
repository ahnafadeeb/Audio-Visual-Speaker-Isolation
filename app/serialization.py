"""JSON that survives the round trip to the browser.

Several numbers this pipeline reports can legitimately be non-finite, and they
occur in exactly the places we most want to look at:

  * ``dsp.silence_stats`` on a channel that is *perfectly* muted has no nonzero
    samples, so its residual floor is ``-inf``.
  * ``dsp.measure_leakage_db`` on a target that is never silent has no
    measurement window; ``dsp.whisper_db`` returns ``None`` for the same
    reason, and also when the window exists but the output is bit-exact
    zero across it.

``-inf`` is the correct real-number answer and an invalid JSON one.  RFC 8259
has no infinity or NaN literal, and the three consumers in this app disagree
about it in three different ways:

  * Starlette's ``JSONResponse.render`` calls ``json.dumps(content,
    ensure_ascii=False, allow_nan=False, ...)`` -- non-finite values **raise**,
    so ``GET /api/jobs/{id}`` answers 500.  That is the deep-link boot path
    (``/?job=demo``), i.e. the way the demo is opened.
  * Bare ``json.dumps`` -- the SSE encoder and the ``meta.json`` writer --
    defaults to ``allow_nan=True`` and emits the non-standard ``-Infinity``
    token.  The browser's ``JSON.parse`` **throws** on it, so the terminal
    ``done`` event is dropped and the progress bar hangs at the end of a
    *successful* job.
  * Python's own ``json.loads`` accepts ``-Infinity``, so ``meta.json`` round
    trips inside the process and the corruption is invisible from the server.

The trigger is success.  A channel that achieves this project's headline
guarantee -- 100% bit-exact zeros -- is precisely the channel whose statistics
cannot be serialised.

:func:`json_safe` maps non-finite floats to ``None`` (JSON ``null``) rather than
to a sentinel like ``-400.0``.  ``null`` cannot be mistaken for a measurement; a
large negative number can, and for ``measure_leakage_db`` / ``whisper_db`` it
would be mistaken for the *best possible* result when it actually means "not
measurable".

Producers are fixed at the source as well (they return ``None`` directly).  This
module is the boundary net: it keeps a future non-finite value -- a NaN
confidence out of a degenerate score matrix, a stray ``inf`` in a config dump --
from reaching the wire.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

__all__ = ["json_safe", "dumps"]


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Containers are rebuilt rather than mutated, so the caller's structure is
    left untouched.  numpy scalars and arrays are unwrapped on the way through
    -- ``np.float32('-inf')`` is *not* an instance of ``float`` and would
    otherwise slip past the check and straight into the encoder.

    Anything else is returned as-is; the encoder's own ``default=`` hook still
    applies to it.
    """
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()                  # np.float32 -> float, np.int64 -> int
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def dumps(obj: Any, **kw: Any) -> str:
    """``json.dumps`` with the non-finite hazard removed first.

    ``allow_nan=False`` is set deliberately.  After :func:`json_safe` there is
    nothing left for it to reject, so if it ever *does* raise, that is a real
    bug in ``json_safe`` -- and we want it loud, in the server log, rather than
    silently emitting a token the browser will choke on.
    """
    kw.setdefault("allow_nan", False)
    return json.dumps(json_safe(obj), **kw)
