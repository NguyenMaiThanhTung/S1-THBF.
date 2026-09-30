# Static THBF — Strict S1 Implementation

Python/NumPy implementation of the **S1: Static THBF First** blocks in the working draft.

## Files

- `config.py` — static S1 parameters.
- `em.py` — Eq. (3), (4), (5), plus no-EM broadside `g0`.
- `rf.py` — Eq. (8); Eq. (9)-(10) only assemble valid `F_RF`.
- `channel.py` — Eq. (12), direct Eq. (14), effective-channel helper Eq. (18)-(19).
- `precoding.py` — Eq. (20) and Eq. (11) power normalization.
- `metrics.py` — Eq. (22), Eq. (23).
- `scheduling.py` — greedy-add user selection; structure follows Ref. [30], while each candidate set is evaluated with this paper's RZF Eq. (20) and gross SE Eq. (23).
- `static_model.py` — complete static chain for a **fixed** EM state `q` and RF beam configuration `b`.
- `acceptance.py` — checks all S1 acceptance items that are fully specified by the draft.
- `tests/test_s1.py` — unit tests. Eq. (6) and Eq. (13) are constructed **only here** to verify direct Eq. (14).
- `SPEC_GAP.md` — documents the unresolved `q*_t` + deterministic full-refresh rule.

## Important scope note

This project does **not** invent an equation for `q*_t` or a deterministic full-refresh routine. The working draft still refers to `q*_t` from `(??)` and states that the full-refresh procedure must be implemented, validated, and frozen. Therefore the corresponding acceptance check is explicitly marked as skipped until the supervisor supplies/finalizes that routine.

This is deliberate: the S1 mathematical model is implemented without silently adding a missing algorithm.

## Run

```bash
python acceptance.py
python -m pytest -q
```
