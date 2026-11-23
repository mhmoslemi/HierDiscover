"""
PUCT/UCT state archive.

For each candidate state s in the archive, PUCT uses:

    score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))

UCT uses the same c and reward scale but removes the prior:

    score(s) = Q(s) + c * scale * sqrt(log(1 + T) / (1 + n(s)))

  Q(s)   = max child search value seen so far from s (or its own search value)
  P(s)   = rank-based prior over the archive.
  scale  = range of search values over the non-seed archive
  T      = total expansions performed
  n(s)   = expansions of s OR ANY DESCENDANT (so a successful lineage
           gets dampened, not just one node)

When selecting a batch of B parents, we also exclude the full ancestor
AND descendant set of each picked state from being picked again in the
same batch ("lineage blocking").

After each batch:
  - Push the new children into the archive
  - Keep top-K children per parent (K=2 by default)
  - Cap the archive at MAX_BUFFER_SIZE by search value (seeds always kept)

State carries two optional problem-agnostic payloads:
  - raw_score:   the TRUE metric for prompt display (separate from `value`).
  - construction: an injected global threaded parent -> child for warm starts.

Archive reads, updates, and checkpoint serialization are guarded by an RLock.
"""

import threading
import uuid
import numpy as np
from dataclasses import asdict, dataclass, field, replace

from config_validation import PARENT_SELECTION_MODES


AC1_CONSTRUCTION_LIMITS = (1_000, 100_000)


def parent_selection_mode(config):
    """Resolve the new selector while accepting old ``uct: true`` configs."""
    mode = config.get("parent_selection")
    if mode is None:
        mode = "uct" if config.get("uct", False) else "puct"
    if not isinstance(mode, str) or mode.lower() not in PARENT_SELECTION_MODES:
        raise ValueError(f"parent_selection must be one of {PARENT_SELECTION_MODES}")
    return mode.lower()


def make_sampler(*, parent_selection="puct", **kwargs):
    mode = parent_selection_mode({"parent_selection": parent_selection})
    return PUCTSampler(use_uct=(mode == "uct"), **kwargs)


@dataclass
class State:
    id: str
    timestep: int          # training step when this state was first created
    value: float           # reward of this state (always higher=better)
    code: str              # the code that produced it
    parents: list = field(default_factory=list)   # [{"id": ..., "timestep": ...}]
    is_seed: bool = False  # True for initial seed states
    raw_score: float = None      # true metric for prompt display (may equal value)
    construction: list = None    # injected global threaded parent -> child
    observation: str = ""        # evaluator output shown when this state is a parent
    previous_raw_score: float | None = None
    poisson: float | None = None
    construction_length: int | None = None

    @staticmethod
    def make(timestep, value, code, parents=None, is_seed=False,
             raw_score=None, construction=None, observation="",
             previous_raw_score=None, poisson=None, construction_length=None):
        return State(
            id=str(uuid.uuid4()),
            timestep=timestep,
            value=value,
            code=code,
            parents=parents or [],  # all the parents not just one
            is_seed=is_seed,
            raw_score=raw_score,
            construction=construction,
            observation=str(observation or ""),
            previous_raw_score=previous_raw_score,
            poisson=poisson,
            construction_length=construction_length,
        )


class PUCTSampler:
    def __init__(
        self,
        num_seeds: int,
        puct_c: float = 1.0,
        max_buffer_size: int = 1000,
        topk_children: int = 2,
        seed_value: float = 0.0,
        seed_states: list = None,
        use_uct: bool = False,
        ac1_archive_seed: int | None = None,
        refresh_ac1_seed=None,
    ):
        self.puct_c = float(puct_c)
        self.use_uct = bool(use_uct)
        self.max_buffer_size = max_buffer_size
        self.topk_children = topk_children
        self.ac1_archive_seed = ac1_archive_seed
        self._refresh_ac1_seed = refresh_ac1_seed
        self._last_seed_refresh_step = None
        self._last_seed_refresh_picks = []

        # PUCT statistics
        self._n = {}   # state.id -> visit count (incl. descendants)
        self._m = {}   # state.id -> best child reward seen
        self._T = 0    # total expansions

        # Archive starts with seeds
        self._states = []
        self._seed_ids = set()

        self._lock = threading.RLock()

        if seed_states:
            for ss in seed_states:
                s = State.make(
                    timestep=0,
                    value=getattr(ss, "value", seed_value),
                    code=getattr(ss, "code", "") or "",
                    is_seed=True,
                    raw_score=getattr(ss, "raw_score", None),
                    construction=getattr(ss, "construction", None),
                    poisson=getattr(ss, "poisson", None),
                )
                self._states.append(s)
                self._seed_ids.add(s.id)
        else:
            for _ in range(num_seeds):
                s = State.make(timestep=0, value=seed_value, code="", is_seed=True)
                self._states.append(s)
                self._seed_ids.add(s.id)

        # Independent records of the raw extrema. They are updated before code
        # deduplication and archive pruning, so "best ever" cannot regress when
        # a stochastic rerun of identical code obtains a better native metric.
        self._best_raw_min = None
        self._best_raw_max = None
        self._record_raw_states(self._states)

        # Stats from last sample_states call (for printing)
        self.last_picks_info = []

    @property
    def selection_mode(self):
        return "uct" if self.use_uct else "puct"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finite_raw(state):
        if state.raw_score is None:
            return None
        try:
            raw = float(state.raw_score)
        except (TypeError, ValueError):
            return None
        return raw if np.isfinite(raw) else None

    def _record_raw_states(self, states):
        for state in states:
            raw = self._finite_raw(state)
            if raw is None:
                continue
            if (self._best_raw_min is None
                    or raw < float(self._best_raw_min.raw_score)):
                self._best_raw_min = state
            if (self._best_raw_max is None
                    or raw > float(self._best_raw_max.raw_score)):
                self._best_raw_max = state

    def _search_value(self, state):
        if self.ac1_archive_seed is not None:
            raw = self._finite_raw(state)
            return -raw if raw is not None else -np.inf
        return state.value if state.value is not None else -np.inf

    def _archive_key(self, state):
        if self.ac1_archive_seed is not None and state.construction:
            return tuple(state.construction)
        return ("code", state.code) if state.code else None

    def _accepts_construction(self, state):
        if self.ac1_archive_seed is None:
            return True
        length = state.construction_length
        if length is None and state.construction is not None:
            length = len(state.construction)
        low, high = AC1_CONSTRUCTION_LIMITS
        return not length or low <= length <= high

    def _cap_archive(self):
        if len(self._states) <= self.max_buffer_size:
            return
        if self.ac1_archive_seed is not None:
            # Use upstream's cutoff ordering, then preserve insertion order.
            values = [self._search_value(s) for s in self._states]
            keep = {i for i, s in enumerate(self._states) if s.id in self._seed_ids}
            for i in np.argsort(values)[::-1]:
                if len(keep) >= self.max_buffer_size:
                    break
                keep.add(i)
            self._states = [self._states[i] for i in sorted(keep)]
        else:
            seeds = [s for s in self._states if s.id in self._seed_ids]
            others = [s for s in self._states if s.id not in self._seed_ids]
            others.sort(key=self._search_value, reverse=True)
            self._states = seeds + others[:self.max_buffer_size - len(seeds)]

    def _build_children_map(self):
        children = {}
        for s in self._states:
            for p in s.parents:
                children.setdefault(p["id"], set()).add(s.id)
        return children

    def _full_lineage(self, state, children_map):
        """Ancestors + descendants + self."""
        lineage = {state.id}
        for p in state.parents:
            lineage.add(p["id"])
        queue = [state.id]
        visited = {state.id}
        while queue:
            sid = queue.pop(0)
            for c in children_map.get(sid, []):
                if c not in visited:
                    visited.add(c)
                    lineage.add(c)
                    queue.append(c)
        return lineage

    def _scale(self):
        vals = np.array([
            self._search_value(s) for s in self._states
            if s.id not in self._seed_ids and s.value is not None
        ])
        if vals.size == 0 and self.ac1_archive_seed is not None:
            vals = np.array([self._search_value(s) for s in self._states])
        if vals.size == 0:
            return 1.0
        return float(max(vals.max() - vals.min(), 1e-6))

    def _prior(self, values):
        """Rank-based: best gets (N), worst gets 1; normalize to a distribution."""
        if len(values) == 0:
            return np.array([])
        N = len(values)
        order = np.argsort(np.argsort(-values))  # rank 0 = best
        w = (N - order).astype(np.float64)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _ranked_states(self):
        """Compute PUCT or UCT scores for the current archive."""
        values = np.array([self._search_value(s) for s in self._states])
        scale = self._scale()
        if self.use_uct:
            # This archive has one global expansion count rather than a local
            # tree-parent count, so T and n(s) are the direct archive analogues
            # of the usual UCT N(parent) and N(parent, action).
            P = np.zeros(len(values), dtype=np.float64)
            uct_numerator = np.log1p(self._T)
        else:
            P = self._prior(values)
            sqrtT = np.sqrt(1.0 + self._T)

        scored = []
        for i, s in enumerate(self._states):
            n = self._n.get(s.id, 0)
            m = self._m.get(s.id, values[i])
            Q = m if n > 0 else values[i]
            if self.use_uct:
                bonus = self.puct_c * scale * np.sqrt(
                    uct_numerator / (1.0 + n))
            else:
                bonus = self.puct_c * scale * P[i] * sqrtT / (1.0 + n)
            scored.append((Q + bonus, values[i], s, n, Q, P[i], bonus))

        # Sort by (score, value) descending
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored

    def _pick_info(self, entry):
        return {
            "value": entry[1], "n": entry[3], "Q": entry[4],
            "P": None if self.use_uct else entry[5],
            "bonus": entry[6], "score": entry[0],
            "selection_mode": self.selection_mode,
            "is_seed": entry[2].id in self._seed_ids,
        }

    def sample_states(self, num_states: int, *, step_idx: int | None = None):
        """Pick `num_states` parent states for the next batch."""
        if not self._states:
            return []
        if (self.ac1_archive_seed is not None and step_idx is not None
                and self._last_seed_refresh_step == step_idx):
            return list(self._last_seed_refresh_picks)
        scored = self._ranked_states()

        # Lineage blocking across the batch
        children_map = self._build_children_map()
        blocked = set()
        picks = []
        info = []
        for entry in scored:
            s = entry[2]
            if s.id in blocked:
                continue
            picks.append(s)
            info.append(self._pick_info(entry))
            blocked.update(self._full_lineage(s, children_map))
            if len(picks) >= num_states:
                break

        # Only the legacy archive tops up with states from blocked lineages.
        if self.ac1_archive_seed is None and len(picks) < num_states:
            picked_ids = {s.id for s in picks}
            for entry in scored:
                if len(picks) >= num_states:
                    break
                s = entry[2]
                if s.id in picked_ids:
                    continue
                picks.append(s)
                info.append(self._pick_info(entry))

        if self.ac1_archive_seed is not None:
            # Recreate the same random constructions from the pre-step checkpoint.
            entropy = None if step_idx is None else [self.ac1_archive_seed, step_idx]
            rng = np.random.default_rng(entropy)
            refreshed = {}
            for index, state in enumerate(picks):
                if state.id in self._seed_ids:
                    seed = self._refresh_ac1_seed(rng)
                    refreshed[state.id] = picks[index] = replace(
                        state, construction=seed.construction, raw_score=seed.raw_score,
                        value=seed.value)
            self._states = [refreshed.get(s.id, s) for s in self._states]
            self._record_raw_states(refreshed.values())
            self._last_seed_refresh_step = step_idx
            self._last_seed_refresh_picks = list(picks)
        self.last_picks_info = info
        return picks

    def update(self, children_with_parents):
        """Push valid children into the archive and update search statistics."""
        children_with_parents = list(children_with_parents)
        with self._lock:
            for child, parent in children_with_parents:
                child.previous_raw_score = parent.raw_score
            # Do this before deduplication/pruning: every successfully evaluated
            # child counts toward the run's problem-native best-ever value.
            self._record_raw_states(
                child for child, _parent in children_with_parents
                if child.value is not None
            )

            # The best child updates PUCT even if admission later rejects it.
            parent_best = {}
            for child, parent in children_with_parents:
                if child.value is None:
                    continue
                pid = parent.id
                parent_best[pid] = max(parent_best.get(pid, -np.inf), self._search_value(child))
            for pid, best in parent_best.items():
                self._m[pid] = max(self._m.get(pid, best), best)

            existing_keys = {self._archive_key(s) for s in self._states}
            existing_keys.discard(None)
            new_states = []
            for child, parent in children_with_parents:
                if child.value is None or not self._accepts_construction(child):
                    continue
                key = self._archive_key(child)
                if key is not None and key in existing_keys:
                    continue  # exact dup
                child.parents = (
                    [{"id": parent.id, "timestep": parent.timestep}]
                    + (parent.parents or [])
                )
                new_states.append(child)
                if key is not None:
                    existing_keys.add(key)

            self._states.extend(new_states)

            # Enforce top-K children per parent (excluding seeds)
            if self.topk_children > 0:
                by_parent = {}
                no_parent = []
                for s in self._states:
                    if s.id in self._seed_ids or not s.parents:
                        no_parent.append(s)
                        continue
                    pid = s.parents[0]["id"]
                    by_parent.setdefault(pid, []).append(s)
                filtered = list(no_parent)
                for children in by_parent.values():
                    children.sort(key=self._search_value, reverse=True)
                    filtered.extend(children[: self.topk_children])
                self._states = filtered

            self._cap_archive()

            retained_ids = {state.id for state in self._states}
            return {
                "submitted": len(children_with_parents),
                "unique_new": len(new_states),
                "retained_new": sum(
                    state.id in retained_ids for state in new_states),
                "archive_size": len(self._states),
            }

    def record_expansion(self, parent: State, count: int = 1):
        """Count a parent batch by attempted rollouts, including failed candidates."""
        anc_ids = [parent.id] + [p["id"] for p in (parent.parents or [])]
        for aid in anc_ids:
            self._n[aid] = self._n.get(aid, 0) + count
        self._T += count

    def reassign_seed_id(self, state: State, new_id: str):
        """Bind a freshly recreated step-zero seed to its durable artifact ID."""
        with self._lock:
            if state.id not in self._seed_ids:
                raise ValueError("only fresh seed identities can be rebound")
            self._seed_ids.remove(state.id)
            state.id = str(new_id)
            self._seed_ids.add(state.id)

    def best_state(self):
        with self._lock:
            non_seeds = [s for s in self._states if s.id not in self._seed_ids]
            if not non_seeds:
                return None
            return max(non_seeds,
                       key=lambda s: s.value if s.value is not None else -np.inf)

    def best_raw_state(self, maximize: bool = True):
        """Best state ever evaluated in the problem's native metric direction.

        Seeds are included because a generated candidate has not necessarily
        beaten the initial construction. The record is independent of archive
        deduplication/pruning and is included in sampler checkpoints.
        """
        with self._lock:
            return self._best_raw_max if maximize else self._best_raw_min

    def archive_size(self):
        return len(self._states)

    def state_dict(self):
        """Return all search state required to continue in another process."""
        with self._lock:
            return {
                "version": 1,
                "states": [asdict(s) for s in self._states],
                "seed_ids": sorted(self._seed_ids),
                "n": dict(self._n),
                "m": dict(self._m),
                "T": int(self._T),
                "selection_mode": self.selection_mode,
                "ac1_archive": self.ac1_archive_seed is not None,
                "best_raw_min": (asdict(self._best_raw_min)
                                 if self._best_raw_min is not None else None),
                "best_raw_max": (asdict(self._best_raw_max)
                                 if self._best_raw_max is not None else None),
            }

    def load_state_dict(self, payload):
        """Restore a snapshot produced by :meth:`state_dict`."""
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported or invalid sampler checkpoint")
        if bool(payload.get("ac1_archive")) != (self.ac1_archive_seed is not None):
            raise ValueError("sampler checkpoint uses different AC1 archive scoring rules")
        saved_mode = payload.get("selection_mode")
        current_mode = self.selection_mode
        if saved_mode is not None and saved_mode != current_mode:
            raise ValueError(
                f"sampler checkpoint uses {saved_mode}, but this run uses "
                f"{current_mode}")
        states = payload.get("states")
        if not isinstance(states, list) or not states:
            raise ValueError("sampler checkpoint contains no states")

        restored = [State(**item) for item in states]
        restored_ids = {s.id for s in restored}
        seed_ids = set(payload.get("seed_ids", ()))
        if not seed_ids.issubset(restored_ids):
            raise ValueError("sampler checkpoint references missing seed states")

        with self._lock:
            self._states = restored
            self._seed_ids = seed_ids
            self._n = {str(k): int(v) for k, v in payload.get("n", {}).items()}
            self._m = {str(k): float(v) for k, v in payload.get("m", {}).items()}
            self._T = int(payload.get("T", 0))
            raw_min = payload.get("best_raw_min")
            raw_max = payload.get("best_raw_max")
            self._best_raw_min = State(**raw_min) if raw_min else None
            self._best_raw_max = State(**raw_max) if raw_max else None
            # Backward compatibility for checkpoints written before the raw
            # extrema were persisted.
            if self._best_raw_min is None or self._best_raw_max is None:
                self._record_raw_states(self._states)
            self.last_picks_info = []
            self._last_seed_refresh_step = None
            self._last_seed_refresh_picks = []

    def import_legacy_states(self, states, total_expansions: int = 0):
        """Best-effort archive import for runs created before checkpoints."""
        with self._lock:
            existing_keys = {self._archive_key(s) for s in self._states}
            existing_keys.discard(None)
            for state in states:
                self._record_raw_states([state])
                key = self._archive_key(state)
                if (not self._accepts_construction(state)
                        or (key is not None and key in existing_keys)):
                    continue
                state.parents = []
                self._states.append(state)
                if key is not None:
                    existing_keys.add(key)
            self._cap_archive()
            self._T = max(self._T, int(total_expansions))

if __name__ == "__main__":
    sampler = PUCTSampler(num_seeds=3, puct_c=1.0)
    print("Initial archive:", sampler.archive_size())

    picks = sampler.sample_states(2)
    print(f"Picked {len(picks)} parents (all seeds):", [s.is_seed for s in picks])

    children = []
    for p in picks:
        sampler.record_expansion(p)
        child = State.make(timestep=1, value=np.random.rand() * 2.5,
                           code=f"code_{uuid.uuid4().hex[:6]}")
        children.append((child, p))
    sampler.update(children)
    print("After update:", sampler.archive_size())

    picks2 = sampler.sample_states(2)
    print(f"Pick info: {sampler.last_picks_info}")
    print("Best state value:", sampler.best_state().value)

    ac1 = PUCTSampler(num_seeds=1, topk_children=1, ac1_archive_seed=0)
    parent = ac1._states[0]
    first = State.make(1, 1.0, "first", raw_score=1.5, construction=[0.5] * 1_000)
    duplicate = State.make(1, 1.0, "duplicate", raw_score=1.5, construction=first.construction)
    short = State.make(1, 2.0, "short", raw_score=1.4, construction=[0.5] * 999)
    slower = State.make(1, 3.0, "slower", raw_score=1.6, construction=[1.0] * 1_000)
    stats = ac1.update((child, parent) for child in (first, duplicate, short, slower))
    assert stats["unique_new"] == 2
    assert ac1._states == [parent, first]
    assert ac1._m[parent.id] == -1.4
