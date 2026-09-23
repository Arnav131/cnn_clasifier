"""Multi-Objective Golden Eagle Optimizer (MOGEO).

This is a real population-based metaheuristic -- NOT random search and NOT
grid search. It implements the core mechanics of the Golden Eagle Optimizer
(Mohammed & Rashid, 2021) -- per-eagle attack/cruise flight vectors, a
generation-dependent attack propensity schedule, and per-eagle memory of
visited points -- extended to the multi-objective case using the standard
NSGA-II machinery (Pareto dominance, fast non-dominated sorting, crowding
distance) to maintain a non-dominated archive and select flight targets
("prey") when there is no single global best.

Objectives (both maximized): internal validation accuracy and internal
validation macro-F1, evaluated ONLY on the dev_val split. final_test is
never touched here.

The optimizer is resumable: state (population, archive, RNG, in-progress
generation) is checkpointed to disk after every single candidate evaluation,
so a Colab disconnect loses at most one partially-evaluated candidate.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import HParams

Objectives = Tuple[float, float]  # (val_acc, val_macro_f1), both maximized


# ---------------------------------------------------------------------------
# Search space: normalized [0, 1]^7 <-> HParams
# ---------------------------------------------------------------------------
class HyperParamSpace:
    DIM = 7
    BASE_CHANNEL_CHOICES = [16, 32, 64]
    BATCH_SIZE_CHOICES = [16, 32, 64]

    NUM_BLOCKS_RANGE = (3, 5)
    DROPOUT_RANGE = (0.2, 0.6)
    LOG_LR_RANGE = (-4.0, -2.5)       # lr in [1e-4, ~3.16e-3]
    LOG_WD_RANGE = (-5.0, -2.0)       # weight_decay in [1e-5, 1e-2]
    LABEL_SMOOTHING_RANGE = (0.0, 0.1)

    @classmethod
    def decode(cls, vector: np.ndarray) -> HParams:
        v = np.clip(vector, 0.0, 1.0)

        lo, hi = cls.NUM_BLOCKS_RANGE
        num_blocks = int(round(lo + v[0] * (hi - lo)))

        idx = min(int(v[1] * len(cls.BASE_CHANNEL_CHOICES)), len(cls.BASE_CHANNEL_CHOICES) - 1)
        base_channels = cls.BASE_CHANNEL_CHOICES[idx]

        lo, hi = cls.DROPOUT_RANGE
        dropout = lo + v[2] * (hi - lo)

        lo, hi = cls.LOG_LR_RANGE
        lr = float(10 ** (lo + v[3] * (hi - lo)))

        lo, hi = cls.LOG_WD_RANGE
        weight_decay = float(10 ** (lo + v[4] * (hi - lo)))

        idx = min(int(v[5] * len(cls.BATCH_SIZE_CHOICES)), len(cls.BATCH_SIZE_CHOICES) - 1)
        batch_size = cls.BATCH_SIZE_CHOICES[idx]

        lo, hi = cls.LABEL_SMOOTHING_RANGE
        label_smoothing = lo + v[6] * (hi - lo)

        return HParams(
            num_blocks=num_blocks,
            base_channels=base_channels,
            dropout=float(dropout),
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            label_smoothing=float(label_smoothing),
        )


# ---------------------------------------------------------------------------
# NSGA-II style dominance / non-dominated sorting / crowding distance
# ---------------------------------------------------------------------------
def dominates(a: Objectives, b: Objectives) -> bool:
    """True if `a` Pareto-dominates `b` (both objectives maximized)."""
    not_worse = all(x >= y for x, y in zip(a, b))
    strictly_better = any(x > y for x, y in zip(a, b))
    return not_worse and strictly_better


def fast_non_dominated_sort(objs: List[Objectives]) -> List[List[int]]:
    n = len(objs)
    dominated_count = [0] * n
    dominates_list: List[List[int]] = [[] for _ in range(n)]
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                dominates_list[p].append(q)
            elif dominates(objs[q], objs[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # last one is always empty
    return fronts


def crowding_distance(front: List[int], objs: List[Objectives]) -> Dict[int, float]:
    distances = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distances[i] = float("inf")
        return distances

    num_obj = len(objs[0])
    for m in range(num_obj):
        front_sorted = sorted(front, key=lambda i: objs[i][m])
        min_v, max_v = objs[front_sorted[0]][m], objs[front_sorted[-1]][m]
        distances[front_sorted[0]] = float("inf")
        distances[front_sorted[-1]] = float("inf")
        denom = (max_v - min_v) if max_v > min_v else 1e-12
        for k in range(1, len(front_sorted) - 1):
            prev_v = objs[front_sorted[k - 1]][m]
            next_v = objs[front_sorted[k + 1]][m]
            distances[front_sorted[k]] += (next_v - prev_v) / denom
    return distances


# ---------------------------------------------------------------------------
# Eagle / archive individual representation
# ---------------------------------------------------------------------------
@dataclass
class Individual:
    position: np.ndarray
    objectives: Objectives
    hparams: Dict
    candidate_id: str = ""
    extra: Dict = field(default_factory=dict)


def build_archive(pool: List[Individual], max_size: int) -> List[Individual]:
    """Non-dominated front of `pool`, truncated to `max_size` by crowding distance."""
    if not pool:
        return []
    objs = [ind.objectives for ind in pool]
    fronts = fast_non_dominated_sort(objs)
    front0 = fronts[0]
    if len(front0) <= max_size:
        return [pool[i] for i in front0]
    dist = crowding_distance(front0, objs)
    ranked = sorted(front0, key=lambda i: dist[i], reverse=True)
    return [pool[i] for i in ranked[:max_size]]


def tournament_leader(archive: List[Individual], rng: np.random.Generator, k: int = 2) -> Individual:
    """Binary-tournament leader ('prey') selection favoring high crowding distance,
    i.e. favoring diverse, less-crowded regions of the Pareto front."""
    if len(archive) == 1:
        return archive[0]
    objs = [ind.objectives for ind in archive]
    dist = crowding_distance(list(range(len(archive))), objs)
    candidates = rng.choice(len(archive), size=min(k, len(archive)), replace=False)
    best_idx = max(candidates, key=lambda i: dist[i])
    return archive[best_idx]


# ---------------------------------------------------------------------------
# MOGEO optimizer
# ---------------------------------------------------------------------------
class MOGEO:
    def __init__(
        self,
        objective_fn: Callable[[HParams, str], Tuple[float, float, Dict]],
        pop_size: int = 8,
        generations: int = 6,
        seed: int = 42,
        archive_size: Optional[int] = None,
        p_min: float = 0.2,
        p_max: float = 0.9,
        cruise_radius: float = 0.15,
        state_path: Optional[str] = None,
        on_candidate_evaluated: Optional[Callable[[int, int, Individual], None]] = None,
    ):
        self.objective_fn = objective_fn
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        self.archive_size = archive_size or (2 * pop_size)
        self.p_min = p_min
        self.p_max = p_max
        self.cruise_radius = cruise_radius
        self.state_path = state_path
        self.on_candidate_evaluated = on_candidate_evaluated

        self.rng = np.random.default_rng(seed)
        self.dim = HyperParamSpace.DIM

        self.population: List[Individual] = []
        self.archive: List[Individual] = []
        self.generation = -1  # last FULLY completed generation
        self.partial: List[Individual] = []  # in-progress generation's evaluated eagles
        self.serial = 0

        if self.state_path:
            self._try_load_state()

    # -- persistence -------------------------------------------------------
    def _save_state(self) -> None:
        if not self.state_path:
            return
        state = {
            "population": self.population,
            "archive": self.archive,
            "generation": self.generation,
            "partial": self.partial,
            "serial": self.serial,
            "rng_state": self.rng.bit_generator.state,
        }
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(state, f)
        import os
        os.replace(tmp_path, self.state_path)

    def _try_load_state(self) -> bool:
        import os
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path, "rb") as f:
            state = pickle.load(f)
        self.population = state["population"]
        self.archive = state["archive"]
        self.generation = state["generation"]
        self.partial = state["partial"]
        self.serial = state["serial"]
        self.rng.bit_generator.state = state["rng_state"]
        return True

    # -- flight mechanics ----------------------------------------------------
    def _attack_propensity(self, generation: int) -> float:
        t = generation / max(self.generations - 1, 1)
        return self.p_min + t * (self.p_max - self.p_min)

    def _fly(self, position: np.ndarray, leader: np.ndarray, generation: int) -> np.ndarray:
        attack_vec = leader - position
        norm_a = np.linalg.norm(attack_vec)
        if norm_a < 1e-9:
            attack_hat = np.zeros(self.dim)
        else:
            attack_hat = attack_vec / norm_a

        # Cruise vector: random direction orthogonalized against the attack
        # vector (Gram-Schmidt) -- models the eagle circling prey before
        # committing to a strike.
        rand_vec = self.rng.normal(size=self.dim)
        if norm_a >= 1e-9:
            rand_vec = rand_vec - np.dot(rand_vec, attack_hat) * attack_hat
        norm_c = np.linalg.norm(rand_vec)
        cruise_hat = rand_vec / norm_c if norm_c >= 1e-9 else np.zeros(self.dim)

        p = self._attack_propensity(generation)
        r1, r2 = self.rng.uniform(), self.rng.uniform()

        attack_step = r1 * p * norm_a
        cruise_step = r2 * (1 - p) * self.cruise_radius

        velocity = attack_step * attack_hat + cruise_step * cruise_hat
        new_position = position + velocity
        return np.clip(new_position, 0.0, 1.0)

    # -- main loop -----------------------------------------------------------
    def _evaluate(self, position: np.ndarray, generation: int, eagle_idx: int) -> Individual:
        hparams = HyperParamSpace.decode(position)
        candidate_id = f"gen{generation}_eagle{eagle_idx}_s{self.serial}"
        self.serial += 1
        val_acc, val_macro_f1, extra = self.objective_fn(hparams, candidate_id)
        ind = Individual(
            position=position,
            objectives=(val_acc, val_macro_f1),
            hparams=hparams.to_dict(),
            candidate_id=candidate_id,
            extra=extra,
        )
        if self.on_candidate_evaluated is not None:
            self.on_candidate_evaluated(generation, eagle_idx, ind)
        return ind

    def run(self) -> List[Individual]:
        # --- Generation 0: initialize population if not resumed ---
        if self.generation == -1 and not self.population:
            positions = self.rng.uniform(0.0, 1.0, size=(self.pop_size, self.dim))
            resume_from = len(self.partial)
            for i in range(resume_from, self.pop_size):
                ind = self._evaluate(positions[i], generation=0, eagle_idx=i)
                self.partial.append(ind)
                self._save_state()
            # reconstruct positions for eagles already done (resume case)
            self.population = self.partial
            self.archive = build_archive(self.population, self.archive_size)
            self.generation = 0
            self.partial = []
            self._save_state()

        # --- Generations 1..N-1 ---
        for g in range(self.generation + 1, self.generations):
            new_individuals: List[Individual] = list(self.partial)
            start_idx = len(new_individuals)
            for i in range(start_idx, self.pop_size):
                leader = tournament_leader(self.archive, self.rng)
                new_position = self._fly(self.population[i].position, leader.position, g)
                ind = self._evaluate(new_position, generation=g, eagle_idx=i)
                new_individuals.append(ind)
                self.partial = new_individuals
                self._save_state()

            # Greedy elitist replacement: an eagle only moves if its new
            # position is not dominated by its own previous position.
            next_population = []
            for old, new in zip(self.population, new_individuals):
                if dominates(old.objectives, new.objectives):
                    next_population.append(old)
                else:
                    next_population.append(new)

            self.population = next_population
            self.archive = build_archive(self.population + self.archive, self.archive_size)
            self.generation = g
            self.partial = []
            self._save_state()

        return self.archive

    def best_compromise(self) -> Individual:
        """Picks a single solution from the final archive for final retraining:
        the archive member maximizing the mean of (val_acc, val_macro_f1)."""
        if not self.archive:
            raise RuntimeError("Archive is empty -- run() must be called first.")
        return max(self.archive, key=lambda ind: sum(ind.objectives) / len(ind.objectives))
