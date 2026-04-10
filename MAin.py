from __future__ import annotations

import matplotlib
matplotlib.use("TkAgg")

import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Circle

from WildFire_Model import WildFire


# ============================================================
# Experiment parameters
# ============================================================

num_uavs = 3


# ============================================================
# State encoding (HMM hidden states)
# ============================================================

HEALTHY = 0
FIRE = 1
BURNED = 2

WILDFIRE_CMAP = ListedColormap([
    "#2e8b57",  # healthy  (green)
    "#ff8c00",  # fire     (orange)
    "#000000",  # burned   (black)
])

UAV_COLORS = ["white", "cyan", "magenta", "yellow", "lime", "orange"]


# ============================================================
# Parameter dataclasses
# ============================================================

@dataclass
class TransitionParams:
    """Parameters for belief-front extraction and noisy belief transition."""

    # Belief-front extraction thresholds
    front_prob_threshold: float = 0.05
    front_max_points: int = 700
    front_local_radius: int = 2
    front_min_patch_sum: float = 0.04
    intensity_from_prob_scale: float = 60.0

    # Display thresholds for belief rendering
    belief_fire_display_thresh: float = 0.30
    belief_burned_display_thresh: float = 0.55

    eps: float = 1e-6

    # ── Gaussian noise for belief transition model ────────────────────────
    # The belief uses the SAME FireCommander2020 simulator as the ground
    # truth, but with additive/multiplicative Gaussian noise on the inputs
    # to model the UAV's imperfect knowledge of environmental conditions.
    # (Rafiee et al.: process noise w_t ~ N(0, Q_t))

    sigma_wind_speed: float = 0.5           # additive N(0, s^2) on wind speed
    sigma_wind_direction: float = 0.15      # additive N(0, s^2) on wind dir (rad)
    sigma_front_position: float = 1.5       # per-point positional jitter (cells)
    sigma_spread_rate: float = 0.15         # multiplicative log-normal on fuel
    spread_noise_kernel_size: int = 5       # spatial smoothing for fuel noise

    # Spatial diffusion on predicted fire probability (Gaussian blur on
    # b_predicted[:,:,FIRE] to approximate positional uncertainty from a
    # single FARSITE rollout instead of expensive Monte Carlo).
    belief_diffusion_sigma: float = 0.8

    # Information decay: unobserved cells mix toward max entropy [1/3,1/3,1/3]
    # at rate lambda per step.  b(i,j) <- (1-lam)*b(i,j) + lam*[1/3,1/3,1/3]
    info_decay_rate: float = 0.001


@dataclass
class SensorParams:
    pm: float = 0.90
    fov_radius: int = 4

    r_HF: float = 0.75
    r_HB: float = 0.25
    r_FH: float = 0.70
    r_FB: float = 0.30
    r_BH: float = 0.50
    r_BF: float = 0.50

    def __post_init__(self):
        if not 0.0 < self.pm <= 1.0:
            raise ValueError(f"pm must be in (0, 1], got {self.pm}")
        for name, a, b in [("H", self.r_HF, self.r_HB),
                           ("F", self.r_FH, self.r_FB),
                           ("B", self.r_BH, self.r_BF)]:
            if abs(a + b - 1.0) > 1e-6 or a < 0 or b < 0:
                raise ValueError(f"Confusion ratios for {name} invalid: {a}+{b}")


@dataclass
class PlannerParams:
    """SVGD-based receding-horizon ergodic planner parameters."""

    horizon: int = 15
    num_particles: int = 15
    svgd_iters: int = 14
    step_size: float = 0.05
    bandwidth: float = 2.0
    dt: float = 1.0

    # Unicycle dynamics
    v0: float = 4.0
    omega_max: float = 0.60

    # Fourier basis order for ergodic metric
    basis_order_x: int = 5
    basis_order_y: int = 5
    ergodic_lambda: float = 1.0
    fd_eps: float = 5e-3

    # Gaussian prior on controls: p(omega) = N(0, sigma^2 I)
    omega_prior_sigma: float = 0.60
    prior_weight: float = 0.10

    # Inter-agent separation constraint (Lee et al. arXiv:2406.11767 Eq.18)
    separation_distance: float = 8.0
    separation_weight: float = 0.02

    # Boundary wall avoidance constraint (same Eq.18)
    boundary_distance: float = 5.0
    boundary_weight: float   = 0.05

    # Decentralized coefficient-sharing consensus rounds
    consensus_rounds: int = 2

    # EID smoothing kernel (sensor footprint convolution)
    smooth_kernel_size: float = 5
    smooth_sigma: float = 1.0


def generate_uav_initial_states(num_uavs: int, map_size: int) -> np.ndarray:
    """
    Deploy UAVs evenly along the map boundary facing inward.
    Returns shape (num_uavs, 3) => [x, y, theta].
    """
    states = []
    margin = 5.0

    perimeter = 4 * (map_size - 2 * margin)
    spacing = perimeter / num_uavs

    for i in range(num_uavs):
        pos = (i + 0.5) * spacing
        side_len = map_size - 2 * margin

        if pos < side_len:
            x, y = margin + pos, margin
            theta = np.pi / 2
        elif pos < 2 * side_len:
            x, y = map_size - margin, margin + (pos - side_len)
            theta = np.pi
        elif pos < 3 * side_len:
            x, y = map_size - margin - (pos - 2 * side_len), map_size - margin
            theta = -np.pi / 2
        else:
            x, y = margin, map_size - margin - (pos - 3 * side_len)
            theta = 0.0

        states.append([float(x), float(y), float(theta)])

    return np.asarray(states, dtype=float)


# ============================================================
# General helpers
# ============================================================

def clip_xy(x: float, y: float, world_size: int) -> Tuple[float, float]:
    """Clip (x, y) to valid grid range [0, world_size-1]."""
    x = float(np.clip(x, 0.0, world_size - 1.0))
    y = float(np.clip(y, 0.0, world_size - 1.0))
    return x, y


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def normalize_belief(belief: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize belief so each cell sums to 1 across states."""
    denom = np.sum(belief, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return belief / denom


def observation_confusion_matrix(sensor_params: SensorParams) -> np.ndarray:
    pm = float(sensor_params.pm)
    err = 1.0 - pm

    M = np.array(
        [
            [pm,                        sensor_params.r_HF * err, sensor_params.r_HB * err],
            [sensor_params.r_FH * err,  pm,                       sensor_params.r_FB * err],
            [sensor_params.r_BH * err,  sensor_params.r_BF * err, pm                      ],
        ],
        dtype=float,
    )
    return M


def belief_to_display_state(belief_map: np.ndarray) -> np.ndarray:
    """Convert belief to hard state map via argmax for display."""
    return np.argmax(belief_map, axis=-1).astype(np.int32)


def generate_random_hotspots(
    world_size: int,
    num_hotspots: int,
    patch_size: int,
    min_separation: float,
    border_margin: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Generate separated ignition regions as [x_min, x_max, y_min, y_max]."""
    hotspots: List[List[int]] = []
    centers: List[Tuple[float, float]] = []

    tries = 0
    max_tries = 1000

    while len(hotspots) < num_hotspots and tries < max_tries:
        tries += 1
        x = int(rng.integers(border_margin, world_size - patch_size - border_margin))
        y = int(rng.integers(border_margin, world_size - patch_size - border_margin))

        cx = x + patch_size / 2.0
        cy = y + patch_size / 2.0

        ok = True
        for px, py in centers:
            if np.hypot(cx - px, cy - py) < min_separation:
                ok = False
                break

        if ok:
            hotspots.append([x, x + patch_size, y, y + patch_size])
            centers.append((cx, cy))

    if len(hotspots) < num_hotspots:
        raise RuntimeError("Could not generate separated ignition regions.")

    return hotspots


# ============================================================
# Rasterization and true state map
# ============================================================

def rasterize_intensity(points_xyz: np.ndarray, n: int) -> np.ndarray:
    """Rasterize (x, y, intensity) points onto an n x n grid using max-at."""
    I = np.zeros((n, n), dtype=float)
    if points_xyz is None or len(points_xyz) == 0:
        return I

    pts = np.asarray(points_xyz, dtype=float)
    xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, n - 1)
    ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, n - 1)
    vals = np.maximum(pts[:, 2], 0.0)

    np.maximum.at(I, (xs, ys), vals)
    return I


def intensity_to_state_map(
    intensity_map: np.ndarray,
    prev_state_map: Optional[np.ndarray],
    tau_fire: float,
    tau_burn: float,
) -> np.ndarray:
    """Classify each cell as HEALTHY/FIRE/BURNED based on intensity thresholds."""
    state_map = np.full(intensity_map.shape, HEALTHY, dtype=np.int32)

    if prev_state_map is None:
        prev_state_map = np.full(intensity_map.shape, HEALTHY, dtype=np.int32)

    burning_mask = intensity_map > tau_fire
    burned_mask = ((prev_state_map == FIRE) & (intensity_map < tau_burn)) | (prev_state_map == BURNED)

    state_map[burned_mask] = BURNED
    state_map[burning_mask] = FIRE
    return state_map


# ============================================================
# FireCommander2020 simulator wrapper
# ============================================================

def _safe_pruned_list(pruned_points: Optional[np.ndarray]) -> List[List[int]]:
    """Convert pruned points array to list-of-lists for fire_propagation()."""
    if pruned_points is None or len(pruned_points) == 0:
        return []
    return [[int(round(float(p[0]))), int(round(float(p[1])))] for p in pruned_points]


def simulate_fire_one_step(
    wildfire_model,
    world_size: int,
    active_fronts: np.ndarray,
    time_vector: np.ndarray,
    geo_phys_info: Dict[str, np.ndarray],
    previous_terrain_map: np.ndarray,
    decay_rate: float = 0.003,
    pruned_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one step of FARSITE propagation + exponential decay."""
    if active_fronts is None or len(active_fronts) == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0, 3), dtype=float),
        )

    pruned_list = _safe_pruned_list(pruned_points)

    propagated_fronts, _ = wildfire_model.fire_propagation(
        world_Size=world_size,
        ign_points_all=active_fronts.copy(),
        geo_phys_info=geo_phys_info,
        previous_terrain_map=previous_terrain_map.copy(),
        pruned_List=pruned_list,
    )

    next_active_fronts, next_time_vector, burnt_out_fires = wildfire_model.fire_decay(
        terrain_map=propagated_fronts.copy(),
        time_vector=time_vector.copy(),
        geo_phys_info=geo_phys_info,
        decay_rate=decay_rate,
    )

    return next_active_fronts, next_time_vector, burnt_out_fires


# ============================================================
# Belief initialization and belief-front extraction
# ============================================================

def initialize_belief(world_size: int, prior=(0.9985, 0.0010, 0.0005)) -> np.ndarray:
    """Create uniform prior belief map."""
    belief = np.zeros((world_size, world_size, 3), dtype=float)
    belief[:, :, HEALTHY] = prior[0]
    belief[:, :, FIRE] = prior[1]
    belief[:, :, BURNED] = prior[2]
    return normalize_belief(belief)


def initialize_belief_from_initial_fire(
    world_size: int,
    initial_fronts: np.ndarray,
    base_prior=(0.9985, 0.0010, 0.0005),
) -> np.ndarray:
    """Seed initial belief using known ignition fronts (e.g. satellite report)."""
    belief = initialize_belief(world_size, prior=base_prior)

    if initial_fronts is None or len(initial_fronts) == 0:
        return belief

    I0 = rasterize_intensity(initial_fronts, world_size)
    m = float(np.max(I0))
    if m <= 0.0:
        return belief

    fire_support = I0 / m
    fire_seed = np.clip(1.00 * fire_support, 0.0, 0.98)
    fire_seed[fire_seed < 0.015] = 0.0

    burned_seed = np.full((world_size, world_size), base_prior[2], dtype=float)
    healthy_seed = 1.0 - fire_seed - burned_seed
    healthy_seed = np.clip(healthy_seed, 0.0, 1.0)

    belief[:, :, HEALTHY] = healthy_seed
    belief[:, :, FIRE] = fire_seed
    belief[:, :, BURNED] = burned_seed

    return normalize_belief(belief)


def extract_estimated_fronts_from_belief(
    belief_map: np.ndarray,
    transition_params: TransitionParams,
) -> np.ndarray:
    """
    Extract fire front positions from belief map (used only at step 0).
    Returns (K, 3) array of [x, y, intensity_proxy].
    """
    p_fire = belief_map[:, :, FIRE]
    n = p_fire.shape[0]
    r = transition_params.front_local_radius

    candidate_mask = p_fire >= transition_params.front_prob_threshold
    coords = np.argwhere(candidate_mask)

    if len(coords) == 0:
        return np.zeros((0, 3), dtype=float)

    kept: List[Tuple[int, int, float]] = []

    for i, j in coords:
        i0 = max(0, i - r)
        i1 = min(n - 1, i + r)
        j0 = max(0, j - r)
        j1 = min(n - 1, j + r)

        patch = p_fire[i0:i1 + 1, j0:j1 + 1]
        center_val = float(p_fire[i, j])
        patch_sum = float(np.sum(patch))
        local_max = float(np.max(patch))

        if center_val >= 0.90 * local_max and patch_sum >= transition_params.front_min_patch_sum:
            intensity_proxy = transition_params.intensity_from_prob_scale * center_val
            kept.append((int(i), int(j), intensity_proxy))

    if len(kept) == 0:
        return np.zeros((0, 3), dtype=float)

    kept.sort(key=lambda x: x[2], reverse=True)
    kept = kept[:transition_params.front_max_points]

    pts = np.zeros((len(kept), 3), dtype=float)
    for k, (i, j, intensity_proxy) in enumerate(kept):
        pts[k, 0] = float(i)
        pts[k, 1] = float(j)
        pts[k, 2] = float(intensity_proxy)

    return pts


# ============================================================
# Noise injection for belief transition model
# ============================================================

def perturb_geo_phys_info(
    geo_phys_info: Dict[str, np.ndarray],
    transition_params: TransitionParams,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Create a noisy copy of geo-physical info for belief prediction.
    Models the UAV's imperfect environmental knowledge via Gaussian noise
    on wind speed, wind direction, and spread rate.
    """
    noisy_info = {}

    # Wind speed: additive Gaussian
    wind_speed = geo_phys_info["wind_speed"].copy()
    if transition_params.sigma_wind_speed > 0.0:
        noise = rng.normal(0.0, transition_params.sigma_wind_speed, size=wind_speed.shape)
        wind_speed = np.maximum(0.0, wind_speed + noise)
    noisy_info["wind_speed"] = wind_speed

    # Wind direction: additive Gaussian
    wind_direction = geo_phys_info["wind_direction"].copy()
    if transition_params.sigma_wind_direction > 0.0:
        noise = rng.normal(0.0, transition_params.sigma_wind_direction, size=wind_direction.shape)
        wind_direction = wind_direction + noise
    noisy_info["wind_direction"] = wind_direction

    # Spread rate: multiplicative log-normal with spatial smoothing
    # R_noisy = R * exp(eps), preserves R > 0
    spread_rate = geo_phys_info["spread_rate"].copy()
    if transition_params.sigma_spread_rate > 0.0:
        eps_field = rng.normal(0.0, transition_params.sigma_spread_rate, size=spread_rate.shape)

        k = transition_params.spread_noise_kernel_size
        if k >= 3:
            try:
                from scipy.ndimage import gaussian_filter
                eps_field = gaussian_filter(eps_field, sigma=k / 3.0)
            except ImportError:
                pass

        spread_rate = spread_rate * np.exp(eps_field)
        spread_rate = np.maximum(spread_rate, 1e-15)
    noisy_info["spread_rate"] = spread_rate

    return noisy_info


def perturb_front_positions(
    fronts: np.ndarray,
    sigma: float,
    world_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add per-point Gaussian jitter to belief front positions.
    Models imprecise front localisation from the belief probability map.
    """
    if fronts is None or len(fronts) == 0 or sigma <= 0.0:
        return fronts.copy() if fronts is not None else np.zeros((0, 3), dtype=float)

    noisy = fronts.copy()
    n = len(noisy)
    noisy[:, 0] += rng.normal(0.0, sigma, size=n)
    noisy[:, 1] += rng.normal(0.0, sigma, size=n)

    noisy[:, 0] = np.clip(noisy[:, 0], 0.0, world_size - 1.0)
    noisy[:, 1] = np.clip(noisy[:, 1], 0.0, world_size - 1.0)

    return noisy


# ============================================================
# Belief transition (same FARSITE model as ground truth + noise)
# ============================================================

def predict_belief_with_tracked_state(
    belief_map: np.ndarray,
    wildfire_model,
    geo_phys_info: Dict[str, np.ndarray],
    transition_params: TransitionParams,
    world_size: int,
    decay_rate: float,
    tau_fire: float,
    tau_burn: float,
    belief_active_fronts: np.ndarray,
    belief_time_vector: np.ndarray,
    belief_previous_terrain: np.ndarray,
    belief_burnt_out_points: np.ndarray,
    belief_prev_state_map: Optional[np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict next belief using the SAME FireCommander2020 transition model
    as the ground truth, but with Gaussian noise on inputs.

    Pipeline (mirrors ground truth exactly):
      1. Perturb geo-physical inputs (wind, fuel)
      2. Perturb front positions
      3. simulate_fire_one_step() with noisy inputs
      4. Rasterize predicted fronts
      5. Classify into state map
      6. Build per-cell transition tensor T
      7. Chapman-Kolmogorov: b'(s') = sum_s b(s) * T(s'|s)
      8. Spatial diffusion on fire probability

    Returns: (predicted_belief, pred_active_raster, pred_burnt_raster,
              next_fronts, next_time_vec, next_burnt_out, next_prev_terrain,
              next_state_map)
    """
    eps = transition_params.eps
    nx = ny = world_size

    # Empty fronts => identity transition
    if belief_active_fronts is None or len(belief_active_fronts) == 0:
        T = np.zeros((nx, ny, 3, 3), dtype=float)
        T[:, :, HEALTHY, HEALTHY] = 1.0
        T[:, :, FIRE, FIRE]       = 1.0
        T[:, :, BURNED, BURNED]   = 1.0

        predicted = belief_map.copy()

        z = np.zeros((nx, ny), dtype=float)
        empty3 = np.zeros((0, 3), dtype=float)
        empty1 = np.zeros((0,), dtype=float)
        prev_sm = belief_prev_state_map if belief_prev_state_map is not None \
                  else np.full((nx, ny), HEALTHY, dtype=np.int32)
        return (predicted, z, z,
                empty3, empty1, empty3, empty3, prev_sm)

    # Step 1: Perturb geo-physical inputs
    if rng is not None:
        noisy_geo = perturb_geo_phys_info(
            geo_phys_info=geo_phys_info,
            transition_params=transition_params,
            rng=rng,
        )
    else:
        noisy_geo = geo_phys_info

    # Step 2: Perturb front positions
    if rng is not None and transition_params.sigma_front_position > 0.0:
        noisy_fronts = perturb_front_positions(
            fronts=belief_active_fronts,
            sigma=transition_params.sigma_front_position,
            world_size=world_size,
            rng=rng,
        )
    else:
        noisy_fronts = belief_active_fronts.copy()

    # Step 3: Run FARSITE with noisy inputs (same function as ground truth)
    next_belief_fronts, next_belief_time_vector, next_belief_burnt_out = \
        simulate_fire_one_step(
            wildfire_model=wildfire_model,
            world_size=world_size,
            active_fronts=noisy_fronts,
            time_vector=belief_time_vector,
            geo_phys_info=noisy_geo,
            previous_terrain_map=belief_previous_terrain,
            decay_rate=decay_rate,
            pruned_points=belief_burnt_out_points,
        )

    # Step 4: Rasterize predicted fronts
    pred_active_raster_raw = rasterize_intensity(next_belief_fronts, world_size)
    pred_burnt_raster_raw  = rasterize_intensity(next_belief_burnt_out, world_size)

    # Step 5: Classify predicted state map (same thresholds as ground truth)
    next_belief_state_map = intensity_to_state_map(
        intensity_map=pred_active_raster_raw,
        prev_state_map=belief_prev_state_map,
        tau_fire=tau_fire,
        tau_burn=tau_burn,
    )

    # Step 6: Build transition tensor T[i,j,s_curr,s_next]
    current_state_map = np.argmax(belief_map, axis=-1).astype(np.int32)

    T = np.zeros((nx, ny, 3, 3), dtype=float)
    for s_curr in range(3):
        mask_curr = (current_state_map == s_curr)
        for s_next in range(3):
            T[:, :, s_curr, s_next] = (
                mask_curr & (next_belief_state_map == s_next)
            ).astype(float)

    # Physical constraint: HEALTHY cannot jump directly to BURNED
    T[:, :, HEALTHY, BURNED] = 0.0
    h_row_sum = T[:, :, HEALTHY, :].sum(axis=-1, keepdims=True)
    h_row_sum = np.maximum(h_row_sum, eps)
    T[:, :, HEALTHY, :] /= h_row_sum

    # Identity fallback for rows that don't sum to 1
    for s_curr in range(3):
        row_sum = T[:, :, s_curr, :].sum(axis=-1)
        zero_rows = row_sum < eps
        if np.any(zero_rows):
            T[zero_rows, s_curr, s_curr] = 1.0

    # Step 7: Chapman-Kolmogorov predict: b'(s') = sum_s b(s) * T(s'|s)
    pH = belief_map[:, :, HEALTHY]
    pF = belief_map[:, :, FIRE]
    pB = belief_map[:, :, BURNED]

    predH = (
        pH * T[:, :, HEALTHY, HEALTHY]
        + pF * T[:, :, FIRE, HEALTHY]
        + pB * T[:, :, BURNED, HEALTHY]
    )
    predF = (
        pH * T[:, :, HEALTHY, FIRE]
        + pF * T[:, :, FIRE, FIRE]
        + pB * T[:, :, BURNED, FIRE]
    )
    predB = (
        pH * T[:, :, HEALTHY, BURNED]
        + pF * T[:, :, FIRE, BURNED]
        + pB * T[:, :, BURNED, BURNED]
    )

    # Step 8: Spatial diffusion on fire probability (mass-preserving blur)
    sigma_diff = transition_params.belief_diffusion_sigma
    if sigma_diff > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            fire_mass_before = float(np.sum(predF))
            predF = gaussian_filter(predF, sigma=sigma_diff)
            fire_mass_after = float(np.sum(predF))
            if fire_mass_after > 1e-12:
                predF *= (fire_mass_before / fire_mass_after)
        except ImportError:
            pass

    predicted = np.stack([predH, predF, predB], axis=-1)
    predicted = normalize_belief(predicted)

    # Normalised intensity rasters for logging
    active_max = float(np.max(pred_active_raster_raw))
    burnt_max  = float(np.max(pred_burnt_raster_raw))
    pred_active_norm = (pred_active_raster_raw / max(active_max, eps)) if active_max > 0.0 \
                       else np.zeros_like(pred_active_raster_raw)
    pred_burnt_norm  = (pred_burnt_raster_raw  / max(burnt_max,  eps)) if burnt_max  > 0.0 \
                       else np.zeros_like(pred_burnt_raster_raw)

    # Update belief-side previous terrain (mirrors ground truth)
    next_belief_prev_terrain = (
        next_belief_fronts.copy() if len(next_belief_fronts) > 0
        else np.zeros((0, 3), dtype=float)
    )

    return (predicted, pred_active_norm, pred_burnt_norm,
            next_belief_fronts, next_belief_time_vector,
            next_belief_burnt_out, next_belief_prev_terrain,
            next_belief_state_map)


# ============================================================
# Sensor model and Bayesian update
# ============================================================

def containing_cell_from_position(robot_xy: Sequence[float], world_size: int) -> Tuple[int, int]:
    """Map continuous position to discrete grid cell."""
    x = float(robot_xy[0])
    y = float(robot_xy[1])
    i = int(np.floor(np.clip(x, 0.0, world_size - 1.0)))
    j = int(np.floor(np.clip(y, 0.0, world_size - 1.0)))
    return i, j


def chebyshev_fov_mask(world_size: int, robot_xy: Sequence[float], fov_radius: int) -> np.ndarray:
    """Create square FOV boolean mask centred on robot position."""
    i, j = containing_cell_from_position(robot_xy, world_size)

    mask = np.zeros((world_size, world_size), dtype=bool)

    i_min = max(0, i - fov_radius)
    i_max = min(world_size - 1, i + fov_radius)
    j_min = max(0, j - fov_radius)
    j_max = min(world_size - 1, j + fov_radius)

    mask[i_min:i_max + 1, j_min:j_max + 1] = True
    return mask


def sample_square_fov_observation(
    true_state_map: np.ndarray,
    robot_xy: Sequence[float],
    sensor_params: SensorParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample noisy observations within the FOV using the confusion matrix."""
    world_size = true_state_map.shape[0]
    mask = chebyshev_fov_mask(world_size, robot_xy, sensor_params.fov_radius)
    obs_map = np.full_like(true_state_map, fill_value=-1, dtype=np.int32)

    conf = observation_confusion_matrix(sensor_params)
    coords = np.argwhere(mask)

    if len(coords) > 0:
        states = true_state_map[coords[:, 0], coords[:, 1]]
        probs = conf[states]
        draws = np.array([rng.choice([HEALTHY, FIRE, BURNED], p=p) for p in probs], dtype=np.int32)
        obs_map[coords[:, 0], coords[:, 1]] = draws

    return mask, obs_map


def multi_uav_bayesian_fusion(
    predicted_belief: np.ndarray,
    observations: List[Tuple[np.ndarray, np.ndarray]],
    sensor_params: SensorParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Bayesian belief update for multiple UAVs.
    posterior(i,j) ~ predicted_belief(i,j) * P(obs | state)
    Multiple UAVs fused sequentially (product-of-likelihoods).
    """
    updated = predicted_belief.copy()
    conf = observation_confusion_matrix(sensor_params)

    world_size = predicted_belief.shape[0]
    observed_any = np.zeros((world_size, world_size), dtype=bool)
    obs_count = np.zeros((world_size, world_size), dtype=np.int32)

    for mask_i, obs_map_i in observations:
        coords = np.argwhere(mask_i)
        for i, j in coords:
            y_obs = int(obs_map_i[i, j])
            if y_obs < 0:
                continue

            likelihood = conf[:, y_obs]
            posterior = updated[i, j] * likelihood

            s = posterior.sum()
            if s > 0.0:
                updated[i, j] = posterior / s

            observed_any[i, j] = True
            obs_count[i, j] += 1

    updated = normalize_belief(updated)
    return updated, observed_any, obs_count


# ============================================================
# Information maps
# ============================================================

def burning_belief_map(belief_map: np.ndarray) -> np.ndarray:
    """Extract P(FIRE) layer from belief."""
    return belief_map[:, :, FIRE].copy()


def entropy_map(belief_map: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-cell Shannon entropy H(b) = -sum p*log(p)."""
    p = np.clip(belief_map, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def mutual_information_map(
    belief_map: np.ndarray,
    sensor_params: SensorParams,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-cell mutual information MI(state; observation | belief)."""
    conf = observation_confusion_matrix(sensor_params)
    mi = np.zeros(belief_map.shape[:2], dtype=float)

    for y in range(3):
        py = np.sum(belief_map * conf[:, y][None, None, :], axis=-1)
        py = np.clip(py, eps, 1.0)

        for s in range(3):
            term = belief_map[:, :, s] * conf[s, y] * np.log(np.clip(conf[s, y], eps, 1.0) / py)
            mi += term

    return np.maximum(mi, 0.0)


def shift2d_zero(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift 2D array by (dx, dy), filling with zeros."""
    nx, ny = arr.shape
    out = np.zeros_like(arr)

    src_x_min = max(0, -dx)
    src_x_max = min(nx, nx - dx)
    src_y_min = max(0, -dy)
    src_y_max = min(ny, ny - dy)

    dst_x_min = max(0, dx)
    dst_x_max = min(nx, nx + dx)
    dst_y_min = max(0, dy)
    dst_y_max = min(ny, ny + dy)

    out[dst_x_min:dst_x_max, dst_y_min:dst_y_max] = arr[src_x_min:src_x_max, src_y_min:src_y_max]
    return out


def expected_burning_neighbors(lambda_map: np.ndarray) -> np.ndarray:
    """Sum of P(FIRE) over 8-connected neighbors."""
    out = np.zeros_like(lambda_map, dtype=float)
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for dx, dy in shifts:
        out += shift2d_zero(lambda_map, dx, dy)
    return out


def wildfire_relevance_map(
    lambda_map: np.ndarray,
    neighbor_map: np.ndarray,
    burned_map: np.ndarray,
    gamma: float = 0.5,
) -> np.ndarray:
    """Physical relevance map (legacy, not used by current EID)."""
    psi = (lambda_map + gamma * neighbor_map) * (1.0 - burned_map)
    return np.maximum(psi, 0.0)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Build a normalised 2D Gaussian kernel."""
    if size % 2 == 0:
        size += 1
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=float)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    K = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    K /= np.sum(K)
    return K


def smooth_map(map2d: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing via scipy convolution."""
    from scipy.ndimage import convolve
    K = gaussian_kernel(size=size, sigma=sigma)
    return convolve(map2d.astype(float), K, mode="nearest")


def _phi_pure_single_step(
    belief: np.ndarray,
    sensor_params: SensorParams,
) -> np.ndarray:
    """
    Single-step EID: Phi(x, t) = MI(x, t) + b^F(x, t)
    (Coffin et al. ICRA 2022 Eq.9 + Mavrommati et al. 2018)
    """
    mi  = mutual_information_map(belief, sensor_params)
    lam = belief[:, :, FIRE]
    return mi + lam


def target_distribution(
    belief_map: np.ndarray,
    sensor_params: SensorParams,
    planner_params: PlannerParams,
    pred_belief_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute horizon-averaged EID for the ergodic planner.

    Phi_avg(x) = (1/(H+1)) * sum_{s=0}^{H} Phi(x, t+s)

    where intermediate beliefs are linearly interpolated between current
    and predicted belief.  After averaging, the EID is smoothed with a
    Gaussian sensor footprint kernel and normalised to a pdf.

    Returns: (lambda_map, sigma_map, mi_map, psi_map, phi_normalised)
    """
    H = planner_params.horizon

    # Logging quantities from current step
    lambda_map = belief_map[:, :, FIRE].copy()
    sigma_map  = entropy_map(belief_map)
    mi_map     = mutual_information_map(belief_map, sensor_params)

    # Step s=0: current belief EID
    phi_acc = _phi_pure_single_step(belief_map, sensor_params).copy()
    num_steps = 1

    # Steps s=1..H: linearly interpolated beliefs
    if pred_belief_map is not None and pred_belief_map.shape == belief_map.shape:
        for s in range(1, H + 1):
            alpha = float(s) / float(H)
            interp_belief = (1.0 - alpha) * belief_map + alpha * pred_belief_map
            row_sums = interp_belief.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
            interp_belief = interp_belief / row_sums
            phi_acc += _phi_pure_single_step(interp_belief, sensor_params)
            num_steps += 1

    # Horizon average
    psi_map = phi_acc / float(num_steps)

    # Sensor footprint smoothing + normalise to pdf
    phi_smooth = smooth_map(
        psi_map,
        size=planner_params.smooth_kernel_size,
        sigma=planner_params.smooth_sigma,
    )

    s_total = np.sum(phi_smooth)
    if s_total <= 0.0:
        phi = np.ones_like(phi_smooth) / phi_smooth.size
    else:
        phi = phi_smooth / s_total

    return lambda_map, sigma_map, mi_map, psi_map, phi


# ============================================================
# Ergodic metric (Fourier-based)
# ============================================================

class ErgodicCache:
    """
    Pre-computes cosine basis functions, spectral weights, and normalisation
    factors for efficient ergodic metric evaluation.
    (Mathew & Mezic 2011; Mavrommati et al. 2018 Eq.4)
    """
    def __init__(self, world_size_x: int, world_size_y: int, order_x: int, order_y: int):
        self.nx = world_size_x
        self.ny = world_size_y
        self.order_x = order_x
        self.order_y = order_y
        self.k_list = [(kx, ky) for kx in range(order_x) for ky in range(order_y)]
        self.K = len(self.k_list)
        self.num_basis = self.K

        xs = np.arange(self.nx, dtype=float)
        ys = np.arange(self.ny, dtype=float)

        # Cosine basis: cos(k * pi * x / L) for each frequency
        self.basis_x = np.stack(
            [np.cos(kx * np.pi * xs / self.nx) for kx in range(order_x)],
            axis=0,
        )
        self.basis_y = np.stack(
            [np.cos(ky * np.pi * ys / self.ny) for ky in range(order_y)],
            axis=0,
        )

        # Spectral weights: Lambda_k = (1 + ||k||^2)^(-3/2)
        self.spectral_weights = np.array(
            [(1.0 + kx * kx + ky * ky) ** (-1.5) for (kx, ky) in self.k_list],
            dtype=float,
        )

        self._kx_idx = np.array([kx for kx, _ in self.k_list], dtype=int)
        self._ky_idx = np.array([ky for _, ky in self.k_list], dtype=int)

        # Normalisation factors h_k: h_k = prod_i sqrt(1 + delta_{k_i,0})
        self._hk = np.array(
            [
                (np.sqrt(2.0) if kx == 0 else 1.0) * (np.sqrt(2.0) if ky == 0 else 1.0)
                for kx, ky in self.k_list
            ],
            dtype=float,
        )

        # Pre-built basis tensors: F_k(i,j) = (1/h_k)*cos(kx*pi*i/nx)*cos(ky*pi*j/ny)
        self._basis_outer = np.stack(
            [
                np.outer(self.basis_x[kx], self.basis_y[ky]) / hk
                for (kx, ky), hk in zip(self.k_list, self._hk)
            ],
            axis=0,
        )

    def phi_fourier_coefficients(self, phi_map: np.ndarray) -> np.ndarray:
        """Compute phi_k = sum_{i,j} phi(i,j) * F_k(i,j)."""
        return np.einsum('kij,ij->k', self._basis_outer, phi_map)

    def trajectory_fourier_coefficients(self, traj_xy: np.ndarray) -> np.ndarray:
        """Compute c_k = (1/T) sum_t F_k(x(t)) for a trajectory."""
        if len(traj_xy) == 0:
            return np.zeros((self.K,), dtype=float)

        x = np.clip(traj_xy[:, 0], 0.0, self.nx - 1.0)
        y = np.clip(traj_xy[:, 1], 0.0, self.ny - 1.0)

        kx_vals = np.arange(self.order_x)[:, None]
        ky_vals = np.arange(self.order_y)[:, None]
        bx = np.cos(kx_vals * np.pi * x[None, :] / self.nx)
        by = np.cos(ky_vals * np.pi * y[None, :] / self.ny)

        raw = np.mean(bx[self._kx_idx] * by[self._ky_idx], axis=1)
        return raw / self._hk

    def ergodic_metric(self, traj_xy: np.ndarray, phi_map: np.ndarray) -> float:
        """E = sum_k Lambda_k * (c_k - phi_k)^2."""
        phi_k = self.phi_fourier_coefficients(phi_map)
        c_k = self.trajectory_fourier_coefficients(traj_xy)
        return float(np.sum(self.spectral_weights * (c_k - phi_k) ** 2))

    def team_ergodic_metric_from_coefficients(
        self,
        local_ck: np.ndarray,
        phi_map: np.ndarray,
        neighbor_cks: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Team ergodic metric using averaged coefficients from all agents."""
        phi_k = self.phi_fourier_coefficients(phi_map)
        all_ck = [np.asarray(local_ck, dtype=float)]
        if neighbor_cks:
            for ck in neighbor_cks:
                if ck is not None and len(ck) == self.num_basis:
                    all_ck.append(np.asarray(ck, dtype=float))

        c_k_team = np.mean(np.stack(all_ck, axis=0), axis=0)
        return float(np.sum(self.spectral_weights * (c_k_team - phi_k) ** 2))

    def collective_ergodic_metric(
        self,
        traj_xy: np.ndarray,
        phi_map: np.ndarray,
        neighbor_cks: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Convenience: compute local c_k from trajectory, then team metric."""
        local_ck = self.trajectory_fourier_coefficients(traj_xy)
        return self.team_ergodic_metric_from_coefficients(
            local_ck=local_ck,
            phi_map=phi_map,
            neighbor_cks=neighbor_cks,
        )


# ============================================================
# Unicycle dynamics + SVGD trajectory optimisation
# ============================================================

def rollout_unicycle(
    x0: Sequence[float],
    omega_seq: np.ndarray,
    dt: float,
    world_size: int,
    v0: float,
) -> np.ndarray:
    """Forward-simulate unicycle: x' = x + v0*cos(theta)*dt, etc."""
    state = np.asarray(x0, dtype=float).copy()
    traj = [state.copy()]

    for omega in omega_seq:
        x, y, theta = state
        x = x + v0 * np.cos(theta) * dt
        y = y + v0 * np.sin(theta) * dt
        theta = wrap_angle(theta + float(omega) * dt)
        x, y = clip_xy(x, y, world_size)
        state = np.array([x, y, theta], dtype=float)
        traj.append(state.copy())

    return np.asarray(traj, dtype=float)


def omega_cost(omega_seq: np.ndarray, weight: float = 1e-3) -> float:
    """Control effort penalty: weight * sum(omega^2)."""
    return float(weight * np.sum(omega_seq ** 2))


def inequality_constraint_penalty(
    traj_xy: np.ndarray,
    world_size: int,
    planner_params: PlannerParams,
    other_trajs_xy: Optional[List[np.ndarray]] = None,
) -> float:
    """
    Quadratic hinge penalty for inequality constraints
    (Lee et al. arXiv:2406.11767 Eq.18):
      - Inter-agent separation: c_sep * max(0, d_min - d_ij)^2
      - Boundary avoidance:     c_wall * max(0, d_margin - d_wall)^2
    """
    total = 0.0

    # Inter-agent separation
    if other_trajs_xy is not None and planner_params.separation_weight > 0.0:
        for other in other_trajs_xy:
            T = min(len(traj_xy), len(other))
            if T == 0:
                continue
            d = np.linalg.norm(traj_xy[:T] - other[:T], axis=1)
            viol = np.maximum(planner_params.separation_distance - d, 0.0)
            total += planner_params.separation_weight * float(np.sum(viol ** 2))

    # Boundary wall avoidance
    if planner_params.boundary_weight > 0.0 and len(traj_xy) > 0:
        x   = traj_xy[:, 0]
        y   = traj_xy[:, 1]
        lim = float(world_size - 1)
        d_wall = np.minimum(
            np.minimum(x, lim - x),
            np.minimum(y, lim - y),
        )
        viol = np.maximum(planner_params.boundary_distance - d_wall, 0.0)
        total += planner_params.boundary_weight * float(np.sum(viol ** 2))

    return total


def fully_connected_ck_consensus(
    local_ck_matrix: np.ndarray,
    rounds: int = 1,
) -> np.ndarray:
    """Fully connected consensus: each agent averages all agents' c_k vectors."""
    ck = np.asarray(local_ck_matrix, dtype=float).copy()
    if ck.ndim != 2:
        raise ValueError("local_ck_matrix must be shaped [num_uavs, num_basis].")

    n = ck.shape[0]
    if n <= 1:
        return ck

    P = np.full((n, n), 1.0 / n, dtype=float)
    for _ in range(max(int(rounds), 1)):
        ck = P @ ck
    return ck


def objective_of_omega(
    x0: Sequence[float],
    omega_seq: np.ndarray,
    phi_map: np.ndarray,
    planner_params: PlannerParams,
    ergodic_cache: ErgodicCache,
    neighbor_cks: Optional[List[np.ndarray]] = None,
    neighbor_reference_trajs: Optional[List[np.ndarray]] = None,
    return_local_ck: bool = False,
    ck_bar_i: Optional[np.ndarray] = None,
    steps_in_memory: int = 0,
) -> float | Tuple[float, np.ndarray]:
    """Total cost = ergodic metric + control cost + constraint penalties."""
    traj_full = rollout_unicycle(
        x0=x0,
        omega_seq=omega_seq,
        dt=planner_params.dt,
        world_size=phi_map.shape[0],
        v0=planner_params.v0,
    )
    traj_xy = traj_full[:, :2]
    ck_plan = ergodic_cache.trajectory_fourier_coefficients(traj_xy)

    # Blend past history c_bar_k with current plan (Mavrommati 2018 Eq.17)
    H = len(omega_seq)
    if ck_bar_i is not None and steps_in_memory > 0:
        total_duration = steps_in_memory + H
        full_ck = (steps_in_memory * ck_bar_i + H * ck_plan) / total_duration
    else:
        full_ck = ck_plan

    # Team ergodic metric via coefficient sharing
    e_cost = ergodic_cache.team_ergodic_metric_from_coefficients(
        local_ck=full_ck,
        phi_map=phi_map,
        neighbor_cks=neighbor_cks,
    )
    u_cost = omega_cost(omega_seq)

    # Inequality constraint penalties (separation + boundary)
    c_cost = inequality_constraint_penalty(
        traj_xy=traj_xy,
        world_size=phi_map.shape[0],
        planner_params=planner_params,
        other_trajs_xy=neighbor_reference_trajs,
    )

    total_cost = e_cost + u_cost + c_cost
    if return_local_ck:
        return total_cost, full_ck
    return total_cost


def finite_difference_gradient_omega(
    x0: Sequence[float],
    omega_seq: np.ndarray,
    phi_map: np.ndarray,
    planner_params: PlannerParams,
    ergodic_cache: ErgodicCache,
    neighbor_cks: Optional[List[np.ndarray]] = None,
    neighbor_reference_trajs: Optional[List[np.ndarray]] = None,
    ck_bar_i: Optional[np.ndarray] = None,
    steps_in_memory: int = 0,
) -> np.ndarray:
    """Central-difference gradient of objective w.r.t. omega sequence."""
    eps = planner_params.fd_eps
    grad = np.zeros_like(omega_seq)

    for t in range(len(omega_seq)):
        perturbed_p = omega_seq.copy()
        perturbed_p[t] += eps
        obj_p = float(objective_of_omega(
            x0, perturbed_p, phi_map, planner_params, ergodic_cache,
            neighbor_cks, neighbor_reference_trajs,
            ck_bar_i=ck_bar_i, steps_in_memory=steps_in_memory,
        ))

        perturbed_m = omega_seq.copy()
        perturbed_m[t] -= eps
        obj_m = float(objective_of_omega(
            x0, perturbed_m, phi_map, planner_params, ergodic_cache,
            neighbor_cks, neighbor_reference_trajs,
            ck_bar_i=ck_bar_i, steps_in_memory=steps_in_memory,
        ))

        grad[t] = (obj_p - obj_m) / (2.0 * eps)

    return grad


def rbf_kernel_and_grad(
    particles_flat: np.ndarray,
    bandwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """RBF kernel with median heuristic bandwidth (Liu & Wang 2016)."""
    diffs = particles_flat[:, None, :] - particles_flat[None, :, :]
    sq = np.sum(diffs * diffs, axis=-1)

    # Median heuristic: h = median(||x_i - x_j||^2) / log(N)
    N = particles_flat.shape[0]
    off_diag = sq[~np.eye(N, dtype=bool)]
    median_sq = float(np.median(off_diag)) if len(off_diag) > 0 else 0.0
    if median_sq > 1e-10 and N > 1:
        h = median_sq / max(np.log(float(N)), 1e-8)
    else:
        h = max(bandwidth, 1e-6)

    K = np.exp(-sq / h)
    gradK = (-2.0 / h) * K[:, :, None] * diffs
    return K, gradK


def svgd_optimize_omega(
    x0: Sequence[float],
    phi_map: np.ndarray,
    planner_params: PlannerParams,
    rng: np.random.Generator,
    ergodic_cache: ErgodicCache,
    initial_omega: Optional[np.ndarray] = None,
    neighbor_cks: Optional[List[np.ndarray]] = None,
    neighbor_reference_trajs: Optional[List[np.ndarray]] = None,
    ck_bar_i: Optional[np.ndarray] = None,
    steps_in_memory: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SVGD-based trajectory optimisation over omega sequences.
    Returns (best_omega, all_particles, particle_variance, best_ck).
    """
    H = planner_params.horizon
    N = planner_params.num_particles

    # Initialise particles from warm-start or random
    if initial_omega is None:
        particles = rng.normal(loc=0.0, scale=0.15, size=(N, H))
    else:
        init = np.asarray(initial_omega, dtype=float)
        particles = np.repeat(init[None, :], N, axis=0)
        particles += rng.normal(loc=0.0, scale=0.05, size=particles.shape)

    prior_sigma = max(float(planner_params.omega_prior_sigma), 1e-6)

    # SVGD iterations
    for _ in range(planner_params.svgd_iters):
        K, gradK = rbf_kernel_and_grad(particles, planner_params.bandwidth)

        score = np.zeros_like(particles)
        for i in range(N):
            obj_grad = finite_difference_gradient_omega(
                x0=x0,
                omega_seq=particles[i],
                phi_map=phi_map,
                planner_params=planner_params,
                ergodic_cache=ergodic_cache,
                neighbor_cks=neighbor_cks,
                neighbor_reference_trajs=neighbor_reference_trajs,
                ck_bar_i=ck_bar_i,
                steps_in_memory=steps_in_memory,
            )
            prior_score = -(particles[i] / (prior_sigma ** 2))
            score[i] = (
                -planner_params.ergodic_lambda * obj_grad
                + planner_params.prior_weight * prior_score
            )

        # SVGD update: phi(x) = (1/N) sum_j [K(x_j,x)*score(x_j) + grad_K(x_j,x)]
        phi_svgd = np.zeros_like(particles)
        for i in range(N):
            acc = np.zeros(H, dtype=float)
            for j in range(N):
                acc += K[j, i] * score[j] + gradK[j, i]
            phi_svgd[i] = acc / N

        particles = particles + planner_params.step_size * phi_svgd
        particles = np.clip(particles, -planner_params.omega_max, planner_params.omega_max)

    # Select best particle
    scores = np.zeros((N,), dtype=float)
    local_cks = np.zeros((N, ergodic_cache.num_basis), dtype=float)
    for i in range(N):
        score_i, ck_i = objective_of_omega(
            x0=x0,
            omega_seq=particles[i],
            phi_map=phi_map,
            planner_params=planner_params,
            ergodic_cache=ergodic_cache,
            neighbor_cks=neighbor_cks,
            neighbor_reference_trajs=neighbor_reference_trajs,
            return_local_ck=True,
            ck_bar_i=ck_bar_i,
            steps_in_memory=steps_in_memory,
        )
        scores[i] = float(score_i)
        local_cks[i] = ck_i

    best_idx = int(np.argmin(scores))
    particle_variance = np.var(particles, axis=0)
    return particles[best_idx].copy(), particles.copy(), particle_variance.copy(), local_cks[best_idx].copy()


def shift_omega_horizon(omega_seq: np.ndarray, fill_zero: bool = True) -> np.ndarray:
    """Shift omega sequence left by 1 for warm-starting next step."""
    shifted = np.zeros_like(omega_seq)
    shifted[:-1] = omega_seq[1:]
    shifted[-1] = 0.0 if fill_zero else omega_seq[-1]
    return shifted


# ============================================================
# Visualization
# ============================================================

def draw_drone_icon(
    ax,
    x: float,
    y: float,
    size: float = 2.4,
    color: str = "white",
    zorder: int = 10,
):
    """Draw a quadrotor icon on the given axes."""
    artists = []

    body_r = 0.34 * size
    arm_start = 0.50 * size
    arm_end = 0.95 * size
    rotor_half = 0.14 * size
    rotor_offset = 1.10 * size

    body = Circle((x, y), body_r, facecolor=color, edgecolor="black", linewidth=1.0, zorder=zorder)
    ax.add_patch(body)
    artists.append(body)

    angles_deg = [45, 135, 225, 315]
    for ang_deg in angles_deg:
        ang = np.deg2rad(ang_deg)

        x1 = x + arm_start * np.cos(ang)
        y1 = y + arm_start * np.sin(ang)
        x2 = x + arm_end * np.cos(ang)
        y2 = y + arm_end * np.sin(ang)

        line, = ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=zorder,
        )
        artists.append(line)

        xr = x + rotor_offset * np.cos(ang)
        yr = y + rotor_offset * np.sin(ang)

        prop_ang = ang + np.pi / 4.0
        dx = rotor_half * np.cos(prop_ang)
        dy = rotor_half * np.sin(prop_ang)

        prop, = ax.plot(
            [xr - dx, xr + dx],
            [yr - dy, yr + dy],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=zorder,
        )
        artists.append(prop)

    return artists


# ============================================================
# Environment
# ============================================================

class FinalWildfireMonitoringEnv:
    """
    Main environment: manages ground truth fire, belief state, sensor
    observations, Bayesian updates, and EID computation.
    """
    def __init__(
        self,
        world_size: int = 100,
        duration: int = 200,
        fireAreas_Num: int = 3,
        seed=None,
    ):
        self.world_size = int(world_size)
        self.duration = int(duration)
        self.fireAreas_Num = int(fireAreas_Num)

        if seed == "time":
            seed = int(time.time() * 1e6) % (2**32 - 1)
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.rng = np.random.default_rng(0 if seed is None else int(seed))
        self._fig = None
        self._axs = None
        self._cb = []
        self._burn_scatter = None

        # Fire classification thresholds (shared by ground truth and belief)
        self.tau_fire = 0.5
        self.tau_burn = 0.08

    def reset(
        self,
        transition_params: TransitionParams,
        sensor_params: SensorParams,
        planner_params: PlannerParams,
        init_states: Optional[np.ndarray] = None,
        num_uavs: Optional[int] = None,
        decay_rate: float = 0.003,
    ):
        self.transition_params = transition_params
        self.sensor_params = sensor_params
        self.planner_params = planner_params
        self.decay_rate = float(decay_rate)

        n = self.world_size

        # Generate random fire ignition hotspots
        hotspots = generate_random_hotspots(
            world_size=n,
            num_hotspots=self.fireAreas_Num,
            patch_size=10,
            min_separation=5.0,
            border_margin=10,
            rng=self.rng,
        )

        # Initialise FireCommander2020 simulator
        self.wildfire = WildFire(
            terrain_sizes=[n, n],
            hotspot_areas=hotspots,
            num_ign_points=100,
            duration=self.duration,
            time_step=1,
            radiation_radius=50,
            weak_fire_threshold=0.05,
            flame_height=10,
            flame_angle=np.pi / 3,
        )

        self.geo_phys_info = self.wildfire.geo_phys_info_init(
            max_fuel_coeff=5,
            avg_wind_speed=2.0,
            avg_wind_direction=np.pi / 10,
        )

        # Ground truth fire state
        self.active_fronts = self.wildfire.hotspot_init()
        self.previous_terrain_map = self.active_fronts.copy()
        self.time_vector = np.zeros(self.active_fronts.shape[0], dtype=float)
        self.burnt_out_points = np.zeros((0, 3), dtype=float)

        self.true_intensity_map = rasterize_intensity(self.active_fronts, n)
        self.true_state_map = intensity_to_state_map(
            self.true_intensity_map,
            prev_state_map=None,
            tau_fire=self.tau_fire,
            tau_burn=self.tau_burn,
        )

        # Belief state (seeded from known initial ignition)
        self.belief_map = initialize_belief_from_initial_fire(
            world_size=n,
            initial_fronts=self.active_fronts,
            base_prior=(0.9985, 0.0010, 0.0005),
        )

        # Belief-side fire state (parallel to ground truth, diverges via noise)
        self.belief_active_fronts = self.active_fronts.copy()
        self.belief_time_vector = np.zeros(self.belief_active_fronts.shape[0], dtype=float)
        self.belief_previous_terrain = self.belief_active_fronts.copy()
        self.belief_burnt_out_points = np.zeros((0, 3), dtype=float)
        self.belief_state_map = intensity_to_state_map(
            rasterize_intensity(self.belief_active_fronts, n),
            prev_state_map=None,
            tau_fire=self.tau_fire,
            tau_burn=self.tau_burn,
        )

        self.pred_belief_active_raster = np.zeros((n, n), dtype=float)
        self.pred_belief_burnt_raster = np.zeros((n, n), dtype=float)

        # UAV initialisation
        if init_states is None:
            if num_uavs is None:
                num_uavs = 3
            init_states = generate_uav_initial_states(num_uavs=num_uavs, map_size=n)
        else:
            init_states = np.asarray(init_states, dtype=float)
            if init_states.ndim != 2 or init_states.shape[1] != 3:
                raise ValueError("init_states must have shape (num_uavs, 3)")

        self.robot_states = init_states.copy()
        self.num_uavs = self.robot_states.shape[0]

        for i in range(self.num_uavs):
            self.robot_states[i, 0], self.robot_states[i, 1] = clip_xy(
                self.robot_states[i, 0], self.robot_states[i, 1], n
            )
            self.robot_states[i, 2] = wrap_angle(self.robot_states[i, 2])

        self.trajs = [[self.robot_states[i].copy()] for i in range(self.num_uavs)]
        self.t = 0

        # Initial EID (no prediction available yet)
        self.lambda_map, self.sigma_map, self.mi_map, self.psi_map, self.phi_map = target_distribution(
            belief_map      = self.belief_map,
            sensor_params   = self.sensor_params,
            planner_params  = self.planner_params,
            pred_belief_map = None,
        )
        self._phi_map_prev = self.phi_map.copy()

        self.last_obs_count = np.zeros((n, n), dtype=np.int32)
        self.cumulative_observed_mask = np.zeros((n, n), dtype=bool)

        print("Hotspots:", hotspots)
        print("Number of UAVs:", self.num_uavs)
        return self._obs()

    def step(self, omegas: np.ndarray, planner_params: PlannerParams):
        self.t += 1
        omegas = np.asarray(omegas, dtype=float)

        # ── Belief prediction (FARSITE + noise) ──────────────────────────
        (belief_pred, pred_active_raster, pred_burnt_raster,
         next_b_fronts, next_b_time, next_b_burnt, next_b_prev_terrain,
         next_b_state_map) = predict_belief_with_tracked_state(
            belief_map=self.belief_map,
            wildfire_model=self.wildfire,
            geo_phys_info=self.geo_phys_info,
            transition_params=self.transition_params,
            world_size=self.world_size,
            decay_rate=self.decay_rate,
            tau_fire=self.tau_fire,
            tau_burn=self.tau_burn,
            belief_active_fronts=self.belief_active_fronts,
            belief_time_vector=self.belief_time_vector,
            belief_previous_terrain=self.belief_previous_terrain,
            belief_burnt_out_points=self.belief_burnt_out_points,
            belief_prev_state_map=self.belief_state_map,
            rng=self.rng,
        )

        # Update belief-side fire state
        self.belief_active_fronts = next_b_fronts
        self.belief_time_vector = next_b_time
        self.belief_burnt_out_points = next_b_burnt
        self.belief_previous_terrain = next_b_prev_terrain
        self.belief_state_map = next_b_state_map

        self.pred_belief_active_raster = pred_active_raster
        self.pred_belief_burnt_raster = pred_burnt_raster

        # ── Ground truth fire propagation ────────────────────────────────
        next_active_fronts, next_time_vector, next_burnt_out = simulate_fire_one_step(
            wildfire_model=self.wildfire,
            world_size=self.world_size,
            active_fronts=self.active_fronts,
            time_vector=self.time_vector,
            geo_phys_info=self.geo_phys_info,
            previous_terrain_map=self.previous_terrain_map,
            decay_rate=self.decay_rate,
            pruned_points=self.burnt_out_points,
        )

        self.active_fronts = next_active_fronts
        self.time_vector = next_time_vector
        self.burnt_out_points = next_burnt_out
        self.previous_terrain_map = (
            self.active_fronts.copy() if len(self.active_fronts) > 0 else np.zeros((0, 3), dtype=float)
        )

        self.true_intensity_map = rasterize_intensity(self.active_fronts, self.world_size)
        self.true_state_map = intensity_to_state_map(
            self.true_intensity_map,
            prev_state_map=self.true_state_map,
            tau_fire=self.tau_fire,
            tau_burn=self.tau_burn,
        )

        # ── Move UAVs ────────────────────────────────────────────────────
        for i in range(self.num_uavs):
            x, y, theta = self.robot_states[i]
            x = x + self.planner_params.v0 * np.cos(theta) * self.planner_params.dt
            y = y + self.planner_params.v0 * np.sin(theta) * self.planner_params.dt
            theta = wrap_angle(theta + float(omegas[i]) * self.planner_params.dt)
            x, y = clip_xy(x, y, self.world_size)
            self.robot_states[i] = np.array([x, y, theta], dtype=float)
            self.trajs[i].append(self.robot_states[i].copy())

        # ── Sensor observations + Bayesian fusion ────────────────────────
        all_observations: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.num_uavs):
            observed_mask_i, observation_map_i = sample_square_fov_observation(
                true_state_map=self.true_state_map,
                robot_xy=self.robot_states[i, :2],
                sensor_params=self.sensor_params,
                rng=self.rng,
            )
            all_observations.append((observed_mask_i, observation_map_i))

        self.belief_map, team_observed_mask, obs_count = multi_uav_bayesian_fusion(
            predicted_belief=belief_pred,
            observations=all_observations,
            sensor_params=self.sensor_params,
        )

        # ── Information decay for unobserved cells ───────────────────────
        lam = self.transition_params.info_decay_rate
        if lam > 0.0:
            unobserved = ~team_observed_mask
            if np.any(unobserved):
                max_ent = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
                self.belief_map[unobserved] = (
                    (1.0 - lam) * self.belief_map[unobserved]
                    + lam * max_ent[None, :]
                )
                self.belief_map = normalize_belief(self.belief_map)

        self.last_obs_count = obs_count
        self.cumulative_observed_mask |= team_observed_mask

        # ── Compute EID ──────────────────────────────────────────────────
        self.lambda_map, self.sigma_map, self.mi_map, self.psi_map, phi_raw = target_distribution(
            belief_map      = self.belief_map,
            sensor_params   = self.sensor_params,
            planner_params  = self.planner_params,
            pred_belief_map = belief_pred,
        )

        # Temporal EMA on EID to prevent abrupt jumps from cluster burnout
        alpha_ema = 0.5
        if hasattr(self, '_phi_map_prev') and self._phi_map_prev is not None:
            self.phi_map = alpha_ema * phi_raw + (1.0 - alpha_ema) * self._phi_map_prev
            s = np.sum(self.phi_map)
            if s > 0.0:
                self.phi_map = self.phi_map / s
            else:
                self.phi_map = np.ones_like(self.phi_map) / self.phi_map.size
        else:
            self.phi_map = phi_raw
        self._phi_map_prev = self.phi_map.copy()

        return self._obs()

    def _obs(self):
        return {
            "t": self.t,
            "true_intensity_map": self.true_intensity_map,
            "true_state_map": self.true_state_map,
            "belief_map": self.belief_map,
            "pred_belief_active_raster": self.pred_belief_active_raster,
            "pred_belief_burnt_raster": self.pred_belief_burnt_raster,
            "phi_map": self.phi_map,
            "robot_states": self.robot_states.copy(),
            "trajs": [np.asarray(tr, dtype=float) for tr in self.trajs],
            "burnt_out_points": self.burnt_out_points.copy(),
        }

    def render_live(
        self,
        preview_trajs: Optional[List[np.ndarray]] = None,
        inference_vars: Optional[np.ndarray] = None,
        pause: float = 0.05,
    ):
        """Live matplotlib rendering: EID | Belief State | Ground Truth."""
        phi_show = self.phi_map
        belief_state_show = belief_to_display_state(self.belief_map)
        gt_state_show = self.true_state_map

        if self._fig is None:
            plt.ion()
            self._fig, self._axs = plt.subplots(1, 3, figsize=(16, 5))

            titles = [
                "EID  Φ(s,t)",
                "Belief State",
                "Ground Truth State",
            ]

            for ax, title in zip(self._axs, titles):
                ax.set_title(title)
                ax.set_xlim(0, self.world_size - 1)
                ax.set_ylim(0, self.world_size - 1)
                ax.set_aspect("equal")

            self._im0 = self._axs[0].imshow(phi_show, origin="lower", cmap="inferno")
            self._im1 = self._axs[1].imshow(
                belief_state_show,
                origin="lower",
                cmap=WILDFIRE_CMAP,
                vmin=0,
                vmax=2,
                interpolation="nearest",
            )
            self._im2 = self._axs[2].imshow(
                gt_state_show,
                origin="lower",
                cmap=WILDFIRE_CMAP,
                vmin=0,
                vmax=2,
                interpolation="nearest",
            )

            self._cb.append(plt.colorbar(self._im0, ax=self._axs[0], fraction=0.046, pad=0.03))

            self._traj_lines0 = []
            self._traj_lines1 = []
            self._preview_lines = []
            self._fov_rects0 = []
            self._fov_rects1 = []
            self._drone_artists0: List[List] = []
            self._drone_artists1: List[List] = []

            for i in range(self.num_uavs):
                color = UAV_COLORS[i % len(UAV_COLORS)]

                line0, = self._axs[0].plot([], [], color=color, linewidth=2.0)
                line1, = self._axs[1].plot([], [], color=color, linewidth=1.7)
                prev, = self._axs[0].plot([], [], color=color, linewidth=1.6, alpha=0.9, linestyle="None")

                r0 = Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=1.2, zorder=6)
                r1 = Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=1.2, zorder=6)

                self._axs[0].add_patch(r0)
                self._axs[1].add_patch(r1)

                self._traj_lines0.append(line0)
                self._traj_lines1.append(line1)
                self._preview_lines.append(prev)
                self._fov_rects0.append(r0)
                self._fov_rects1.append(r1)
                self._drone_artists0.append([])
                self._drone_artists1.append([])

        self._im0.set_data(phi_show)

        # Rescale colorbar to current data range each frame
        phi_min = float(np.min(phi_show))
        phi_max = float(np.max(phi_show))
        if phi_max - phi_min < 1e-15:
            self._im0.set_clim(vmin=0.0, vmax=max(phi_max, 1e-12))
        else:
            self._im0.set_clim(vmin=phi_min, vmax=phi_max)

        self._im1.set_data(belief_state_show)
        self._im2.set_data(gt_state_show)

        for i in range(self.num_uavs):
            for art in self._drone_artists0[i]:
                try:
                    art.remove()
                except Exception:
                    pass
            for art in self._drone_artists1[i]:
                try:
                    art.remove()
                except Exception:
                    pass
            self._drone_artists0[i] = []
            self._drone_artists1[i] = []

            traj = np.asarray(self.trajs[i], dtype=float)
            self._traj_lines0[i].set_data(traj[:, 1], traj[:, 0])
            self._traj_lines1[i].set_data(traj[:, 1], traj[:, 0])

            if (
                preview_trajs is not None
                and i < len(preview_trajs)
                and preview_trajs[i] is not None
                and len(preview_trajs[i]) > 1
            ):
                self._preview_lines[i].set_data(preview_trajs[i][:, 1], preview_trajs[i][:, 0])
            else:
                self._preview_lines[i].set_data([], [])

            x_row, y_col, _ = self.robot_states[i]

            self._drone_artists0[i] = draw_drone_icon(
                self._axs[0], x=y_col, y=x_row, size=2.3, color=UAV_COLORS[i % len(UAV_COLORS)], zorder=9
            )
            self._drone_artists1[i] = draw_drone_icon(
                self._axs[1], x=y_col, y=x_row, size=2.3, color=UAV_COLORS[i % len(UAV_COLORS)], zorder=9
            )

            ci, cj = containing_cell_from_position(self.robot_states[i, :2], self.world_size)
            r = self.sensor_params.fov_radius

            i0 = max(0, ci - r)
            i1 = min(self.world_size - 1, ci + r)
            j0 = max(0, cj - r)
            j1 = min(self.world_size - 1, cj + r)

            for rect in [self._fov_rects0[i], self._fov_rects1[i]]:
                rect.set_xy((j0, i0))
                rect.set_width(max(1.0, j1 - j0 + 1))
                rect.set_height(max(1.0, i1 - i0 + 1))

        if self._burn_scatter is not None:
            self._burn_scatter.remove()
            self._burn_scatter = None

        if self.burnt_out_points is not None and len(self.burnt_out_points) > 0:
            bx = self.burnt_out_points[:, 0]
            by = self.burnt_out_points[:, 1]
            self._burn_scatter = self._axs[2].scatter(
                by, bx, c="black", s=8, marker="s", zorder=6
            )

        if inference_vars is not None and len(inference_vars) > 0:
            ergodic_cost = float(np.mean(inference_vars))
            title = f"t={self.t} | ergodic cost={ergodic_cost:.4f}"
        else:
            title = f"t={self.t}"

        self._fig.suptitle(title, fontsize=13)
        self._fig.canvas.draw_idle()
        plt.pause(pause)

        self._fig.suptitle(title, fontsize=13)
        self._fig.canvas.draw_idle()
        plt.pause(pause)


# ============================================================
# Main loop
# ============================================================

if __name__ == "__main__":
    transition_params = TransitionParams(
        front_prob_threshold=0.08,
        front_max_points=400,
        front_local_radius=2,
        front_min_patch_sum=0.02,
        intensity_from_prob_scale=80.0,
        belief_fire_display_thresh=0.30,
        belief_burned_display_thresh=0.55,
        eps=1e-6,
        sigma_wind_speed=0.5,
        sigma_wind_direction=0.15,
        sigma_front_position=0.5,
        sigma_spread_rate=0.15,
        spread_noise_kernel_size=5,
        belief_diffusion_sigma=1.5,
        info_decay_rate=0.005,
    )

    sensor_params = SensorParams(
        pm=0.90,
        fov_radius=4,
    )

    planner_params = PlannerParams(
        horizon=10,
        num_particles=14,
        svgd_iters=10,
        step_size=0.05,
        bandwidth=2.0,
        dt=1.0,
        v0=3.5,
        omega_max=0.6,
        basis_order_x=5,
        basis_order_y=5,
        ergodic_lambda=1.0,
        fd_eps=5e-3,
        separation_distance=8.0,
        separation_weight=0.02,
        boundary_distance=5.0,
        boundary_weight=0.10,
        smooth_kernel_size=5,
        smooth_sigma=1.0,
    )

    env = FinalWildfireMonitoringEnv(
        world_size=100,
        duration=200,
        fireAreas_Num=3,
        seed="time",
    )

    desired_num_uavs = num_uavs

    init_states = generate_uav_initial_states(
        num_uavs=desired_num_uavs,
        map_size=env.world_size,
    )

    obs = env.reset(
        transition_params=transition_params,
        sensor_params=sensor_params,
        planner_params=planner_params,
        init_states=init_states,
        num_uavs=desired_num_uavs,
        decay_rate=0.00015,
    )

    ergodic_cache = ErgodicCache(
        world_size_x=env.world_size,
        world_size_y=env.world_size,
        order_x=planner_params.basis_order_x,
        order_y=planner_params.basis_order_y,
    )

    num_uavs = env.num_uavs
    warm_start_omegas = np.zeros((num_uavs, planner_params.horizon), dtype=float)
    local_ck_memory = np.zeros((num_uavs, ergodic_cache.num_basis), dtype=float)
    for i in range(num_uavs):
        init_ref_traj = rollout_unicycle(
            x0=obs["robot_states"][i],
            omega_seq=warm_start_omegas[i],
            dt=planner_params.dt,
            world_size=env.world_size,
            v0=planner_params.v0,
        )[:, :2]
        local_ck_memory[i] = ergodic_cache.trajectory_fourier_coefficients(init_ref_traj)

    consensus_ck_memory = fully_connected_ck_consensus(
        local_ck_memory,
        rounds=planner_params.consensus_rounds,
    )

    # Ergodic memory: accumulates past trajectory coefficients (Mavrommati 2018 Eq.17)
    M_ERG = 2 * planner_params.horizon
    ck_bar = np.zeros((num_uavs, ergodic_cache.num_basis), dtype=float)
    steps_in_memory = 0

    for step in range(120):
        best_omega_seqs = np.zeros((num_uavs, planner_params.horizon), dtype=float)
        preview_trajs: List[np.ndarray] = [None] * num_uavs  # type: ignore
        inference_vars = np.zeros((num_uavs,), dtype=float)
        local_ck_updates = np.zeros_like(local_ck_memory)

        # Generate reference trajectories from warm-starts for separation constraints
        reference_preview_trajs: List[np.ndarray] = []
        for i in range(num_uavs):
            ref_traj_i = rollout_unicycle(
                x0=obs["robot_states"][i],
                omega_seq=warm_start_omegas[i],
                dt=planner_params.dt,
                world_size=env.world_size,
                v0=planner_params.v0,
            )[:, :2]
            reference_preview_trajs.append(ref_traj_i)

        # Per-agent SVGD optimisation
        for i in range(num_uavs):
            neighbor_cks = [consensus_ck_memory[j].copy() for j in range(num_uavs) if j != i]
            neighbor_reference_trajs = [reference_preview_trajs[j] for j in range(num_uavs) if j != i]

            best_omega_seq_i, particles_i, particle_var_i, best_ck_i = svgd_optimize_omega(
                x0=obs["robot_states"][i],
                phi_map=obs["phi_map"],
                planner_params=planner_params,
                rng=env.rng,
                ergodic_cache=ergodic_cache,
                initial_omega=warm_start_omegas[i],
                neighbor_cks=neighbor_cks,
                neighbor_reference_trajs=neighbor_reference_trajs,
                ck_bar_i=ck_bar[i],
                steps_in_memory=steps_in_memory,
            )

            best_omega_seqs[i] = best_omega_seq_i
            local_ck_updates[i] = best_ck_i
            inference_vars[i] = float(np.mean(particle_var_i))

            preview_traj_i = rollout_unicycle(
                x0=obs["robot_states"][i],
                omega_seq=best_omega_seq_i,
                dt=planner_params.dt,
                world_size=env.world_size,
                v0=planner_params.v0,
            )[:, :2]
            preview_trajs[i] = preview_traj_i

        # Consensus update
        local_ck_memory = local_ck_updates.copy()
        consensus_ck_memory = fully_connected_ck_consensus(
            local_ck_memory,
            rounds=planner_params.consensus_rounds,
        )

        # Execute first control action
        omegas0 = best_omega_seqs[:, 0]
        obs = env.step(omegas0, planner_params)

        # Update ergodic memory c_bar_k (Mavrommati 2018 Eq.17)
        H = planner_params.horizon
        for i in range(num_uavs):
            executed_pos = obs["robot_states"][i, :2].reshape(1, 2)
            ck_executed = ergodic_cache.trajectory_fourier_coefficients(executed_pos)
            if steps_in_memory + H > 0:
                old_total = steps_in_memory + H
                new_total = steps_in_memory + 1 + H
                ck_bar[i] = (old_total * ck_bar[i] + ck_executed) / new_total

        steps_in_memory = min(steps_in_memory + 1, M_ERG)

        env.render_live(
            preview_trajs=preview_trajs,
            inference_vars=inference_vars,
            pause=0.05,
        )

        # Warm-start next step
        for i in range(num_uavs):
            warm_start_omegas[i] = shift_omega_horizon(best_omega_seqs[i], fill_zero=True)

        # Logging
        p_fire = env.belief_map[:, :, FIRE]
        p_burned = env.belief_map[:, :, BURNED]
        print(
            f"[step={step:03d}] "
            f"entropy={np.mean(env.sigma_map):.4f} | "
            f"belief_fire_max={np.max(p_fire):.4f} | "
            f"belief_fire_mean={np.mean(p_fire):.4f} | "
            f"belief_burned_max={np.max(p_burned):.4f} | "
            f"pred_active_max={np.max(env.pred_belief_active_raster):.4f}"
        )

        if len(env.active_fronts) == 0 and np.sum(env.true_intensity_map) <= 0.0:
            print("Wildfire extinguished / decayed out. Stopping.")
            break

    plt.ioff()
    plt.show()
