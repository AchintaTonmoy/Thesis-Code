"""
# *******************<><><><><>**************************
# * Updated Wildfire Environment Python Translation      *
# * Supports physics-informed belief prediction helpers  *
# *******************<><><><><>**************************
#
# Original simulator concept implemented by:
#   Esmaeil Seraj, CORE Robotics Lab, Georgia Tech
#
# Updated for thesis integration:
#   - safer indexing / bounds handling
#   - cleaner propagation / decay implementation
#   - added support-map utilities for reduced probabilistic
#     transition model in belief prediction
#
# Recommended transition-model parameters (to use in main script):
#   alpha_neighbor  = 0.90
#   alpha_intensity = 1.10
#   alpha_wind      = 0.35
#   rho             = 0.08
#   eps             = 1e-6
#
# Belief-prediction form:
#   p_ign  = 1 - exp(-(a_n*N + a_I*I + a_w*W))
#   p_surv = exp(-rho*dt / (R + eps))
#
# where:
#   N = neighborhood fire pressure (from belief)
#   I = front/intensity influence map (from active fronts)
#   W = wind alignment / downwind favorability map
#   R = spread-rate map
"""

from __future__ import annotations

import numpy as np


class WildFire:
    def __init__(
        self,
        terrain_sizes=None,
        hotspot_areas=None,
        num_ign_points=None,
        duration=None,
        time_step=1.0,
        radiation_radius=10.0,
        weak_fire_threshold=0.5,
        flame_height=3.0,
        flame_angle=np.pi / 3,
    ):
        if terrain_sizes is None or hotspot_areas is None or num_ign_points is None or duration is None:
            raise ValueError("WildFire environment requires terrain_sizes, hotspot_areas, num_ign_points, and duration.")

        self.terrain_sizes = [int(terrain_sizes[0]), int(terrain_sizes[1])]
        self.initial_terrain_map = np.zeros(shape=self.terrain_sizes, dtype=float)

        # hotspot format: [[x_min, x_max, y_min, y_max], ...]
        self.hotspot_areas = hotspot_areas
        self.num_ign_points = int(num_ign_points)
        self.duration = int(duration)

        self.time_step = float(time_step)
        self.radiation_radius = float(radiation_radius)
        self.weak_fire_threshold = float(weak_fire_threshold)

        self.flame_height = float(flame_height)
        self.flame_angle = float(flame_angle)

    # ============================================================
    # Internal utilities
    # ============================================================

    def _clip_index(self, x: float, y: float) -> tuple[int, int]:
        i = int(np.clip(round(float(x)), 0, self.terrain_sizes[0] - 1))
        j = int(np.clip(round(float(y)), 0, self.terrain_sizes[1] - 1))
        return i, j

    @staticmethod
    def _normalize_map01(map2d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mmin = float(np.min(map2d))
        mmax = float(np.max(map2d))
        if mmax - mmin < eps:
            return np.zeros_like(map2d, dtype=float)
        return (map2d - mmin) / (mmax - mmin + eps)

    # ============================================================
    # Hotspot initialization
    # ============================================================

    def hotspot_init(self) -> np.ndarray:
        """
        Generate initial hotspot ignition points with intensity.

        Returns:
            ign_points: [K, 3] array, each row = [x, y, intensity]
        """
        ign_points_all = np.zeros((0, 2), dtype=float)

        for hotspot in self.hotspot_areas:
            x_min, x_max, y_min, y_max = hotspot
            ign_x = np.random.randint(low=x_min, high=x_max, size=(self.num_ign_points, 1))
            ign_y = np.random.randint(low=y_min, high=y_max, size=(self.num_ign_points, 1))
            pts = np.concatenate([ign_x, ign_y], axis=1)
            ign_points_all = np.concatenate([ign_points_all, pts], axis=0)

        ign_points = np.zeros((ign_points_all.shape[0], 3), dtype=float)

        for k, point in enumerate(ign_points_all):
            diffs = np.tile(point, (ign_points_all.shape[0], 1)) - ign_points_all
            dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
            idx = np.where(dists <= self.radiation_radius)[0]
            fire_intensity = self.fire_intensity(
                current_fire_spot=point,
                heat_source_spots=ign_points_all[idx.tolist(), :].tolist(),
            )
            ign_points[k] = np.array([point[0], point[1], fire_intensity], dtype=float)

        return ign_points

    # ============================================================
    # Fire intensity and flame length
    # ============================================================

    def fire_intensity(
        self,
        current_fire_spot=None,
        heat_source_spots=None,
        deviation_min: int = 9,
        deviation_max: int = 11,
    ) -> float:
        """
        Compute accumulated fire intensity at a fire location from nearby heat sources.
        """
        if current_fire_spot is None or heat_source_spots is None:
            raise ValueError("current_fire_spot and heat_source_spots are required.")

        x = float(current_fire_spot[0])
        y = float(current_fire_spot[1])

        x_dev = float(np.random.randint(low=deviation_min, high=deviation_max) + np.random.normal())
        y_dev = float(np.random.randint(low=deviation_min, high=deviation_max) + np.random.normal())

        x_dev = max(abs(x_dev), 1e-6)
        y_dev = max(abs(y_dev), 1e-6)

        cos_term = np.cos(self.flame_angle)
        if abs(cos_term) < 1e-9:
            intensity_coeff = (259.833 * (self.flame_height ** 2.174)) / 1e3
        else:
            intensity_coeff = (259.833 * ((self.flame_height / cos_term) ** 2.174)) / 1e3

        intensity_terms = []
        for spot in heat_source_spots:
            x_f = float(spot[0])
            y_f = float(spot[1])
            term = (1.0 / (2.0 * np.pi * x_dev * y_dev)) * np.exp(
                -0.5 * (((x - x_f) ** 2) / (x_dev ** 2) + ((y - y_f) ** 2) / (y_dev ** 2))
            )
            intensity_terms.append(term)

        accumulated_intensity = float(sum(intensity_terms)) * intensity_coeff
        return 1e3 * accumulated_intensity

    @staticmethod
    def fire_flame_length(accumulated_intensity=None) -> float:
        """
        Flame length as a function of fire intensity.
        """
        if accumulated_intensity is None:
            raise ValueError("accumulated_intensity is required.")
        return 0.0775 * (float(accumulated_intensity) ** 0.46)

    # ============================================================
    # Geo-physical information
    # ============================================================

    def geo_phys_info_init(
        self,
        max_fuel_coeff: float = 7.0,
        avg_wind_speed: float = 5.0,
        avg_wind_direction: float = np.pi / 8,
    ) -> dict:
        """
        Generate terrain spread-rate map and stochastic wind profile.
        """
        min_fuel_coeff = 1e-15
        fuel_rng = max_fuel_coeff - min_fuel_coeff

        spread_rate = fuel_rng * np.random.rand(self.terrain_sizes[0], self.terrain_sizes[1]) + min_fuel_coeff

        # keep wind speed nonnegative
        wind_speed = np.maximum(
            0.0,
            np.random.normal(avg_wind_speed, 2.0, size=(self.terrain_sizes[0], 1)),
        )

        wind_direction = np.random.normal(avg_wind_direction, 2.0, size=(self.terrain_sizes[0], 1))

        return {
            "spread_rate": spread_rate.astype(float),
            "wind_speed": wind_speed.astype(float),
            "wind_direction": wind_direction.astype(float),
        }

    # ============================================================
    # Wildfire propagation
    # ============================================================

    def fire_propagation(
        self,
        world_Size,
        ign_points_all=None,
        geo_phys_info=None,
        previous_terrain_map=None,
        pruned_List=None,
    ):
        """
        Simplified FARSITE-style wildfire propagation.

        Args:
            ign_points_all: [K, 3] active fire fronts
            geo_phys_info: dict with spread_rate, wind_speed, wind_direction
            previous_terrain_map: previous full terrain map / fronts
            pruned_List: list of integer [x, y] points removed due to burnout

        Returns:
            new_fire_front: [K, 3]
            current_geo_phys_info: [K, 3] with [R, U, Theta] for each front
        """
        if ign_points_all is None or geo_phys_info is None or previous_terrain_map is None or pruned_List is None:
            raise ValueError("fire_propagation requires ign_points_all, geo_phys_info, previous_terrain_map, and pruned_List.")

        spread_rate = geo_phys_info["spread_rate"]
        wind_speed = geo_phys_info["wind_speed"]
        wind_direction = geo_phys_info["wind_direction"]

        current_geo_phys_info = np.zeros((ign_points_all.shape[0], 3), dtype=float)
        new_fire_front = np.zeros((ign_points_all.shape[0], 3), dtype=float)

        for counter, point in enumerate(ign_points_all):
            x = float(point[0])
            y = float(point[1])

            # keep only visible / valid points
            if not ((x <= (world_Size - 1)) and (y <= (world_Size - 1)) and (x > 0) and (y > 0)):
                continue

            i, j = self._clip_index(x, y)
            R = float(spread_rate[i, j])

            # row-wise stochastic wind sampling, matching original behavior
            row_idx = np.random.randint(low=0, high=self.terrain_sizes[0])
            U = float(wind_speed[row_idx, 0])
            Theta = float(wind_direction[row_idx, 0])

            current_geo_phys_info[counter] = np.array([R, U, Theta], dtype=float)

            # Simplified FARSITE
            LB = 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397

            inner = np.abs(LB ** 2 - 1.0)
            root_term = np.sqrt(inner)
            denom = LB - root_term
            if abs(denom) < 1e-9:
                HB = 1.0
            else:
                HB = (LB + root_term) / denom

            if abs(HB) < 1e-9:
                C = 0.0
            else:
                C = 0.5 * (R - (R / HB))

            x_diff = C * np.sin(Theta)
            y_diff = C * np.cos(Theta)

            if [int(x), int(y)] not in pruned_List:
                x_new = x + x_diff * self.time_step
                y_new = y + y_diff * self.time_step
            else:
                x_new = x
                y_new = y

            # current-front source contribution
            diff1 = np.tile(point, (ign_points_all.shape[0], 1)) - ign_points_all
            dist1 = np.sqrt(diff1[:, 0] ** 2 + diff1[:, 1] ** 2)
            idx1 = np.where(dist1 <= self.radiation_radius)[0]
            fire_intensity1 = self.fire_intensity(
                current_fire_spot=point,
                heat_source_spots=ign_points_all[idx1.tolist(), :].tolist(),
            )

            # previous-terrain contribution
            if previous_terrain_map.shape[0] > 0:
                diff2 = np.tile(point, (previous_terrain_map.shape[0], 1)) - previous_terrain_map
                dist2 = np.sqrt(diff2[:, 0] ** 2 + diff2[:, 1] ** 2)
                idx2 = np.where(dist2 <= self.radiation_radius)[0]
                fire_intensity2 = self.fire_intensity(
                    current_fire_spot=point,
                    heat_source_spots=previous_terrain_map[idx2.tolist(), :].tolist(),
                )
            else:
                fire_intensity2 = 0.0

            fire_intensity = fire_intensity1 + fire_intensity2
            new_fire_front[counter] = np.array([x_new, y_new, fire_intensity], dtype=float)

        return new_fire_front, current_geo_phys_info

    # ============================================================
    # Dynamic decay
    # ============================================================

    def fire_decay(self, terrain_map=None, time_vector=None, geo_phys_info=None, decay_rate: float = 0.01):
        """
        Exponential fire-intensity decay with local spread-rate dependence.
        """
        if terrain_map is None or geo_phys_info is None or time_vector is None:
            raise ValueError("fire_decay requires terrain_map, geo_phys_info, and time_vector.")

        spread_rate = geo_phys_info["spread_rate"]

        step_vector = self.time_step * np.ones(terrain_map.shape[0], dtype=float)
        updated_time_vector = time_vector + step_vector

        updated_terrain_map = np.zeros((terrain_map.shape[0], 3), dtype=float)

        for counter, spot in enumerate(terrain_map):
            x = float(spot[0])
            y = float(spot[1])
            intensity = float(spot[2])

            i, j = self._clip_index(x, y)
            R = float(spread_rate[i, j])

            I_new = intensity * np.exp(-float(decay_rate) * updated_time_vector[counter] / max(R, 1e-12))
            updated_terrain_map[counter] = np.array([x, y, I_new], dtype=float)

        updated_terrain_map, updated_time_vector, burnt_out_fires_new = self.pruning_fire_map(
            updated_terrain_map=updated_terrain_map,
            updated_time_vector=updated_time_vector,
        )

        return updated_terrain_map, updated_time_vector, burnt_out_fires_new

    # ============================================================
    # Pruning
    # ============================================================

    def pruning_fire_map(self, updated_terrain_map=None, updated_time_vector=None):
        """
        Remove weak / burnt-out fire spots.
        """
        if updated_terrain_map is None or updated_time_vector is None:
            raise ValueError("pruning_fire_map requires updated_terrain_map and updated_time_vector.")

        burnt_out_idx = np.where(updated_terrain_map[:, 2] < self.weak_fire_threshold)
        burnt_out_fires_new = updated_terrain_map[burnt_out_idx]

        updated_terrain_map = np.delete(updated_terrain_map, burnt_out_idx, axis=0)
        updated_time_vector = np.delete(updated_time_vector, burnt_out_idx)

        return updated_terrain_map, updated_time_vector, burnt_out_fires_new

    # ============================================================
    # Support maps for reduced probabilistic transition model
    # ============================================================

    def spread_rate_map(self, geo_phys_info: dict) -> np.ndarray:
        """
        Direct access to local spread-rate map R(i,j).
        Used in:
            p_surv(i,j,t) = exp(-rho*dt / (R(i,j) + eps))
        """
        return np.asarray(geo_phys_info["spread_rate"], dtype=float).copy()

    def front_intensity_influence_map(
        self,
        active_fronts: np.ndarray | None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Build front/intensity influence map from active fronts.

        Each active front contributes a Gaussian-like influence weighted
        by its current fire intensity.

        Returns:
            influence_map: [Nx, Ny]
        """
        nx, ny = self.terrain_sizes
        influence = np.zeros((nx, ny), dtype=float)

        if active_fronts is None or len(active_fronts) == 0:
            return influence

        xs = np.arange(nx, dtype=float)
        ys = np.arange(ny, dtype=float)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")

        sigma = max(1.0, self.radiation_radius / 2.0)

        for p in np.asarray(active_fronts, dtype=float):
            x0, y0, inten = float(p[0]), float(p[1]), max(float(p[2]), 0.0)
            d2 = (XX - x0) ** 2 + (YY - y0) ** 2
            influence += inten * np.exp(-0.5 * d2 / (sigma ** 2))

        if normalize:
            influence = self._normalize_map01(influence)

        return influence

    def wind_alignment_map(
        self,
        active_fronts: np.ndarray | None,
        geo_phys_info: dict,
        decay_length: float = 10.0,
        normalize: bool = True,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Build a wind-favorability map.

        Cells lying roughly downwind from active fronts get higher values.
        This is a reduced proxy for spread tendency, not a raw simulator rollout.

        Returns:
            W_map: [Nx, Ny]
        """
        nx, ny = self.terrain_sizes
        out = np.zeros((nx, ny), dtype=float)

        if active_fronts is None or len(active_fronts) == 0:
            return out

        wind_speed_field = geo_phys_info["wind_speed"]
        wind_dir_field = geo_phys_info["wind_direction"]

        xs = np.arange(nx, dtype=float)
        ys = np.arange(ny, dtype=float)
        XX, YY = np.meshgrid(xs, ys, indexing="ij")

        for p in np.asarray(active_fronts, dtype=float):
            x0 = float(p[0])
            y0 = float(p[1])

            i, _ = self._clip_index(x0, y0)
            U = float(wind_speed_field[i, 0])
            Theta = float(wind_dir_field[i, 0])

            # same directional convention used in propagation
            wx = np.sin(Theta)
            wy = np.cos(Theta)

            dx = XX - x0
            dy = YY - y0

            norm = np.sqrt(dx ** 2 + dy ** 2) + eps
            proj = (dx * wx + dy * wy) / norm
            align = np.maximum(proj, 0.0)  # only downwind contribution
            dist_decay = np.exp(-norm / max(decay_length, eps))

            out += max(U, 0.0) * align * dist_decay

        if normalize:
            out = self._normalize_map01(out)

        return out

    def transition_support_maps(
        self,
        active_fronts: np.ndarray | None,
        geo_phys_info: dict,
    ) -> dict:
        """
        Convenience function returning all support maps needed by the
        recommended reduced probabilistic transition model.

        Returns:
            {
                "spread_rate_map": R_map,
                "intensity_influence_map": I_map,
                "wind_alignment_map": W_map
            }
        """
        R_map = self.spread_rate_map(geo_phys_info)
        I_map = self.front_intensity_influence_map(active_fronts, normalize=True)
        W_map = self.wind_alignment_map(active_fronts, geo_phys_info, normalize=True)

        return {
            "spread_rate_map": R_map,
            "intensity_influence_map": I_map,
            "wind_alignment_map": W_map,
        }