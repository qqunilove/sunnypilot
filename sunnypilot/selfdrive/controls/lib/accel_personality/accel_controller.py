"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom
import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params

AccelPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ACCEL_PERSONALITY_OPTIONS = [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]

MAX_ACCEL_PROFILES = {
  AccelPersonality.eco:    [2.00, 1.55, 1.10, 0.78, 0.56, 0.38, 0.24, 0.10, 0.05],
  AccelPersonality.normal: [2.00, 1.65, 1.25, 0.95, 0.72, 0.52, 0.32, 0.13, 0.07],
  AccelPersonality.sport:  [2.00, 1.80, 1.45, 1.15, 0.90, 0.68, 0.44, 0.18, 0.09],
}
MAX_ACCEL_BREAKPOINTS = [0.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0, 32.0, 42.0]

# Cruise decel floor: base * exp(-decay * v_ego)
MIN_ACCEL_BASE = {
  AccelPersonality.eco:    -0.35,
  AccelPersonality.normal: -0.55,
  AccelPersonality.sport:  -0.90,
}
MIN_ACCEL_DECAY = 0.022  # smooth exponential fade with speed

JERK_ACCEL = 0.50  # accel ceiling rate (m/s² per s)

# Decel floor engagement: very slow — brake authority drifts in, no bite
_DECEL_ON_BP = [0.0,  8.0,  18.0,  32.0]
_DECEL_ON_V  = [0.20, 0.13,  0.08,  0.06]  # m/s² per s

# Decel floor release: slightly faster — lets MPC back off cleanly without trailing clamp
_DECEL_OFF_BP = [0.0,  8.0,  18.0,  32.0]
_DECEL_OFF_V  = [0.22, 0.16,  0.12,  0.09]  # m/s² per s

_MIN_MAX_GAP = 0.05


class AccelPersonalityController:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self.first_run = True
    self.last_max_accel = 2.0
    self.last_min_accel = 0.0
    self._cache_v: float | None = None
    self._cache_min: float = 0.0
    self._cache_max: float = 2.0

    val = self.params.get('AccelPersonality')
    self._accel_personality = val if val is not None else AccelPersonality.normal
    self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  def update(self, sm=None):
    self.frame += 1
    self._cache_v = None
    if self.frame % max(1, int(1.0 / DT_MDL)) == 0:
      val = self.params.get('AccelPersonality')
      self._accel_personality = val if val is not None else AccelPersonality.normal
      self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  @property
  def accel_personality(self) -> int:
    return self._accel_personality

  def get_accel_personality(self) -> int:
    return int(self._accel_personality)

  def set_accel_personality(self, personality: int):
    if personality in ACCEL_PERSONALITY_OPTIONS:
      self._accel_personality = personality
      self.params.put('AccelPersonality', personality)

  def cycle_accel_personality(self) -> int:
    current = self._accel_personality
    idx = ACCEL_PERSONALITY_OPTIONS.index(current) if current in ACCEL_PERSONALITY_OPTIONS else 0
    next_p = ACCEL_PERSONALITY_OPTIONS[(idx + 1) % len(ACCEL_PERSONALITY_OPTIONS)]
    self.set_accel_personality(next_p)
    return int(next_p)

  def _step(self, v_ego: float) -> tuple[float, float]:
    target_max = float(np.interp(v_ego, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[self._accel_personality]))
    target_min = float(MIN_ACCEL_BASE[self._accel_personality] * np.exp(-MIN_ACCEL_DECAY * v_ego))

    if self.first_run:
      self.last_max_accel = target_max
      self.last_min_accel = target_min
      self.first_run = False
      return target_min, target_max

    a_step = JERK_ACCEL * DT_MDL
    new_max = float(np.clip(target_max, self.last_max_accel - a_step, self.last_max_accel + a_step))

    tightening = target_min < self.last_min_accel
    d_rate = float(np.interp(v_ego, _DECEL_ON_BP, _DECEL_ON_V)) if tightening \
         else float(np.interp(v_ego, _DECEL_OFF_BP, _DECEL_OFF_V))
    new_min = float(np.clip(target_min, self.last_min_accel - d_rate * DT_MDL, self.last_min_accel + d_rate * DT_MDL))
    new_min = min(new_min, new_max - _MIN_MAX_GAP)

    self.last_max_accel = new_max
    self.last_min_accel = new_min
    return new_min, new_max

  def get_accel_limits(self, v_ego: float) -> tuple[float, float]:
    v_ego = max(0.0, v_ego)
    if self._cache_v is not None and abs(self._cache_v - v_ego) < 0.01:
      return self._cache_min, self._cache_max
    self._cache_min, self._cache_max = self._step(v_ego)
    self._cache_v = v_ego
    return self._cache_min, self._cache_max

  def get_min_accel(self, v_ego: float) -> float:
    return self.get_accel_limits(v_ego)[0]

  def get_max_accel(self, v_ego: float) -> float:
    return self.get_accel_limits(v_ego)[1]

  def is_enabled(self) -> bool:
    return self._enabled

  def set_enabled(self, enabled: bool):
    self._enabled = bool(enabled)
    self.params.put_bool('AccelPersonalityEnabled', self._enabled)

  def toggle_enabled(self) -> bool:
    self.set_enabled(not self._enabled)
    return self._enabled

  def reset(self, personality: int = None):
    new_p = personality if personality in ACCEL_PERSONALITY_OPTIONS else AccelPersonality.normal
    self._accel_personality = new_p
    self.params.put('AccelPersonality', new_p)
    self.frame = 0
    self.last_max_accel = 2.0
    self.last_min_accel = 0.0
    self._cache_v = None
    self.first_run = True
