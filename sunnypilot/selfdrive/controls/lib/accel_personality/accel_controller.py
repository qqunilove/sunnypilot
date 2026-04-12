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
  AccelPersonality.eco:    [1.60, 1.30, 0.95, 0.70, 0.55, 0.42, 0.30, 0.14, 0.08],
  AccelPersonality.normal: [1.80, 1.50, 1.15, 0.90, 0.72, 0.55, 0.36, 0.16, 0.09],
  AccelPersonality.sport:  [2.00, 1.85, 1.50, 1.20, 0.95, 0.72, 0.46, 0.20, 0.10],
}
MAX_ACCEL_BREAKPOINTS = [0.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0, 32.0, 42.0]

MIN_ACCEL_PROFILES = {
  AccelPersonality.eco:    [-0.45, -0.55, -0.65, -0.75, -0.85, -0.95, -1.10, -1.35],
  AccelPersonality.normal: [-0.55, -0.70, -0.90, -1.10, -1.35, -1.60, -1.85, -2.05],
  AccelPersonality.sport:  [-0.70, -0.90, -1.20, -1.45, -1.80, -2.10, -2.35, -2.60],
}
MIN_ACCEL_BREAKPOINTS = [0.0, 3.0, 6.0, 10.0, 14.0, 20.0, 28.0, 40.0]

JERK_ACCEL_INC = 1.8
JERK_ACCEL_DEC = 1.6
JERK_DECEL_INC = 1.0
JERK_DECEL_DEC = 0.6

_MIN_MAX_GAP = 0.05


class AccelPersonalityController:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self.last_max_accel = 2.0
    self.last_min_accel = 0.0
    self.first_run = True
    val = self.params.get('AccelPersonality')
    self._accel_personality = val if val is not None else AccelPersonality.normal
    self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  def update(self, sm=None):
    self.frame += 1
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
    next_personality = ACCEL_PERSONALITY_OPTIONS[(idx + 1) % len(ACCEL_PERSONALITY_OPTIONS)]
    self.set_accel_personality(next_personality)
    return int(next_personality)

  def get_accel_limits(self, v_ego: float) -> tuple[float, float]:
    v_ego = max(0.0, v_ego)
    target_max = float(np.interp(v_ego, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[self.accel_personality]))
    target_min = float(np.interp(v_ego, MIN_ACCEL_BREAKPOINTS, MIN_ACCEL_PROFILES[self.accel_personality]))
    if target_min > self.last_min_accel:
      target_min *= 0.95

    if self.first_run:
      self.last_max_accel = target_max
      self.last_min_accel = target_min
      self.first_run = False
      return float(target_min), float(target_max)

    max_accel_step_up = JERK_ACCEL_INC * DT_MDL
    max_accel_step_down = JERK_ACCEL_DEC * DT_MDL
    min_accel_step_up = JERK_DECEL_INC * DT_MDL
    min_accel_step_down = JERK_DECEL_DEC * DT_MDL
    self.last_max_accel = float(np.clip(target_max, self.last_max_accel - max_accel_step_down, self.last_max_accel + max_accel_step_up))
    self.last_min_accel = float(np.clip(target_min, self.last_min_accel - min_accel_step_down, self.last_min_accel + min_accel_step_up))

    self.last_min_accel = min(self.last_min_accel, self.last_max_accel - _MIN_MAX_GAP)

    return self.last_min_accel, self.last_max_accel

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
    new_personality = personality if personality in ACCEL_PERSONALITY_OPTIONS else AccelPersonality.normal
    self._accel_personality = new_personality
    self.params.put('AccelPersonality', new_personality)
    self.frame = 0
    self.last_max_accel = 2.0
    self.last_min_accel = 0.0
    self.first_run = True