"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import log
import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params

LongPersonality = log.LongitudinalPersonality

FOLLOW_BREAKPOINTS =          [0.,   4.0,  6.0,  11.,  18.,  30.,  40.]

FOLLOW_PROFILES = {
  LongPersonality.relaxed:    [1.60, 1.65, 1.82, 1.88, 2.05, 1.96, 2.12],
  LongPersonality.standard:   [1.35, 1.40, 1.58, 1.64, 1.75, 1.68, 1.78],
  LongPersonality.aggressive: [1.10, 1.15, 1.35, 1.40, 1.50, 1.44, 1.52],
}

SMOOTHING_SPEED_REF = 36.0

# When multiplier needs to grow (lead is slowing — brake side)
_ALPHA_SLOW_BASE  = 0.965   # near-standstill
_ALPHA_SLOW_RANGE = 0.020   # +range at highway = 0.985 max

# When multiplier needs to shrink (lead pulling away — throttle side)
_ALPHA_FAST_BASE  = 0.920
_ALPHA_FAST_RANGE = 0.040   # 0.960 at highway

_ALPHA_MAX = 0.988

PERSONALITY_CHANGE_COOLDOWN_S = 2.0


class FollowDistanceController:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self.current_multiplier = 1.45
    self.first_run = True
    self.personality_change_cooldown = 0
    self.personality_cooldown_frames = int(PERSONALITY_CHANGE_COOLDOWN_S / DT_MDL)
    val = self.params.get('LongitudinalPersonality')
    self._personality = val if val is not None else LongPersonality.standard
    self._enabled = self.params.get_bool('DynamicFollow')

  def _get_alpha(self, v_ego: float, target: float) -> float:
    speed_frac = np.clip(v_ego / SMOOTHING_SPEED_REF, 0.0, 1.0)
    increasing = target > self.current_multiplier
    if increasing:
      alpha = _ALPHA_SLOW_BASE + _ALPHA_SLOW_RANGE * speed_frac
    else:
      alpha = _ALPHA_FAST_BASE + _ALPHA_FAST_RANGE * speed_frac
    return float(min(_ALPHA_MAX, alpha))

  def is_enabled(self) -> bool:
    return self._enabled

  def set_enabled(self, enabled: bool):
    self._enabled = enabled
    self.params.put_bool('DynamicFollow', enabled)

  def toggle(self) -> bool:
    current = self._enabled
    self.set_enabled(not current)
    return not current

  @property
  def personality(self) -> int:
    return self._personality

  def get_personality(self) -> int:
    return int(self._personality)

  def set_personality(self, personality: int):
    if personality not in [LongPersonality.relaxed, LongPersonality.standard, LongPersonality.aggressive]:
      return
    self._personality = personality
    self.params.put('LongitudinalPersonality', personality)
    self.personality_change_cooldown = self.personality_cooldown_frames

  def cycle_personality(self) -> int:
    personalities = [LongPersonality.relaxed, LongPersonality.standard, LongPersonality.aggressive]
    current_idx = personalities.index(self._personality)
    next_p = personalities[(current_idx + 1) % len(personalities)]
    self.set_personality(next_p)
    return int(next_p)

  def get_follow_distance_multiplier(self, v_ego: float) -> float:
    v_ego = max(0.0, v_ego)
    target = float(np.interp(v_ego, FOLLOW_BREAKPOINTS, FOLLOW_PROFILES[self._personality]))

    if self.first_run:
      self.current_multiplier = target
      self.first_run = False
      return float(self.current_multiplier)

    if self.personality_change_cooldown > 0:
      return float(self.current_multiplier)

    alpha = self._get_alpha(v_ego, target)
    self.current_multiplier = alpha * self.current_multiplier + (1.0 - alpha) * target
    return float(self.current_multiplier)

  def reset(self):
    self._personality = LongPersonality.standard
    self.params.put('LongitudinalPersonality', LongPersonality.standard)
    self.frame = 0
    self.current_multiplier = 1.45
    self.first_run = True
    self.personality_change_cooldown = 0

  def update(self):
    self.frame += 1
    if self.personality_change_cooldown > 0:
      self.personality_change_cooldown -= 1
    if self.frame % max(1, int(1.0 / DT_MDL)) == 0:
      val = self.params.get('LongitudinalPersonality')
      self._personality = val if val is not None else LongPersonality.standard
      self._enabled = self.params.get_bool('DynamicFollow')
