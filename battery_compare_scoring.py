import math
from typing import Dict, List, Tuple

import numpy as np


BASE_WEIGHTS: Dict[str, float] = {"mAh": 0.40, "Wh": 0.25, "IR_DC": 0.20, "DeltaV": 0.15}
RECOMMENDED_WEIGHTS: Dict[str, float] = {"mAh": 0.00, "Wh": 0.60, "IR_DC": 0.40, "DeltaV": 0.00}
COMPROMISE_WEIGHTS: Dict[str, float] = {"mAh": 0.20, "Wh": 0.50, "IR_DC": 0.30, "DeltaV": 0.00}


def normalize_higher_better(value: float, v_min: float, v_max: float) -> float:
    if math.isnan(value):
        return 50.0
    if v_max <= v_min:
        return 100.0
    return 10.0 + 90.0 * ((value - v_min) / (v_max - v_min))


def normalize_lower_better(value: float, v_min: float, v_max: float) -> float:
    if math.isnan(value):
        return 50.0
    if v_max <= v_min:
        return 100.0
    return 10.0 + 90.0 * ((v_max - value) / (v_max - v_min))


def _min_max(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if not mask.any():
        return float("nan"), float("nan")
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def compute_compare_scores(
    cap_mah: List[float],
    energy_wh: List[float],
    ir_dc_mohm: List[float],
    delta_v_start: List[float],
    *,
    weights: Dict[str, float],
) -> Dict[str, List[float]]:
    n = len(cap_mah)
    if not (len(energy_wh) == n and len(ir_dc_mohm) == n and len(delta_v_start) == n):
        raise ValueError("All input lists must have the same length.")

    m_min, m_max = _min_max(cap_mah)
    w_min, w_max = _min_max(energy_wh)
    ir_min, ir_max = _min_max(ir_dc_mohm)
    dv_min, dv_max = _min_max(delta_v_start)

    score_mah: List[float] = [normalize_higher_better(float(v), m_min, m_max) for v in cap_mah]
    score_wh: List[float] = [normalize_higher_better(float(v), w_min, w_max) for v in energy_wh]
    score_ir: List[float] = [normalize_lower_better(float(v), ir_min, ir_max) for v in ir_dc_mohm]
    score_dv: List[float] = [normalize_lower_better(float(v), dv_min, dv_max) for v in delta_v_start]

    final_score: List[float] = []
    for i in range(n):
        final_score.append(
            weights.get("mAh", 0.0) * score_mah[i]
            + weights.get("Wh", 0.0) * score_wh[i]
            + weights.get("IR_DC", 0.0) * score_ir[i]
            + weights.get("DeltaV", 0.0) * score_dv[i]
        )

    return {
        "Score mAh": score_mah,
        "Score Wh": score_wh,
        "Score IR DC": score_ir,
        "Score ΔV": score_dv,
        "Final Score": final_score,
    }
