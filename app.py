import io
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from battery_compare_scoring import (
    BASE_WEIGHTS,
    COMPROMISE_WEIGHTS,
    RECOMMENDED_WEIGHTS,
    compute_compare_scores,
)


# Номінальна ємність (mAh) за моделями — орієнтовні заводські значення (1S Li-ion).
# Користувач може завжди обрати «Вручну» для точного значення.
IPHONE_BATTERY_MAH: Dict[str, int] = {
    "iPhone X": 2716,
    "iPhone Xs": 2658,
    "iPhone Xs Max": 3174,
    "iPhone 11": 3110,
    "iPhone 11 Pro": 3046,
    "iPhone 11 Pro Max": 3969,
    "iPhone 12 mini": 2227,
    "iPhone 12": 2815,
    "iPhone 12 Pro": 2815,
    "iPhone 12 Pro Max": 3687,
    "iPhone 13 mini": 2438,
    "iPhone 13": 3227,
    "iPhone 13 Pro": 3095,
    "iPhone 13 Pro Max": 4352,
    "iPhone 14": 3279,
    "iPhone 14 Plus": 4325,
    "iPhone 14 Pro": 3200,
    "iPhone 14 Pro Max": 4323,
    "iPhone 15": 3349,
    "iPhone 15 Plus": 4383,
    "iPhone 15 Pro": 3274,
    "iPhone 15 Pro Max": 4441,
}


def _detect_header_row_and_sep(text: str) -> Tuple[int, str]:
    lines = text.splitlines()
    header_idx = None
    header_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("DATE") and "VOLTAGE" in stripped:
            header_idx = i
            header_line = stripped
            break
    if header_idx is None or header_line is None:
        raise ValueError("Не вдалося знайти рядок заголовка (очікується 'DATE,VOLTAGE(V),...').")

    candidates = [",", ";", "\t"]
    best_sep = max(candidates, key=lambda s: header_line.count(s))
    return header_idx, best_sep


def load_battery_csv(file_bytes: bytes, filename_hint: str = "data.csv") -> pd.DataFrame:
    text = file_bytes.decode("utf-8-sig", errors="ignore")
    header_idx, sep = _detect_header_row_and_sep(text)

    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(buf, sep=sep, skiprows=header_idx, header=0, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]

    date_candidates = [c for c in df.columns if c.strip().upper() == "DATE"]
    if date_candidates:
        x_col = date_candidates[0]
        s = df[x_col].astype(str)
        s2 = s.str.replace("_", " ", regex=False)
        parsed = pd.to_datetime(s2, errors="coerce", format="%Y-%m-%d %H:%M:%S")
        mask = parsed.isna()
        if mask.any():
            parsed2 = pd.to_datetime(s2[mask], errors="coerce")
            parsed.loc[mask] = parsed2
        df[x_col] = parsed

    for c in df.columns:
        if c == "DATE":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_from_path(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не знайдено: {path}")
    with open(path, "rb") as f:
        return load_battery_csv(f.read(), filename_hint=os.path.basename(path))


def downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, math.ceil(len(df) / max_points))
    return df.iloc[::step].reset_index(drop=True)


def _plotly_vline_x(t: Union[pd.Timestamp, datetime, Any]) -> datetime:
    """
    Нормалізуємо час до datetime (для Plotly time-осі).
    """
    if t is None or (isinstance(t, (float, int)) and pd.isna(t)):
        raise ValueError("Некоректний час для вертикальної лінії")
    ts = pd.Timestamp(t)
    if pd.isna(ts):
        raise ValueError("Некоректний час для вертикальної лінії")
    return ts.to_pydatetime()


def _add_vertical_time_marker(
    fig: go.Figure,
    t: datetime,
    *,
    color: str,
    label: str,
) -> None:
    """
    Вертикальна лінія на time-осі.
    Не використовуємо fig.add_vline(x=...) — у Plotly він викликає sum/mean по x
    і падає на pandas.Timestamp / datetime у нових версіях pandas/plotly.
    """
    fig.add_shape(
        type="line",
        x0=t,
        x1=t,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=color, width=2, dash="dash"),
    )
    fig.add_annotation(
        x=t,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(color=color, size=11),
        yshift=0,
    )


def discharge_current_a(series: pd.Series) -> pd.Series:
    """Струм розряду як додатне значення (А): для від'ємного струму в CSV — беремо -I."""
    return series.abs()


@dataclass
class DischargeMetrics:
    t_start_pos: int
    t_end_pos: int
    v_start_idle_v: float  # середнє за 2 с до старту (усі точки у вікні)
    v_after_avg_v: float  # середнє за 2 с після старту
    v_cutoff_v: float  # напруга в точці відсічки
    i_discharge_avg_a: float  # середній струм розряду на інтервалі тесту
    capacity_mah: float
    energy_wh: float
    v_avg_load_v: float
    sag_v: float  # ΔV = V_before − V_after (за ТЗ)
    r_internal_ohm: float  # R = ΔV / I; I = i_stable_sag_a або середній струм розряду
    duration_s: float
    notes: str
    limited_before_window: bool = False  # True якщо V_before з обмежених точок, не з повного 2 с вікна
    cutoff_not_reached: bool = False  # True якщо U≤cutoff у файлі не було — кінець по останній точці
    i_stable_sag_a: float = float("nan")  # струм після стабілізації (для R=ΔV/I без вікна до старту)
    v_before_manual_used: bool = False  # True якщо V_before взято з UI (тільки коли нема даних у CSV до старту)
    no_idle_data_no_manual: bool = False  # True якщо нема даних до старту і користувач не ввів V_before вручну


def _current_for_internal_r(m: "DischargeMetrics") -> float:
    """Для розрахунку R: стабільний струм після просадки, якщо задано, інакше середній струм розряду."""
    if not math.isnan(m.i_stable_sag_a) and abs(m.i_stable_sag_a) > 1e-12:
        return m.i_stable_sag_a
    return m.i_discharge_avg_a


def _compute_sag_limited_no_idle(
    work: pd.DataFrame,
    t_start_pos: int,
    i_threshold: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Немає даних до старту навантаження (немає вікна «до»):
    старт — перший рядок з |I| ≥ порога; V_before = max(V) у перших до 5 точках;
    I_mean — середнє за перші 5–10 точок; стабілізація: |I − I_mean| ≤ 10%·I_mean;
    V_after — середнє за 2–3 точки після виходу струму на стабільний рівень;
    I_stable — середній |I| у цьому ж короткому вікні;
    ΔV = V_before − V_after.
    """
    n = len(work)
    if t_start_pos >= n:
        return None
    max_init = 10
    end_init = min(t_start_pos + max_init, n)
    init = work.iloc[t_start_pos:end_init]
    if len(init) < 1:
        return None

    n_vmax = min(5, len(init))
    v_before = float(init["VOLTAGE(V)"].iloc[:n_vmax].astype(float).max())

    i_init = discharge_current_a(init["CURRENT(A)"])
    i_mean = float(i_init.mean())
    if math.isnan(i_mean) or i_mean < 1e-9:
        i_mean = max(float(i_threshold), 1e-6)
    tol = 0.1 * i_mean

    scan_from = t_start_pos + min(5, max(0, len(init) - 1))
    if scan_from >= n:
        scan_from = t_start_pos

    stable_start_pos: Optional[int] = None
    for j in range(scan_from, n):
        ia = float(discharge_current_a(work.iloc[j : j + 1]["CURRENT(A)"]).iloc[0])
        if abs(ia - i_mean) <= tol:
            stable_start_pos = j
            break
    if stable_start_pos is None:
        stable_start_pos = min(scan_from, n - 1)

    t_stable = work.at[stable_start_pos, "DATE"]
    if pd.isna(t_stable):
        return None

    # Візьмемо 2–3 точки ПІСЛЯ стабілізації струму (а не включаючи сам момент стабілізації).
    after_start = stable_start_pos + 1
    after_end = min(after_start + 3, n)
    stable_slice = work.iloc[after_start:after_end]
    if stable_slice.empty or len(stable_slice) < 2:
        return None

    v_after = float(stable_slice["VOLTAGE(V)"].astype(float).mean())
    i_stable = float(discharge_current_a(stable_slice["CURRENT(A)"]).mean())
    if math.isnan(v_after) or math.isnan(i_stable):
        return None

    sag = v_before - v_after
    return (v_before, v_after, sag, i_stable)


def compute_discharge_metrics(
    df: pd.DataFrame,
    v_cutoff: float,
    i_threshold: float,
    before_sec: float,
    after_sec: float,
    v_before_manual: Optional[float] = None,
) -> Tuple[Optional[DischargeMetrics], str]:
    """
    t_start: перший рядок, де |I| >= i_threshold (початок значного розряду).
    Вікна: [t_start - before_sec, t_start), [t_start, t_start + after_sec].
    t_end: перший рядок після старту, де U <= v_cutoff.
    """
    required = ["DATE", "VOLTAGE(V)", "CURRENT(A)", "E_CAPACITY(mAh)", "E_QUANTITY(Wh)"]
    for c in required:
        if c not in df.columns:
            return None, f"У файлі немає колонки «{c}»."

    work = df.sort_values("DATE").reset_index(drop=True)
    work = work.dropna(subset=["DATE", "VOLTAGE(V)", "CURRENT(A)"])
    if len(work) < 3:
        return None, "Замало рядків після очищення."

    i_abs = discharge_current_a(work["CURRENT(A)"])
    mask = i_abs >= i_threshold
    if not mask.any():
        return None, f"Не знайдено старту: жодного рядка з |I| ≥ {i_threshold} A."

    t_start_pos = int(np.flatnonzero(mask.to_numpy())[0])
    t_start_time = work.at[t_start_pos, "DATE"]
    if pd.isna(t_start_time):
        return None, "Некоректний час у точці старту."

    td_before = pd.Timedelta(seconds=before_sec)
    td_after = pd.Timedelta(seconds=after_sec)

    before_mask = (work["DATE"] >= t_start_time - td_before) & (work["DATE"] < t_start_time)
    after_mask = (work["DATE"] >= t_start_time) & (work["DATE"] <= t_start_time + td_after)

    before_df = work.loc[before_mask]
    after_df = work.loc[after_mask]

    limited_before_window = False
    warn_limited = "Запис почався після старту навантаження — просадка розрахована приблизно."

    i_stable_sag_a = float("nan")
    v_before_manual_used = False
    no_idle_data_no_manual = False

    # Наявність даних "до старту" у CSV (у нас старт = перший рядок з |I| >= i_threshold).
    # Якщо t_start_pos == 0, то "до старту" в CSV немає взагалі.
    idle_before_any = bool((t_start_pos > 0) and (i_abs.iloc[:t_start_pos] < i_threshold).any())

    if not before_df.empty:
        v_before = float(before_df["VOLTAGE(V)"].mean())
        if after_df.empty:
            return None, "Немає даних за 2 с після старту."
        v_after_avg = float(after_df["VOLTAGE(V)"].mean())
        sag = v_before - v_after_avg
    else:
        limited_before_window = True
        if idle_before_any:
            # Дефіцит "before_sec", але дані до старту (з |I| < I_threshold) у CSV є.
            pre = work.iloc[:t_start_pos]
            ia_pre = discharge_current_a(pre["CURRENT(A)"])
            idle_pre = pre.loc[ia_pre < i_threshold]
            if not idle_pre.empty:
                take = idle_pre.iloc[:5]
                v_before = float(take["VOLTAGE(V)"].mean())
            else:
                take = pre.iloc[: min(5, len(pre))]
                v_before = float(take["VOLTAGE(V)"].mean())
            if after_df.empty:
                return None, "Немає даних за 2 с після старту."
            v_after_avg = float(after_df["VOLTAGE(V)"].mean())
            sag = v_before - v_after_avg
        else:
            # Немає даних до старту (CSV стартує вже під навантаженням).
            # Якщо користувач дав V_before — рахуємо ΔV та IR, інакше ставимо nan і покажемо warning у UI.
            res = _compute_sag_limited_no_idle(work, t_start_pos, i_threshold)
            if res is None:
                return None, "Немає даних для розрахунку просадки після виходу струму на стабільний рівень."
            _v_before_est, v_after_avg, _sag_est, i_stable_sag_a = res

            if v_before_manual is not None and not math.isnan(v_before_manual):
                v_before = float(v_before_manual)
                sag = v_before - v_after_avg
                v_before_manual_used = True
            else:
                v_before = float("nan")
                sag = float("nan")
                no_idle_data_no_manual = True

    # Відсічка: перший рядок після старту з U <= v_cutoff; якщо ніколи не досягнуто — останній рядок файлу
    cutoff_not_reached = False
    cutoff_approx_from_last = False
    warn_cutoff = (
        f"Тест завершено раніше, ніж напруга досягла {v_cutoff} V — результати можуть бути трохи завищені."
    )

    # Толеранс на "приблизно дорівнює cutoff" (щоб не ловити warning через дискретність/округлення).
    cutoff_eps_v = 0.005

    t_end_pos: Optional[int] = None
    for i in range(t_start_pos + 1, len(work)):
        v = work.at[i, "VOLTAGE(V)"]
        if pd.notna(v) and float(v) <= v_cutoff:
            t_end_pos = i
            break
    if t_end_pos is None:
        t_end_pos = len(work) - 1
        if t_end_pos < t_start_pos:
            return None, "Немає даних після старту для розрахунку кінця тесту."

        v_last = work.at[t_end_pos, "VOLTAGE(V)"]
        if pd.notna(v_last) and float(v_last) <= v_cutoff + cutoff_eps_v:
            # Якщо остання точка майже дорівнює cutoff — вважаємо, що відсічку досягнуто.
            cutoff_not_reached = False
            cutoff_approx_from_last = True
        else:
            cutoff_not_reached = True

    segment = work.iloc[t_start_pos : t_end_pos + 1]
    i_seg = discharge_current_a(segment["CURRENT(A)"])
    v_seg = segment["VOLTAGE(V)"]

    # Середній струм розряду на активному інтервалі (де |I| >= поріг)
    active = i_seg >= i_threshold
    if active.any():
        i_discharge_avg = float(i_seg.loc[active].mean())
    else:
        i_discharge_avg = float(i_seg.mean())

    cap0 = work.at[t_start_pos, "E_CAPACITY(mAh)"]
    cap1 = work.at[t_end_pos, "E_CAPACITY(mAh)"]
    e0 = work.at[t_start_pos, "E_QUANTITY(Wh)"]
    e1 = work.at[t_end_pos, "E_QUANTITY(Wh)"]

    capacity_mah = float(cap1 - cap0) if pd.notna(cap0) and pd.notna(cap1) else float("nan")
    energy_wh = float(e1 - e0) if pd.notna(e0) and pd.notna(e1) else float("nan")

    v_avg_load = float(v_seg.loc[i_seg >= i_threshold].mean()) if (i_seg >= i_threshold).any() else float(v_seg.mean())
    v_cutoff_actual = float(work.at[t_end_pos, "VOLTAGE(V)"])

    # ΔV = V_before − V_after; R = ΔV / I_stable (без вікна до старту) або / I_розряду
    i_for_r = (
        i_stable_sag_a
        if not math.isnan(i_stable_sag_a) and abs(i_stable_sag_a) > 1e-12
        else i_discharge_avg
    )
    if abs(i_for_r) > 1e-12:
        r_int = sag / i_for_r
    else:
        r_int = float("nan")

    t0 = work.at[t_start_pos, "DATE"]
    t1 = work.at[t_end_pos, "DATE"]
    duration_s = float((t1 - t0).total_seconds()) if pd.notna(t0) and pd.notna(t1) else float("nan")

    notes = (
        f"Старт: |I|≥{i_threshold} A; відсічка: U≤{v_cutoff} V; "
        f"вікна: {before_sec} с до / {after_sec} с після старту."
    )
    if limited_before_window:
        # Якщо немає даних до старту і ми НЕ маємо ручної напруги — ΔV та IR не рахуємо,
        # тож тут не показуємо повідомлення про "приблизний" прорахунок.
        if not no_idle_data_no_manual and not v_before_manual_used:
            notes = notes + " " + warn_limited
        if v_before_manual_used and not math.isnan(i_stable_sag_a):
            notes += (
                " Просадка розрахована по ручному V_before та стабільному V_after після виходу струму на рівень."
            )
    if cutoff_not_reached:
        notes = notes + " " + warn_cutoff
    if cutoff_approx_from_last:
        notes = notes + " Відсічка визначена по останній точці (U майже дорівнює cutoff)."

    return (
        DischargeMetrics(
            t_start_pos=t_start_pos,
            t_end_pos=t_end_pos,
            v_start_idle_v=v_before,
            v_after_avg_v=v_after_avg,
            v_cutoff_v=v_cutoff_actual,
            i_discharge_avg_a=i_discharge_avg,
            capacity_mah=capacity_mah,
            energy_wh=energy_wh,
            v_avg_load_v=v_avg_load,
            sag_v=sag,
            r_internal_ohm=r_int,
            duration_s=duration_s,
            notes=notes,
            limited_before_window=limited_before_window,
            cutoff_not_reached=cutoff_not_reached,
            i_stable_sag_a=i_stable_sag_a,
            v_before_manual_used=v_before_manual_used,
            no_idle_data_no_manual=no_idle_data_no_manual,
        ),
        "",
    )


def score_capacity_pct(pct: float) -> float:
    """Оцінка 1–10 за відсотком від номінальної ємності."""
    if math.isnan(pct):
        return float("nan")
    if pct >= 95:
        return 10.0
    if pct >= 90:
        return 9.0
    if pct >= 85:
        return 8.0
    if pct >= 80:
        return 7.0
    if pct >= 75:
        return 6.0
    if pct >= 70:
        return 5.0
    if pct >= 60:
        return 4.0
    return 3.0


def score_energy_pct(pct: float) -> float:
    """Оцінка 1–10 за відсотком від номінальної енергії (інші пороги для <75%)."""
    if math.isnan(pct):
        return float("nan")
    if pct >= 95:
        return 10.0
    if pct >= 90:
        return 9.0
    if pct >= 85:
        return 8.0
    if pct >= 80:
        return 7.0
    if pct >= 75:
        return 6.0
    if pct >= 70:
        return 5.0
    return 4.0


def score_v_at_50pct(v: float) -> float:
    """Оцінка напруги при 50% номіналу (окрема шкала за ТЗ v2.1)."""
    if math.isnan(v):
        return float("nan")
    if v > 3.75:
        return 10.0
    if v >= 3.70:
        return 9.0
    if v >= 3.65:
        return 8.0
    if v >= 3.60:
        return 7.0
    if v >= 3.55:
        return 6.0
    if v >= 3.50:
        return 5.0
    if v >= 3.45:
        return 4.0
    return 3.0


def score_vavg(v: float) -> float:
    if math.isnan(v):
        return float("nan")
    if v > 3.75:
        return 10.0
    if v >= 3.70:
        return 9.0
    if v >= 3.65:
        return 8.0
    if v >= 3.60:
        return 7.0
    if v >= 3.55:
        return 6.0
    if v >= 3.50:
        return 5.0
    if v >= 3.40:
        return 4.0
    return 3.0


def score_sag(sag_v: float) -> float:
    if math.isnan(sag_v):
        return float("nan")
    if sag_v < 0.08:
        return 10.0
    if sag_v < 0.12:
        return 9.0
    if sag_v < 0.15:
        return 8.0
    if sag_v < 0.18:
        return 7.0
    if sag_v < 0.22:
        return 6.0
    if sag_v < 0.25:
        return 5.0
    if sag_v <= 0.30:
        return 4.0
    return 3.0


def score_ir_mohm(r_mohm: float) -> float:
    """Оцінка 1–10 за внутрішнім опором (мОм): нижче mΩ — краще. Чутливіша сітка, ніж раніше."""
    if math.isnan(r_mohm):
        return float("nan")
    if r_mohm < 60:
        return 10.0
    if r_mohm < 80:
        return 9.0
    if r_mohm < 100:
        return 8.0
    if r_mohm < 120:
        return 7.0
    if r_mohm < 150:
        return 6.0
    if r_mohm < 180:
        return 5.0
    if r_mohm < 220:
        return 4.0
    if r_mohm < 260:
        return 3.0
    if r_mohm < 320:
        return 2.0
    return 1.0


def category_from_score(score: float) -> str:
    """9–10 Ідеал, 8–9 Дуже хороша, … (за загальним балом)."""
    if math.isnan(score):
        return "—"
    if score >= 9:
        return "Ідеал"
    if score >= 8:
        return "Дуже хороша"
    if score >= 7:
        return "Норм"
    if score >= 6:
        return "Середня"
    if score >= 5:
        return "Слабка"
    return "Під заміну"


def score_band_color_hex(score: float) -> str:
    """
    Колір за оцінкою (UX v2.2):
    8–10 зелений, 6–8 жовтий, 4–6 оранжевий, <4 червоний.
    """
    if math.isnan(score):
        return "#95a5a6"
    if score >= 8:
        return "#2ecc71"
    if score >= 6:
        return "#f1c40f"
    if score >= 4:
        return "#e67e22"
    return "#e74c3c"


def score_band_label_ua(score: float) -> str:
    """Підпис для індикатора (узгоджено з шкалою)."""
    if math.isnan(score):
        return "—"
    if score >= 8:
        return "добре (8–10)"
    if score >= 6:
        return "норм (6–8)"
    if score >= 4:
        return "середньо (4–6)"
    return "погано (<4)"


def c_rate_label_and_color(c: float) -> Tuple[str, str, str]:
    """
    C-rate: колір узгоджений зі шкалою (зел/жовт/черв).
    0.2–0.7C норма; 0.7–1C підвищене, але допустиме; >1C стрес; <0.2C слабкий тест.
    """
    if math.isnan(c):
        return "—", "#95a5a6", "—"
    if c < 0.2:
        return "слабкий тест (низький струм)", "#f1c40f", "yellow"
    if c <= 0.7:
        return "нормальний тест", "#2ecc71", "green"
    if c <= 1.0:
        return "підвищене, але допустиме навантаження", "#f1c40f", "yellow"
    return "стрес-тест", "#e74c3c", "red"


def headline_from_score(score: float) -> str:
    """Один головний рядок замість дублювання з «Категорія»."""
    if math.isnan(score):
        return "—"
    if score >= 8:
        return "✅ АКБ в дуже хорошому стані"
    if score >= 6:
        return "⚠️ АКБ в нормальному стані"
    if score >= 4:
        return "⚠️ АКБ в середньому стані"
    return "❌ АКБ потребує заміни"


def r_mohm_from_metrics(m: DischargeMetrics) -> float:
    i_use = _current_for_internal_r(m)
    if abs(i_use) < 1e-12:
        return float("nan")
    return abs(m.sag_v / i_use) * 1000.0


def style_scores_dataframe(df: pd.DataFrame) -> Any:
    """Підсвітка колонки «Оцінка» за score (4-смугова шкала)."""

    def _style_row(row: pd.Series) -> List[str]:
        keys = list(row.index)
        out = [""] * len(keys)
        if "Оцінка" not in keys or "Метрика" not in keys:
            return out
        ots = row["Оцінка"]
        metric = str(row["Метрика"])
        if not isinstance(ots, str):
            return out
        s = ots.strip()
        if s == "—" or not s.endswith("/10"):
            return out
        try:
            sc = float(s.replace("/10", "").strip())
        except ValueError:
            return out
        i = keys.index("Оцінка")
        h = score_band_color_hex(sc)
        out[i] = f"color: {h}; font-weight: 600"
        return out

    try:
        return df.style.apply(_style_row, axis=1)  # type: ignore[arg-type]
    except Exception:
        return df


def ideal_deviation_lines(
    pct_mah: float,
    r_mohm: float,
    sag_v: float,
    has_nominal: bool,
) -> List[str]:
    """Людяні формулювання: «на X% нижче/вище … (краще/гірше)»."""
    out: List[str] = []
    if has_nominal and not math.isnan(pct_mah):
        if pct_mah > 100:
            d = pct_mah - 100.0
            out.append(f"Ємність на **{d:.0f}%** вище за 100% номіналу **(краще)**")
        elif pct_mah < 100:
            d = 100.0 - pct_mah
            out.append(f"Ємність на **{d:.0f}%** нижча за номінал **(гірше)**")
        else:
            out.append("Ємність **на рівні** номіналу")
    if not math.isnan(r_mohm):
        ideal_ir = 80.0
        if r_mohm < ideal_ir:
            pct = (ideal_ir - r_mohm) / ideal_ir * 100.0
            out.append(
                f"Внутрішній опір на **{pct:.0f}%** нижчий за орієнтир (~{ideal_ir:.0f} mΩ) **(краще)**"
            )
        elif r_mohm > ideal_ir:
            pct = (r_mohm - ideal_ir) / ideal_ir * 100.0
            out.append(
                f"Внутрішній опір на **{pct:.0f}%** вищий за орієнтир (~{ideal_ir:.0f} mΩ) **(гірше)**"
            )
        else:
            out.append(f"Внутрішній опір **на рівні** орієнтиру (~{ideal_ir:.0f} mΩ)")
    if not math.isnan(sag_v):
        ideal_sag = 0.1
        if sag_v < ideal_sag:
            pct = (ideal_sag - sag_v) / ideal_sag * 100.0
            out.append(
                f"Просадка на **{pct:.0f}%** менша за орієнтир (~{ideal_sag:.1f} V) **(краще)**"
            )
        elif sag_v > ideal_sag:
            pct = (sag_v - ideal_sag) / ideal_sag * 100.0
            out.append(
                f"Просадка на **{pct:.0f}%** більша за орієнтир (~{ideal_sag:.1f} V) **(гірше)**"
            )
        else:
            out.append(f"Просадка **на рівні** орієнтиру (~{ideal_sag:.1f} V)")
    return out


def compute_voltage_at_nominal_fraction(
    df: pd.DataFrame,
    t_start_pos: int,
    t_end_pos: int,
    nominal_mah: float,
    fraction: float = 0.5,
) -> float:
    """Напруга при досягненні fraction×номіналу ємності (за лічильником mAh у CSV)."""
    if nominal_mah <= 0:
        return float("nan")
    work = df.sort_values("DATE").reset_index(drop=True)
    seg = work.iloc[t_start_pos : t_end_pos + 1]
    if seg.empty or "E_CAPACITY(mAh)" not in seg.columns or "VOLTAGE(V)" not in seg.columns:
        return float("nan")
    cap = seg["E_CAPACITY(mAh)"].astype(float)
    v = seg["VOLTAGE(V)"].astype(float)
    if cap.isna().all() or v.isna().all():
        return float("nan")
    c0 = float(cap.iloc[0])
    target = c0 + nominal_mah * fraction
    mask = (cap >= target).to_numpy()
    if mask.any():
        first_i = int(np.flatnonzero(mask)[0])
        return float(v.iloc[first_i])
    return float("nan")


def _join_issue_phrases(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} та {parts[1]}"
    return ", ".join(parts[:-1]) + f" та {parts[-1]}"


def build_verdict_text(
    m: DischargeMetrics,
    r_mohm: float,
    pct_mah: float,
    has_nominal: bool,
    total_score: float,
) -> str:
    """Комбінований вердикт: 2–3 ключові моменти + висновок (ТЗ v2.1)."""
    issues: List[str] = []
    if not math.isnan(r_mohm) and r_mohm > 180:
        issues.append("підвищений внутрішній опір")
    if m.sag_v > 0.18:
        issues.append("є просадка під навантаженням")
    if has_nominal and not math.isnan(pct_mah) and pct_mah < 80:
        issues.append("знижена ємність")

    if not has_nominal or math.isnan(pct_mah):
        opener = "За результатами тесту"
    elif pct_mah >= 80:
        opener = f"АКБ має нормальну ємність (~{pct_mah:.0f}%)"
    else:
        opener = f"АКБ має знижену ємність (~{pct_mah:.0f}% від номіналу)"

    if issues:
        joined = _join_issue_phrases(issues[:3])
        if has_nominal and not math.isnan(pct_mah) and pct_mah >= 80:
            body = f"{opener}, але {joined}."
        elif has_nominal and not math.isnan(pct_mah) and pct_mah < 80:
            body = f"{opener}; також {joined}."
        else:
            body = f"{opener}: {joined}."
    else:
        body = f"{opener}."

    if math.isnan(total_score):
        summary = " Задайте номінал АКБ для повного оцінювання."
    elif total_score >= 8:
        summary = " Підійде для впевненого повсякденного використання."
    elif total_score >= 6:
        summary = " Підійде для звичайного використання, але під навантаженням можливі просадки."
    else:
        summary = " Рекомендовано розглянути заміну або обмежити навантаження."

    return (body + summary).strip()


def weighted_total_score_fixed(
    s_mah: float,
    s_wh: float,
    s_v: float,
    s_sag: float,
    s_ir: float,
) -> float:
    """
    Завжди 5 компонентів, сума ваг = 8:
    Score = (mAh×1 + Wh×1.5 + Vavg×1.5 + Sag×2 + IR×2) / 8
    """
    if any(math.isnan(x) for x in (s_mah, s_wh, s_v, s_sag, s_ir)):
        return float("nan")
    return (s_mah * 1.0 + s_wh * 1.5 + s_v * 1.5 + s_sag * 2.0 + s_ir * 2.0) / 8.0


def build_scores_and_table(
    m: DischargeMetrics,
    nominal_mah: float,
    nominal_wh: float,
    *,
    has_nominal: bool,
    c_rate: float = float("nan"),
    v_at_50_nominal: float = float("nan"),
) -> Tuple[pd.DataFrame, Dict[str, float], float, str, str, str, str]:
    """
    has_nominal: задано номінал mAh (і Wh) — усі 5 оцінок і загальний бал за оцінками, не за сирими значеннями.
    c_rate: I / (номінал/1000); v_at_50_nominal: напруга при 0.5×номінал mAh на лічильнику (опційно).
    """
    # R з тих самих ΔV та I, що в таблиці: I = I_stable після просадки або середній струм
    delta_v = m.sag_v
    i_dis = _current_for_internal_r(m)
    if abs(i_dis) > 1e-12:
        r_ohm_calc = delta_v / i_dis
    else:
        r_ohm_calc = float("nan")
    r_mohm = abs(r_ohm_calc) * 1000.0 if not math.isnan(r_ohm_calc) else float("nan")

    pct_mah = (m.capacity_mah / nominal_mah * 100.0) if nominal_mah > 0 and not math.isnan(m.capacity_mah) else float("nan")
    pct_wh = (m.energy_wh / nominal_wh * 100.0) if nominal_wh > 0 and not math.isnan(m.energy_wh) else float("nan")

    if has_nominal and not math.isnan(pct_mah):
        s_mah = score_capacity_pct(pct_mah)
    else:
        s_mah = float("nan")
    if has_nominal and not math.isnan(pct_wh):
        s_wh = score_energy_pct(pct_wh)
    else:
        s_wh = float("nan")

    s_v = score_vavg(m.v_avg_load_v)
    sag_for_score = max(0.0, m.sag_v) if not math.isnan(m.sag_v) else float("nan")
    s_sag = score_sag(sag_for_score)
    s_ir = score_ir_mohm(r_mohm)

    s_v50 = (
        score_v_at_50pct(v_at_50_nominal)
        if has_nominal and not math.isnan(v_at_50_nominal)
        else float("nan")
    )

    scores: Dict[str, float] = {
        "mAh": s_mah,
        "Wh": s_wh,
        "Vavg": s_v,
        "Sag": s_sag,
        "IR": s_ir,
        "V50": s_v50,
    }

    def fmt_score(x: float) -> str:
        if math.isnan(x):
            return "—"
        return f"{x:.0f}/10"

    def fmt_val_mah() -> str:
        if math.isnan(m.capacity_mah):
            return "—"
        if has_nominal and not math.isnan(pct_mah):
            return f"{m.capacity_mah:.0f} mAh ({pct_mah:.0f}%)"
        return f"{m.capacity_mah:.0f} mAh"

    def fmt_val_wh() -> str:
        if math.isnan(m.energy_wh):
            return "—"
        if has_nominal and not math.isnan(pct_wh):
            return f"{m.energy_wh:.3f} Wh ({pct_wh:.0f}%)"
        return f"{m.energy_wh:.4f} Wh"

    if has_nominal and not math.isnan(c_rate):
        _crate_txt, crate_hex, _ = c_rate_label_and_color(c_rate)
        crate_val_plain = f"{c_rate:.2f}C — {_crate_txt}"
        crate_html = (
            f'<p style="margin:0.4rem 0 0 0"><span style="color:{crate_hex};font-weight:600;font-size:1.05rem">'
            f"C-rate: {c_rate:.2f}C — {_crate_txt}</span></p>"
        )
    else:
        crate_val_plain = "—"
        crate_html = ""

    if m.v_before_manual_used:
        label_v_start = "Стартова напруга (введено вручну)"
    elif m.no_idle_data_no_manual:
        label_v_start = "Стартова напруга (немає даних до старту)"
    else:
        label_v_start = (
            "Стартова напруга (запис почався під навантаженням — оцінка приблизна)"
            if m.limited_before_window
            else "Стартова напруга (середнє за 2 с до старту)"
        )
    label_cutoff = (
        "Кінцева напруга (cutoff не досягнуто)"
        if m.cutoff_not_reached
        else "Напруга відсічки"
    )
    rows: List[Dict[str, Any]] = [
        {
            "Метрика": label_v_start,
            "Значення": f"{m.v_start_idle_v:.4f} V" if not math.isnan(m.v_start_idle_v) else "—",
            "Оцінка": "—",
        },
        {
            "Метрика": label_cutoff,
            "Значення": f"{m.v_cutoff_v:.4f} V",
            "Оцінка": "—",
        },
        {
            "Метрика": "Середній струм розряду",
            "Значення": f"{m.i_discharge_avg_a:.4f} A",
            "Оцінка": "—",
        },
        {
            "Метрика": "Віддана ємність",
            "Значення": fmt_val_mah(),
            "Оцінка": fmt_score(s_mah) if has_nominal else "—",
        },
        {
            "Метрика": "Віддана енергія",
            "Значення": fmt_val_wh(),
            "Оцінка": fmt_score(s_wh) if has_nominal else "—",
        },
        {
            "Метрика": "Середня напруга під навантаженням",
            "Значення": f"{m.v_avg_load_v:.4f} V",
            "Оцінка": fmt_score(s_v),
        },
        {
            "Метрика": "Просадка напруги на старті",
            "Значення": f"{m.sag_v:.4f} V" if not math.isnan(m.sag_v) else "—",
            "Оцінка": fmt_score(s_sag),
        },
        {
            "Метрика": "Внутрішній опір",
            "Значення": f"{r_mohm:.1f} mΩ" if not math.isnan(r_mohm) else "—",
            "Оцінка": fmt_score(s_ir),
        },
        {
            "Метрика": "Час тесту",
            "Значення": f"{m.duration_s:.1f} с ({m.duration_s / 60:.2f} хв)",
            "Оцінка": "—",
        },
        {
            "Метрика": "C-rate (режим навантаження)",
            "Значення": crate_val_plain,
            "Оцінка": "—",
        },
    ]

    if has_nominal and not math.isnan(v_at_50_nominal):
        rows.append(
            {
                "Метрика": "Напруга при 50% номіналу (лічильник mAh)",
                "Значення": f"{v_at_50_nominal:.4f} V",
                "Оцінка": fmt_score(s_v50),
            }
        )

    if has_nominal:
        total = weighted_total_score_fixed(s_mah, s_wh, s_v, s_sag, s_ir)
        w_note = "(5/5 оцінок: score_mAh…score_IR, вага 1+1.5+1.5+2+2 = 8)"
    else:
        total = float("nan")
        w_note = ""

    cat = category_from_score(total) if has_nominal else "—"
    verdict = build_verdict_text(m, r_mohm, pct_mah, has_nominal, total)

    return pd.DataFrame(rows), scores, total, w_note, cat, verdict, crate_html


def _detect_temperature_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        u = str(c).upper()
        if "NTC" in u or "TEMP" in u or "ТЕМП" in u:
            return c
    return None


def _batch_compare_score_100(
    cap_mah: List[float],
    energy_wh: List[float],
    r_mohm: List[float],
    sag_v: List[float],
) -> Dict[str, List[float]]:
    return compute_compare_scores(
        cap_mah,
        energy_wh,
        r_mohm,
        sag_v,
        weights=BASE_WEIGHTS,
    )


def _compare_auto_summary(
    name: str,
    cap: float,
    wh: float,
    r_mohm: float,
    sag: float,
    v_load: float,
    caps: List[float],
    whs: List[float],
    irs: List[float],
    sags: List[float],
    vloads: List[float],
) -> str:
    parts: List[str] = []
    if not any(math.isnan(x) for x in caps) and not math.isnan(cap):
        if cap >= max(caps) - 1e-6:
            parts.append("найкраща віддана ємність у порівнянні")
        elif cap <= min(caps) + 1e-6:
            parts.append("найнижча ємність у порівнянні")
    if not any(math.isnan(x) for x in whs) and not math.isnan(wh):
        if wh >= max(whs) - 1e-9:
            parts.append("найкраща енергія")
    if not any(math.isnan(x) for x in irs) and not math.isnan(r_mohm):
        if r_mohm <= min(irs) + 1e-6:
            parts.append("найнижчий внутрішній опір")
        elif r_mohm >= max(irs) - 1e-6:
            parts.append("найвищий внутрішній опір")
    if not any(math.isnan(x) for x in sags) and not math.isnan(sag):
        if sag <= min(sags) + 1e-6:
            parts.append("мінімальна стартова просадка")
        elif sag >= max(sags) - 1e-6:
            parts.append("вища просадка напруги")
    if not any(math.isnan(x) for x in vloads) and not math.isnan(v_load):
        if v_load >= max(vloads) - 1e-6:
            parts.append("вища середня напруга під навантаженням")
    if not parts:
        return f"{name}: типовий профіль у межах порівняння."
    return f"{name}: " + "; ".join(parts[:4]) + "."


def style_compare_table(
    df: pd.DataFrame,
    highlight_cols: List[str],
    *,
    invert_green_red: Optional[List[str]] = None,
) -> Any:
    """Зелений = найкраще, червоний = найгірше. Для колонок у invert_green_red «краще» = менше (mΩ, ΔV)."""
    invert_set = set(invert_green_red or [])

    def _highlight(sty_df: pd.DataFrame) -> pd.DataFrame:
        ret = pd.DataFrame("", index=sty_df.index, columns=sty_df.columns)
        for col in highlight_cols:
            if col not in sty_df.columns:
                continue
            s = pd.to_numeric(sty_df[col], errors="coerce")
            if s.dropna().empty:
                continue
            hi = float(s.max())
            lo = float(s.min())
            if math.isnan(hi) or math.isnan(lo) or hi == lo:
                continue
            best = lo if col in invert_set else hi
            worst = hi if col in invert_set else lo
            for i in sty_df.index:
                v = s.loc[i]
                if pd.isna(v):
                    continue
                fv = float(v)
                if abs(fv - best) < 1e-9 * max(1.0, abs(best)):
                    ret.loc[i, col] = "background-color: #d5f5e3; color: #145a32; font-weight: 600"
                elif abs(fv - worst) < 1e-9 * max(1.0, abs(worst)):
                    ret.loc[i, col] = "background-color: #fadbd8; color: #922b21; font-weight: 600"
        return ret

    try:
        return df.style.apply(_highlight, axis=None)  # type: ignore[arg-type]
    except Exception:
        return df


def run_full_analysis(
    df: pd.DataFrame,
    *,
    v_cutoff: float,
    i_threshold: float,
    before_sec: float,
    after_sec: float,
    v_before_manual: Optional[float],
    nominal_mah: float,
    nominal_wh: float,
) -> Tuple[Optional[DischargeMetrics], str, float, float, float, Dict[str, float], float, str, str, str, pd.DataFrame]:
    """Повертає (metrics, err, c_rate, v50, r_mohm, scores, total_10, w_note, verdict, crate_html, tab_df)."""
    has_nominal = nominal_mah > 0 and nominal_wh > 0
    m, err = compute_discharge_metrics(
        df,
        v_cutoff,
        i_threshold,
        before_sec,
        after_sec,
        v_before_manual=v_before_manual,
    )
    if err or m is None:
        empty = pd.DataFrame()
        return (
            None,
            err or "Помилка аналізу",
            float("nan"),
            float("nan"),
            float("nan"),
            {},
            float("nan"),
            "",
            "",
            "",
            empty,
        )

    r_mohm = r_mohm_from_metrics(m)
    c_rate = float("nan")
    v50 = float("nan")
    if has_nominal:
        c_rate = m.i_discharge_avg_a / (nominal_mah / 1000.0)
        v50 = compute_voltage_at_nominal_fraction(df, m.t_start_pos, m.t_end_pos, nominal_mah, fraction=0.5)

    tab_df, scores, total, w_note, _cat, verdict, crate_html = build_scores_and_table(
        m,
        nominal_mah=nominal_mah,
        nominal_wh=nominal_wh,
        has_nominal=has_nominal,
        c_rate=c_rate,
        v_at_50_nominal=v50,
    )
    return m, "", c_rate, v50, r_mohm, scores, total, w_note, verdict, crate_html, tab_df


def _elapsed_seconds_from_start(seg: pd.DataFrame) -> pd.Series:
    t = seg["DATE"]
    if len(seg) == 0:
        return pd.Series(dtype=float)
    t0 = t.iloc[0]
    return (t - t0).dt.total_seconds()


@dataclass
class BatteryFilenameMeta:
    """Мета з імені файлу за єдиним стандартом (токени латиницею, через _)."""

    weight_g: Optional[float] = None  # W72g
    u_cell_v: Optional[float] = None  # Ucell4.35V — номінальна/повна напруга комірки
    u_storage_v: Optional[float] = None  # Ustor3.85V — напруга зберігання
    ir_mohm: Optional[float] = None  # IR52mOhm — внутрішній опір до тесту
    u_start_v: Optional[float] = None  # Ustart4.20V — напруга на старті розряду


# Порівняння АКБ: DC — R = ΔV/I з логу розряду; AC — з імені файлу (типово вимір AC ~1 kHz).
COMPARE_COL_IR_DC = "IR DC (мОм)"
COMPARE_COL_IR_AC = "IR AC (мОм)"


def parse_battery_filename_standard(stem: str) -> BatteryFilenameMeta:
    """
    Розбір імені без розширення. Рекомендований шаблон (фрагменти через підкреслення):

    <будь-який_префікс>_W72g_Ucell4.35V_Ustor3.85V_IR52mOhm_Ustart4.20V

    Допускаються коми як десятковий роздільник у числах (у тексті замінюються на крапку).

    Токени зазвичай йдуть через «_»; після g / V / mOhm не можна покладатися на \\b у regex,
    бо «_» у Python є «словесним» символом — кінець токена задаємо як (?=_|$).
    """
    s = stem.strip().replace(",", ".")
    meta = BatteryFilenameMeta()

    def _f(pat: str) -> Optional[float]:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # Не використовуємо \b після літер: у Python «_» — це \w, тому після g/V/mOhm перед «_»
    # межі слова немає (на відміну від кінця рядка — тому раніше працював лише останній токен).
    _end = r"(?=_|$)"
    meta.weight_g = _f(rf"W_?(\d+(?:\.\d+)?)\s*g{_end}")
    meta.u_cell_v = _f(rf"Ucell_?(\d+(?:\.\d+)?)\s*V{_end}")
    meta.u_storage_v = _f(rf"Ustor_?(\d+(?:\.\d+)?)\s*V{_end}")
    meta.ir_mohm = _f(rf"IR_?(\d+(?:\.\d+)?)\s*mOhm{_end}")
    meta.u_start_v = _f(rf"Ustart_?(\d+(?:\.\d+)?)\s*V{_end}")
    return meta


def battery_filename_meta_summary(meta: BatteryFilenameMeta) -> Tuple[str, bool]:
    """Короткий текст і чи всі п’ять полів знайдені."""
    parts: List[str] = []
    if meta.weight_g is not None:
        parts.append(f"вага {meta.weight_g:g} г")
    if meta.u_cell_v is not None:
        parts.append(f"Ucell {meta.u_cell_v:.2f} В")
    if meta.u_storage_v is not None:
        parts.append(f"Ustor {meta.u_storage_v:.2f} В")
    if meta.ir_mohm is not None:
        parts.append(f"IR {meta.ir_mohm:g} мОм")
    if meta.u_start_v is not None:
        parts.append(f"Ustart {meta.u_start_v:.2f} В")
    ok = (
        meta.weight_g is not None
        and meta.u_cell_v is not None
        and meta.u_storage_v is not None
        and meta.ir_mohm is not None
        and meta.u_start_v is not None
    )
    if not parts:
        return "у імені немає токенів стандарту (W…g, Ucell…V, …)", False
    return "; ".join(parts), ok


def effective_compare_start_voltage_v(m: "DischargeMetrics", filename_stem: str) -> float:
    """
    Стартова напруга для таблиці порівняння: з аналізу CSV (вікно до старту або вручну в боковій панелі).
    Якщо там немає числа (запис почався вже під навантаженням і V_before не задано) — беремо **Ustart** з імені файлу.
    """
    if not math.isnan(m.v_start_idle_v):
        return float(m.v_start_idle_v)
    meta = parse_battery_filename_standard(filename_stem)
    if meta.u_start_v is not None:
        return float(meta.u_start_v)
    return float("nan")


def effective_compare_sag_v(m: "DischargeMetrics", filename_stem: str) -> float:
    """
    Просадка ΔV для таблиці порівняння: з аналізу CSV; якщо немає (немає V_before) —
    **Ustart з імені файлу мінус середня напруга одразу після старту** з CSV (v_after_avg_v).
    """
    if not math.isnan(m.sag_v):
        return float(m.sag_v)
    meta = parse_battery_filename_standard(filename_stem)
    if meta.u_start_v is not None and not math.isnan(m.v_after_avg_v):
        return float(meta.u_start_v - m.v_after_avg_v)
    return float("nan")


def r_mohm_discharge_for_compare(m: "DischargeMetrics", filename_stem: str) -> float:
    """
    IR DC (мОм): R = ΔV / I у постійному струмі (з просадки при старті розряду та струму I).
    Узгоджено з колонкою «Просадка»: спочатку ΔU/I з метрик CSV; якщо sag у CSV немає — та сама відновлена ΔU
    (Ustart з імені − U одразу після старту з логу), що й для просадки.
    """
    r0 = r_mohm_from_metrics(m)
    if not math.isnan(r0):
        return float(r0)
    sag_e = effective_compare_sag_v(m, filename_stem)
    i_use = _current_for_internal_r(m)
    if math.isnan(sag_e) or abs(i_use) < 1e-12:
        return float("nan")
    return abs(sag_e / i_use) * 1000.0


def _compare_bar_colors(values: List[float], *, higher_is_better: bool) -> List[str]:
    """Зелений = краще, жовтий = середнє, червоний = гірше (у межах поточної вибірки)."""
    n = len(values)
    if n == 0:
        return []
    arr = np.array(values, dtype=float)
    valid = np.flatnonzero(~np.isnan(arr))
    if valid.size == 0:
        return ["#95a5a6"] * n
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    out: List[str] = []
    if mx <= mn:
        return ["#27ae60"] * n

    def _mix(t: float) -> str:
        t = max(0.0, min(1.0, t))
        bad = (231, 76, 60)
        mid = (241, 196, 15)
        good = (39, 174, 96)
        if t <= 0.5:
            u = t * 2.0
            rgb = tuple(int(bad[i] + (mid[i] - bad[i]) * u) for i in range(3))
        else:
            u = (t - 0.5) * 2.0
            rgb = tuple(int(mid[i] + (good[i] - mid[i]) * u) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    for v in values:
        if math.isnan(v):
            out.append("#bdc3c7")
            continue
        r = (float(v) - mn) / (mx - mn)
        if not higher_is_better:
            r = 1.0 - r
        out.append(_mix(r))
    return out


def _fmt_hover_val(x: Any, fmt: str, empty: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return empty
    if math.isnan(v):
        return empty
    return format(v, fmt)


def _build_compare_point_hover_texts(
    seg_p: pd.DataFrame,
    tsec: pd.Series,
    *,
    short_name: str,
    u_start_v: float,
    nominal_mah: float,
    duration_s: float,
    temp_col: Optional[str],
) -> List[str]:
    """
    Текст hover для кожної точки (після даунсемплу): напруга, струм, потужність, накопичені mAh/Wh,
    просадка від старту, ΔU між сусідами, dU/dt, температура, % тесту, % від номіналу, C-rate, локальний IR DC.
    """
    n = len(seg_p)
    if n == 0:
        return []

    v = seg_p["VOLTAGE(V)"].astype(float).to_numpy()
    if "CURRENT(A)" in seg_p.columns:
        ia = discharge_current_a(seg_p["CURRENT(A)"]).astype(float).to_numpy()
    else:
        ia = np.full(n, float("nan"))

    if "POWER(W)" in seg_p.columns:
        pw = seg_p["POWER(W)"].astype(float).to_numpy()
        mask_pw = np.isnan(pw) & np.isfinite(v) & np.isfinite(ia)
        pw = np.where(mask_pw, v * ia, pw)
    else:
        pw = v * ia

    if "E_CAPACITY(mAh)" in seg_p.columns:
        cap = seg_p["E_CAPACITY(mAh)"].astype(float).to_numpy()
        c0 = float(cap[0])
        d_cap = cap - c0
    else:
        d_cap = np.full(n, float("nan"))

    if "E_QUANTITY(Wh)" in seg_p.columns:
        eq = seg_p["E_QUANTITY(Wh)"].astype(float).to_numpy()
        e0 = float(eq[0])
        d_wh = eq - e0
    else:
        d_wh = np.full(n, float("nan"))

    if not math.isnan(u_start_v):
        sag = u_start_v - v
    else:
        sag = np.full(n, float("nan"))

    t = tsec.astype(float).to_numpy()
    d_u_prev = np.concatenate([[float("nan")], np.diff(v)])
    dt = np.concatenate([[float("nan")], np.diff(t)])
    with np.errstate(divide="ignore", invalid="ignore"):
        dudt = np.where(np.isfinite(dt) & (np.abs(dt) > 1e-9), d_u_prev / dt, float("nan"))

    pct_test = np.full(n, float("nan"))
    if duration_s is not None and not math.isnan(duration_s) and duration_s > 1e-9:
        pct_test = np.clip(t / duration_s * 100.0, 0.0, 100.0)

    pct_nom = np.full(n, float("nan"))
    if nominal_mah > 1e-9:
        pct_nom = np.clip(d_cap / nominal_mah * 100.0, 0.0, 1000.0)

    ah_nom = nominal_mah / 1000.0
    c_rate = np.where(np.isfinite(ia) & (ah_nom > 1e-12), ia / ah_nom, float("nan"))

    with np.errstate(divide="ignore", invalid="ignore"):
        ir_local = np.where(
            np.isfinite(d_u_prev) & (np.abs(ia) > 1e-12),
            np.abs(d_u_prev) / np.abs(ia) * 1000.0,
            float("nan"),
        )

    if temp_col and temp_col in seg_p.columns:
        temps = seg_p[temp_col].astype(float).to_numpy()
    else:
        temps = np.full(n, float("nan"))

    out: List[str] = []
    for i in range(n):
        lines = [f"<b>{short_name}</b>"]
        lines.append(f"Час від старту: {_fmt_hover_val(t[i], '.1f')} с")
        lines.append(f"Напруга: {_fmt_hover_val(v[i], '.3f')} В")
        lines.append(f"Струм: {_fmt_hover_val(ia[i], '.3f')} А")
        lines.append(f"Потужність: {_fmt_hover_val(pw[i], '.2f')} Вт")
        lines.append(f"Ємність (накопич.): {_fmt_hover_val(d_cap[i], '.0f')} mAh")
        lines.append(f"Енергія (накопич.): {_fmt_hover_val(d_wh[i], '.3f')} Wh")
        lines.append(f"Просадка від старту: {_fmt_hover_val(sag[i], '.3f')} В")
        lines.append(f"ΔU від попередньої точки: {_fmt_hover_val(d_u_prev[i], '.4f')} В")
        lines.append(f"Швидкість зміни U (dU/dt): {_fmt_hover_val(dudt[i], '.5f')} В/с")
        if np.isfinite(temps[i]):
            lines.append(f"Температура: {temps[i]:.1f} °C")
        lines.append(f"Пройдено тесту: {_fmt_hover_val(pct_test[i], '.1f')} %")
        if nominal_mah > 1e-9:
            lines.append(f"Віддано від номіналу: {_fmt_hover_val(pct_nom[i], '.1f')} %")
        lines.append(f"C-rate: {_fmt_hover_val(c_rate[i], '.2f')}C")
        lines.append(f"IR DC (локально, |ΔU|/I): {_fmt_hover_val(ir_local[i], '.2f')} мОм")
        out.append("<br>".join(lines))
    return out


# --- Streamlit UI ---

st.set_page_config(page_title="АКБ · аналіз розряду (CSV тестера)", layout="wide")
st.title("Аналіз розряду АКБ · лог CSV тестера")
st.caption("CSV з тестера: U, I, ємність (mAh), енергія (Wh), час (DATE). Номінал АКБ — обов’язково для оцінок mAh/Wh та загального балу.")

st.sidebar.header("Параметри аналізу розряду")
v_cutoff = st.sidebar.number_input("V_cutoff (напруга відсічки)", value=3.2, min_value=2.5, max_value=4.5, step=0.05)
i_threshold = st.sidebar.number_input("I_threshold (поріг струму для старту)", value=0.2, min_value=0.01, max_value=10.0, step=0.05)
before_sec = st.sidebar.number_input("Вікно до старту (с)", value=2.0, min_value=0.1, max_value=30.0, step=0.5)
after_sec = st.sidebar.number_input("Вікно після старту (с)", value=2.0, min_value=0.1, max_value=30.0, step=0.5)

_v_before_manual_text = st.sidebar.text_input(
    "Стартова напруга до початку тесту (V)",
    value="",
    help="Заповнюється тільки якщо запис почався вже під навантаженням",
)
v_before_manual: Optional[float] = None
if _v_before_manual_text.strip():
    try:
        v_before_manual = float(_v_before_manual_text.replace(",", ".").strip())
    except ValueError:
        v_before_manual = None
        st.sidebar.error("Невірний формат стартової напруги. Вкажіть число, напр. 4.2")

st.sidebar.markdown("**Номінал АКБ (обов’язково для оцінок mAh / Wh / загальний бал)**")
nominal_mode = st.sidebar.radio(
    "Спосіб задання номіналу",
    ["Вручну (mAh)", "Модель iPhone"],
    horizontal=True,
)
if nominal_mode == "Вручну (mAh)":
    nominal_mah = float(
        st.sidebar.number_input(
            "Номінал (mAh)",
            min_value=0.0,
            value=2658.0,
            step=50.0,
            help="Ємність нового акумулятора для вашої моделі (для % та оцінок).",
        )
    )
else:
    iphone_keys = sorted(IPHONE_BATTERY_MAH.keys())
    picked = st.sidebar.selectbox("Модель iPhone (автопідстановка mAh)", iphone_keys, index=iphone_keys.index("iPhone Xs") if "iPhone Xs" in iphone_keys else 0)
    nominal_mah = float(IPHONE_BATTERY_MAH[picked])
    st.sidebar.caption(f"Номінал для **{picked}**: **{nominal_mah:.0f} mAh**")

nominal_wh_override = st.sidebar.number_input(
    "Номінальна енергія (Wh), 0 = з mAh×3.7 В",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Якщо 0 — Wh_номінал = (mAh/1000)×3.7 (типово 1S Li-ion для iPhone).",
)
if nominal_wh_override > 0:
    nominal_wh_eff = float(nominal_wh_override)
else:
    nominal_wh_eff = (nominal_mah * 3.7 / 1000.0) if nominal_mah > 0 else 0.0

has_nominal = nominal_mah > 0 and nominal_wh_eff > 0
if not has_nominal:
    st.sidebar.warning("Задайте номінал mAh > 0 (або оберіть модель iPhone).")

st.sidebar.header("Швидкодія графіків")
max_points = st.sidebar.slider("Макс. точок на графік", min_value=1000, max_value=25000, value=8000, step=500)

tab_single, tab_compare = st.tabs(["Один АКБ", "Порівняння АКБ"])

if "single_path_df" not in st.session_state:
    st.session_state.single_path_df = None  # type: ignore[assignment]
if "single_path_err" not in st.session_state:
    st.session_state.single_path_err = None  # type: ignore[assignment]
if "single_path_name" not in st.session_state:
    st.session_state.single_path_name = ""  # type: ignore[assignment]


def _single_battery_block(
    df: pd.DataFrame,
    *,
    metrics: "DischargeMetrics",
    err: str,
    tab_df: pd.DataFrame,
    score_dict: Dict[str, float],
    total_score: float,
    w_note: str,
    verdict: str,
    crate_html: str,
    c_rate_val: float,
    r_mohm_display: float,
) -> None:
    if err:
        st.error(err)
        return
    st.success(metrics.notes)
    if metrics.no_idle_data_no_manual:
        st.warning(
            "Немає даних до старту і не задано стартову напругу вручну — просадку та внутрішній опір пораховано не буде."
        )
    elif metrics.limited_before_window and not metrics.v_before_manual_used:
        st.info("Запис почався після старту навантаження — просадка розрахована приблизно.")
    if metrics.cutoff_not_reached:
        st.info(
            f"Тест завершено раніше, ніж напруга досягла {v_cutoff} V — результати можуть бути трохи завищені."
        )

    if not has_nominal:
        st.warning(
            "**Номінал АКБ не задано.** Оберіть модель iPhone або введіть номінал (mAh) у боковій панелі — "
            "без цього неможливо порахувати оцінки ємності/енергії та загальний бал (5/5)."
        )

    pct_for_ideal = (
        metrics.capacity_mah / nominal_mah * 100.0
        if has_nominal and not math.isnan(metrics.capacity_mah)
        else float("nan")
    )

    st.subheader("Таблиця метрик та оцінок")
    st.dataframe(style_scores_dataframe(tab_df), use_container_width=True, hide_index=True)
    st.caption(
        "Технічні визначення: просадка — ΔV між напругою до та одразу після старту навантаження; "
        "внутрішній опір — R = ΔV / I; C-rate — струм відносно номіналу (C = I / (номінал/1000))."
    )
    if crate_html:
        st.markdown(crate_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Порівняння з ідеалом")
    ideal_lines = ideal_deviation_lines(pct_for_ideal, r_mohm_display, metrics.sag_v, has_nominal)
    if ideal_lines:
        for line in ideal_lines:
            st.markdown(line)
    else:
        st.caption("Задайте номінал і переконайтесь, що тест завершився коректно.")

    st.markdown("---")
    col_s1, col_s3 = st.columns(2)
    with col_s1:
        ts = total_score
        hl = headline_from_score(ts)
        hl_hex = score_band_color_hex(ts) if not math.isnan(ts) else "#888"
        st.markdown(
            f"<p style='font-size:1.15rem;font-weight:700;margin:0 0 0.5rem 0;color:{hl_hex}'>{hl}</p>",
            unsafe_allow_html=True,
        )
        st.metric(
            "Загальний бал (зважений)",
            f"{ts:.2f} / 10" if not math.isnan(ts) else "—",
        )
        with st.expander("Як рахується загальний бал"):
            st.markdown(
                "Загальний бал — середньозважена **оцінка** (1–10), не сирі вольти чи mΩ: "
                "**(score_mAh×1 + score_Wh×1.5 + score_V×1.5 + score_Sag×2 + score_IR×2) / 8**."
            )
            if w_note and has_nominal:
                st.caption(w_note)
    with col_s3:
        ts2 = total_score
        ind_hex = score_band_color_hex(ts2) if not math.isnan(ts2) else "#95a5a6"
        ind_label = score_band_label_ua(ts2)
        st.markdown(
            f"<p style='font-size:1.1rem;margin-top:0.5rem'>Загальний індикатор: "
            f"<span style='color:{ind_hex};font-weight:bold'>{ind_label}</span></p>",
            unsafe_allow_html=True,
        )

    with st.expander("Пояснення шкали оцінок (1–10)"):
        st.markdown(
            "- **8–10** — добре (зелений)  \n"
            "- **6–8** — нормально (жовтий)  \n"
            "- **4–6** — середньо (оранжевий)  \n"
            "- **<4** — погано (червоний)  \n\n"
            "Колір у таблиці, прогрес-барах і загальному індикаторі завжди від **оцінки** (score), "
            "не від «кращого/гіршого» сирого значення без контексту."
        )

    st.success(f"**Вердикт:** {verdict}")

    st.markdown("**Деталізація оцінок (1–10)**")
    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    subs = [
        ("Ємність", score_dict.get("mAh", float("nan"))),
        ("Енергія", score_dict.get("Wh", float("nan"))),
        ("U середн.", score_dict.get("Vavg", float("nan"))),
        ("Просадка", score_dict.get("Sag", float("nan"))),
        ("R внутр.", score_dict.get("IR", float("nan"))),
        ("U @50%", score_dict.get("V50", float("nan"))),
    ]
    for col, (name, sc) in zip([pc1, pc2, pc3, pc4, pc5, pc6], subs):
        with col:
            st.caption(name)
            if math.isnan(sc):
                st.markdown("<span style='color:#888'>—</span>", unsafe_allow_html=True)
            else:
                h = score_band_color_hex(sc)
                st.markdown(
                    f"<span style='color:{h};font-weight:600'>{sc:.0f}/10</span>",
                    unsafe_allow_html=True,
                )
                pct_w = max(0.0, min(100.0, sc * 10.0))
                st.markdown(
                    f'<div style="background:#e8e8e8;border-radius:4px;height:8px;margin-top:4px">'
                    f'<div style="width:{pct_w}%;background:{h};height:8px;border-radius:4px"></div></div>',
                    unsafe_allow_html=True,
                )

    st.info(
        "**Примітка:** порівняння між тестами коректне лише за **однакового струму розряду**, **однакового cutoff** "
        "та **однакової стартової напруги** (умови навантаження)."
    )

    work = df.sort_values("DATE").reset_index(drop=True)
    work_plot = downsample_df(work, max_points=max_points)

    # Маркери часу для вертикальних ліній (datetime, не pandas.Timestamp — сумісність Plotly)
    t_s = _plotly_vline_x(work.at[metrics.t_start_pos, "DATE"])
    t_e = _plotly_vline_x(work.at[metrics.t_end_pos, "DATE"])

    st.subheader("Графіки")

    # 1) Головний: Voltage vs Capacity (інтервал тесту)
    seg = work.iloc[metrics.t_start_pos : metrics.t_end_pos + 1]
    seg_plot = downsample_df(seg, max_points=max_points)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=seg_plot["E_CAPACITY(mAh)"],
            y=seg_plot["VOLTAGE(V)"],
            mode="lines",
            name="U(E)",
        )
    )
    fig1.update_layout(
        title="Головний: напруга vs віддана ємність (інтервал старт → відсічка)",
        xaxis_title="E_CAPACITY (mAh)",
        yaxis_title="VOLTAGE (V)",
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=work_plot["DATE"], y=work_plot["VOLTAGE(V)"], mode="lines", name="U(t)")
        )
        _add_vertical_time_marker(fig2, t_s, color="green", label="старт")
        _add_vertical_time_marker(fig2, t_e, color="red", label="відсічка")
        fig2.update_layout(title="Додатковий: напруга vs час", yaxis_title="V (V)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(x=work_plot["DATE"], y=work_plot["CURRENT(A)"], mode="lines", name="I(t)")
        )
        _add_vertical_time_marker(fig3, t_s, color="green", label="старт")
        _add_vertical_time_marker(fig3, t_e, color="red", label="відсічка")
        fig3.update_layout(title="Контрольний: струм vs час", yaxis_title="I (A)", height=400)
        st.plotly_chart(fig3, use_container_width=True)


with tab_single:
    st.subheader("Джерело даних")
    single_src = st.radio(
        "Завантаження",
        ["Завантажити файл", "Шлях до файлу (локально)"],
        horizontal=True,
        key="single_src_radio",
    )
    df_single: Optional[pd.DataFrame] = None
    single_filename_hint: str = ""
    load_err_single: Optional[str] = None
    if single_src == "Завантажити файл":
        uploaded_single = st.file_uploader("CSV файл", type=["csv"], key="single_csv_uploader")
        if uploaded_single is not None:
            try:
                df_single = load_battery_csv(uploaded_single.getvalue(), uploaded_single.name)
                single_filename_hint = uploaded_single.name
            except Exception as ex:
                load_err_single = str(ex)
    else:
        path_in = st.text_input("Повний шлях до CSV", key="single_path_input")
        if st.button("Завантажити з шляху", key="single_path_btn"):
            try:
                st.session_state.single_path_df = load_from_path(path_in.strip())
                st.session_state.single_path_err = None
                st.session_state.single_path_name = os.path.basename(path_in.strip())
            except Exception as ex:
                st.session_state.single_path_df = None
                st.session_state.single_path_err = str(ex)
                st.session_state.single_path_name = ""
        if st.session_state.single_path_err:
            st.error(st.session_state.single_path_err)
        df_single = st.session_state.single_path_df
        single_filename_hint = str(st.session_state.single_path_name or "")

    if load_err_single:
        st.error(load_err_single)

    if df_single is None:
        st.info("Завантажте CSV або вкажіть шлях і натисніть «Завантажити з шляху».")
    else:
        run_single = st.button("Тестувати", type="primary", key="run_single_analysis")
        if not run_single:
            st.info("Натисніть «Тестувати», щоб побачити результат аналізу.")
        else:
            v_before_single = v_before_manual
            if v_before_single is None and single_filename_hint:
                stem_single = os.path.splitext(single_filename_hint)[0]
                meta_single = parse_battery_filename_standard(stem_single)
                if meta_single.u_start_v is not None:
                    v_before_single = float(meta_single.u_start_v)
                    st.info(
                        f"Стартову напругу взято з імені файлу (Ustart): {v_before_single:.3f} V. "
                        "Щоб перевизначити, введіть значення в полі sidebar."
                    )
            (
                metrics,
                err_fa,
                c_rate_val,
                _v50_unused,
                r_mohm_display,
                score_dict,
                total_score,
                w_note,
                verdict,
                crate_html,
                tab_df,
            ) = run_full_analysis(
                df_single,
                v_cutoff=v_cutoff,
                i_threshold=i_threshold,
                before_sec=before_sec,
                after_sec=after_sec,
                v_before_manual=v_before_single,
                nominal_mah=nominal_mah,
                nominal_wh=nominal_wh_eff,
            )
            _single_battery_block(
                df_single,
                metrics=metrics,  # type: ignore[arg-type]
                err=err_fa,
                tab_df=tab_df,
                score_dict=score_dict,
                total_score=total_score,
                w_note=w_note,
                verdict=verdict,
                crate_html=crate_html,
                c_rate_val=c_rate_val,
                r_mohm_display=r_mohm_display,
            )

    if df_single is not None:
        st.divider()
        st.subheader("Довільні графіки (усі колонки)")
        df = df_single
        if "DATE" in df.columns:
            default_x = "DATE"
        else:
            default_x = df.columns[0]

        x_col = st.selectbox(
            "Вісь X",
            options=list(df.columns),
            index=list(df.columns).index(default_x) if default_x in df.columns else 0,
            key="single_x",
        )
        numeric_cols = [c for c in df.columns if c != x_col]
        preferred = [
            "VOLTAGE(V)",
            "CURRENT(A)",
            "POWER(W)",
            "E_CAPACITY(mAh)",
            "E_QUANTITY(Wh)",
        ]
        default_y = [c for c in preferred if c in numeric_cols]
        if not default_y:
            default_y = numeric_cols[: min(3, len(numeric_cols))]

        y_cols = st.multiselect("Параметри Y", options=numeric_cols, default=default_y, key="single_y")
        separate_plots = st.checkbox("Окремі підграфіки", value=True, key="single_sep")
        chart_mode = st.selectbox(
            "Стиль", options=["lines", "markers", "lines+markers"], index=0, key="single_mode"
        )
        enable_downsample = st.checkbox("Зменшити точки (довільні графіки)", value=True, key="single_ds")
        max_points2 = st.slider(
            "Макс. точок", min_value=500, max_value=20000, value=5000, step=500, key="mp2"
        )

        df_plot = df.copy()
        if x_col in df_plot.columns and pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            df_plot = df_plot.dropna(subset=[x_col])
        if y_cols:
            if enable_downsample:
                df_plot = downsample_df(df_plot, max_points=max_points2)
            if separate_plots:
                fig = make_subplots(
                    rows=len(y_cols),
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    subplot_titles=y_cols,
                )
                for i, y in enumerate(y_cols, start=1):
                    fig.add_trace(
                        go.Scatter(x=df_plot[x_col], y=df_plot[y], mode=chart_mode, name=y), row=i, col=1
                    )
                fig.update_layout(height=260 * len(y_cols) + 80, showlegend=False)
            else:
                fig = go.Figure()
                for y in y_cols:
                    fig.add_trace(go.Scatter(x=df_plot[x_col], y=df_plot[y], mode=chart_mode, name=y))
                fig.update_layout(showlegend=True)
            fig.update_xaxes(title_text=x_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Оберіть хоча б один параметр Y для довільних графіків.")

    st.caption(
        "Задайте номінальні mAh та (за потреби) Wh у боковій панелі для оцінок ємності/енергії; "
        "C-rate див. рядок у таблиці після зміщення номіналу."
    )


with tab_compare:
    st.subheader("Порівняння кількох АКБ")

    with st.expander("Єдиний стандарт імені CSV-файлу (стартові дані АКБ)", expanded=False):
        st.markdown(
            "Щоб порівняння було відтворюваним, **закладайте в ім’я файлу** (без пробілів) такі дані **латиницею**, "
            "фрагменти через **`_`:**"
        )
        st.markdown(
            "| Токен | Значення |\n"
            "|-------|----------|\n"
            "| `W###g` | Вага, г (напр. `W45g`) |\n"
            "| `Ucell#.#V` | Напруга комірки (номінал / повний заряд), В |\n"
            "| `Ustor#.#V` | Напруга зберігання, В |\n"
            "| `IR###mOhm` | **IR AC** до тесту (мОм), зазвичай вимір **AC** (наприклад 1 kHz) |\n"
            "| `Ustart#.#V` | Напруга на **старті** розряду (перед навантаженням), В |\n"
        )
        st.markdown("Перед числом можна ставити підкреслення: `Ucell_4.35V`. Допускається кома в числах у назві файлу.")
        st.code(
            "BrandA_W45g_Ucell4.35V_Ustor3.85V_IR52mOhm_Ustart4.20V.csv",
            language="text",
        )
        st.caption(
            "Довільний префікс (`BrandA`, `HighCopy_1` тощо) — на початку; далі обов’язково впізнавані токени "
            "у будь-якому порядку, але з переліченими префіксами."
        )

    up_multi = st.file_uploader(
        "CSV файли (2–5 шт.)",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_csv",
    )
    n_files = len(up_multi) if up_multi else 0
    if up_multi is not None and n_files == 0:
        st.caption("Оберіть файли у діалоговому вікні.")
    elif n_files == 1:
        st.warning("Завантажте щонайменше **2** CSV для порівняння.")
    elif n_files > 5:
        st.error("Можна не більше **5** файлів одночасно.")
    elif n_files >= 2:
        meta_rows: List[Dict[str, Any]] = []
        for f in up_multi:
            stem = os.path.splitext(f.name)[0]
            meta = parse_battery_filename_standard(stem)
            _, ok = battery_filename_meta_summary(meta)
            meta_rows.append(
                {
                    "Файл": f.name,
                    "Вага (г)": meta.weight_g if meta.weight_g is not None else "—",
                    "Ucell (В)": meta.u_cell_v if meta.u_cell_v is not None else "—",
                    "Ustor (В)": meta.u_storage_v if meta.u_storage_v is not None else "—",
                    "IR AC (ім’я)": meta.ir_mohm if meta.ir_mohm is not None else "—",
                    "Ustart (В)": meta.u_start_v if meta.u_start_v is not None else "—",
                    "Стандарт": "OK" if ok else "неповний",
                }
            )
        st.markdown("**Розпізнано з імені файлу** (контроль стандарту)")
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)
        if any(r["Стандарт"] != "OK" for r in meta_rows):
            st.info(
                "Для кожного файлу додайте в ім’я токени **W…g**, **Ucell…V**, **Ustor…V**, **IR…mOhm**, **Ustart…V** — "
                "див. розділ «Єдиний стандарт імені» вище."
            )

        name_inputs: List[str] = []
        st.markdown(
            "**Повна назва в таблиці** (за замовчуванням — без розширення імені файлу). "
            "На **графіках** використовуються короткі позначки **АКБ 1 … АКБ N** у порядку вибраних файлів."
        )
        score_mode_label = st.selectbox(
            "Формула рейтингу",
            options=[
                "Базова: 40% mAh + 25% Wh + 20% IR DC + 15% ΔV",
                "Рекомендована: 60% Wh + 40% IR DC",
                "Компроміс: 50% Wh + 30% IR DC + 20% mAh",
            ],
            index=0,
            key="compare_score_mode",
        )
        nc = min(5, n_files)
        cols_nm = st.columns(nc)
        for i in range(nc):
            fn = up_multi[i].name
            default_name = os.path.splitext(fn)[0]
            with cols_nm[i]:
                st.caption(up_multi[i].name[:48] + ("…" if len(up_multi[i].name) > 48 else ""))
                name_inputs.append(
                    st.text_input("Назва АКБ", value=default_name, key=f"cmp_name_{i}")
                )

        if st.button("Порівняти", type="primary", key="run_compare"):
            rows_out: List[Dict[str, Any]] = []
            series_for_plots: List[Tuple[str, str, str, str, pd.DataFrame, DischargeMetrics]] = []
            errs: List[str] = []
            akb_seq = 0
            for i, f in enumerate(up_multi):
                label = (name_inputs[i] or "").strip() or os.path.splitext(f.name)[0]
                try:
                    df_i = load_battery_csv(f.getvalue(), f.name)
                except Exception as ex:
                    errs.append(f"{f.name}: {ex}")
                    continue
                m, err_fa, c_rate_v, _v50, _r_mohm_tab_unused, score_d, _tot10, _wn, _ver, _ch, _tdf = run_full_analysis(
                    df_i,
                    v_cutoff=v_cutoff,
                    i_threshold=i_threshold,
                    before_sec=before_sec,
                    after_sec=after_sec,
                    v_before_manual=v_before_manual,
                    nominal_mah=nominal_mah,
                    nominal_wh=nominal_wh_eff,
                )
                if err_fa or m is None:
                    errs.append(f"{f.name}: {err_fa or 'помилка аналізу'}")
                    continue
                akb_seq += 1
                short_lbl = f"АКБ {akb_seq}"
                smah = score_d.get("mAh", float("nan"))
                swh = score_d.get("Wh", float("nan"))
                stem_f = os.path.splitext(f.name)[0]
                fn_meta = parse_battery_filename_standard(stem_f)
                ir_from_name = float(fn_meta.ir_mohm) if fn_meta.ir_mohm is not None else float("nan")
                rows_out.append(
                    {
                        "Коротка назва": short_lbl,
                        "№ АКБ": akb_seq,
                        "Назва": label,
                        "Стартова напруга (V)": effective_compare_start_voltage_v(m, stem_f),
                        "Кінцева напруга (V)": m.v_cutoff_v,
                        "Середній струм (A)": m.i_discharge_avg_a,
                        "Ємність (mAh)": m.capacity_mah,
                        "Енергія (Wh)": m.energy_wh,
                        "Середня напруга (V)": m.v_avg_load_v,
                        "Просадка на старті (V)": effective_compare_sag_v(m, stem_f),
                        COMPARE_COL_IR_DC: r_mohm_discharge_for_compare(m, stem_f),
                        COMPARE_COL_IR_AC: ir_from_name,
                        "Час тесту (с)": m.duration_s,
                        "C-rate": c_rate_v,
                        "Оцінка mAh": smah,
                        "Оцінка Wh": swh,
                    }
                )
                series_for_plots.append((short_lbl, label, stem_f, f.name, df_i, m))

            for e in errs:
                st.error(e)

            if not rows_out:
                st.warning("Немає жодного успішно обробленого файлу.")
            else:
                caps = [float(r["Ємність (mAh)"]) for r in rows_out]
                whs = [float(r["Енергія (Wh)"]) for r in rows_out]
                irs = [float(r[COMPARE_COL_IR_DC]) for r in rows_out]
                sags = [float(r["Просадка на старті (V)"]) for r in rows_out]
                vloads = [float(r["Середня напруга (V)"]) for r in rows_out]
                if score_mode_label.startswith("Рекомендована"):
                    score_weights = RECOMMENDED_WEIGHTS
                elif score_mode_label.startswith("Компроміс"):
                    score_weights = COMPROMISE_WEIGHTS
                else:
                    score_weights = BASE_WEIGHTS

                score_pack = compute_compare_scores(caps, whs, irs, sags, weights=score_weights)
                scores_100 = score_pack["Final Score"]

                for i, r in enumerate(rows_out):
                    r["Score mAh"] = round(float(score_pack["Score mAh"][i]), 2)
                    r["Score Wh"] = round(float(score_pack["Score Wh"][i]), 2)
                    r["Score IR DC"] = round(float(score_pack["Score IR DC"][i]), 2)
                    r["Score ΔV"] = round(float(score_pack["Score ΔV"][i]), 2)
                    r["Загальний бал (0–100)"] = round(float(scores_100[i]), 2)

                compare_df = pd.DataFrame(rows_out)
                compare_df = compare_df.sort_values("Загальний бал (0–100)", ascending=False, na_position="last")
                compare_df.insert(0, "Рейтинг", range(1, len(compare_df) + 1))

                summ_texts: List[str] = []
                for _, r in compare_df.iterrows():
                    summ_texts.append(
                        _compare_auto_summary(
                            f"{r['Коротка назва']} — {r['Назва']}",
                            float(r["Ємність (mAh)"]),
                            float(r["Енергія (Wh)"]),
                            float(r[COMPARE_COL_IR_DC]),
                            float(r["Просадка на старті (V)"]),
                            float(r["Середня напруга (V)"]),
                            caps,
                            whs,
                            irs,
                            sags,
                            vloads,
                        )
                    )

                st.subheader("Підсумкова таблиця")
                st.caption(
                    "**IR DC** — опір у **постійному струмі**: **R = ΔV / I** (просадка при підключенні навантаження та струм розряду). "
                    "Приклад: без навантаження 4.20 В, під струмом 2 А — 4.08 В → ΔV = 0.12 В → R = ΔV / I. "
                    "**IR AC** — окремо, з токена **IR…mOhm** у імені (типовий вимір **AC**, наприклад 1 kHz); з **IR DC не змішується**."
                )
                hl_cols = [
                    "Ємність (mAh)",
                    "Енергія (Wh)",
                    COMPARE_COL_IR_DC,
                    COMPARE_COL_IR_AC,
                    "Просадка на старті (V)",
                    "Загальний бал (0–100)",
                ]
                inv = [COMPARE_COL_IR_DC, COMPARE_COL_IR_AC, "Просадка на старті (V)"]
                tbl_show = compare_df.drop(columns=["№ АКБ"], errors="ignore")
                st.dataframe(
                    style_compare_table(tbl_show, hl_cols, invert_green_red=inv),
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Автоматичні висновки"):
                    for t in summ_texts:
                        st.markdown(f"- {t}")

                short_set = set(compare_df["Коротка назва"].astype(str))

                def _kpi_pick(col: str, *, mode: str) -> str:
                    s = compare_df[col]
                    if not s.notna().any():
                        return "—"
                    if mode == "max":
                        return str(compare_df.loc[s.idxmax(), "Коротка назва"])
                    if mode == "min":
                        return str(compare_df.loc[s.idxmin(), "Коротка назва"])
                    return "—"

                st.subheader("Підсумок (KPI)")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Найкраща ємність", _kpi_pick("Ємність (mAh)", mode="max"))
                with k2:
                    s_irf = compare_df[COMPARE_COL_IR_DC]
                    if s_irf.notna().any():
                        st.metric("Найнижчий IR DC", str(compare_df.loc[s_irf.idxmin(), "Коротка назва"]))
                    else:
                        st.metric("Найнижчий IR DC", "—")
                with k3:
                    st.metric("Найкращий загальний бал", _kpi_pick("Загальний бал (0–100)", mode="max"))
                with k4:
                    st.metric("Найменша просадка", _kpi_pick("Просадка на старті (V)", mode="min"))

                st.subheader("Графіки порівняння")
                st.caption(
                    "**Hover** на лінійних графіках (напруга / U vs mAh / температура): параметри **поточної точки** "
                    "(час, U, I, P, накопичені mAh/Wh, просадка від старту, ΔU, dU/dt, % тесту, % від номіналу, C-rate, локальний IR DC; температура — якщо є колонка в CSV)."
                )

                _legend_top = dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    x=0.5,
                    xanchor="center",
                )

                fig_ut = go.Figure()
                fig_uc = go.Figure()
                any_temp = False
                fig_temp = go.Figure()
                for short_lbl, _long_lbl, stem_f, fname, df_i, met in series_for_plots:
                    if short_lbl not in short_set:
                        continue
                    row_s = compare_df.loc[compare_df["Коротка назва"] == short_lbl].iloc[0]
                    try:
                        u_start_row = float(row_s["Стартова напруга (V)"])
                    except (TypeError, ValueError, KeyError):
                        u_start_row = float("nan")
                    work = df_i.sort_values("DATE").reset_index(drop=True)
                    seg = work.iloc[met.t_start_pos : met.t_end_pos + 1]
                    if seg.empty or "DATE" not in seg.columns:
                        continue
                    seg_p = downsample_df(seg, max_points=max_points)
                    tsec = _elapsed_seconds_from_start(seg_p)
                    tc = _detect_temperature_column(df_i)
                    nom_m = float(nominal_mah) if has_nominal else 0.0
                    ht_list = _build_compare_point_hover_texts(
                        seg_p,
                        tsec,
                        short_name=short_lbl,
                        u_start_v=u_start_row,
                        nominal_mah=nom_m,
                        duration_s=float(met.duration_s),
                        temp_col=tc,
                    )
                    hover_kw = dict(
                        hovertext=ht_list,
                        hovertemplate="%{hovertext}<extra></extra>",
                        hoverinfo="text",
                    )
                    ln = dict(width=3)
                    fig_ut.add_trace(
                        go.Scatter(
                            x=tsec,
                            y=seg_p["VOLTAGE(V)"],
                            mode="lines",
                            name=short_lbl,
                            line=ln,
                            **hover_kw,
                        )
                    )
                    fig_uc.add_trace(
                        go.Scatter(
                            x=seg_p["E_CAPACITY(mAh)"],
                            y=seg_p["VOLTAGE(V)"],
                            mode="lines",
                            name=short_lbl,
                            line=ln,
                            **hover_kw,
                        )
                    )
                    if tc is not None and tc in seg_p.columns:
                        any_temp = True
                        fig_temp.add_trace(
                            go.Scatter(
                                x=tsec,
                                y=seg_p[tc],
                                mode="lines",
                                name=short_lbl,
                                line=ln,
                                **hover_kw,
                            )
                        )

                fig_ut.update_layout(
                    title="Напруга vs час від старту тесту (с)",
                    xaxis_title="Час від старту (с)",
                    yaxis_title="U (V)",
                    height=380,
                    legend=_legend_top,
                    margin=dict(t=100),
                )
                fig_uc.update_layout(
                    title="Напруга vs віддана ємність (інтервал тесту)",
                    xaxis_title="E_CAPACITY (mAh)",
                    yaxis_title="U (V)",
                    height=380,
                    legend=_legend_top,
                    margin=dict(t=100),
                )

                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(fig_ut, use_container_width=True)
                with g2:
                    st.plotly_chart(fig_uc, use_container_width=True)

                cdf_ord = compare_df.sort_values("№ АКБ")
                xcat = list(cdf_ord["Коротка назва"])
                y_mah = [float(x) for x in cdf_ord["Ємність (mAh)"]]
                y_wh = [float(x) for x in cdf_ord["Енергія (Wh)"]]
                c_mah = _compare_bar_colors(y_mah, higher_is_better=True)
                c_wh = _compare_bar_colors(y_wh, higher_is_better=True)
                # mAh і Wh мають різний масштаб — друга вісь Y, щоб grouped bar був читабельний
                fig_g = go.Figure(
                    data=[
                        go.Bar(
                            name="mAh",
                            x=xcat,
                            y=y_mah,
                            yaxis="y",
                            marker_color=c_mah,
                            text=[f"{v:.0f} mAh" for v in y_mah],
                            textposition="outside",
                            textfont=dict(size=11),
                        ),
                        go.Bar(
                            name="Wh",
                            x=xcat,
                            y=y_wh,
                            yaxis="y2",
                            marker_color=c_wh,
                            text=[f"{v:.2f} Wh" for v in y_wh],
                            textposition="outside",
                            textfont=dict(size=11),
                        ),
                    ]
                )
                fig_g.update_layout(
                    barmode="group",
                    title="Ємність та енергія (grouped: mAh зліва, Wh справа)",
                    height=360,
                    margin=dict(b=60),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
                    yaxis=dict(title="mAh", side="left", showgrid=True),
                    yaxis2=dict(
                        title="Wh",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                )

                cdf_irf = compare_df.dropna(subset=[COMPARE_COL_IR_DC]).sort_values(
                    COMPARE_COL_IR_DC, ascending=True
                )
                if len(cdf_irf) > 0:
                    y_irf = list(cdf_irf["Коротка назва"])
                    x_irf = [float(x) for x in cdf_irf[COMPARE_COL_IR_DC]]
                    c_irf = _compare_bar_colors(x_irf, higher_is_better=False)
                    fig_irf = go.Figure(
                        go.Bar(
                            x=x_irf,
                            y=y_irf,
                            orientation="h",
                            marker_color=c_irf,
                            text=[f"{v:.1f} мОм" for v in x_irf],
                            textposition="outside",
                        )
                    )
                    fig_irf.update_layout(
                        title="IR DC: R = ΔV / I (з логу розряду)",
                        xaxis_title="mΩ",
                        height=max(240, 60 + 40 * len(y_irf)),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=80, r=80),
                    )
                else:
                    fig_irf = go.Figure()
                    fig_irf.update_layout(
                        title="IR DC: немає ΔV/I у жодного файлу",
                        height=200,
                        annotations=[
                            dict(
                                text="Задайте вікно до старту або V_before у sidebar",
                                xref="paper",
                                yref="paper",
                                x=0.5,
                                y=0.5,
                                showarrow=False,
                            )
                        ],
                    )

                cdf_irn = compare_df.dropna(subset=[COMPARE_COL_IR_AC]).sort_values(
                    COMPARE_COL_IR_AC, ascending=True
                )
                if len(cdf_irn) > 0:
                    y_irn = list(cdf_irn["Коротка назва"])
                    x_irn = [float(x) for x in cdf_irn[COMPARE_COL_IR_AC]]
                    c_irn = _compare_bar_colors(x_irn, higher_is_better=False)
                    fig_irn = go.Figure(
                        go.Bar(
                            x=x_irn,
                            y=y_irn,
                            orientation="h",
                            marker_color=c_irn,
                            text=[f"{v:.1f} мОм" for v in x_irn],
                            textposition="outside",
                        )
                    )
                    fig_irn.update_layout(
                        title="IR AC: з імені файлу (токен IR…mOhm, типово AC ~1 kHz)",
                        xaxis_title="mΩ",
                        height=max(240, 60 + 40 * len(y_irn)),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=80, r=80),
                    )
                else:
                    fig_irn = go.Figure()
                    fig_irn.update_layout(
                        title="IR AC: токен IR…mOhm не знайдено",
                        height=200,
                    )

                g3, g4 = st.columns(2)
                with g3:
                    st.plotly_chart(fig_g, use_container_width=True)
                with g4:
                    st.plotly_chart(fig_irf, use_container_width=True)
                    st.plotly_chart(fig_irn, use_container_width=True)

                y_sag = [float(x) for x in cdf_ord["Просадка на старті (V)"]]
                y_vavg = [float(x) for x in cdf_ord["Середня напруга (V)"]]
                y_sc = [float(x) for x in cdf_ord["Загальний бал (0–100)"]]
                c_sag = _compare_bar_colors(y_sag, higher_is_better=False)
                c_vavg = _compare_bar_colors(y_vavg, higher_is_better=True)
                c_sc = _compare_bar_colors(y_sc, higher_is_better=True)

                fig_sag = go.Figure(
                    go.Bar(
                        x=xcat,
                        y=y_sag,
                        marker_color=c_sag,
                        text=[f"{v:.3f} V" for v in y_sag],
                        textposition="outside",
                    )
                )
                fig_sag.update_layout(
                    title="Просадка на старті (ΔV)",
                    yaxis_title="V",
                    height=320,
                    margin=dict(b=60),
                )

                fig_vavg = go.Figure(
                    go.Bar(
                        x=xcat,
                        y=y_vavg,
                        marker_color=c_vavg,
                        text=[f"{v:.2f} V" for v in y_vavg],
                        textposition="outside",
                    )
                )
                fig_vavg.update_layout(
                    title="Середня напруга під навантаженням",
                    yaxis_title="V",
                    height=320,
                    margin=dict(b=60),
                )

                fig_sc = go.Figure(
                    go.Bar(
                        x=xcat,
                        y=y_sc,
                        marker_color=c_sc,
                        text=[f"{v:.1f}" for v in y_sc],
                        textposition="outside",
                    )
                )
                fig_sc.update_layout(
                    title="Загальний бал порівняння (0–100)",
                    yaxis_title="Бал",
                    height=320,
                    margin=dict(b=60),
                )

                g5, g6 = st.columns(2)
                with g5:
                    st.plotly_chart(fig_sag, use_container_width=True)
                with g6:
                    st.plotly_chart(fig_vavg, use_container_width=True)

                g7, g8 = st.columns(2)
                with g7:
                    st.plotly_chart(fig_sc, use_container_width=True)
                with g8:
                    if any_temp:
                        fig_temp.update_layout(
                            title="Температура vs час від старту тесту",
                            xaxis_title="Час від старту (с)",
                            yaxis_title="Темп.",
                            height=320,
                            legend=_legend_top,
                            margin=dict(t=100),
                        )
                        st.plotly_chart(fig_temp, use_container_width=True)
                    else:
                        st.caption("Колонку температури (NTC/TEMP) не знайдено — графік температури пропущено.")

                if not has_nominal:
                    st.warning(
                        "Для оцінок mAh/Wh і зваженого балу порівняння задайте номінал АКБ у боковій панелі."
                    )
