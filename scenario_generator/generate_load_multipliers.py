import argparse
import contextlib
import inspect
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


SEASONS = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}


@contextlib.contextmanager
def temporary_cwd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextlib.contextmanager
def temporary_sys_path(path: Path):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    else:
        inserted = False
    try:
        yield
    finally:
        if inserted and path_str in sys.path:
            sys.path.remove(path_str)


def _safe_call(obj: Any, attr_names: Iterable[str], *args, **kwargs):
    for attr in attr_names:
        candidate = getattr(obj, attr, None)
        if callable(candidate):
            return candidate(*args, **kwargs)
    return None


def _safe_get(obj: Any, attr_names: Iterable[str]):
    for attr in attr_names:
        candidate = getattr(obj, attr, None)
        if candidate is None:
            continue
        if callable(candidate):
            try:
                return candidate()
            except TypeError:
                continue
        return candidate
    return None


class DSSAdapter:
    def compile(self, dss_path: Path) -> None:
        raise NotImplementedError

    def solve(self) -> None:
        raise NotImplementedError

    def load_names(self) -> List[str]:
        raise NotImplementedError

    def set_active_load(self, name: str) -> None:
        raise NotImplementedError

    def active_load_full_name(self) -> str:
        raise NotImplementedError

    def active_load_bus(self) -> str:
        raise NotImplementedError

    def active_load_kw(self) -> float:
        raise NotImplementedError

    def active_load_kvar(self) -> float:
        raise NotImplementedError

    def active_load_kv(self) -> float:
        raise NotImplementedError

    def active_load_phases(self) -> int:
        raise NotImplementedError

    def load_count(self) -> int:
        return len(self.load_names())


class OpenDSSDirectAdapter(DSSAdapter):
    def __init__(self, dss_obj: Any):
        self.dss = dss_obj

    def compile(self, dss_path: Path) -> None:
        if hasattr(self.dss, "Command"):
            self.dss.Command(f"compile '{dss_path}'")
            return
        text = getattr(self.dss, "Text", None)
        if text is not None and hasattr(text, "Command"):
            text.Command(f"compile '{dss_path}'")
            return
        raise RuntimeError("Unable to issue compile command for OpenDSSDirect backend.")

    def solve(self) -> None:
        if hasattr(self.dss, "Command"):
            self.dss.Command("solve")
            return
        text = getattr(self.dss, "Text", None)
        if text is not None and hasattr(text, "Command"):
            text.Command("solve")
            return
        raise RuntimeError("Unable to issue solve command for OpenDSSDirect backend.")

    def load_names(self) -> List[str]:
        names = self.dss.Loads.AllNames()
        return [n for n in names if n]

    def set_active_load(self, name: str) -> None:
        self.dss.Loads.Name(name)

    def active_load_full_name(self) -> str:
        full_name = self.dss.CktElement.Name()
        if full_name:
            return str(full_name)
        return f"Load.{self.dss.Loads.Name()}"

    def active_load_bus(self) -> str:
        bus = self.dss.CktElement.BusNames()[0]
        return str(bus).split(".")[0]

    def active_load_kw(self) -> float:
        return float(self.dss.Loads.kW())

    def active_load_kvar(self) -> float:
        return float(self.dss.Loads.kvar())

    def active_load_kv(self) -> float:
        return float(self.dss.Loads.kV())

    def active_load_phases(self) -> int:
        return int(self.dss.CktElement.NumPhases())


class PyDSSInterfaceAdapter(DSSAdapter):
    def __init__(self, dss_obj: Any):
        self.dss = dss_obj

    def compile(self, dss_path: Path) -> None:
        if _safe_call(self.dss, ["text"], f"compile '{dss_path}'") is not None:
            return
        if _safe_call(self.dss, ["Text"], f"compile '{dss_path}'") is not None:
            return
        text_obj = _safe_get(self.dss, ["Text"])
        if text_obj is not None:
            if _safe_call(text_obj, ["Command", "command"], f"compile '{dss_path}'") is not None:
                return
        raise RuntimeError("Unable to issue compile command for py-dss-interface backend.")

    def solve(self) -> None:
        if _safe_call(self.dss, ["solution_solve", "Solution_Solve"]) is not None:
            return
        if _safe_call(self.dss, ["text", "Text"], "solve") is not None:
            return
        text_obj = _safe_get(self.dss, ["Text"])
        if text_obj is not None:
            if _safe_call(text_obj, ["Command", "command"], "solve") is not None:
                return
        raise RuntimeError("Unable to issue solve command for py-dss-interface backend.")

    def load_names(self) -> List[str]:
        names = _safe_call(self.dss, ["loads_allnames", "Loads_AllNames"])
        if names is None:
            loads_obj = _safe_get(self.dss, ["Loads", "loads"])
            if loads_obj is not None:
                names = _safe_call(loads_obj, ["AllNames", "all_names"]) or _safe_get(
                    loads_obj, ["AllNames", "all_names"]
                )
        return [str(n) for n in (names or []) if str(n)]

    def set_active_load(self, name: str) -> None:
        if _safe_call(self.dss, ["loads_write_name", "Loads_Write_Name"], name) is not None:
            return
        loads_obj = _safe_get(self.dss, ["Loads", "loads"])
        if loads_obj is not None:
            if _safe_call(loads_obj, ["Name", "name"], name) is not None:
                return
        raise RuntimeError(f"Unable to set active load: {name}")

    def active_load_full_name(self) -> str:
        name = _safe_call(self.dss, ["cktelement_name", "CktElement_Name"])
        if name:
            return str(name)
        ckt_obj = _safe_get(self.dss, ["CktElement", "cktelement"])
        if ckt_obj is not None:
            name = _safe_call(ckt_obj, ["Name", "name"]) or _safe_get(ckt_obj, ["Name", "name"])
            if name:
                return str(name)
        load_name = _safe_call(self.dss, ["loads_read_name", "Loads_Read_Name"])
        if not load_name:
            loads_obj = _safe_get(self.dss, ["Loads", "loads"])
            if loads_obj is not None:
                load_name = _safe_call(loads_obj, ["Name", "name"]) or _safe_get(
                    loads_obj, ["Name", "name"]
                )
        return f"Load.{load_name}"

    def active_load_bus(self) -> str:
        buses = _safe_call(self.dss, ["cktelement_read_bus_names", "CktElement_Read_BusNames"])
        if not buses:
            ckt_obj = _safe_get(self.dss, ["CktElement", "cktelement"])
            if ckt_obj is not None:
                buses = _safe_call(ckt_obj, ["BusNames", "bus_names"]) or _safe_get(
                    ckt_obj, ["BusNames", "bus_names"]
                )
        if not buses:
            return ""
        return str(buses[0]).split(".")[0]

    def active_load_kw(self) -> float:
        kw = _safe_call(self.dss, ["loads_read_kw", "Loads_Read_kW"])
        if kw is None:
            loads_obj = _safe_get(self.dss, ["Loads", "loads"])
            kw = _safe_call(loads_obj, ["kW", "kw"]) if loads_obj is not None else None
        return float(kw or 0.0)

    def active_load_kvar(self) -> float:
        kvar = _safe_call(self.dss, ["loads_read_kvar", "Loads_Read_kvar"])
        if kvar is None:
            loads_obj = _safe_get(self.dss, ["Loads", "loads"])
            kvar = _safe_call(loads_obj, ["kvar", "kVar", "Kvar"]) if loads_obj is not None else None
        return float(kvar or 0.0)

    def active_load_kv(self) -> float:
        kv = _safe_call(self.dss, ["loads_read_kv", "Loads_Read_kV"])
        if kv is None:
            loads_obj = _safe_get(self.dss, ["Loads", "loads"])
            kv = _safe_call(loads_obj, ["kV", "kv"]) if loads_obj is not None else None
        return float(kv or 0.0)

    def active_load_phases(self) -> int:
        phases = _safe_call(self.dss, ["cktelement_num_phases", "CktElement_NumPhases"])
        if phases is None:
            ckt_obj = _safe_get(self.dss, ["CktElement", "cktelement"])
            phases = _safe_call(ckt_obj, ["NumPhases", "num_phases"]) if ckt_obj is not None else None
        return int(phases or 0)


def detect_dss_adapter(namespace: Dict[str, Any]) -> DSSAdapter:
    dss_obj = namespace.get("dss")
    if dss_obj is None:
        for value in namespace.values():
            if value is None:
                continue
            if hasattr(value, "Loads") and hasattr(value, "Circuit"):
                dss_obj = value
                break
            if any(hasattr(value, attr) for attr in ["loads_allnames", "solution_solve", "text"]):
                dss_obj = value
                break

    if dss_obj is None:
        raise RuntimeError("Could not detect a DSS engine object after executing scenario_generation.py")

    module_name = str(getattr(dss_obj, "__name__", "")).lower()
    if "opendssdirect" in module_name or hasattr(dss_obj, "Command"):
        return OpenDSSDirectAdapter(dss_obj)

    if any(hasattr(dss_obj, attr) for attr in ["loads_allnames", "solution_solve", "text"]):
        return PyDSSInterfaceAdapter(dss_obj)

    if hasattr(dss_obj, "Loads") and hasattr(dss_obj, "Circuit"):
        return OpenDSSDirectAdapter(dss_obj)

    raise RuntimeError("DSS backend detected, but unsupported interface shape was found.")


def _candidate_master_paths(namespace: Dict[str, Any], scenario_dir: Path, provided_master: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []

    if provided_master:
        candidates.append(provided_master if provided_master.is_absolute() else (scenario_dir / provided_master))

    for value in namespace.values():
        if isinstance(value, (str, Path)) and str(value).lower().endswith(".dss"):
            p = Path(value)
            candidates.append(p if p.is_absolute() else scenario_dir / p)

    path_helper_cls = namespace.get("PathHelper")
    if inspect.isclass(path_helper_cls):
        for system_id in ["123Bus", "J1", "13Bus", "34Bus", "37Bus", "8500"]:
            try:
                helper_obj = path_helper_cls(system_id)
                dss_file_path = getattr(helper_obj, "dss_file_path", None)
                if dss_file_path:
                    p = Path(dss_file_path)
                    candidates.append(p if p.is_absolute() else scenario_dir / p)
            except Exception:
                continue

    for pattern in ["**/*Master*.dss", "**/*master*.dss", "**/*.dss"]:
        for p in scenario_dir.glob(pattern):
            candidates.append(p)

    unique: List[Path] = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key not in seen:
            seen.add(key)
            unique.append(rp)
    return unique


def _attempt_main_invocation(namespace: Dict[str, Any]) -> None:
    main_func = namespace.get("main")
    if not callable(main_func):
        return

    try:
        sig = inspect.signature(main_func)
    except Exception:
        return

    if len(sig.parameters) == 0:
        try:
            main_func()
        except Exception:
            pass
        return

    kwargs = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            return
        kwargs[name] = param.default

    try:
        main_func(**kwargs)
    except Exception:
        pass


def load_and_compile_scenario(scenario_path: Path, master_path: Optional[Path]) -> DSSAdapter:
    scenario_dir = scenario_path.parent
    with temporary_cwd(scenario_dir), temporary_sys_path(scenario_dir):
        namespace = runpy.run_path(str(scenario_path), run_name="scenario_runtime")

    adapter = detect_dss_adapter(namespace)

    # Try to solve immediately in case scenario script already compiled the circuit.
    try:
        adapter.solve()
    except Exception:
        pass

    if adapter.load_count() > 0:
        return adapter

    # Try invoking main() with defaults if present.
    _attempt_main_invocation(namespace)
    try:
        adapter.solve()
    except Exception:
        pass
    if adapter.load_count() > 0:
        return adapter

    candidates = _candidate_master_paths(namespace, scenario_dir, master_path)
    compile_errors: List[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            adapter.compile(candidate)
            adapter.solve()
            if adapter.load_count() > 0:
                return adapter
        except Exception as exc:
            compile_errors.append(f"{candidate}: {exc}")

    error_lines = [
        "Unable to compile and solve an OpenDSS circuit from scenario_generation.py.",
        "Tried executing scenario file, optional main(), and candidate .dss master files.",
    ]
    if compile_errors:
        error_lines.append("Compile attempts with errors:")
        error_lines.extend(compile_errors[:10])
    raise RuntimeError("\n".join(error_lines))


def infer_load_type(short_name: str, kw: float) -> str:
    name = short_name.lower()

    if any(token in name for token in ["ev", "charger", "vehicle"]):
        return "EV-Charger"
    if kw >= 150:
        return "Industrial"
    if any(token in name for token in ["ind", "mfg"]):
        return "Industrial"
    if any(token in name for token in ["com", "off", "retail", "shop"]):
        return "Commercial"
    if "mix" in name:
        return "Mixed-Use"
    if kw <= 15:
        return "Residential"
    if 15 < kw < 150:
        return "Commercial"
    return "Unknown"


def extract_load_info(adapter: DSSAdapter) -> pd.DataFrame:
    rows = []
    for short_name in adapter.load_names():
        adapter.set_active_load(short_name)
        full_name = adapter.active_load_full_name()
        if not full_name.lower().startswith("load."):
            full_name = f"Load.{short_name}"
        rows.append(
            {
                "Name": full_name,
                "ShortName": short_name,
                "Bus": adapter.active_load_bus(),
                "kW": adapter.active_load_kw(),
                "kvar": adapter.active_load_kvar(),
                "kV": adapter.active_load_kv(),
                "Phases": adapter.active_load_phases(),
            }
        )

    load_info = pd.DataFrame(rows)
    if load_info.empty:
        raise RuntimeError("No loads were found in the active DSS circuit.")

    load_info["Type"] = [infer_load_type(sn, kw) for sn, kw in zip(load_info["ShortName"], load_info["kW"])]
    return load_info


def apply_type_overrides(load_info: pd.DataFrame, override_csv: Optional[Path]) -> pd.DataFrame:
    if override_csv is None:
        return load_info

    override_df = pd.read_csv(override_csv)
    required = {"ShortName", "Type"}
    if not required.issubset(set(override_df.columns)):
        missing = sorted(required - set(override_df.columns))
        raise ValueError(f"Override file is missing required columns: {missing}")

    override_map = dict(zip(override_df["ShortName"].astype(str), override_df["Type"].astype(str)))
    updated = load_info.copy()
    updated["Type"] = [override_map.get(sn, t) for sn, t in zip(updated["ShortName"], updated["Type"])]
    return updated


def build_timeframe(year: int) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    start = pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0)
    end = pd.Timestamp(year=year + 1, month=1, day=1, hour=0, minute=0)
    index = pd.date_range(start=start, end=end, freq="15min", inclusive="left")

    holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
    date_norm = index.normalize()

    tags = pd.DataFrame(index=index)
    tags["day_of_week"] = index.dayofweek
    tags["is_weekend"] = (index.dayofweek >= 5) | date_norm.isin(holidays)
    tags["season"] = [SEASONS[m] for m in index.month]
    tags["hour"] = index.hour
    tags["minute"] = index.minute
    tags["qidx"] = tags["hour"] * 4 + (tags["minute"] // 15)
    tags["date"] = date_norm

    return index, tags


def gaussian_bump(hours: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((hours - center) / width) ** 2)


def residential_curve(weekend: bool, season: str) -> np.ndarray:
    h = np.arange(96) / 4.0
    curve = np.full(96, 0.18)

    if weekend:
        curve += gaussian_bump(h, 10.0, 1.7, 0.36)
        curve += gaussian_bump(h, 14.0, 3.5, 0.22)
        curve += gaussian_bump(h, 19.5, 2.0, 0.52)
    else:
        curve += gaussian_bump(h, 8.0, 1.2, 0.40)
        curve += gaussian_bump(h, 19.5, 2.0, 0.70)
        curve += gaussian_bump(h, 13.0, 2.8, 0.06)

    if season == "Summer":
        curve *= 1.18
        curve += gaussian_bump(h, 15.5, 1.5, 0.18)
    elif season == "Winter":
        curve += gaussian_bump(h, 19.0, 1.5, 0.08)
        curve *= 1.05

    return np.clip(curve, 0.0, 1.0)


def commercial_curve(weekend: bool, season: str) -> np.ndarray:
    h = np.arange(96) / 4.0
    curve = np.full(96, 0.13)

    if weekend:
        curve += gaussian_bump(h, 12.5, 3.5, 0.18)
        curve += gaussian_bump(h, 16.0, 2.8, 0.10)
    else:
        curve += gaussian_bump(h, 8.0, 1.6, 0.38)
        curve += gaussian_bump(h, 12.5, 3.0, 0.52)
        curve += gaussian_bump(h, 16.5, 2.5, 0.35)

    if season == "Summer":
        curve *= 1.20
    elif season == "Winter":
        curve *= 1.05

    return np.clip(curve, 0.0, 1.0)


def industrial_curve(weekend: bool, season: str) -> np.ndarray:
    h = np.arange(96) / 4.0
    curve = np.full(96, 0.34)

    if weekend:
        curve += gaussian_bump(h, 12.0, 4.5, 0.36)
    else:
        curve += gaussian_bump(h, 8.0, 2.0, 0.28)
        curve += gaussian_bump(h, 14.0, 4.5, 0.42)
        curve += gaussian_bump(h, 19.0, 2.5, 0.18)

    if season == "Summer":
        curve *= 1.05
    elif season == "Winter":
        curve *= 0.95

    return np.clip(curve, 0.0, 1.0)


def ev_curve(weekend: bool, season: str) -> np.ndarray:
    h = np.arange(96) / 4.0
    curve = np.full(96, 0.05)

    if weekend:
        curve += gaussian_bump(h, 14.0, 2.8, 0.34)
        curve += gaussian_bump(h, 20.0, 2.8, 0.30)
    else:
        curve += gaussian_bump(h, 8.5, 0.9, 0.26)
        curve += gaussian_bump(h, 20.0, 2.2, 0.95)

    if season == "Winter":
        curve *= 1.10

    return np.clip(curve, 0.0, 1.0)


def mixed_use_curve(weekend: bool, season: str) -> np.ndarray:
    return np.clip(0.5 * residential_curve(weekend, season) + 0.5 * commercial_curve(weekend, season), 0.0, 1.0)


def build_base_curves(load_type: str) -> Dict[Tuple[bool, str], np.ndarray]:
    curves = {}
    for weekend in [False, True]:
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            if load_type == "Residential":
                c = residential_curve(weekend, season)
            elif load_type == "Commercial":
                c = commercial_curve(weekend, season)
            elif load_type == "Industrial":
                c = industrial_curve(weekend, season)
            elif load_type == "Mixed-Use":
                c = mixed_use_curve(weekend, season)
            elif load_type == "EV-Charger":
                c = ev_curve(weekend, season)
            else:
                c = mixed_use_curve(weekend, season)
            curves[(weekend, season)] = c
    return curves


def build_annual_profile(load_type: str, tags: pd.DataFrame) -> np.ndarray:
    curves = build_base_curves(load_type)
    profile = np.zeros(len(tags), dtype=float)

    for weekend in [False, True]:
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            mask = (tags["is_weekend"].to_numpy() == weekend) & (tags["season"].to_numpy() == season)
            qidx = tags.loc[mask, "qidx"].to_numpy(dtype=int)
            profile[mask] = curves[(weekend, season)][qidx]

    return np.clip(profile, 0.0, 1.0)

def _inject_event_days(
    daily_scale: np.ndarray,
    day_seasons: np.ndarray,
    rng: np.random.Generator,
    event_prob: float = 0.04,
    suppress_prob: float = 0.02,
) -> np.ndarray:
    """Amplify or suppress entire days to model extreme weather / anomalous demand.

    ~4 % of days receive a demand *surge* (heat-wave, cold-snap, large event),
    ~2 % of remaining days receive a demand *suppression* (unusually mild weather).
    Summer/Winter surges are stronger than Spring/Fall ones.
    A random 70-100 % subset of loads is affected each event day so that not
    every load responds identically.
    """
    n_days, n_loads = daily_scale.shape
    out = daily_scale.copy()

    is_event = rng.random(n_days) < event_prob
    is_suppress = (~is_event) & (rng.random(n_days) < suppress_prob)

    for d in np.where(is_event)[0]:
        if day_seasons[d] in ("Summer", "Winter"):
            factor = rng.uniform(1.25, 1.65)          # up to +65 %
        else:
            factor = rng.uniform(1.10, 1.35)
        affected = rng.random(n_loads) < rng.uniform(0.7, 1.0)
        out[d, affected] *= factor

    for d in np.where(is_suppress)[0]:
        factor = rng.uniform(0.45, 0.75)              # down to -55 %
        affected = rng.random(n_loads) < rng.uniform(0.5, 0.9)
        out[d, affected] *= factor

    return out


def _inject_point_outliers(
    multipliers: np.ndarray,
    rng: np.random.Generator,
    spike_prob: float = 0.002,
    dip_prob: float = 0.001,
) -> np.ndarray:
    """Inject rare per-timestep spikes and dips (equipment transients, etc.).

    Each element has an independent probability of being hit by a
    multiplicative spike (×1.4 – 2.5) or dip (×0.05 – 0.4).
    """
    n_steps, n_loads = multipliers.shape
    out = multipliers.copy()

    spike_mask = rng.random((n_steps, n_loads)) < spike_prob
    out[spike_mask] *= rng.uniform(1.4, 2.5, size=int(spike_mask.sum()))

    dip_mask = rng.random((n_steps, n_loads)) < dip_prob
    out[dip_mask] *= rng.uniform(0.05, 0.4, size=int(dip_mask.sum()))

    return out


def _inject_outage_periods(
    multipliers: np.ndarray,
    day_codes: np.ndarray,
    load_indices: List[int],
    rng: np.random.Generator,
    outage_prob_per_load: float = 0.08,
    max_outage_days: int = 10,
    residual: float = 0.02,
) -> np.ndarray:
    """Zero-out contiguous multi-day blocks for selected loads.

    Models vacations, maintenance, or temporary outages.  Each eligible
    load has an independent probability of experiencing one outage block
    of 1-``max_outage_days`` days during the year.
    """
    out = multipliers.copy()
    n_days = day_codes.max() + 1
    for i in load_indices:
        if rng.random() < outage_prob_per_load:
            n_outage = rng.integers(1, max_outage_days + 1)
            start_day = rng.integers(0, max(1, n_days - n_outage))
            outage_days = np.arange(start_day, start_day + n_outage)
            mask = np.isin(day_codes, outage_days)
            out[mask, i] = residual
    return out


def generate_multiplier_matrix(
    load_info: pd.DataFrame,
    tags: pd.DataFrame,
    seed: int,
    clip_hi: float = 1.5,
) -> np.ndarray:
    n_steps = len(tags)
    n_loads = len(load_info)
    rng = np.random.default_rng(seed)

    day_codes, _ = pd.factorize(tags["date"])
    n_days = day_codes.max() + 1
    load_types = load_info["Type"].tolist()

    # ---- (1) Per-load volatility so loads of the same type differ --------
    load_volatility = rng.uniform(0.5, 2.0, size=n_loads)

    # ---- (2) Daily scaling: Student-t (df=5) for heavier tails -----------
    #      Old: N(1, 0.03).  New: t(df=5) × 0.07 × per-load volatility
    daily_raw = rng.standard_t(df=5, size=(n_days, n_loads))
    daily_scale = 1.0 + 0.07 * daily_raw * load_volatility[np.newaxis, :]

    # ---- (3) Event days (weather extremes, anomalous demand) -------------
    day_seasons = tags.groupby(day_codes)["season"].first().to_numpy()
    daily_scale = _inject_event_days(daily_scale, day_seasons, rng)

    # ---- Base type profiles (unchanged) ----------------------------------
    unique_types = sorted(set(load_types) | {"Unknown"})
    type_profiles = {t: build_annual_profile(t, tags) for t in unique_types}

    # ---- Residential spatial correlation ---------------------------------
    res_indices = [i for i, t in enumerate(load_types) if t == "Residential"]
    rho = 0.5
    z_common = (
        rng.standard_t(df=5, size=n_steps) if res_indices else np.zeros(n_steps)
    )

    multipliers = np.zeros((n_steps, n_loads), dtype=float)

    for i, row in load_info.reset_index(drop=True).iterrows():
        load_type = row["Type"] if row["Type"] in type_profiles else "Unknown"
        base = type_profiles[load_type]

        # ---- (4) Intra-day noise: Student-t, doubled sigma ---------------
        #      Old: sigma = 0.04 * base, Gaussian
        #      New: sigma = 0.08 * base * volatility, t(df=5)
        sigma = 0.08 * base * load_volatility[i]
        z_ind = rng.standard_t(df=5, size=n_steps)

        if i in res_indices:
            z = np.sqrt(rho) * z_common + np.sqrt(1.0 - rho) * z_ind
        else:
            z = z_ind

        multipliers[:, i] = base * daily_scale[day_codes, i] + sigma * z

    # ---- (5) Random point-level spikes and dips --------------------------
    multipliers = _inject_point_outliers(multipliers, rng)

    # ---- (6) Outage / vacation blocks for residential loads --------------
    multipliers = _inject_outage_periods(multipliers, day_codes, res_indices, rng)

    # ---- (7) Clip ceiling raised so peaks can exceed nominal -------------
    return np.clip(multipliers, 0.0, clip_hi)

def save_verification_plot(
    out_path: Path,
    index: pd.DatetimeIndex,
    load_info: pd.DataFrame,
    multipliers: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(multipliers, index=index, columns=load_info["ShortName"].tolist())
    agg = df.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)

    week_end = min(len(agg), 96 * 7)
    axes[0].plot(agg.iloc[:week_end].to_numpy(), linewidth=1.3)
    axes[0].set_title("Average Feeder Multiplier - First Week (15-minute)")
    axes[0].set_xlabel("15-minute interval")
    axes[0].set_ylabel("Multiplier")
    axes[0].grid(alpha=0.3)

    sample_cols = load_info["ShortName"].head(min(6, len(load_info))).tolist()
    for col in sample_cols:
        axes[1].plot(df[col].iloc[:week_end].to_numpy(), linewidth=1.0, alpha=0.85, label=col)
    axes[1].set_title("Sample Load Multipliers - First Week")
    axes[1].set_xlabel("15-minute interval")
    axes[1].set_ylabel("Multiplier")
    axes[1].grid(alpha=0.3)
    if sample_cols:
        axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DSS loads from scenario_generation.py and generate annual QSTS load multipliers."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_generation.py",
        help="Path to scenario_generation.py (default: ./scenario_generation.py)",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Optional master .dss file to compile if scenario script does not compile automatically.",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help="Optional CSV with columns [ShortName, Type] to override inferred load types.",
    )
    parser.add_argument("--year", type=int, default=2025, help="Target year (default: 2025)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--output",
        type=str,
        default="LoadMultipliers.csv",
        help="Output CSV file (default: LoadMultipliers.csv)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamp column in output CSV.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of verification plot image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scenario_path = Path(args.scenario).resolve()
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    override_path = Path(args.overrides).resolve() if args.overrides else None
    output_path = Path(args.output).resolve()
    master_path = Path(args.master).resolve() if args.master else None

    adapter = load_and_compile_scenario(scenario_path, master_path)

    load_info = extract_load_info(adapter)
    load_info = apply_type_overrides(load_info, override_path)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print("\nExtracted load_info:\n")
    print(load_info.to_string(index=False))

    load_info_out = output_path.parent / "load_info_extracted.csv"
    load_info.to_csv(load_info_out, index=False)

    index, tags = build_timeframe(args.year)
    multipliers = generate_multiplier_matrix(load_info, tags, args.seed)

    out_df = pd.DataFrame(multipliers, columns=load_info["ShortName"].tolist())
    out_df = out_df.round(4)

    if args.timestamps:
        out_df.insert(0, "timestamp", index.astype(str))

    out_df.to_csv(output_path, index=False)

    if not args.no_plots:
        plot_path = output_path.with_name(output_path.stem + "_verification.png")
        save_verification_plot(plot_path, index, load_info, multipliers)
        print(f"\nVerification plot saved to: {plot_path}")

    print(f"\nSaved load metadata to: {load_info_out}")
    print(f"Saved multipliers to: {output_path}")
    print(f"Total timesteps: {len(index)}")


if __name__ == "__main__":
    main()
