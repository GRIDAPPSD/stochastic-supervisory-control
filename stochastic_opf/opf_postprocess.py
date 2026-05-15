import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import opendssdirect as dss
import pandas as pd

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def compare_with_opendss(network_data, results):
    """Compare OPF optimization results with OpenDSS power flow solution.
    
    Args:
        network_data: Network data dict from NetworkExtractor
        results: OPF results dict from OptimalPowerFlow.solve()
    """
    # extract all node voltages from results where voltages are in pu
    voltage_optimization = {}
    for bus, bus_info in network_data['buses'].items():
        for phase in bus_info['phases']:
            voltage = results['voltages'][bus][phase]
            node_name = bus + '.' + str(phase)
            if bus_info['kv_base'] < 0.5:
                continue
            voltage_optimization[node_name] = voltage
    
    dss.run_command("solve")
    
    # Extract OpenDSS power flows
    print("\nExtracting OpenDSS power flows...")
    opendss_flows = {}
    
    # Get all lines
    lines_list = dss.Lines.AllNames()
    for line_name in lines_list:
        dss.Lines.Name(line_name)
        dss.Circuit.SetActiveElement(f"Line.{line_name}")
        
        # Get powers (kW and kVAR) for each phase
        powers = dss.CktElement.Powers()  # Returns [P1, Q1, P2, Q2, ...] in kW and kVAR
        num_phases = dss.Lines.Phases()

        # find the actual phases of the line
        bus_names = dss.CktElement.BusNames()
        bus2_full = bus_names[1]
        bus2 = bus2_full.split('.')[0].upper()
        phases_bus2 = [int(p) for p in bus2_full.split('.')[1:]] if '.' in bus2_full else []
        if not phases_bus2:
            phases_bus2 = [1, 2, 3]
        
        opendss_flows[line_name] = {}
        for i, phase in enumerate(phases_bus2):
            # Powers are in kW and kVAR, keep the same units for comparison
            p_kw = powers[2*i]  # Real power in kW
            q_kvar = powers[2*i + 1]  # Reactive power in kVAR
            opendss_flows[line_name][phase] = {
                'P': p_kw,
                'Q': q_kvar
            }
    
    # Compare optimization results with OpenDSS flows
    print("\n" + "="*80)
    print("COMPARISON: Optimization vs OpenDSS")
    print("="*80)
    
    comparison_data = []
    for line_name in results['power_flows']:
        if line_name in opendss_flows:
            for phase in results['power_flows'][line_name]:
                opt_p = results['power_flows'][line_name][phase]['P']
                opt_q = results['power_flows'][line_name][phase]['Q']
                dss_p = opendss_flows[line_name][phase]['P']
                dss_q = opendss_flows[line_name][phase]['Q']
                
                error_p = opt_p - dss_p
                error_q = opt_q - dss_q
                
                # Calculate percentage error (avoid division by zero)
                pct_error_p = (error_p / abs(dss_p) * 100) if abs(dss_p) > 1e-6 else 0.0
                pct_error_q = (error_q / abs(dss_q) * 100) if abs(dss_q) > 1e-6 else 0.0

                comparison_data.append({
                    'line': line_name,
                    'phase': phase,
                    'opt_P_kW': opt_p,
                    'dss_P_kW': dss_p,
                    'error_P_kW': error_p,
                    'pct_error_P': pct_error_p,
                    'opt_Q_kVAR': opt_q,
                    'dss_Q_kVAR': dss_q,
                    'error_Q_kVAR': error_q,
                    'pct_error_Q': pct_error_q,
                    'label': f"{line_name}_ph{phase}"
                })
    
    # Create DataFrame for easier analysis
    comparison_df = pd.DataFrame(comparison_data)

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot the comparison with dual y-axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Plot 1: Real Power with dual y-axis
    ax1 = axes[0]        
    ax1.plot(comparison_df['dss_P_kW'], comparison_df['opt_P_kW'], '.', color='steelblue')
    # plot a y=x line
    ax1.plot([comparison_df['dss_P_kW'].min(), comparison_df['dss_P_kW'].max()],
             [comparison_df['dss_P_kW'].min(), comparison_df['dss_P_kW'].max()], 'r--')
    ax1.set_xlabel('OpenDSS P (kW)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Optimization P (kW)', fontsize=10, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Reactive Power
    ax2 = axes[1]        
    ax2.plot(comparison_df['dss_Q_kVAR'], comparison_df['opt_Q_kVAR'], '.', color='steelblue')
    # plot a y=x line
    ax2.plot([comparison_df['dss_Q_kVAR'].min(), comparison_df['dss_Q_kVAR'].max()],
             [comparison_df['dss_Q_kVAR'].min(), comparison_df['dss_Q_kVAR'].max()], 'r--')
    ax2.set_xlabel('OpenDSS Q (kVAR)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Optimization Q (kVAR)', fontsize=10, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'opf_vs_opendss_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Flow Comparison plot saved as '{output_path}'")
    plt.show()

    # compare optimization voltage and Opendss voltage
    dss_voltages = dss.Circuit.AllBusMagPu()
    dss_node_names = dss.Circuit.AllNodeNames()
    voltage_opendss = {}
    for idx, dss_node in enumerate(dss_node_names):
        voltage_opendss[dss_node.upper()] = dss_voltages[idx]
        if dss_node.upper() in voltage_optimization:
            plt.scatter(voltage_optimization[dss_node.upper()], voltage_opendss[dss_node.upper()], facecolors='none', edgecolors='r')
    
    # plot a y=x line for reference
    plt.plot([min(voltage_optimization.values()), max(voltage_optimization.values())],
             [min(voltage_optimization.values()), max(voltage_optimization.values())], 'k--')
    plt.xlabel('Optimization Voltage (p.u.)', fontsize=10, fontweight='bold')
    plt.ylabel('OpenDSS Voltage (p.u.)', fontsize=10, fontweight='bold')
    plt.title('Voltage Comparison', fontsize=10, fontweight='bold')
    plt.grid(True)
    output_path = output_dir / 'voltage_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Voltage comparison plot saved as '{output_path}'")
    plt.show()

def results_viz(network_data, results):
    """Plot circuit subplots: line loading heatmap and voltage profile heatmap.

    Args:
        network_data: Network data dict from NetworkExtractor
        results: OPF results dict from OptimalPowerFlow.solve()
    """

    power_base = 1 / 3  # MVA per phase (same as OptimalPowerFlow)

    # --- Bus coordinates ---
    bus_coords = {
        name: (info.get('x', 0.0), info.get('y', 0.0))
        for name, info in network_data['buses'].items()
    }

    # --- Line loading (max % across phases) ---
    line_loading = {}
    for line in network_data['lines']:
        line_name = line['name']
        if not line['enabled'] or line_name not in results['power_flows']:
            continue
        # Rating was modified in-place by OPF: line['rating'] = original_mva / power_base
        original_rating_mva = line['rating'] * power_base  # MVA per phase
        max_loading = 0.0
        for flow in results['power_flows'][line_name].values():
            s_mva = math.sqrt(flow['P'] ** 2 + flow['Q'] ** 2) / 1000
            if original_rating_mva > 1e-9:
                max_loading = max(max_loading, s_mva / original_rating_mva * 100)
        line_loading[line_name] = max_loading

    # --- Bus voltage (mean across phases, p.u.) ---
    bus_voltage = {
        bus: float(np.mean(list(phases.values())))
        for bus, phases in results['voltages'].items()
        if phases
    }

    # --- Build line segment geometry ---
    seg_coords = []
    loading_vals = []
    seg_bus1 = []
    for line in network_data['lines']:
        if not line['enabled']:
            continue
        bus1, bus2 = line['bus1'], line['bus2']
        x1, y1 = bus_coords.get(bus1, (0.0, 0.0))
        x2, y2 = bus_coords.get(bus2, (0.0, 0.0))
        # Skip elements without valid coordinates
        if (x1 == 0.0 and y1 == 0.0) or (x2 == 0.0 and y2 == 0.0):
            continue
        seg_coords.append([(x1, y1), (x2, y2)])
        loading_vals.append(line_loading.get(line['name'], 0.0))
        seg_bus1.append(bus1)

    loading_vals = np.array(loading_vals)

    # --- Bus scatter geometry (buses with valid coords) ---
    valid_buses = [
        (name, x, y)
        for name, (x, y) in bus_coords.items()
        if not (x == 0.0 and y == 0.0)
    ]
    bus_names_plot = [b[0] for b in valid_buses]
    bus_x = np.array([b[1] for b in valid_buses])
    bus_y = np.array([b[2] for b in valid_buses])
    bus_v = np.array([bus_voltage.get(n, 1.0) for n in bus_names_plot])

    # --- Create figure with two subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # ── Subplot 1: Line Loading Heatmap ──────────────────────────────────────
    ax1 = axes[0]
    vmax_load = max(100.0, float(loading_vals.max()) if len(loading_vals) else 100.0)
    norm_load = Normalize(vmin=0.0, vmax=vmax_load)
    cmap_load = plt.cm.RdYlGn_r  # green = lightly loaded, red = overloaded

    lc1 = LineCollection(seg_coords, cmap=cmap_load, norm=norm_load, linewidths=1.0, zorder=2)
    lc1.set_array(loading_vals)
    ax1.add_collection(lc1)

    # ax1.scatter(bus_x, bus_y, s=2, c='#333333', zorder=4)

    cbar1 = fig.colorbar(lc1, ax=ax1, shrink=0.75, pad=0.02)
    cbar1.set_label('Line Loading (%)', fontsize=10, fontweight='bold')

    ax1.autoscale_view()
    ax1.set_aspect('equal', adjustable='datalim')
    ax1.margins(0.05)
    ax1.set_title('Line Loading Heatmap', fontsize=10, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_off()
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Voltage Profile Heatmap ───────────────────────────────────
    ax2 = axes[1]
    norm_v = Normalize(vmin=0.95, vmax=1.05)
    cmap_v = plt.cm.RdYlGn_r  # green = nominal (~1.0), red = deviation

    # Draw topology in grey as background
    lc2_bg = LineCollection(seg_coords, colors='#cccccc', linewidths=1.0, zorder=2)
    ax2.add_collection(lc2_bg)

    sc2 = ax2.scatter(bus_x, bus_y, s=5, c=bus_v, cmap=cmap_v, norm=norm_v,
                      edgecolors='k', linewidths=0.4, zorder=4)

    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label('Voltage (p.u.)', fontsize=10, fontweight='bold')

    ax2.autoscale_view()
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.margins(0.05)
    ax2.set_title('Voltage Profile Heatmap', fontsize=10, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axis_off()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'circuit_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Circuit heatmap plot saved as '{output_path}'")
    plt.show()

def line_loading_analysis(line_name, phase):
    """Plot time series loading values for a specific line and phase.
    
    Args:
        line_name: Name of the line (e.g., 'temp_sub', 'oh_b13552')
        phase: Phase number (1, 2, or 3)
    """
    # Load line aging data
    output_dir = Path(__file__).resolve().parent / "output"
    line_aging_path = output_dir / "line_aging_data.json"
    
    if not line_aging_path.exists():
        print(f"Error: Line aging data file not found at {line_aging_path}")
        return
    
    with open(line_aging_path, 'r') as f:
        line_aging_data = json.load(f)
    
    # Extract loading percentages for the given line and phase
    scenarios = []
    loading_values = []
    
    for scenario_num in sorted(line_aging_data.keys(), key=int):
        scenario_data = line_aging_data[scenario_num]
        
        # Check if the line exists in this scenario
        if line_name in scenario_data:
            line_data = scenario_data[line_name]
            
            # Check if the phase exists for this line
            if str(phase) in line_data:
                loading_pct = line_data[str(phase)]['loading_percentage']
                scenarios.append(int(scenario_num))
                loading_values.append(loading_pct)
    
    # Check if data was found
    if not loading_values:
        print(f"Error: No data found for line '{line_name}' phase {phase}")
        print(f"Available lines in scenario 0: {list(line_aging_data['0'].keys())}")
        return
    
    # Create the plot
    plt.figure(figsize=(3.5, 2))
    plt.plot(scenarios, loading_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Scenario Number', fontsize=8, fontweight='bold')
    plt.ylabel('Line Loading (%)', fontsize=8, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / f'line_loading_{line_name}_phase{phase}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def steady_state_temperature(
    I,                      # Current (A)
    Ta_C,                   # Ambient temperature (°C)
    D0_m      = 0.00462,    # Conductor diameter (m)        [Drake ACSR default]
    R_low     = 8.153e-4,   # Resistance at 25°C (Ω/m)     [Drake ACSR default]
    R_high    = 9.702e-4,   # Resistance at 75°C (Ω/m)     [Drake ACSR default]
    Vw_ms     = 0.61,       # Wind speed (m/s)
    phi_deg   = 90.0,       # Wind angle to conductor (°)
    qs        = 26.8,       # Solar heat gain (W/m)
    emissivity= 0.5,        # Surface emissivity
):
    """Returns steady-state conductor temperature (°C) for the given current and ambient."""
 
    def R(T):
        return R_low + (R_high - R_low) * (T - 25) / (75 - 25)
 
    def qc(T):
        Tf  = (T + Ta_C) / 2
        rho = (1.293 * 273.15 / (273.15 + Tf)) * math.exp(0)
        mu  = (1.458e-6 * (Tf + 273.15)**1.5) / (Tf + 273.15 + 110.4)
        kf  = 2.42e-2 + 7.2e-5 * Tf
        NRe = D0_m * Vw_ms * rho / mu
        phi = math.radians(phi_deg)
        Ka  = 1.194 - math.cos(phi) + 0.194*math.cos(2*phi) + 0.368*math.sin(2*phi)
        dT  = T - Ta_C
        return max(
            Ka * (1.01 + 1.35 * NRe**0.52) * kf * dT,
            Ka * 0.754 * NRe**0.6           * kf * dT,
            3.645 * rho**0.5 * D0_m**0.75  * dT**1.25,
        )
 
    def qr(T):
        return 17.8 * D0_m * emissivity * ((T + 273.15)**4 - (Ta_C + 273.15)**4) / 1e8
 
    # Bisection: find T where heat_balance_current(T) == I
    lo, hi = Ta_C, Ta_C + 250.0
    for _ in range(100):
        T   = (lo + hi) / 2
        net = qc(T) + qr(T) - qs
        I_calc = math.sqrt(max(net, 0) / R(T))
        if I_calc > I:
            hi = T
        else:
            lo = T
        if hi - lo < 0.001:
            break
 
    return round((lo + hi) / 2, 2)


def conductor_temperature_evolution(line_name, phase, T_ambient=25):
    """Calculate and plot conductor temperature and line loading evolution across scenarios.

    Reads per-scenario loading percentages from the line aging data JSON, computes
    the steady-state conductor temperature for each scenario using
    ``steady_state_temperature``, and produces a two-subplot figure showing
    (1) conductor temperature (°C) and (2) line loading (%) vs. scenario number.
    The figure is saved to the output directory as a PNG file.

    Args:
        line_name (str): Name of the line to analyse (e.g. ``'oh_b18930'``).
        phase (int): Phase number (1, 2, or 3).
        T_ambient (float): Ambient temperature in °C used for the thermal model.
            Defaults to 25.
    """
    # Load line aging data
    output_dir = Path(__file__).resolve().parent / "output"
    line_aging_path = output_dir / "line_loading_data_scenario_0.json"

    line_info_path = output_dir / "line_info.json"
    with open(line_info_path, 'r') as f:
        line_info = json.load(f)
    
    if not line_aging_path.exists():
        print(f"Error: Line aging data file not found at {line_aging_path}")
        return
    
    with open(line_aging_path, 'r') as f:
        line_aging_data = json.load(f)
    
    # Extract loading percentages for the given line and phase
    timestamps = []
    loading_values = []
    
    # Account for timestamps where the data may be missing.
    # We want to use the data from previous timestamps if data is missing for the current timestamp.
    for timestamp in range(len(line_aging_data)):
        # Use existing data if the timestamp is present, otherwise carry forward last known value
        if str(timestamp) in line_aging_data:
            loading_data = line_aging_data[str(timestamp)]
            if line_name in loading_data:
                line_data = loading_data[line_name]
                if str(phase) in line_data:
                    timestamps.append(timestamp)
                    loading_values.append(line_data[str(phase)])
            
    # Check if data was found
    if not loading_values:
        print(f"Error: No data found for line '{line_name}' phase {phase}")
        print(f"Available lines in scenario 0: {list(line_aging_data['0'].keys())}")
        return

    # Calculate temperature evolution
    temperatures = []
    for L_t in loading_values:
        # Assume rated current corresponds to 100% loading
        I = L_t / 100 * line_info[line_name]['rating']
        T_ss = steady_state_temperature(I, T_ambient)
        temperatures.append(T_ss)

    # Plot temperature evolution and loading using subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(timestamps, temperatures, marker='o', linestyle='-')
    axes[0].set_xlabel("Timestamp")
    axes[0].set_ylabel("Conductor Temperature (°C)")
    axes[0].set_title(f"Temperature Evolution\nLine '{line_name}' Phase {phase}")
    axes[0].grid(True)

    axes[1].plot(timestamps, loading_values, marker='s', linestyle='-', color='orange')
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel("Line Loading (%)")
    axes[1].set_title(f"Line Loading\nLine '{line_name}' Phase {phase}")
    axes[1].grid(True)

    plt.tight_layout()
    output_path = output_dir / f'conductor_temperature_{line_name}_phase{phase}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Conductor temperature plot saved as '{output_path}'")
    plt.show()
    return temperatures, loading_values


def arrhenius_life(T_profile, dt=1.0):
    """
    Calculate thermal aging of a conductor using Arrhenius Life-Temperature Relationship.
    
    Parameters
    ----------
    T_profile : array-like
        Conductor temperature profile [°C]
    dt : float
        Time step between samples [hours] (default = 1.0)
    
    Returns
    -------
    results : dict
        Contains life, damage, loss-of-life metrics
    """
    
    # ---------- Arrhenius Constants ----------
    T_rated_C = 90.0            # Rated temperature [°C]
    L_rated   = 30 * 365 * 24   # Rated life at T_rated [hours] (30 years)
    Ea        = 1.10            # Activation energy [eV]
    kB        = 8.617e-5        # Boltzmann constant [eV/K]
    
    # ---------- Convert to Kelvin ----------
    T_profile = np.asarray(T_profile, dtype=float)
    T_K       = T_profile + 273.15
    T_rated_K = T_rated_C + 273.15
    
    # ---------- Pre-exponential constant (calibrated to rated life) ----------
    A = L_rated / np.exp(Ea / (kB * T_rated_K))
    
    # ---------- Instantaneous life at each temperature ----------
    life_at_T = A * np.exp(Ea / (kB * T_K))                # [hours]
    
    # ---------- Cumulative thermal damage (Miner's rule) ----------
    damage_increments  = dt / life_at_T
    cumulative_damage  = np.cumsum(damage_increments)
    total_damage       = cumulative_damage[-1]
    
    # ---------- Aging acceleration factor (IEEE C57.91 style) ----------
    F_AA       = np.exp(Ea / kB * (1.0/T_rated_K - 1.0/T_K))
    LoL_hours  = np.sum(F_AA * dt)                         # equivalent rated hours
    
    # ---------- Summary ----------
    results = {
        # "life_at_T_hours"       : life_at_T,
        # "life_at_T_years"       : life_at_T / 8760.0,
        "cumulative_damage"     : cumulative_damage,
        "total_damage"          : total_damage,
        "remaining_life_frac"   : max(0.0, 1.0 - total_damage),
        "expected_life_years"   : (1.0 / total_damage) 
                                   if total_damage > 0 else np.inf,
        # "aging_factor_FAA"      : F_AA,
        # "loss_of_life_hours"    : LoL_hours,
        # "loss_of_life_days"     : LoL_hours / 24.0,
    }

    # plot cummulative damage increment and total damage
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(results['cumulative_damage'], marker='o', linestyle='-')
    plt.xlabel("Scenario")
    plt.ylabel("Cumulative Damage")
    plt.title("Cumulative Thermal Damage")
    plt.grid(True)
    output_dir = Path(__file__).resolve().parent / "output"
    output_path = output_dir / f'cumulative_damage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Post process line loading - specify line name and phase
    # Example: line_loading_analysis('temp_sub', 1)
    # Example: line_loading_analysis('oh_b18916', 2)
    # line_loading_analysis('oh_b18948', 2)
    
    # Calculate and plot conductor temperature evolution for a specific line and phase
    temperatures, loading_values = conductor_temperature_evolution('oh_b18948', 2, T_ambient=25)
    # Calculate relative aging rates from temperatures
    # aging_results = arrhenius_life(temperatures)
    # print(aging_results)

