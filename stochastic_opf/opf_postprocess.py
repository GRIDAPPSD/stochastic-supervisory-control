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
