from pathlib import Path

import networkx as nx
import numpy as np
import opendssdirect as dss
import pandas as pd

from opf import OptimalPowerFlow
from opf_postprocess import results_viz, compare_with_opendss


class PathHelper: 
    def __init__(self, systemID="123Bus"): 
        self.current_dir = self.get_script_directory() 
        self.feeder_models_dir = self.current_dir.parent / "feeder-models" 

        if not self.feeder_models_dir.exists():
            raise FileNotFoundError(f"Feeder models directory not found at: {self.feeder_models_dir}")

        if systemID == "123Bus": 
            self.dss_file_path = self.feeder_models_dir / "123Bus" / "IEEE123Master.dss" 
        elif systemID == "J1":
            self.dss_file_path = self.feeder_models_dir / "J1" / "Master_withPV.dss"
        else:
            raise ValueError(f"Unknown systemID: {systemID}. Valid options are '123Bus' and 'J1'.")

        if not self.dss_file_path.exists(): 
            raise FileNotFoundError(f"DSS file not found at: {self.dss_file_path}")

    def get_script_directory(self): 
        return Path(__file__).resolve().parent


class NetworkExtractor:
    """
    Class for extracting and validating network data from OpenDSS.
    """ 
    def __init__(self, dss_file_path): 
        """
        Initialize the NetworkExtractor with an OpenDSS file.
        
        Args:
            dss_file_path: Path to the OpenDSS master file
        """
        self.dss_file_path = dss_file_path
        dss.run_command(f"compile '{self.dss_file_path}'")
        dss.run_command("solve")
    
    def check_radiality(self):
        """
        Check if the network is radial (tree structure with no loops).
        
        Returns:
            bool: True if network is radial, False otherwise
        """
        # Create a graph to represent the network
        G = nx.Graph()
        branch_found = dss.Topology.First()
        while branch_found:
            # Get the buses connected by the current branch
            bus1 = dss.CktElement.BusNames()[0].split('.')[0].upper()
            bus2 = dss.CktElement.BusNames()[1].split('.')[0].upper()
            if bus1 != bus2:  # Avoid self-loops
                G.add_edge(bus1, bus2)
            branch_found = dss.Topology.Next()
        
        # Check if the graph is a tree (radial)
        is_radial = nx.is_tree(G)
        if not is_radial:
            print("The network is not radial. It contains loops.")
        return is_radial
    
    def extract_network_data(self):
        """
        Extract network data from OpenDSS.
        Handles multi-phase systems (1, 2, and 3-phase).
        
        Returns:
            Dictionary with network data structure
        """
        network_data = {
            'buses': {},
            'lines': [],
            'transformers': [],
            'loads': {},
            'ders': {},
            'capacitors': [],
            'source_bus': None
        }
        
        # Extract source bus
        dss.Topology.First()
        source_bus = dss.CktElement.BusNames()[0].split('.')[0].upper()
        network_data['source_bus'] = source_bus
        
        # Extract all buses and their phases
        all_buses = dss.Circuit.AllBusNames()
        for bus_name in all_buses:
            dss.Circuit.SetActiveBus(bus_name)
            bus_name_upper = bus_name.upper()
            network_data['buses'][bus_name_upper] = {
                'name': bus_name_upper,
                'phases': dss.Bus.Nodes(),  # Actual phase numbers present
                'num_phases': len(dss.Bus.Nodes()),
                'kv_base': dss.Bus.kVBase(),
                'x': dss.Bus.X(),
                'y': dss.Bus.Y(),
                'distance': dss.Bus.Distance()
            }
        
        # Extract lines with phase information
        lines_list = dss.Lines.AllNames()
        for line_name in lines_list:
            dss.Lines.Name(line_name)
            bus_names = dss.CktElement.BusNames()
            
            # Parse bus names and phases
            bus1_full = bus_names[0]
            bus2_full = bus_names[1]
            bus1 = bus1_full.split('.')[0].upper()
            bus2 = bus2_full.split('.')[0].upper()
            
            # Get phases for this line (as integers)
            phases_bus1 = [int(p) for p in bus1_full.split('.')[1:]] if '.' in bus1_full else []
            phases_bus2 = [int(p) for p in bus2_full.split('.')[1:]] if '.' in bus2_full else []
            
            # If no phases specified, use all phases
            if not phases_bus1:
                phases_bus1 = network_data['buses'][bus1]['phases']
            if not phases_bus2:
                phases_bus2 = network_data['buses'][bus2]['phases']
            
            # Get line parameters
            r_matrix = np.array(dss.Lines.RMatrix()).reshape(dss.Lines.Phases(), dss.Lines.Phases())
            x_matrix = np.array(dss.Lines.XMatrix()).reshape(dss.Lines.Phases(), dss.Lines.Phases())
            
            kv_base = network_data['buses'][bus1]['kv_base']
            mva_rating = kv_base * dss.Lines.NormAmps() / 1000  
            line_data = {
                'name': line_name,
                'bus1': bus1,
                'bus2': bus2,
                'phases': phases_bus1,  # Phases for this line
                'num_phases': dss.Lines.Phases(),
                'R_matrix': r_matrix,  # Resistance matrix (Ohms)
                'X_matrix': x_matrix,  # Reactance matrix (Ohms)
                'length': dss.Lines.Length(),
                'rating': mva_rating,
                'enabled': dss.CktElement.Enabled()
            }
            network_data['lines'].append(line_data)

        # Extract transformers
        transformer_list = dss.Transformers.AllNames()
        for xfmr_name in transformer_list:
            dss.Transformers.Name(xfmr_name)
            bus_names = dss.CktElement.BusNames()
            
            # Parse bus names and phases for transformer
            bus1_full = bus_names[0]
            bus2_full = bus_names[1] if len(bus_names) > 1 else bus1_full
            bus1 = bus1_full.split('.')[0].upper()
            bus2 = bus2_full.split('.')[0].upper()
            
            # Get phases for transformer (as integers)
            phases_bus1 = [int(p) for p in bus1_full.split('.')[1:]] if '.' in bus1_full else []
            phases_bus2 = [int(p) for p in bus2_full.split('.')[1:]] if '.' in bus2_full else []
            
            num_phases = dss.CktElement.NumPhases()
            
            # If no phases specified, assume 1,2,3 for 3-phase or appropriate for num_phases
            if not phases_bus1:
                phases_bus1 = list(range(1, num_phases + 1))
            if not phases_bus2:
                phases_bus2 = list(range(1, num_phases + 1))
            
            # check if this transformer is a regulator
            # regulator is connected between buses with same kvbase. Sometime they might be off by a small margin
            if abs(network_data['buses'][bus1]['kv_base'] - network_data['buses'][bus2]['kv_base']) < 0.1:
                is_regulator = True
            else:
                is_regulator = False
            
            xfmr_data = {
                'name': xfmr_name,
                'bus1': bus1,
                'bus2': bus2,
                'phases': phases_bus1,  # Phases on primary side
                'num_phases': num_phases,
                'kV1': network_data['buses'][bus1]['kv_base'],
                'kV2': network_data['buses'][bus2]['kv_base'],  # Note: may need to access winding 2
                'kVA': dss.Transformers.kVA(),
                'R': dss.Transformers.R(),
                'X': dss.Transformers.Xhl(),
                'enabled': dss.CktElement.Enabled(),
                'is_regulator': is_regulator
            }
            network_data['transformers'].append(xfmr_data)
        
        # Extract loads (per bus, per phase) and build base_loads list
        base_loads = []
        loads_list = dss.Loads.AllNames()
        for load_name in loads_list:
            dss.Loads.Name(load_name)
            bus_name = dss.CktElement.BusNames()[0].split('.')[0].upper()
            if bus_name not in network_data['loads']:
                network_data['loads'][bus_name] = {}
            
            # find the actual phases of the load (as integers)
            phases = [int(p) for p in dss.CktElement.BusNames()[0].split('.')[1:]]  # Get phases from bus name
            # Get actual phase numbers
            bus_phases = network_data['buses'][bus_name]['phases']
            if not phases:
                phases = list(bus_phases)  # Use all bus phases if not specified
            kW = dss.Loads.kW()
            kvar = dss.Loads.kvar()

            base_loads.append({
                'name': load_name,
                'bus': bus_name,
                'phases': phases,
                'kW': kW,
                'kvar': kvar
            })
            
            # Distribute load across phases
            kw_per_phase = kW / len(phases)
            kvar_per_phase = kvar / len(phases)
            
            for _, phase in enumerate(phases):
                if phase not in network_data['loads'][bus_name]:
                    network_data['loads'][bus_name][phase] = {'P': 0, 'Q': 0}
                network_data['loads'][bus_name][phase]['P'] += kw_per_phase
                network_data['loads'][bus_name][phase]['Q'] += kvar_per_phase
        
        # Extract DER parameters (e.g., PV systems) and build base_ders list
        base_ders = []
        der_list = dss.PVsystems.AllNames()
        for der_name in der_list:
            dss.PVsystems.Name(der_name)
            bus_name = dss.CktElement.BusNames()[0].split('.')[0].upper()
            if bus_name not in network_data['ders']:
                network_data['ders'][bus_name] = {}
            
            # find the actual phases of the DER (as integers)
            phases = [int(p) for p in dss.CktElement.BusNames()[0].split('.')[1:]]  # Get phases from bus name
            # Get actual phase numbers
            bus_phases = network_data['buses'][bus_name]['phases']
            if not phases:
                phases = list(bus_phases)  # Use all bus phases if not specified
            kw = dss.PVsystems.Pmpp()
            kva = dss.PVsystems.kVARated()

            base_ders.append({
                'name': der_name,
                'bus': bus_name,
                'phases': phases,
                'kW': kw,
                'kVA': kva
            })
            
            # Distribute DER output across phases
            kw_per_phase = kw / len(phases)
            kva_per_phase = kva / len(phases)
            
            for _, phase in enumerate(phases):
                if phase not in network_data['ders'][bus_name]:
                    network_data['ders'][bus_name][phase] = {'P': 0, 'Q': 0, 'kVA': 0}
                network_data['ders'][bus_name][phase]['P'] += kw_per_phase
                network_data['ders'][bus_name][phase]['kVA'] += kva_per_phase
        
        # Extract capacitors (as objects with controllable status)
        capacitors_list = dss.Capacitors.AllNames()
        for cap_name in capacitors_list:
            dss.Capacitors.Name(cap_name)
            bus_name = dss.CktElement.BusNames()[0].split('.')[0].upper()
            
            # Get capacitor phases from bus name (as integers)
            phases = [int(p) for p in dss.CktElement.BusNames()[0].split('.')[1:]]  # Get phases from bus name
            bus_phases = network_data['buses'][bus_name]['phases']
            if not phases:
                phases = bus_phases  # Use all bus phases if not specified
            
            # Get capacitor kvar (positive value = injection)
            kvar = dss.Capacitors.kvar()
            
            # Distribute capacitor kvar across phases
            kvar_per_phase = kvar / len(phases)
            
            # Store capacitor as an object
            capacitor_data = {
                'name': cap_name,
                'bus': bus_name,
                'phases': phases,  # List of phase integers
                'kvar_per_phase': kvar_per_phase,  # kvar per phase
                'total_kvar': kvar,  # Total kvar
                'num_phases': len(phases)
            }
            network_data['capacitors'].append(capacitor_data)
        
        return network_data, base_loads, base_ders


def apply_load_multipliers(network_data, base_loads, multipliers):
    """Rebuild network_data['loads'] from base loads with scenario multipliers applied.
    
    Args:
        network_data: Network data dict
        base_loads: List of base load dicts (name, bus, phases, kW, kvar)
        multipliers: Dict mapping load name -> multiplier value
    """
    network_data['loads'] = {}
    for load in base_loads:
        mult = multipliers.get(load['name'], 1.0)
        kW = load['kW'] * mult
        kvar = load['kvar'] * mult
        bus = load['bus']

        if bus not in network_data['loads']:
            network_data['loads'][bus] = {}

        kW_per_phase = kW / len(load['phases'])
        kvar_per_phase = kvar / len(load['phases'])

        for phase in load['phases']:
            if phase not in network_data['loads'][bus]:
                network_data['loads'][bus][phase] = {'P': 0, 'Q': 0}
            network_data['loads'][bus][phase]['P'] += kW_per_phase
            network_data['loads'][bus][phase]['Q'] += kvar_per_phase


def apply_der_multipliers(network_data, base_ders, multiplier):
    """Rebuild network_data['ders'] from base DERs with a single multiplier applied.
    
    Args:
        network_data: Network data dict
        base_ders: List of base DER dicts (name, bus, phases, kW, kVA)
        multiplier: Single float multiplier applied to all DERs (e.g. weather-dependent)
    """
    network_data['ders'] = {}
    for der in base_ders:
        kW = der['kW'] * multiplier
        kVA = der['kVA']
        bus = der['bus']

        if bus not in network_data['ders']:
            network_data['ders'][bus] = {}

        kW_per_phase = kW / len(der['phases'])
        kva_per_phase = kVA / len(der['phases'])

        for phase in der['phases']:
            if phase not in network_data['ders'][bus]:
                network_data['ders'][bus][phase] = {'P': 0, 'Q': 0, 'kVA': 0}
            network_data['ders'][bus][phase]['P'] += kW_per_phase
            network_data['ders'][bus][phase]['kVA'] += kva_per_phase


def main(systemID):
    try: 
        paths = PathHelper(systemID) 
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return
    
    print(f"Loading DSS file from: {paths.dss_file_path}")
    
    # Step 1: Initialize network extractor and check radiality
    extractor = NetworkExtractor(paths.dss_file_path)
    radial = extractor.check_radiality()
    
    if not radial:
        print("OPF requires radial network. Exiting.")
        return
    
    # Step 2: Extract network data from OpenDSS
    print("\nExtracting network data from OpenDSS...")
    network_data, base_loads, base_ders = extractor.extract_network_data()
    
    print(f"Number of buses: {len(network_data['buses'])}")
    print(f"Number of lines: {len(network_data['lines'])}")
    print(f"Source bus: {network_data['source_bus']}")

    # Step 3: Read scenario file and solve OPF for each scenario
    scenario_file = paths.current_dir.parent / "scenario_generator"/ "scenarios" / "LoadMultipliers_0.csv"
    load_multipliers_df = pd.read_csv(scenario_file)
    print(f"\nLoaded {len(load_multipliers_df)} scenarios from {scenario_file.name}")

    for scenario_idx, row in load_multipliers_df.iterrows():
        print(f"\n{'='*80}")
        print(f"Scenario {scenario_idx}")
        print(f"{'='*80}")

        # # Apply load multipliers for this scenario
        # multipliers = row.to_dict()
        # apply_load_multipliers(network_data, base_loads, multipliers)

        # # Apply DER multiplier (single weather-dependent value for all DERs)
        # der_multiplier = np.random.uniform(0, 1)
        # apply_der_multipliers(network_data, base_ders, der_multiplier)
        # print(f"DER multiplier: {der_multiplier:.4f}")

        print("\nFormulating OPF problem...")
        opf = OptimalPowerFlow(network_data)
        opf.formulate_opf()

        print("Solving OPF...")
        results = opf.solve(verbose=True)

        if results['status'] == 'ok':

            print("\nSubstation power injection:")
            for phase, power in results['substation_power'].items():
                print(f"  Phase {phase}: P={power['P']:.4f} kW, Q={power['Q']:.4f} kVAr")

            print("\nCapacitor Status (Optimal Control Decisions):")
            if results['capacitor_status']:
                for cap_name, cap_data in results['capacitor_status'].items():
                    status = cap_data['status']
                    status_str = "ON" if status == 1 else "OFF"
                    print(f"  {cap_name}: {status_str} (status={status})")
            else:
                print("  No capacitors in the system")

            if results['regulator_tap']:
                print("\nRegulator Tap Positions (Optimal Control Decisions):")
                for reg_name, tap_positions in results['regulator_tap'].items():
                    tap_indices = [i for i, val in enumerate(tap_positions) if val == 1]
                    print(f"  {reg_name} is at tap position: {tap_indices[0]}")
            else:
                print("  No Regulators in the system")

            if results['der_q']:
                print("\nDER Reactive Power Outputs (Optimal Control Decisions):")
                for der_bus, der_data in results['der_q'].items():
                    for phase, q in der_data.items():
                        voltage = results['voltages'][der_bus][phase]
                        print(f"  {der_bus} Phase {phase}: Q={q['Q']:.4f} kVAR, V={voltage:.4f} p.u.")
            # postprocessing.compare_with_opendss(network_data, results)
            results_viz(network_data, results)

            return        
        else:
            print(f"  OPF failed for scenario {scenario_idx}: {results.get('message', 'Unknown error')}")

    return


if __name__ == "__main__":
    systemID = "J1"
    # systemID = "123Bus"

    main(systemID)