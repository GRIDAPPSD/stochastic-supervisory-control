import numpy as np
import networkx as nx
import opendssdirect as dss
from pathlib import Path

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
        dss.Command(f"compile '{self.dss_file_path}'")
        dss.Command("solve")
    
    def check_radiality(self):
        """
        Check if the network is radial (tree structure with no loops).
        
        Returns:
            bool: True if network is radial, False otherwise
        """
        # Create a graph to represent the network
        G = nx.Graph()
        flag = dss.Topology.First()
        while flag:
            # Get the buses connected by the current branch
            bus1 = dss.CktElement.BusNames()[0].split('.')[0].upper()
            bus2 = dss.CktElement.BusNames()[1].split('.')[0].upper()
            if bus1 != bus2:  # Avoid self-loops
                G.add_edge(bus1, bus2)
            flag = dss.Topology.Next()
        
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
            'capacitors': [],  # List of capacitor objects
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
                'y': dss.Bus.Y()
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
            
            # Get phases for this line
            phases_bus1 = bus1_full.split('.')[1:] if '.' in bus1_full else []
            phases_bus2 = bus2_full.split('.')[1:] if '.' in bus2_full else []
            
            # If no phases specified, use all phases
            if not phases_bus1:
                phases_bus1 = [str(p) for p in network_data['buses'][bus1]['phases']]
            if not phases_bus2:
                phases_bus2 = [str(p) for p in network_data['buses'][bus2]['phases']]
            
            # Get line parameters
            r_matrix = np.array(dss.Lines.RMatrix()).reshape(dss.Lines.Phases(), dss.Lines.Phases())
            x_matrix = np.array(dss.Lines.XMatrix()).reshape(dss.Lines.Phases(), dss.Lines.Phases())
            
            line_data = {
                'name': line_name,
                'bus1': bus1,
                'bus2': bus2,
                'phases': phases_bus1,  # Phases for this line
                'num_phases': dss.Lines.Phases(),
                'R_matrix': r_matrix,  # Resistance matrix (Ohms)
                'X_matrix': x_matrix,  # Reactance matrix (Ohms)
                'length': dss.Lines.Length(),
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
            
            # Get phases for transformer
            phases_bus1 = bus1_full.split('.')[1:] if '.' in bus1_full else []
            phases_bus2 = bus2_full.split('.')[1:] if '.' in bus2_full else []
            
            num_phases = dss.CktElement.NumPhases()
            
            # If no phases specified, assume 1,2,3 for 3-phase or appropriate for num_phases
            if not phases_bus1:
                phases_bus1 = [str(i) for i in range(1, num_phases + 1)]
            if not phases_bus2:
                phases_bus2 = [str(i) for i in range(1, num_phases + 1)]
            
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
        
        # Extract loads (per bus, per phase)
        loads_list = dss.Loads.AllNames()
        for load_name in loads_list:
            dss.Loads.Name(load_name)
            bus_name = dss.CktElement.BusNames()[0].split('.')[0].upper()
            if bus_name not in network_data['loads']:
                network_data['loads'][bus_name] = {}
            
            # find the actual phases of the load
            phases = dss.CktElement.BusNames()[0].split('.')[1:]  # Get phases from bus name
            # Get actual phase numbers
            bus_phases = network_data['buses'][bus_name]['phases']
            if not phases:
                phases = [str(p) for p in bus_phases]  # Use all bus phases if not specified
            kW = dss.Loads.kW()
            kvar = dss.Loads.kvar()
            
            # Distribute load across phases
            kW_per_phase = kW / len(phases)
            kvar_per_phase = kvar / len(phases)
            
            for i, phase in enumerate(phases):
                if phase not in network_data['loads'][bus_name]:
                    network_data['loads'][bus_name][phase] = {'P': 0, 'Q': 0}
                network_data['loads'][bus_name][phase]['P'] += kW_per_phase / 1000  # Convert to MW
                network_data['loads'][bus_name][phase]['Q'] += kvar_per_phase / 1000  # Convert to MVAr
        
        # Extract capacitors (as objects with controllable status)
        capacitors_list = dss.Capacitors.AllNames()
        for cap_name in capacitors_list:
            dss.Capacitors.Name(cap_name)
            bus_name = dss.CktElement.BusNames()[0].split('.')[0].upper()
            
            # Get capacitor phases from bus name
            phases = dss.CktElement.BusNames()[0].split('.')[1:]  # Get phases from bus name
            bus_phases = network_data['buses'][bus_name]['phases']
            if not phases:
                phases = [str(p) for p in bus_phases]  # Use all bus phases if not specified
            
            # Get capacitor kvar (positive value = injection)
            kvar = dss.Capacitors.kvar()
            
            # Distribute capacitor kvar across phases
            kvar_per_phase = kvar / len(phases)
            
            # Store capacitor as an object
            capacitor_data = {
                'name': cap_name,
                'bus': bus_name,
                'phases': phases,  # List of phase strings
                'kvar_per_phase': kvar_per_phase / 1000,  # MVAr per phase
                'total_kvar': kvar / 1000,  # Total MVAr
                'num_phases': len(phases)
            }
            network_data['capacitors'].append(capacitor_data)
        
        return network_data


def main(systemID):
    try: 
        paths = PathHelper(systemID) 
    except (ValueError, FileNotFoundError) as e:
        print(e)
        return
    
 
    return

if __name__ == "__main__":
    systemID = "J1"
    systemID = "123Bus"
    # systemID = "124Bus"

    main(systemID)