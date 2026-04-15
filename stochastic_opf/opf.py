import json
import math
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


class OptimalPowerFlow:
    def __init__(self, network_data):
        """
        Initialize OPF with network data extracted from OpenDSS.
        
        Args:
            network_data: Dictionary containing buses, lines, transformers, loads, etc.
        """
        self.network_data = network_data
        self.buses = network_data['buses']
        self.lines = network_data['lines']
        self.transformers = network_data['transformers']
        self.loads = network_data['loads']
        self.ders = network_data['ders']
        self.capacitors = network_data.get('capacitors', [])
        self.source_bus = network_data['source_bus']
        self.power_base = 1 / 3 # Base power in MVA per phase
        # convert line ratings from MVA to pu and save it as dictionary
        self.line_ampacity = {}
        for line in self.lines:
            line['rating'] = line['rating'] / self.power_base
            self.line_ampacity[line['name']] = line['rating']
        # Pyomo model
        self.model = None
        
    def setup_optimization_variables(self):
        """
        Create Pyomo variables for the OPF problem.
        Variables per phase: voltage magnitude, real power flow, reactive power flow
        """
        m = self.model
        
        # Create index sets for buses and their phases
        bus_phase_set = []
        for bus_name, bus_data in self.buses.items():
            for phase in bus_data['phases']:
                bus_phase_set.append((bus_name, phase))
        m.BusPhaseSet = pyo.Set(initialize=bus_phase_set)
        
        # Voltage magnitude squared for each bus and phase
        m.v = pyo.Var(m.BusPhaseSet, within=pyo.NonNegativeReals, doc="Voltage magnitude squared")
        
        # Create index sets for lines and their phases
        line_phase_set = []
        for line in self.lines:
            if not line['enabled']:
                continue
            line_name = line['name']
            for phase in line['phases']:
                line_phase_set.append((line_name, phase))
        m.LinePhaseSet = pyo.Set(initialize=line_phase_set)
        
        # Real and reactive power flow for each line and phase
        m.P_line = pyo.Var(m.LinePhaseSet, within=pyo.Reals, doc="Real power flow")
        m.Q_line = pyo.Var(m.LinePhaseSet, within=pyo.Reals, doc="Reactive power flow")
        
        # Slack variables for line limit violations (non-negative)
        # m.line_violation = pyo.Var(m.LinePhaseSet, within=pyo.NonNegativeReals, doc="Line limit violation slack variable")
        # m.ViolSegSet = pyo.RangeSet(0, 3)  # 4 severity tiers
        N_PWL = 10
        self.N_PWL = N_PWL
        self.x_max_fraction = 2.0  # Segments span [0, 200% overload]; last segment is unbounded
        m.ViolSegSet = pyo.RangeSet(0, N_PWL - 1)
        
        m.line_viol_seg = pyo.Var(
            m.LinePhaseSet, m.ViolSegSet,
            within=pyo.NonNegativeReals,
            doc="Piecewise violation segments"
        )
        # Create index sets for transformers and their phases
        xfmr_phase_set = []
        for xfmr in self.transformers:
            xfmr_name = xfmr['name']
            for phase in xfmr['phases']:
                xfmr_phase_set.append((xfmr_name, phase))
        m.XfmrPhaseSet = pyo.Set(initialize=xfmr_phase_set)
        
        # Real and reactive power flow for each transformer and phase
        m.P_xfmr = pyo.Var(m.XfmrPhaseSet, within=pyo.Reals, doc="Real power flow through transformer")
        m.Q_xfmr = pyo.Var(m.XfmrPhaseSet, within=pyo.Reals, doc="Reactive power flow through transformer")
        
        # Substation injection (slack bus)
        source_phases = self.buses[self.source_bus]['phases']
        sub_phase_set = [(self.source_bus, phase) for phase in source_phases]
        m.SubPhaseSet = pyo.Set(initialize=sub_phase_set)
        m.P_sub = pyo.Var(m.SubPhaseSet, within=pyo.Reals, doc="Substation real power injection")
        m.Q_sub = pyo.Var(m.SubPhaseSet, within=pyo.Reals, doc="Substation reactive power injection")
        
        # Capacitor status variables (binary: 0=off, 1=on)
        cap_names = [cap['name'] for cap in self.capacitors]
        m.CapSet = pyo.Set(initialize=cap_names)
        m.cap_status = pyo.Var(m.CapSet, within=pyo.Binary, doc="Capacitor status")
        
        # Auxiliary variables for linearization of cap_status * v (binary × continuous)
        # Q_cap_aux[bus_name, phase] represents the total linearized capacitor injection at that bus-phase
        # Build set of (bus, phase) that have capacitors - phases are already integers
        cap_bus_phase_set = set()
        for capacitor in self.capacitors:
            bus_name = capacitor['bus']
            for phase in capacitor['phases']:
                cap_bus_phase_set.add((bus_name, phase))
        m.CapBusPhaseSet = pyo.Set(initialize=list(cap_bus_phase_set))
        m.Q_cap_aux = pyo.Var(m.CapBusPhaseSet, within=pyo.NonNegativeReals, doc="Auxiliary variable for capacitor Q injection")
        
        # Create mapping from (bus, phase) to list of capacitors at that location
        self.cap_map = {}
        for capacitor in self.capacitors:
            bus_name = capacitor['bus']
            for phase in capacitor['phases']:
                key = (bus_name, phase)
                if key not in self.cap_map:
                    self.cap_map[key] = []
                self.cap_map[key].append(capacitor)
        
        # Regulator tap position variables (integer)
        reg_names = [xfmr['name'] for xfmr in self.transformers if xfmr['is_regulator']]
        m.RegSet = pyo.Set(initialize=reg_names)
        reg_tap_indices = [(reg_name, i) for reg_name in reg_names for i in range(33)]
        m.RegTapSet = pyo.Set(initialize=reg_tap_indices)
        m.reg_tap = pyo.Var(m.RegTapSet, within=pyo.Binary, doc="Regulator tap position")

        # Build set of (bus, phase) that have DERs (inverters) - phases are already integers
        der_bus_phase_set = []
        for bus_name, phase_data in self.ders.items():
            for phase in phase_data.keys():
                der_bus_phase_set.append((bus_name, phase))
        m.DerBusPhaseSet = pyo.Set(initialize=der_bus_phase_set)

        # Using 6 breakpoints: 0.88, 0.92, 0.98, 1.02, 1.08, 1.12 - (indices 0..5)
        N_BP = 6
        m.VVBpSet = pyo.RangeSet(0, N_BP - 1)          # 0,1,2,3,4,5
        m.VVSegSet = pyo.RangeSet(0, N_BP - 2)          # 0,1,2,3,4  (5 segments)

        # SOS2 lambda weights: one per (bus, phase, breakpoint)
        m.lam_vv = pyo.Var(m.DerBusPhaseSet, m.VVBpSet,
                        within=pyo.NonNegativeReals,
                        doc="SOS2 piecewise weights for volt-VAR curve")

        # Binary segment selector: one per (bus, phase, segment)
        m.seg_vv = pyo.Var(m.DerBusPhaseSet, m.VVSegSet,
                        within=pyo.Binary,
                        doc="Active segment indicator for volt-VAR curve")

        # Reactive power output of each inverter (signed: + = inject, - = absorb)
        m.Q_inv = pyo.Var(m.DerBusPhaseSet, within=pyo.Reals,
                        doc="Inverter reactive power output (p.u.)")

    def add_voltvar_constraints(self):
        """
        Encode the Category-B IEEE 1547 volt-VAR curve as piecewise-linear
        constraints using SOS2 (lambda) variables.

        Category-B breakpoints (V in p.u., Q in fraction of Q_max):
            V:  [0.88,  0.92,  0.98,  1.02,  1.08,  1.12]
            Q:  [+1.0,  +1.0,   0.0,   0.0,  -1.0,  -1.0]

        The outer flat segments (0.88 and 1.12) act as saturation sentinels
        so the curve is valid for any voltage in [0.88, 1.12].

        Because m.v stores voltage SQUARED, the breakpoints are also squared.
        """
        m = self.model

        # --- Category B breakpoints ---
        V_bp  = [0.88**2, 0.92**2, 1.00**2, 1.02**2, 1.08**2, 1.12**2]  # v = V^2
        Q_frac = [1.0,     1.0,     0.0,     0.0,    -1.0,    -1.0]       # Q/Q_max

        N_BP  = len(V_bp)
        N_SEG = N_BP - 1

        for bus_name, phase in m.DerBusPhaseSet:
            # Q_max for this DER in p.u.  (use rated kVA as Q_max; adjust as needed)
            der_kva = self.ders[bus_name][phase].get('kVA', 0)
            # Q_max to be at 44% of rated kVA
            Q_max_pu = 0.44 * der_kva / (1000.0 * self.power_base)  # kVA → p.u.

            bp = (bus_name, phase)
            # ----------------------------------------------------------
            # C1: Convexity — weights must sum to 1
            # ----------------------------------------------------------
            cname = f"VV_Conv_{bus_name}_ph{phase}"
            setattr(m, cname,
                    pyo.Constraint(expr=sum(m.lam_vv[bp[0], bp[1], k]
                                            for k in range(N_BP)) == 1))

            # ----------------------------------------------------------
            # C2: Segment selection — exactly one segment active
            # ----------------------------------------------------------
            cname = f"VV_SegSum_{bus_name}_ph{phase}"
            setattr(m, cname,
                    pyo.Constraint(expr=sum(m.seg_vv[bp[0], bp[1], s]
                                            for s in range(N_SEG)) == 1))

            # ----------------------------------------------------------
            # C3 & C4: SOS2 adjacency — lambda[k] can only be nonzero
            #          if seg_vv[k-1]=1 or seg_vv[k]=1 (adjacent segments)
            #
            #   lambda[0]   <= seg[0]
            #   lambda[k]   <= seg[k-1] + seg[k]   for k = 1 .. N_SEG-1
            #   lambda[N_BP-1] <= seg[N_SEG-1]
            # ----------------------------------------------------------
            for k in range(N_BP):
                if k == 0:
                    rhs = m.seg_vv[bus_name, phase, 0]
                elif k == N_BP - 1:
                    rhs = m.seg_vv[bus_name, phase, N_SEG - 1]
                else:
                    rhs = m.seg_vv[bus_name, phase, k-1] + m.seg_vv[bus_name, phase, k]

                cname = f"VV_SOS2_{bus_name}_ph{phase}_k{k}"
                setattr(m, cname,
                        pyo.Constraint(expr=m.lam_vv[bus_name, phase, k] <= rhs))

            # ----------------------------------------------------------
            # C5: Voltage interpolation
            #   v[bus, phase] == sum_k( V_bp[k] * lambda[k] )
            # ----------------------------------------------------------
            v_expr = sum(V_bp[k] * m.lam_vv[bus_name, phase, k] for k in range(N_BP))
            cname = f"VV_Vinterp_{bus_name}_ph{phase}"
            setattr(m, cname,
                    pyo.Constraint(expr=m.v[bus_name, phase] == v_expr))

            # ----------------------------------------------------------
            # C6: Q interpolation
            #   Q_inv[bus, phase] == Q_max * sum_k( Q_frac[k] * lambda[k])
            # ----------------------------------------------------------
            q_expr = Q_max_pu * sum(Q_frac[k] * m.lam_vv[bus_name, phase, k]
                                    for k in range(N_BP))
            cname = f"VV_Qinterp_{bus_name}_ph{phase}"
            setattr(m, cname,
                    pyo.Constraint(expr=m.Q_inv[bus_name, phase] == q_expr))

    def add_power_balance_constraints(self):
        """ 
        Add linearized power balance equations (LinDistFlow model).
        For each bus and phase: sum of incoming power = sum of outgoing power + load
        """
        m = self.model
        
        def power_balance_P_rule(m, bus_name, phase):
            # Find incoming and outgoing lines for this bus and phase
            P_in = []
            P_out = []
            
            # Check if this is the source bus
            if bus_name == self.source_bus:
                P_in.append(m.P_sub[bus_name, phase])
            
            # Iterate through lines
            for line in self.lines:
                if not line['enabled']:
                    continue
                
                if phase not in line['phases']:
                    continue
                
                line_name = line['name']
                
                # Line flows into this bus (bus2 == current bus)
                if line['bus2'] == bus_name:
                    P_in.append(m.P_line[line_name, phase])
                
                # Line flows out of this bus (bus1 == current bus)
                if line['bus1'] == bus_name:
                    P_out.append(m.P_line[line_name, phase])
            
            # Iterate through transformers
            for xfmr in self.transformers:
                if not xfmr['enabled']:
                    continue
                
                if phase not in xfmr['phases']:
                    continue
                
                xfmr_name = xfmr['name']
                
                # Transformer flows into this bus (bus2 == current bus - secondary side)
                if xfmr['bus2'] == bus_name:
                    P_in.append(m.P_xfmr[xfmr_name, phase])
                
                # Transformer flows out of this bus (bus1 == current bus - primary side)
                if xfmr['bus1'] == bus_name:
                    P_out.append(m.P_xfmr[xfmr_name, phase])
            
            # Get load at this bus and phase
            P_load = 0
            if bus_name in self.loads and phase in self.loads[bus_name]:
                # base power is in MVA and actual power is in kW
                P_load = self.loads[bus_name][phase]['P'] / (1000 * self.power_base)
            
            # Get der at this bus and phase
            P_der = 0
            if bus_name in self.ders and phase in self.ders[bus_name]:
                # base power is in MVA and actual power is in kW
                P_der = self.ders[bus_name][phase]['P'] / (1000 * self.power_base)

            # Power balance: sum(P_in) = sum(P_out) + P_load - P_der
            if P_in or P_out:
                return sum(P_in) == sum(P_out) + P_load - P_der
            else:
                return pyo.Constraint.Skip
        
        def power_balance_Q_rule(m, bus_name, phase):
            # Find incoming and outgoing lines for this bus and phase
            Q_in = []
            Q_out = []
            
            # Check if this is the source bus
            if bus_name == self.source_bus:
                Q_in.append(m.Q_sub[bus_name, phase])
            
            # Iterate through lines
            for line in self.lines:
                if not line['enabled']:
                    continue
                
                if phase not in line['phases']:
                    continue
                
                line_name = line['name']
                
                # Line flows into this bus (bus2 == current bus)
                if line['bus2'] == bus_name:
                    Q_in.append(m.Q_line[line_name, phase])
                
                # Line flows out of this bus (bus1 == current bus)
                if line['bus1'] == bus_name:
                    Q_out.append(m.Q_line[line_name, phase])
            
            # Iterate through transformers
            for xfmr in self.transformers:
                if not xfmr['enabled']:
                    continue
                
                if phase not in xfmr['phases']:
                    continue
                
                xfmr_name = xfmr['name']
                
                # Transformer flows into this bus (bus2 == current bus - secondary side)
                if xfmr['bus2'] == bus_name:
                    Q_in.append(m.Q_xfmr[xfmr_name, phase])
                
                # Transformer flows out of this bus (bus1 == current bus - primary side)
                if xfmr['bus1'] == bus_name:
                    Q_out.append(m.Q_xfmr[xfmr_name, phase])
            
            # Get load at this bus and phase
            Q_load = 0
            if bus_name in self.loads and phase in self.loads[bus_name]:
                # base power is in MVA and actual power is in kW
                Q_load = self.loads[bus_name][phase]['Q'] / ( 1000 * self.power_base )
            
            # Get capacitor reactive power injection at this bus and phase
            # Use linearized auxiliary variable instead of direct multiplication
            Q_cap = 0
            if (bus_name, phase) in m.CapBusPhaseSet:
                Q_cap = m.Q_cap_aux[bus_name, phase]
            
            # Get inverter Q contribution at this bus and phase
            Q_inv = 0
            if bus_name in self.ders and phase in self.ders[bus_name]:
                if (bus_name, phase) in m.DerBusPhaseSet:
                    Q_inv = m.Q_inv[bus_name, phase]
            
            # Power balance: sum(Q_in) = sum(Q_out) + Q_load - Q_cap - Q_inv
            if Q_in or Q_out:
                return sum(Q_in) == sum(Q_out) + Q_load - Q_cap - Q_inv
            else:
                return pyo.Constraint.Skip
        
        m.PowerBalanceP = pyo.Constraint(m.BusPhaseSet, rule=power_balance_P_rule)
        m.PowerBalanceQ = pyo.Constraint(m.BusPhaseSet, rule=power_balance_Q_rule)
        
        # Add Big-M linearization constraints for Q_cap_aux = (cap_status * kvar_per_phase * v) for the capacitor at bus-phase
        # Since all terms are non-negative, we only need 4 constraints:
        # (1) Q_cap_aux <= kvar_per_phase * v  
        # (2) Q_cap_aux <= M * cap_status  (can't exceed M)
        # (3) Q_cap_aux >= kvar_per_phase * v - M * (1 - cap_status)
        # (4) Q_cap_aux >= 0 (defined by variable bounds)
        
        # Use a large M value (conservative upper bound on kvar_per_phase * v)
        # M = 500  # Large constant
        
        def bigM_cap_upper1_rule(m, bus_name, phase):
            # Q_cap_aux <= kvar_per_phase * v
            capacitor = self.cap_map.get((bus_name, phase), [])[0]
            if not capacitor:
                return pyo.Constraint.Skip
            cap_kvar = capacitor['kvar_per_phase'] / (1000 * self.power_base)

            return m.Q_cap_aux[bus_name, phase] <= cap_kvar  * m.v[bus_name, phase]
        
        def bigM_cap_upper2_rule(m, bus_name, phase):
            # Q_cap_aux <= M * cap_status
            capacitor = self.cap_map.get((bus_name, phase), [])[0]
            if not capacitor:
                return pyo.Constraint.Skip
            cap_kvar = capacitor['kvar_per_phase'] / (1000 * self.power_base)
            M = cap_kvar * (1.05 ** 2)  # Tighten M based on max possible value of kvar_per_phase * v
            return m.Q_cap_aux[bus_name, phase] <= M * m.cap_status[capacitor['name']]
        
        def bigM_cap_lower_rule(m, bus_name, phase):
            # Q_cap_aux >= kvar_per_phase * v - M * (1 - cap_status)
            capacitor = self.cap_map.get((bus_name, phase), [])[0]
            if not capacitor:
                return pyo.Constraint.Skip

            cap_kvar = capacitor['kvar_per_phase'] / (1000 * self.power_base)
            M = cap_kvar * (1.05 ** 2)  # Tighten M based on max possible value of kvar_per_phase * v
            return m.Q_cap_aux[bus_name, phase] >= cap_kvar * m.v[bus_name, phase] - M * (1 - m.cap_status[capacitor['name']])
        
        m.BigM_Cap_Upper1 = pyo.Constraint(m.CapBusPhaseSet, rule=bigM_cap_upper1_rule)
        m.BigM_Cap_Upper2 = pyo.Constraint(m.CapBusPhaseSet, rule=bigM_cap_upper2_rule)
        m.BigM_Cap_Lower = pyo.Constraint(m.CapBusPhaseSet, rule=bigM_cap_lower_rule)
        
        # Fix all capacitors to ON/OFF status for some debugging purposes
        # def cap_status_rule(m, cap_name):
        #     return m.cap_status[cap_name] == 1
        # m.CapStatusFix = pyo.Constraint(m.CapSet, rule=cap_status_rule)
    
    def add_voltage_drop_constraints(self):
        """
        Add linearized voltage drop equations.
        For lines: V_j^2 = V_i^2 - H_ij^p P_ij - H_ij^q Q_ij
        For transformers: V_j^2 = V_i^2 ; for now lossless transformer
        For regulators: V_j^2 = A_reg * V_i^2
        """
        m = self.model
        # z_base = kv ** 2 / MVA
        
        # Voltage drop for lines
        for line in self.lines:
            if not line['enabled']:
                continue
            
            bus1 = line['bus1']
            bus2 = line['bus2']
            line_name = line['name']
            
            # Get impedance matrices scaled by line length
            R = line['R_matrix'] * line['length']  # Resistance matrix (Ohms)
            X = line['X_matrix'] * line['length']  # Reactance matrix (Ohms)
            n_phases = len(line['phases'])
            
            # Construct H matrices
            H_P = np.zeros((n_phases, n_phases))
            H_Q = np.zeros((n_phases, n_phases))
            sqrt3 = np.sqrt(3)
            z_base = (self.buses[bus1]['kv_base'] ** 2) / self.power_base
            for i in range(n_phases):
                for j in range(n_phases):
                    if i == j:
                        # Diagonal terms
                        H_P[i, j] = -2 * R[i, j] / z_base
                        H_Q[i, j] = -2 * X[i, j] / z_base
                    else:
                        # Off-diagonal terms with sqrt(3) coupling
                        diff = (j - i) % 3
                        if diff == 1:
                            H_P[i, j] = (R[i, j] - sqrt3 * X[i, j]) / z_base
                            H_Q[i, j] = (X[i, j] + sqrt3 * R[i, j]) / z_base
                        else:  # diff == 2
                            H_P[i, j] = (R[i, j] + sqrt3 * X[i, j]) / z_base
                            H_Q[i, j] = (X[i, j] - sqrt3 * R[i, j]) / z_base
            
            # Create voltage drop constraints directly for this line
            for i, phase in enumerate(line['phases']):
                # Calculate voltage drop expression
                voltage_drop = 0
                for j, phase_j in enumerate(line['phases']):
                    voltage_drop += H_P[i, j] * m.P_line[line_name, phase_j]
                    voltage_drop += H_Q[i, j] * m.Q_line[line_name, phase_j]
                
                # Create constraint: v_bus2 = v_bus1 + voltage_drop
                constraint_name = f"VoltageDrop_{line_name}_ph{phase}"
                setattr(m, constraint_name,
                       pyo.Constraint(expr=m.v[bus2, phase] == m.v[bus1, phase] + voltage_drop))

        # Voltage drop for transformers (including regulators)
        tap_values = np.linspace(0.9, 1.1, 33)
        B_values = [float(b**2) for b in tap_values]
        v_upper = (1.05 ** 2) * B_values[-1]  # Conservative upper bound for voltage squared with highest tap
        for xfmr in self.transformers:
            if not xfmr['enabled']:
                continue

            bus1 = xfmr['bus1']
            bus2 = xfmr['bus2']
            xfmr_name = xfmr['name']

            # Check if regulator and compute tap values once
            if xfmr['is_regulator']:               
                # Add tap selection constraints (exactly one tap must be selected)
                tap_sum_constraint_name = f"RegTapSum_{xfmr_name}"
                setattr(m, tap_sum_constraint_name,
                       pyo.Constraint(expr=sum(m.reg_tap[xfmr_name, i] for i in range(len(B_values))) == 1))

                # Fix tap position at 16 (mid-point)
                # tap_fix_constraint_name = f"RegTapFix_{xfmr_name}"
                # setattr(m, tap_fix_constraint_name,
                #        pyo.Constraint(expr=m.reg_tap[xfmr_name, 16] == 1))

                # Add per-phase voltage constraints for regulator
                for phase in xfmr['phases']:
                    # Create auxiliary variables and constraints for each tap position
                    # z_i represents: reg_tap[i] * (B_values[i] * v[bus1])
                    # Since B_values[i] > 0, v[bus1] > 0, and z_i >= 0 (NonNegativeReals), we need 4 constraints per tap:
                    # (1) z_i <= B_values[i] * v[bus1]  (upper bound when reg_tap = 1)
                    # (2) z_i <= B_values[i] * v_upper * reg_tap[i]  (upper bound when reg_tap = 0, forces z_i = 0)
                    # (3) z_i >= B_values[i] * v[bus1] - B_values[i] * v_upper * (1 - reg_tap[i])  (lower bound when reg_tap = 1)
                    # (4) z_i >= 0  (non-negativity constraint, already enforced by variable domain)
                    z_vars = []
                    for i in range(len(B_values)):
                        z_var_name = f"z_{xfmr_name}_{phase}_{i}"
                        setattr(m, z_var_name, pyo.Var(within=pyo.NonNegativeReals))
                        z_i = getattr(m, z_var_name)
                        z_vars.append(z_i)
 
                        B_i = B_values[i]

                        # Constraint 1: Variable upper bound (tight when tap is selected)
                        constraint1_name = f"RegLin1_{xfmr_name}_{phase}_{i}"
                        setattr(m, constraint1_name,
                               pyo.Constraint(expr=z_i <= B_i * m.v[bus1, phase]))

                        # Constraint 2: Big-M upper bound (forces z_i = 0 when tap is not selected)
                        constraint2_name = f"RegLin2_{xfmr_name}_{phase}_{i}"
                        setattr(m, constraint2_name,
                               pyo.Constraint(expr=z_i <= B_i * v_upper * m.reg_tap[xfmr_name, i]))

                        # Constraint 3: Lower bound (forces z_i = B_i * v when tap is selected)
                        constraint3_name = f"RegLin3_{xfmr_name}_{phase}_{i}"
                        setattr(m, constraint3_name,
                               pyo.Constraint(expr=z_i >= B_i * m.v[bus1, phase] - B_i * v_upper * (1 - m.reg_tap[xfmr_name, i])))
                    
                    # Final voltage relationship: v_j = sum(z_i) where z_i = reg_tap[i] * B_i * v[bus1]
                    v_j_expr = sum(z_vars[i] for i in range(len(B_values)))
                    constraint_name = f"RegVoltage_{xfmr_name}_ph{phase}"
                    setattr(m, constraint_name,
                           pyo.Constraint(expr=m.v[bus2, phase] == v_j_expr))
            
            else:
                # Non-regulator transformer (ideal transformer)
                for phase in xfmr['phases']:
                    # TODO: Replace this with actual voltage drop calculation for non-regulator transformers
                    constraint_name = f"XfmrVoltage_{xfmr_name}_ph{phase}"
                    setattr(m, constraint_name,
                           pyo.Constraint(expr=m.v[bus2, phase] == m.v[bus1, phase]))
    
    def add_source_voltage(self, vsrc):
        """
        Define the source bus voltage.
        
        Args:
            vsrc: Source voltage magnitude (p.u.)
        """
        m = self.model
        for phase in self.buses[self.source_bus]['phases']:
            m.v[self.source_bus, phase].fix(vsrc ** 2)

    
    def add_voltage_limits(self, v_min=0.95**2, v_max=1.05**2):
        """
        Add voltage magnitude constraints.
        
        Args:
            v_min: Minimum voltage squared (p.u.)
            v_max: Maximum voltage squared (p.u.)
        """
        m = self.model
        
        # Add voltage lower bound for all non-source buses
        def voltage_min_rule(m, bus_name, phase):
            if bus_name != self.source_bus:
                return m.v[bus_name, phase] >= v_min
            else:
                return pyo.Constraint.Skip
        
        m.VoltageMin = pyo.Constraint(m.BusPhaseSet, rule=voltage_min_rule)
        
        # Add voltage upper bound for all non-source buses
        def voltage_max_rule(m, bus_name, phase):
            if bus_name != self.source_bus:
                return m.v[bus_name, phase] <= v_max
            else:
                return pyo.Constraint.Skip
        
        m.VoltageMax = pyo.Constraint(m.BusPhaseSet, rule=voltage_max_rule)
    
    def add_line_limit_constraints(self):
        """
        Add line thermal limit constraints with slack variables for violations.
        Uses two linear approximations of the circular constraint S^2 = P^2 + Q^2:
        - K*P + Q <= S_max + line_violation
        - P + K*Q <= S_max + line_violation
        where K = sqrt(2) - 1
        This allows violations which are then penalized in the objective function.
        """
        m = self.model
        K = math.sqrt(2) - 1  # Linearization coefficient

        # # Segment widths as fractions of line rating
        # # [0-5%, 5-20%, 20-50%, 50%+]
        # seg_width_frac = [0.05, 0.15, 0.30, None]  # None = unbounded
        
        # # --- Bound each segment's width ---
        # def seg_upper_bound_rule(m, line_name, phase, seg):
        #     if seg_width_frac[seg] is None:
        #         return pyo.Constraint.Skip  # Last segment is unbounded
        #     line_rating = self.line_ampacity[line_name]
        #     return (m.line_viol_seg[line_name, phase, seg] 
        #             <= seg_width_frac[seg] * line_rating)
        
        # m.ViolSegBounds = pyo.Constraint(
        #     m.LinePhaseSet, m.ViolSegSet, rule=seg_upper_bound_rule
        # )

        def seg_upper_bound_rule(m, line_name, phase, seg):
            if seg == self.N_PWL - 1:
                return pyo.Constraint.Skip
            seg_width = self.x_max_fraction / (self.N_PWL - 1)
            return m.line_viol_seg[line_name, phase, seg] <= seg_width

        m.ViolSegBounds = pyo.Constraint(
            m.LinePhaseSet, m.ViolSegSet, rule=seg_upper_bound_rule
        )

        def total_viol(m, line_name, phase):
            return sum(m.line_viol_seg[line_name, phase, seg] 
                   for seg in m.ViolSegSet)
        
        def line_limit_rule_1(m, line_name, phase):
            line_rating = self.line_ampacity[line_name]
            return (K * m.P_line[line_name, phase] + m.Q_line[line_name, phase]
                    <= line_rating * (1 + total_viol(m, line_name, phase)))

        def line_limit_rule_2(m, line_name, phase):
            line_rating = self.line_ampacity[line_name]
            return (m.P_line[line_name, phase] + K * m.Q_line[line_name, phase]
                    <= line_rating * (1 + total_viol(m, line_name, phase)))

        m.LineLimitConstraint1 = pyo.Constraint(m.LinePhaseSet, rule=line_limit_rule_1)
        m.LineLimitConstraint2 = pyo.Constraint(m.LinePhaseSet, rule=line_limit_rule_2)
    
    def formulate_opf(self, penalty_weight=1, vsrc=1.0):
        """
        Formulate the complete OPF problem.
        
        Args:
            penalty_weight: Weight for line violation penalty in objective function (default: 1)
        """
        # Create Pyomo model
        self.model = pyo.ConcreteModel()
        
        # Create variables
        self.setup_optimization_variables()
        
        # Add constraints
        self.add_source_voltage(vsrc)
        self.add_power_balance_constraints()
        self.add_voltage_drop_constraints()
        self.add_voltage_limits()
        self.add_line_limit_constraints()
        self.add_voltvar_constraints()
        
        # Objective: Minimize line limit violations
        # Sum of all line violation slack variables weighted by penalty
        m = self.model
        # violation_penalty = sum(m.line_violation[line_name, phase]
        #                        for line_name, phase in m.LinePhaseSet)
        
        # slopes = [1.0, 5.0, 25.0, 125.0]
    
        # penalty = sum(
        #     slopes[seg] * m.line_viol_seg[line_name, phase, seg]
        #     for line_name, phase in m.LinePhaseSet
        #     for seg in m.ViolSegSet
        # )

        def pwl_slopes():
            """
            Slopes of X^2 PWL approximation.
            First (N_PWL-1) segments span [0, x_max_fraction] uniformly.
            Last segment is unbounded — its slope is set to the next value in the
            X^2 derivative sequence, ensuring penalty keeps increasing steeply.
            """
            seg_width = self.x_max_fraction / (self.N_PWL - 1)
            slopes = []
            for seg in range(self.N_PWL):
                x_mid = (seg + 0.5) * seg_width
                slopes.append(2.0 * x_mid)
            return slopes

        penalty = sum(
            pwl_slopes()[seg] * m.line_viol_seg[line_name, phase, seg]
            for line_name, phase in m.LinePhaseSet
            for seg in m.ViolSegSet
        )
        
        m.obj = pyo.Objective(
            expr=penalty_weight * penalty,
            sense=pyo.minimize
        )

        # m.obj = pyo.Objective(expr=penalty_weight * violation_penalty, sense=pyo.minimize)
        
    def solve(self, solver='mosek', verbose=True):
        """
        Solve the OPF problem.
        
        Args:
            solver: Solver name (e.g., 'gurobi', 'glpk', 'cplex', 'mosek')
            verbose: Whether to print solver output
        
        Returns:
            Dictionary with optimization results
        """
        # Create solver
        opt = SolverFactory(solver)

        # opt.options['mio_tol_rel_gap'] = 0.01
        opt.options['dparam.mio_tol_rel_gap'] = 0.01

        # Write optimization problem to a file in output directory
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "optimization_problem.txt"
        with open(output_path, "w") as f:
            f.write("Pyomo Model:\n")
            self.model.pprint(ostream=f)
        
        # Solve the problem
        if verbose:
            opt_results = opt.solve(self.model, tee=True)
        else:
            opt_results = opt.solve(self.model)
        
        # Extract results
        results = {
            'status': str(opt_results.solver.status),
            'termination_condition': str(opt_results.solver.termination_condition),
            'optimal_value': pyo.value(self.model.obj),
            'voltages': {},
            'power_flows': {},
            'substation_power': {},
            'capacitor_status': {},
            'regulator_tap': {},
            'line_violations': {},
            'der_q': {}
        } 
        
        # Extract voltage results
        if opt_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print(f"\nOptimization status: {pyo.TerminationCondition.optimal}\n")
            for bus_name, phase in self.model.BusPhaseSet:
                if bus_name not in results['voltages']:
                    results['voltages'][bus_name] = {}
                results['voltages'][bus_name][phase] = math.sqrt(pyo.value(self.model.v[bus_name, phase]))
            
            # Extract power flow results for lines (convert pu to kW, pu to kVAr)
            for line_name, phase in self.model.LinePhaseSet:
                if line_name not in results['power_flows']:
                    results['power_flows'][line_name] = {}
                results['power_flows'][line_name][phase] = {
                    'P': pyo.value(self.model.P_line[line_name, phase] * 1000 * self.power_base),  # Convert pu to kW
                    'Q': pyo.value(self.model.Q_line[line_name, phase] * 1000 * self.power_base)  # Convert pu to kVAr
                }

            # Extract power flow results for transformers (convert pu to kW, pu to kVAr)
            for xfmr_name, phase in self.model.XfmrPhaseSet:
                if xfmr_name not in results['power_flows']:
                    results['power_flows'][xfmr_name] = {}
                results['power_flows'][xfmr_name][phase] = {
                    'P': pyo.value(self.model.P_xfmr[xfmr_name, phase]) * 1000 * self.power_base,  # Convert pu to kW
                    'Q': pyo.value(self.model.Q_xfmr[xfmr_name, phase]) * 1000 * self.power_base   # Convert pu to kVAr
                }
            
            # Extract substation power (convert pu to kW, pu to kVAr)
            for source_bus, phase in self.model.SubPhaseSet:
                results['substation_power'][phase] = {
                    'P': pyo.value(self.model.P_sub[source_bus, phase]) * 1000 * self.power_base,  # Convert pu to kW
                    'Q': pyo.value(self.model.Q_sub[source_bus, phase]) * 1000 * self.power_base   # Convert pu to kVAr
                }
            
            # Extract capacitor status
            for cap_name in self.model.CapSet:
                status_value = pyo.value(self.model.cap_status[cap_name])
                results['capacitor_status'][cap_name] = {
                    'status': int(status_value) if status_value is not None else 0,
                    'status_raw': status_value
                }
            
            # Extract regulator tap positions
            for reg_name in self.model.RegSet:
                tap_positions = []
                for i in range(33):
                    tap_value = pyo.value(self.model.reg_tap[reg_name, i])
                    # sometime numerical precision issues cause tap_value to be slightly less than 1
                    tap_positions.append(1 if tap_value > 0.99 else 0)
                results['regulator_tap'][reg_name] = tap_positions

            # Extract line violation slack variables
            total_violations = 0
            for line_name, phase in self.model.LinePhaseSet:
                seg_values = []
                line_total = 0
                for seg in self.model.ViolSegSet:
                    val = pyo.value(self.model.line_viol_seg[line_name, phase, seg])
                    seg_values.append(val)
                    line_total += val
                
                if line_total > 1e-6:
                    if line_name not in results['line_violations']:
                        results['line_violations'][line_name] = {}
                    
                    rating = self.line_ampacity[line_name]
                    # overload_pct = (rating + line_total) / rating * 100 if rating > 0 else 0
                    overload_pct = (1 + line_total)* 100
                    
                    results['line_violations'][line_name][phase] = {
                        'total': line_total,
                        'overload_percent': overload_pct,
                        'tier_0_minor': seg_values[0],
                        'tier_1_moderate': seg_values[1],
                        'tier_2_severe': seg_values[2],
                        'tier_3_critical': seg_values[3],
                    }
                    total_violations += line_total
                    print(f"Line {line_name} phase {phase}: "
                        f"{overload_pct:.3f}% loaded")
            
            # extract DER power outputs (convert pu to kVAr)
            for der_bus, phase in self.model.DerBusPhaseSet:
                if der_bus not in results['der_q']:
                    results['der_q'][der_bus] = {}
                results['der_q'][der_bus][phase] = {
                    'Q': pyo.value(self.model.Q_inv[der_bus, phase]) * 1000 * self.power_base,  # Convert pu to kVAr
                }
            
            # save results to a JSON file inside the output directory
            # Output directory is already created
            output_file = output_dir / "opf_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Print summary of violations
            if total_violations > 1e-6:
                print(f"\nTotal line violations penalty: {results['optimal_value'] :.6f}")
                print(f"Number of violated lines: {sum(len(v) for v in results['line_violations'].values())}")
            else:
                print(f"\nNo line limit violations detected")
        else:
            print(f"\nOptimization did not converge to an optimal solution. Termination condition: {opt_results.solver.termination_condition}")
            exit()

        return results
