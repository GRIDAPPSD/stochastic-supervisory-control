# Stochastic Supervisory Control

Optimal Power Flow (OPF) solver for multi-phase radial distribution networks with stochastic load and DER scenarios. Supports capacitor bank switching, voltage regulator tap control, and DER volt-VAR optimization using the LinDistFlow model.

## Project Structure

```
feeder-models/         # OpenDSS distribution network models
  ├── 123Bus/          # IEEE 123-bus test feeder
  └── J1/              # Real-world feeder with PV systems
scenario_generator/    # Stochastic load & generation scenario creation
stochastic_opf/        # Core OPF formulation and solver
  └── output/          # Results (JSON) and visualizations
```

## Features

- **Multi-phase LinDistFlow** formulation with phase-aware impedance matrices
- **Capacitor bank control** — binary on/off switching via Big-M linearization
- **Voltage regulator taps** — 33-position discrete tap selection
- **DER volt-VAR support** — IEEE 1547 Category-B curve (SOS2 piecewise-linear)
- **Soft thermal limits** — piecewise quadratic penalty for line overloads
- **Scenario generation** — 15-minute resolution load multipliers with seasonal/temporal classification
- **Validation** — OPF results cross-checked against OpenDSS power flow

## Requirements

- Python 3
- [OpenDSS](https://www.epri.com/pages/sa/opendss) via `opendssdirect`
- [Pyomo](http://www.pyomo.org/) with a supported solver (MOSEK, Gurobi, or CPLEX)
- NumPy, Pandas, Matplotlib, NetworkX

## Usage

```bash
cd stochastic_opf
python main.py
```

The solver extracts the network from the OpenDSS model, builds the Pyomo OPF formulation, solves it, and writes results to `stochastic_opf/output/`.

## License

See [LICENSE](LICENSE) for details.
