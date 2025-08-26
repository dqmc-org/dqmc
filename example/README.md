# Configuration File Format

The DQMC framework supports configuration via Boost.Program_options.

## Usage

```bash
# Use configuration file
./build/main --config example/config.cfg -o output/

# Override specific parameters via command line
./build/main --config example/config.cfg --model.type RepulsiveHubbard --mc.beta 4.0
```

## Configuration File Format

The new configuration uses a simple `key = value` format:

```ini
# Model parameters
model.type = AttractiveHubbard
model.hopping_t = 1.0
model.onsite_u = 4.0
model.chemical_potential = 0.0

# Lattice parameters
lattice.type = Square
lattice.size = 4
lattice.size = 4  # For rectangular lattices, repeat for each dimension

# Vector parameters use multiple lines
observables = filling_number
observables = double_occupancy
observables = kinetic_energy
```

## Available Parameters

Use `./build/main --help` to see all available configuration parameters and their descriptions.