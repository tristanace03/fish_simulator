# Fish Simulator: Neuroevolution Survival Game

A Python-based simulation demonstrating neuroevolution, where schools of fish learn to survive predators through artificial neural networks. Watch as fish evolve intelligent evasion strategies over generations!


## Features

- **Neuroevolution**: Fish are controlled by simple neural networks that evolve over generations
- **Real-time Visualization**: Interactive Pygame-based simulation with live rendering
- **Predator Evasion**: Fish learn to avoid predators while staying within the arena boundaries
- **Fitness-based Selection**: Survival time, wall avoidance, and energy efficiency drive evolution
- **Training & Replay Modes**: Train new populations or replay the best evolved behavior
- **Configurable Parameters**: Easily adjust simulation settings, network architecture, and evolution parameters

## How It Works

Each fish has a tiny neural network (8 inputs → 12 hidden → 2 outputs) that processes sensory information:
- Self velocity and heading
- Distance and direction to nearest predator
- Distance to arena walls

The network outputs control turning and speed adjustments. Through generations of evolution:
- Fitter fish (those surviving longer) reproduce
- Mutations introduce variation
- Elitism preserves the best performers

## Requirements

- Python 3.7+
- NumPy
- Pygame

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tristanace03/fish_simulator.git
   cd fish_simulator
   ```

2. Install dependencies:
   ```bash
   pip install numpy pygame
   ```

## Usage

### Training Mode (Default)
Run the simulation to train a new population:
```bash
python src/fish.py
```

The simulation will:
- Run for 200 generations by default
- Save the best neural network to `runs/best.json`
- Automatically switch to replay mode when training completes

### Replay Mode
Watch the best evolved fish in action:
```bash
python src/fish.py  # Then press SPACE during training to switch to replay
# OR modify the main block to call run_replay() directly
```

## Controls

- **SPACE**: Toggle between Training and Replay modes
- **R**: Reset the current replay episode
- **ESC**: Quit the simulation

## Configuration

Modify the `CONFIG` dictionary in `src/fish.py` to adjust:

- **Simulation Settings**: Window size, FPS, arena radius
- **Population**: Number of fish and predators
- **Evolution Parameters**: Generations, mutation rate, elitism fraction
- **Network Architecture**: Input/hidden/output dimensions
- **Fitness Weights**: Penalties and bonuses for different behaviors

## Project Structure

```
fish_simulator/
├── README.md
├── src/
│   └── fish.py          # Main simulation code
└── runs/
    └── best.json        # Saved best neural network
```

## Fitness Function

Fish fitness combines multiple factors:
- **Survival Time**: Base score for staying alive
- **Wall Hits**: Penalty for hitting arena boundaries
- **Jerkiness**: Penalty for erratic movement
- **Predator Distance**: Reward for staying far from predators
- **Speed Efficiency**: Reward fast movement near predators, penalize wasteful speed when safe

