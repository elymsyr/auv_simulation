# MPC Control System Development Repository

This repository contains the full development and testing environment for Model Predictive Control (MPC) systems, including simulation, communication, and learning phases.

## Project Structure

- **connection/**  
  Python modules for testing communication, code definitions, and visualization.
  - `bridge.py`: Communication bridge between test/Model and simulation.
  - `codes.py`: Code definitions and utilities.
  - `comm.py`: Communication test interface.
  - `visualize.py`: Visualization tools for data EnvironmentMap.save and results.

- **simulation/**  
  Simulation environment, assets, and configuration files.
  - `config.json`: Simulation configuration parameters.
  - `Assets/`, `Models/`, `Scenes/`, `Scripts/`, `Settings/`, `TutorialInfo/`: Unity or simulation assets and scripts.

- **test/**
  Testing and validation scripts for different phases.
  - `il/`: Imitation learning tests.
  - `il-map/`: Map-based imitation learning.
  - `map-entegrated-mpc/`: MPC tests with mapping integration.
  - `Model/`: Model-specific tests.
  - `mpc/`: MPC algorithm tests.
  - `ppo/`: Proximal Policy Optimization (PPO) reinforcement learning tests.

- **Library/**, **Logs/**, **Packages/**, **ProjectSettings/**, **Temp/**, **UserSettings/**  
  Standard project folders for dependencies, logs, settings, and temporary files.

## Features

- **MPC Algorithm Development**: Core algorithms and utilities for MPC.
- **Simulation Environment**: Configurable simulation for SIL, MIL, and HIL testing.
- **Imitation Learning**: Scripts and tests for imitation learning phases.
- **Reinforcement Learning**: PPO-based learning and testing.
- **Visualization**: Tools for visualizing simulation and control results.
- **Comprehensive Testing**: Organized test suites for all development phases.

# License
[GNU GENERAL PUBLIC LICENSE](LICENSE)