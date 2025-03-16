# DDPG Controller for Autonomous Underwater Vehicle Yaw Motion Control

This repository contains the implementation of a Deep Deterministic Policy Gradient (DDPG) controller for simulating and controlling autonomous underwater vehicle (AUV) yaw motion. The code provides a complete framework for training, evaluating, and testing DDPG control systems for third-order linear models.

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Training the DDPG Controller](#training-the-ddpg-controller)
6. [Testing and Evaluation](#testing-and-evaluation)
7. [Simulation Environment Parameters](#simulation-environment-parameters)
8. [Performance Metrics](#performance-metrics)
9. [Saving and Loading Models](#saving-and-loading-models)
10. [Troubleshooting](#troubleshooting)

## Introduction

This project implements a DDPG reinforcement learning controller for autonomous underwater vehicle yaw motion control. The system uses a third-order linear model to represent the AUV dynamics and trains a neural network-based controller to track reference signals. The implementation includes actuator dynamics, time delays, and disturbance models to simulate realistic operating conditions.

## System Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- Control (Python Control Systems Library)

## DDPG Hyperparameters and Network Architecture

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Actor Learning Rate | 1e-4 | Learning rate for actor network optimization |
| Critic Learning Rate | 1e-3 | Learning rate for critic network optimization |
| Discount Factor (γ) | 0.99 | Future reward discount factor |
| Soft Update Rate (τ) | 0.005 | Target network update coefficient |
| Batch Size | 64 | Mini-batch size for optimization |
| Replay Buffer Size | 10,000 | Maximum transitions stored in replay memory |
| Hidden Layer Dimension | 64 | Dimension of hidden layers in both networks |
| Exploration Noise | OU Process | Ornstein-Uhlenbeck noise process for exploration |
| OU Noise θ | 0.15 | Mean reversion parameter in OU process |
| OU Noise σ | 0.2 | Volatility parameter in OU process |
| Max Action Value | 20.0 | Maximum allowed control output |
| Episodes | 200 | Number of training episodes |
| Evaluation Interval | 10 | Episodes between evaluations |
| Time Step (dt) | 0.01 | Simulation time step in seconds |
| Episode Length | 25.0 | Maximum episode duration in seconds |
| Optimizer | Adam | Optimization algorithm for both networks |

### Network Architectures

#### Actor Network
```
Actor(
  (net): Sequential(
    (0): Linear(in_features=4, out_features=64)  # Input layer: state dimension to hidden dim
    (1): ReLU()                                  # Activation function
    (2): Linear(in_features=64, out_features=64) # First hidden layer
    (3): ReLU()                                  # Activation function
    (4): Linear(in_features=64, out_features=1)  # Output layer: action dimension
    (5): Tanh()                                  # Output activation scaled by max_action
  )
)
```

#### Critic Network
```
Critic(
  (net): Sequential(
    (0): Linear(in_features=5, out_features=64)  # Input layer: state_dim + action_dim
    (1): ReLU()                                  # Activation function
    (2): Linear(in_features=64, out_features=64) # First hidden layer
    (3): ReLU()                                  # Activation function
    (4): Linear(in_features=64, out_features=1)  # Output layer: single Q-value
  )
)
```

### State and Action Space

| Space | Dimension | Description |
|-------|-----------|-------------|
| State Space | 4 | [output, error, derivative_error, integral_error] |
| Action Space | 1 | Control input to the system (continuous) |
| State Normalization | None | No normalization applied to states |
| Action Range | [-20.0, 20.0] | Output is scaled and clipped to this range |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/auv-ddpg-controller.git
cd auv-ddpg-controller

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib control
```

## Project Structure

- `ddpg_implementation.py` - Main DDPG algorithm implementation
- `train.py` - Training script for the DDPG controller
- `test.py` - Testing and evaluation script
- `everything_env.py` - Environment simulation with realistic disturbances and actuator dynamics
- `models/` - Directory for saving trained models
- `Simulation Outcomes/` - Directory for saving simulation results and plots

## Training the DDPG Controller

To train a new DDPG controller:

```bash
python train.py --episodes 200 --batch-size 64 --hidden-dim 64 --actor-lr 1e-4 --critic-lr 1e-3
```

### Training Parameters

- `--episodes`: Number of training episodes (default: 200)
- `--batch-size`: Size of batch for training (default: 64)
- `--hidden-dim`: Dimension of hidden layers in networks (default: 64)
- `--actor-lr`: Learning rate for the actor network (default: 1e-4)
- `--critic-lr`: Learning rate for the critic network (default: 1e-3)
- `--gamma`: Discount factor (default: 0.99)
- `--tau`: Soft update parameter (default: 0.005)
- `--buffer-size`: Replay buffer size (default: 10000)
- `--eval-interval`: Episodes between evaluations (default: 10)

### Training Process

1. The script initializes the EverythingEnv environment and DDPG agent
2. For each episode:
   - Reset environment to initial state
   - Execute actions according to current policy plus exploration noise
   - Store transitions in replay buffer
   - Update neural networks using sampled mini-batches
3. Evaluate performance every `eval-interval` episodes
4. Save the best model based on evaluation performance

## Testing and Evaluation

To test a trained DDPG controller:

```bash
python test.py --model-path models/best_ddpg_model --reference 1.0 --disturbance 0.0 --noise 0.0 --delay 0.42
```

### Testing Parameters

- `--model-path`: Path to the saved model (default: "models/best_ddpg_model")
- `--reference`: Reference value for tracking (default: 1.0)
- `--max-time`: Maximum simulation time (default: 10.0)
- `--disturbance`: Disturbance value (default: 0.0)
- `--disturbance-time`: Time when disturbance is applied (default: 15.0)
- `--noise`: Standard deviation of measurement noise (default: 0.0)
- `--noise-time`: Time when noise is applied (default: 20.0)
- `--use-actuator`: Use actuator dynamics (default: True)
- `--use-uncertainty`: Apply gain and delay uncertainty (default: True)
- `--gain`: Input gain uncertainty (default: 1.0)
- `--delay`: Input delay in seconds (default: 0.42)

## Simulation Environment Parameters

The `EverythingEnv` class simulates the AUV dynamics with the following configurable parameters:

### System Model

The AUV yaw motion is modeled as a third-order linear system with the following state-space representation:

```
A = [
    [0,     1,      0],
    [0,     0,      1],
    [0, -5.375, -5.235]
]

B = [
    [0],
    [1.816],
    [-3.770]
]

C = [[1, 0, 0]]
D = [[0]]
```

### Environment Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| dt | 0.01 | Time step in seconds |
| TOTAL_TIME | 25.0 | Maximum episode duration in seconds |
| MAX_STEPS | 2500 | Maximum number of steps per episode |
| max_action | 20.0 | Maximum control input magnitude |
| reference | 1.0 | Reference value for tracking |
| disturbance_time | 15.0 | Time when disturbance is applied |
| disturbance_value | 0.0 | Magnitude of step disturbance |
| noise_time | 20.0 | Time when measurement noise starts |
| noise_std | 0.0 | Standard deviation of measurement noise |
| use_actuator | True | Enable actuator dynamics simulation |
| Sat | 20.0 | Actuator saturation limit |
| RSat | 30.0 | Actuator rate limit |
| use_uncertainty | True | Enable input uncertainty (gain and delay) |
| gain | 1.0 | Input gain multiplier |
| delay | 0.42 | Input delay in seconds |
| control_mode | "ddpg" | Control mode (options: "ddpg", "td3", "sine", "step") |

### Actuator Model

The actuator dynamics include both magnitude saturation and rate limiting:

```python
def actuator_model(u_desired, u_ac_prev, dt, RSat, Sat):
    delta_u = u_desired - u_ac_prev
    if delta_u > RSat * dt:
        u_ac_new = u_ac_prev + RSat * dt
    elif delta_u < -RSat * dt:
        u_ac_new = u_ac_prev - RSat * dt
    else:
        u_ac_new = u_desired
    u_ac_new = np.clip(u_ac_new, -Sat, Sat)
    return u_ac_new
```

## Performance Metrics

The system calculates and saves the following performance metrics:

- Steady-state error (e_ss)
- Integral of Squared Error (ISE)
- Integral of Time multiplied by Absolute Error (ITAE)
- Integral of Absolute Control Effort (IACE)
- Integral of Absolute Control Effort Rate (IACER)
- Maximum control value

These metrics are saved to `Criteria_ddpg.txt` after simulation.

### Reward Function

The reward function used during training penalizes both tracking error and control effort:

```
reward = -(10.0 * (integral_error**2) + 1.0 * (action**2))
```

This balances the competing objectives of:
1. Minimizing tracking error (weighted by factor of 10.0)
2. Minimizing control effort (weighted by factor of 1.0)

## Saving and Loading Models

Models are automatically saved during training when performance improves:

```python
agent.save("models/best_ddpg_model")
```

To load a previously trained model:

```python
agent.load("models/best_ddpg_model")
```

The model files include:
- `models/best_ddpg_model_actor.pth` - Actor network weights
- `models/best_ddpg_model_critic.pth` - Critic network weights

## Troubleshooting

### Common Issues:

1. **Training instability**: Try reducing learning rates or increasing batch size
2. **Poor tracking performance**: 
   - Increase number of training episodes
   - Adjust reward function weights
   - Increase network hidden layer dimensions
3. **Simulation errors**:
   - Check system matrices for correct dimensions
   - Verify time step is appropriate for system dynamics
   - Ensure actuator limits are reasonable for the system

### Debugging Tips:

- Enable verbose output during training with `--verbose` flag
- Plot training rewards to identify convergence issues
- Save intermediate models during training to track progress
- Use smaller time steps if numerical instability occurs
