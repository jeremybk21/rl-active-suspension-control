# RL Active Suspension Control

Advanced implementation of deep reinforcement learning for active vehicle suspension control using PyTorch and Stable-Baselines3.

## Project Overview
This project implements reinforcement learning algorithms to control an active vehicle suspension system, aiming to optimize ride comfort and handling performance. The system uses a quarter-car model with nonlinear dynamics and continuous action space control.

## Key Features
- Custom Gymnasium environment with nonlinear quarter-car dynamics
- Implementation of TD3 and PPO algorithms for continuous control
- LQR-based replay buffer initialization for improved learning
- Configurable simulation parameters and road profile generation

## Installation
```bash
# Clone the repository
git clone https://github.com/jeremybk21/rl-active-suspension-control.git
cd rl-active-suspension-control

# Create and activate virtual environment (optional)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
# Train the model
python training/train_NonlinearQuarterCar.py

# Monitor training progress
tensorboard --logdir logs/
```

## Environment Details
The quarter-car model includes:
- 4 state variables: sprung mass position/velocity, unsprung mass position/velocity
- Continuous action space: actuator force [-1000N, 1000N]
- Custom reward function optimizing ride comfort
- Configurable road profile generation

## Results
- Improved ride comfort compared to passive suspension
- Successful disturbance rejection for various road profiles
- Stable control performance across different operating conditions

## License
This project is licensed under the MIT License - see the LICENSE file for details.