# PokerDQN - Reinforcement Learning in Poker

This project implements a reinforcement learning agent using a Deep Q-Network (DQN) to play poker. The goal is to train the agent to learn optimal strategies and make intelligent decisions in a simulated poker environment.

## Table of Contents

- [Description](#Description)
- [Features](#Features)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [Usage](#Usage)
- [Training](#Training)
- [Common Issues](#Common-Issues)
- [Contributions](#contributions)
- [License](#license)
  
## Description

This project uses a DQN algorithm to train a poker-playing agent. The poker environment is based on Texas Hold'em rules, where the agent makes decisions such as betting, checking, or folding based on its past experiences and the rewards received.

The agent is trained by simulating poker games, where it learns to maximize rewards (chips won) over multiple training episodes. The model uses a reinforcement learning approach, balancing exploration and exploitation to improve decision-making over time.

## Features

- Texas Hold'em poker simulation.
- Agent trained via a Deep Q-Network (DQN).
- Training and hyperparameter tuning.
- Action, bet, and reward history tracking.
- Configurable number of rounds and episodes.
- Reinforcement learning based on cumulative experience.

## Requirements

Ensure you have the following dependencies installed before running the project:

- Python 3.8+
- TensorFlow or PyTorch (depending on DQN backend)
- NumPy
- scikit-learn
- gym (for the simulation environment)
- Matplotlib (optional, for results visualization)
- Other packages listed in `requirements.txt`

Install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository:

 ```bash
git clone https://github.com/your-username/PokerDQN.git
cd PokerDQN
pip install -r requirements.txt
 ```
2. Install the dependencies:

```bash
pip install -r requirements.txt
Configure the training parameters in config.yaml according to your preferences.
```
## Usage

To train the agent, run the following command in the terminal:

```bash
python main.py
```
During execution, the agent will play several poker rounds against simulated players and learn from its actions using the DQN algorithm.

Configuration Parameters
Training parameters (such as learning rate, number of episodes, etc.) can be adjusted in the config.yaml file. Make sure to review and modify these values as needed before starting the training process.

## Training

The training process consists of multiple episodes where the agent plays several hands of poker. The agent uses a deep neural network to estimate the Q-values of each possible action and adjusts its decision-making policy to maximize future rewards.

### Common Error Example

If you encounter an error similar to:

valueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions.

This typically occurs when input matrix dimensions are inconsistent. Ensure that the state and action arrays are correctly aligned in terms of dimensions and format.

## Common Issues

### Input Dimensions

If you encounter a `ValueError: setting an array element with a sequence`, ensure that the states stored in the replay buffer have consistent dimensions and formats.

### Hyperparameter Tuning

Training can be sensitive to hyperparameter settings such as the exploration rate (epsilon), learning rate, and neural network size. Adjust these values in the `config.yaml` file if the agent is not learning as expected.

## Contributions

Contributions are welcome! If you would like to contribute, follow these steps:

1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
