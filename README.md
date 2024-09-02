# Okey RL Project

## Overview
This project implements a Reinforcement Learning (RL) agent using Deep Q-Networks (DQN) to play the "Okey" game. The agent is trained to make decisions to maximize the total score based on specific combinations of cards.

## File Structure
- `main.py`: Entry point to run the training.
- `environment.py`: Contains the game environment and logic.
- `model.py`: Defines the neural network architecture.
- `train.py`: Contains the training loop.
- `agent.py`: Defines the DQN agent.
- `replay_buffer.py`: Implements the replay buffer for experience replay.
- `utils.py`: Utility functions for preprocessing and other tasks.
- `config.py`: Configuration file with hyperparameters and settings.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/enesYildiz69/okey_rl_project.git
    cd okey_rl_project
    ```

2. Install dependencies:
    ```bash
    pip install torch numpy
    ```

3. Run the training script:
    ```bash
    python main.py
    ```

4. The trained model will be saved to `dqn_model.pth` after training is complete.

## Future Work
- Improve the game logic for more complex strategies.
- Tune hyperparameters for better performance.
- Implement a more advanced RL algorithm such as Double DQN or Dueling DQN.
