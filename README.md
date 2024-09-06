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
- `play.py`: File to play with the trained model.
- `okey_algorithm_cpp/`: It is the folder for an independent algorithm which also plays the game.

## How to Run DQN model training
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

## How to Run the Algorithm
1. Compile the file
   ```bash
    g++ okey_game_algorithm.cpp -o okey_game -std=c++11
   ```

3. Run
   ```bash
   ./okey_game
   ```

## Future Work
- Improve the game logic for more complex strategies.
- Tune hyperparameters for better performance.
- Implement a more advanced RL algorithm such as Double DQN or Dueling DQN.

