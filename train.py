import torch
import torch.optim as optim
from collections import deque
import random
import numpy as np
from utils import preprocess_state

def train_agent(env, agent, config):
    scores = []
    replay_buffer = deque(maxlen=config.buffer_size)
    optimizer = optim.Adam(agent.model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    for episode in range(config.num_episodes):
        state = preprocess_state(env.reset())
        total_reward = 0

        for t in range(config.max_steps_per_episode):
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)
            next_state, reward, done = env.step(action)
            next_state = preprocess_state(next_state)
            replay_buffer.append((state, action, reward, next_state, done))
            
            if len(replay_buffer) > config.batch_size:
                minibatch = random.sample(replay_buffer, config.batch_size)
                agent.replay(minibatch, optimizer, criterion)

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)
        if episode % config.target_update == 0:
            agent.update_target_network()

        # Dynamic adjustment of the target network update frequency
        if episode % 500 == 0 and episode > 0:
            if total_reward > best_total_reward:
                print(f"Improved total reward from {best_total_reward} to {total_reward}")
                best_total_reward = total_reward
                config.target_update = max(config.target_update + 50, 1000)  # Decrease frequency if improving
                print(f"Increasing target update frequency to {config.target_update}")
            else:
                print(f"Total reward did not improve from {best_total_reward}")
                config.target_update = max(config.target_update - 50, 100)  # Increase frequency if not improving
                print(f"Decreasing target update frequency to {config.target_update}")
    
        print(f"Episode {episode}/{config.num_episodes}, Total Reward: {total_reward}")

    torch.save(agent.model.state_dict(), config.model_save_path)
