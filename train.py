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
    best_total_reward = 0
    # Introduce a learning rate scheduler for decaying learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

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
        if episode % 100 == 0 and episode > 0:
            if total_reward > best_total_reward:
                print(f"Improved total reward from {best_total_reward} to {total_reward}")
                best_total_reward = total_reward
                config.target_update = min(config.target_update + 100, 300) # Decrease frequency if improving
                print(f"Increasing target update frequency to {config.target_update}")
            else:
                print(f"Total reward did not improve from {best_total_reward}")
                config.target_update = max(config.target_update - 100, 100)  # Increase frequency if not improving
                print(f"Decreasing target update frequency to {config.target_update}")
        
        # Update the learning rate using the scheduler
        scheduler.step()
    
        print(f"Episode {episode}/{config.num_episodes}, Total Reward: {total_reward}, Learning Rate: {scheduler.get_last_lr()[0]}, epsilon: {agent.epsilon}")
    torch.save(agent.model.state_dict(), config.model_save_path)

    print(f"Average reward: {np.mean(scores)}")
    # print the action q values for actions
    state = preprocess_state(env.reset())
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.model(state)
    q_values = q_values.squeeze().cpu().numpy()
    print("Action Q-values:")
    for i in range(len(q_values)):
        print(f"Action {i}: {q_values[i]}")
