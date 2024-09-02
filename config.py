class Config:
    def __init__(self):
        self.num_episodes = 10000  # Set to 1 million or more as needed
        self.max_steps_per_episode = 1000  # Maximum steps per game
        self.buffer_size = 10000  # Replay buffer size
        self.batch_size = 64  # Batch size for replay
        self.learning_rate = 0.001
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update = 1000  # Update target network every 1000 episodes
        self.log_interval = 100  # Log progress every 100 episodes
        self.model_save_path = 'dqn_model.pth'  # Path to save the trained model
