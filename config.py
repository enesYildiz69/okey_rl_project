class Config:
    def __init__(self):
        self.num_episodes = 50000  # Set to 1 million or more as needed
        self.max_steps_per_episode = 500  # Maximum steps per game
        self.buffer_size = 50000  # Replay buffer size
        self.batch_size = 128  # Batch size for replay
        self.learning_rate = 0.0005
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.target_update = 500  # Update target network every 500 episodes
        self.log_interval = 100  # Log progress every 100 episodes
        self.model_save_path = 'dqn_model_v2.pth'  # Path to save the trained model
