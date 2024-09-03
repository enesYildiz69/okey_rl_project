class Config:
    def __init__(self):
        self.num_episodes = 600  # TODO This field shall be tuned
        self.max_steps_per_episode = 25  # Maximum steps per game
        self.buffer_size = 200000  # Replay buffer size
        self.batch_size = 128  # Batch size for replay
        self.learning_rate = 0.00025 # TODO This field shall be tuned
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.999 # TODO This field shall be tuned
        self.epsilon_min = 0.01
        self.target_update = 100  # TODO This field shall be tuned
        self.model_save_path = 'dqn_model_v.pth'  # Path to save the trained model
