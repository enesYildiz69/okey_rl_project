from environment import OkeyEnvironment
from agent import DQNAgent
from train import train_agent
from config import Config

def main():
    config = Config()
    env = OkeyEnvironment()
    agent = DQNAgent(env.state_size, env.action_size, config)
    
    train_agent(env, agent, config)

if __name__ == "__main__":
    main()
