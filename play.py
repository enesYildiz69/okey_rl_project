import torch
import numpy as np
from utils import preprocess_state
from environment import OkeyEnvironment
from agent import DQNAgent
from config import Config

def play_game_interactively(model_path):
    # Initialize environment and agent
    env = OkeyEnvironment()
    config = Config()
    agent = DQNAgent(env.state_size, env.action_size, config)

    # Load the trained model
    agent.model.load_state_dict(torch.load(model_path))

    # Get the starting hand from the user
    print("Provide your starting hand (in the format color1,number1 color2,number2 ...):")
    hand_input = input()
    hand = [tuple(map(int, card.split(','))) for card in hand_input.split()]
    env.hand = hand

    while True:
        state = preprocess_state(env.get_state())

        # Agent suggests an action
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        print(f"Suggested action: {action}")

        if action in range(5):
            print(f"Discard card: {env.hand[action]}")
            # Discarding the card
            env.hand[action] = env.dummy_card
            if len(env.deck) > 0:
                print("Please provide the new card from the deck (in format color,number):")
                new_card_input = input()
                new_card = tuple(map(int, new_card_input.split(',')))
                env.hand[action] = new_card
        elif action >= 5 and action <= 18:
            combination, _ = env.valid_combinations[action]
            print(f"Making a combination with: {combination}")
            env.remove_combination_cards_from_hand(*combination)

            # Draw new cards based on how many cards were removed and how many are left in the deck
            num_cards_to_draw = min(3, len(env.deck))
            for i in range(num_cards_to_draw):
                print(f"Please provide the new card {i+1} (in format color,number):")
                new_card_input = input()
                new_card = tuple(map(int, new_card_input.split(',')))
                for j in range(len(env.hand)):
                    if env.hand[j] == env.dummy_card:
                        env.hand[j] = new_card
                        break

        # Check if the game is done
        done = env.check_if_done()
        if done:
            print("Game over!")
            break

        # Display the current hand after the action
        print(f"Your current hand: {env.hand}")

if __name__ == "__main__":
    model_path = "dqn_model_v2.pth"  # Path to your trained model
    play_game_interactively(model_path)
