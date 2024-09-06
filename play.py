import torch
import numpy as np
from environment import OkeyEnvironment
from agent import DQNAgent
from utils import preprocess_state
from config import Config

# Map color initials to numeric values used in the environment
color_to_number = {
    "r": 0,  # red
    "y": 1,  # yellow
    "b": 2   # blue
}

number_to_color = {v: k for k, v in color_to_number.items()}  # Reverse lookup

# Helper function to parse the initial hand from user input
def parse_hand_input(hand_input):
    hand = []
    for card in hand_input.split():
        color, number = card.split(',')
        hand.append((color_to_number[color], int(number)))
    return hand

# Helper function to format the hand for display
def format_hand(hand):
    return [(number_to_color[color], number) for color, number in hand]

# Helper function to parse new card input
def parse_new_card_input(new_card_input):
    color, number = new_card_input.split(',')
    return (color_to_number[color], int(number))

def play_game_interactively(model_path):
    # Initialize environment and agent
    env = OkeyEnvironment()
    config = Config()
    agent = DQNAgent(env.state_size, env.action_size, config)
    
    # Load the trained model
    agent.model.load_state_dict(torch.load(model_path))

    # Get the starting hand from the user
    print("Provide your starting hand (in the format color1,number1 color2,number2 ...):")
    hand_input = input().strip()
    env.hand = parse_hand_input(hand_input)

    total_points = 0

    while True:
        state = preprocess_state(env.get_state())

        # Agent suggests an action
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        print(f"Suggested action: {action}")

        if action in range(0, 8):
            # Same-number combination
            combination = env.valid_combinations[action][0]
            print(f"Making a same-number combination with: {format_hand([combination])}")
            env.remove_combination_cards_from_hand(*combination)
            points = env.calculate_reward_for_action(action, same_color=True)
            num_cards_to_draw = min(3, len(env.deck))
            for i in range(num_cards_to_draw):
                if len(env.deck) > 0:
                    print(f"Please provide the new card {i+1} (in format color,number):")
                    new_card_input = input().strip()
                    new_card = parse_new_card_input(new_card_input)
                    for j in range(5):
                        if env.hand[j] == env.dummy_card:
                            env.hand[j] = new_card
                            break

        elif action in range(8, 14):
            # Same-color sequential combination
            combination = env.valid_combinations[action][0]
            print("combination: ", combination)
            print(f"Making a same-color sequential combination with: {format_hand(combination)}")
            env.remove_combination_cards_from_hand(*combination)
            points = env.calculate_reward_for_action(action, same_color=True)
            num_cards_to_draw = min(3, len(env.deck))
            for i in range(num_cards_to_draw):
                if len(env.deck) > 0:
                    print(f"Please provide the new card {i+1} (in format color,number):")
                    new_card_input = input().strip()
                    new_card = parse_new_card_input(new_card_input)
                    for j in range(5):
                        if env.hand[j] == env.dummy_card:
                            env.hand[j] = new_card
                            break

        elif action in range(14, 20):
            # Different-color sequential combination
            combination = env.valid_combinations[action][0]
            # Ensure combination is a list of tuples for formatting
            if isinstance(combination, tuple):
                combination = [combination]
            print(f"Making a different-color sequential combination with: {format_hand(combination)}")
            env.remove_combination_cards_from_hand(*combination)
            points = env.calculate_reward_for_action(action, same_color=False)
            num_cards_to_draw = min(3, len(env.deck))
            for i in range(num_cards_to_draw):
                if len(env.deck) > 0:
                    print(f"Please provide the new card {i+1} (in format color,number):")
                    new_card_input = input().strip()
                    new_card = parse_new_card_input(new_card_input)
                    for j in range(5):
                        if env.hand[j] == env.dummy_card:
                            env.hand[j] = new_card
                            break

        elif action in range(20, 44):
            # Discard action based on specific card
            card_to_discard = env.get_card_from_action_index(action)
            print(f"Discarding card: {format_hand([card_to_discard])[0]}")
            index = env.hand.index(card_to_discard)
            env.hand[index] = env.dummy_card
            env.discarded.append(card_to_discard)
            points = 0  # No points for discarding
            if len(env.deck) > 0:
                print("Please provide the new card from the deck (in format color,number):")
                new_card_input = input().strip()
                new_card = parse_new_card_input(new_card_input)
                env.hand[index] = new_card

        total_points += points

        # Display the current hand after the action
        print(f"Your current hand: {format_hand(env.hand)}")
        print(f"Points received from this action: {points}, Total points: {total_points}")
        print(f"Remaining cards in the deck: {len(env.deck)}")

        # Check if the game is done
        done = env.check_if_done()
        if done:
            print("Game over!")
            break


if __name__ == "__main__":
    model_path = "dqn_model_v1.pth"  # Path to your trained model
    play_game_interactively(model_path)
