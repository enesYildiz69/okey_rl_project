import torch
import numpy as np
from utils import preprocess_state
from environment import OkeyEnvironment
from agent import DQNAgent
from config import Config

class Play():
    remaining_cards_in_deck = 24

# Mapping for colors to numbers and vice versa
color_to_number = {
    "r": 0,
    "y": 1,
    "b": 2
}

number_to_color = {v: k for k, v in color_to_number.items()}

def format_combination(combination):
    """Ensure the combination is correctly formatted with color and number pairs."""
    if isinstance(combination, list):
        try:
            return [(number_to_color[color], number) for color, number in combination]
        except Exception as e:
            print(f"Error formatting combination: {combination} with error: {e}")
            return []
    else:
        print(f"Unexpected type for combination: {combination}")
        return []

def parse_hand_input(hand_input):
    hand = []
    for card in hand_input.split():
        color, number = card.split(',')
        hand.append((color_to_number[color], int(number)))
    return hand

def format_hand(hand):
    return [(number_to_color[color], number) for color, number in hand]

def parse_new_card_input(new_card_input):
    color, number = new_card_input.split(',')
    return (color_to_number[color], int(number))

def play_game_interactively(model_path):
    # Initialize environment and agent
    env = OkeyEnvironment()
    config = Config()
    agent = DQNAgent(env.state_size, env.action_size, config)
    done = False
    # Load the trained model
    agent.model.load_state_dict(torch.load(model_path))

    # Get the starting hand from the user
    print("Provide your starting hand (in the format color1,number1 color2,number2 ...):")
    Play.remaining_cards_in_deck -= 5
    hand_input = input()
    env.hand = parse_hand_input(hand_input)

    total_points = 0

    while True:
        state = preprocess_state(env.get_state())

        # Agent suggests an action
        valid_actions = env.get_valid_actions()
        action = agent.act(state, valid_actions)
        print(f"Suggested action: {action}")

        if action in range(5):
            print(f"Discard card: {format_hand([env.hand[action]])[0]}")
            # Discarding the card
            env.hand[action] = env.dummy_card
            if Play.remaining_cards_in_deck > 0:
                print("Please provide the new card from the deck (in format color,number):")
                new_card_input = input()
                new_card = parse_new_card_input(new_card_input)
                env.hand[action] = new_card
                Play.remaining_cards_in_deck -= 1
            points = 0  # No points for discarding
        elif action >= 5 and action <= 12:
            combination, _ = env.valid_combinations[action]
            print(f"Making a same-number combination with: {format_combination(combination)}")
            env.remove_combination_cards_from_hand(*combination)
            points = env.calculate_reward_for_action(action, same_color=True)
            num_cards_to_draw = min(3, Play.remaining_cards_in_deck)
            for i in range(num_cards_to_draw):
                print(f"Please provide the new card {i+1} (in format color,number):")
                new_card_input = input()
                new_card = parse_new_card_input(new_card_input)
                for j in range(5):
                    if env.hand[j] == env.dummy_card:
                        env.hand[j] = new_card
                        Play.remaining_cards_in_deck -= 1
                        break
        elif action >= 13 and action <= 18:
            combination, _ = env.valid_combinations[action]
            print(f"Making a same-color sequential combination with: {format_combination(combination)}")
            env.remove_combination_cards_from_hand(*combination)
            points = env.calculate_reward_for_action(action, same_color=True)
            num_cards_to_draw = min(3, Play.remaining_cards_in_deck)
            for i in range(num_cards_to_draw):
                print(f"Please provide the new card {i+1} (in format color,number):")
                new_card_input = input()
                new_card = parse_new_card_input(new_card_input)
                for j in range(len(env.hand)):
                    if env.hand[j] == env.dummy_card:
                        env.hand[j] = new_card
                        Play.remaining_cards_in_deck -= 1
                        break
        elif action >= 19 and action <= 24:
            combination, _ = env.valid_combinations[action]
            print(f"Making a different-color sequential combination with: {format_combination(combination)}")
            env.remove_combination_cards_from_hand(*combination)
            points = env.calculate_reward_for_action(action, same_color=False)
            num_cards_to_draw = min(3, Play.remaining_cards_in_deck)
            for i in range(num_cards_to_draw):
                print(f"Please provide the new card {i+1} (in format color,number):")
                new_card_input = input()
                new_card = parse_new_card_input(new_card_input)
                for j in range(len(env.hand)):
                    if env.hand[j] == env.dummy_card:
                        env.hand[j] = new_card
                        Play.remaining_cards_in_deck -= 1
                        break

        total_points += points

        # Display the current hand after the action
        print(f"Your current hand: {format_hand(env.hand)}, Remaining cards in the deck: {Play.remaining_cards_in_deck}")
        print(f"Points received from this action: {points}, Total points: {total_points}")
        # Check if the game is done
        if Play.remaining_cards_in_deck == 0:
            if len(env.hand) == 0 or all(card == env.dummy_card for card in env.hand) or (not env.has_valid_combination()):
                done = True
        if done:
            print("Game over!")
            break

if __name__ == "__main__":
    model_path = "dqn_model_v3.pth"  # Path to your trained model
    play_game_interactively(model_path)
