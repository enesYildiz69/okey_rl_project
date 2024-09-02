import numpy as np
import random
from collections import Counter

class OkeyEnvironment:
    def __init__(self):
        self.deck = self.initialize_deck()
        self.state_size = 72  # 3 colors * 8 numbers * 3 states (deck, hand, discarded)
        self.action_size = 19  # Example action size: 19 possible actions
        self.valid_combinations = {}
        self.discarded = []
        self.dummy_card = (-1, -1)  # Represent an empty slot

    def initialize_deck(self):
        deck = []
        for color in range(3):
            for number in range(1, 9):
                deck.append((color, number))
        random.shuffle(deck)
        return deck

    def reset(self):
        self.deck = self.initialize_deck()
        self.hand = [self.deck.pop() for _ in range(5)]
        self.discarded = []
        return self.get_state()

    def get_state(self):
        # Encode the current hand and remaining deck into a state tensor
        state = np.zeros((3, 8, 3))
        # Mark cards in the hand
        for card in self.hand:
            if card != self.dummy_card:
                color, number = card
                state[color, number-1, 1] = 1  # Mark card as in hand

        # Mark cards in the deck
        for card in self.deck:
            color, number = card
            state[color, number-1, 0] = 1  # Mark card as in deck

        # Mark used/discarded cards
        for card in self.discarded:
            color, number = card
            state[color, number-1, 2] = 1  # Mark card as discarded

        return state.flatten()  # Flatten for use in the neural network

    def get_valid_actions(self):
        valid_actions = []
        self.valid_combinations = {}

        # Check if deck has cards; if not, discard actions shouldn't be allowed
        if len(self.deck) > 0:
            for i in range(len(self.hand)):
                if self.hand[i] != self.dummy_card:
                    valid_actions.append(i)

        # Check for possible combinations without modifying the hand
        for action, combination in enumerate([
            (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4),
            (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8),
            (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6),
            (5, 6, 7), (6, 7, 8)
        ], start=5):
            success, same_color = self.try_combination(*combination, remove=False)
            if success:
                valid_actions.append(action)
                self.valid_combinations[action] = (combination, same_color)

        return valid_actions

    def step(self, action):
        reward = 0
        if action in self.get_valid_actions():
            hand_before_action = self.hand.copy()
            if action >= 0 and action <= 4:
                discarded_card = self.hand.pop(action)
                self.hand[action] = self.dummy_card
                self.discarded.append(discarded_card)
                if len(self.deck) > 0:
                    new_card = self.deck.pop()
                    self.hand[action] = new_card
                reward = 0
            elif action >= 5 and action <= 18:
                # Use the stored valid combination without rechecking
                combination, same_color = self.valid_combinations[action]
                self.remove_combination_cards_from_hand(*combination)
                reward = self.calculate_reward_for_action(action, same_color)
                
                # Replace the removed cards if the deck still has enough cards
                num_cards_to_draw = min(3, len(self.deck))
                for i in range(len(self.hand)):
                    if self.hand[i] == self.dummy_card and num_cards_to_draw > 0:
                        new_card = self.deck.pop()
                        self.hand[i] = new_card
                        num_cards_to_draw -= 1
                        
            done = self.check_if_done()
        else:
            done = True  # Invalid action should never happen now
        print(f"Action taken: {action},Hand before action {hand_before_action}, Hand after action: {self.hand}, Reward: {reward}")
        new_state = self.get_state()
        return new_state, reward, done

    def calculate_reward_for_action(self, action, same_color):
        # Map actions to rewards
        if action == 5:
            return 20
        elif action == 6:
            return 30
        elif action == 7:
            return 40
        elif action == 8:
            return 50
        elif action == 9:
            return 60
        elif action == 10:
            return 70
        elif action == 11:
            return 80
        elif action == 12:
            return 90
        elif action >= 13 and action <= 18:
            base_rewards = [10, 20, 30, 40, 50, 60]  # Rewards for actions 13-18
            additional_reward = 40 if same_color else 0
            return base_rewards[action - 13] + additional_reward
        else:
            return 0

    def try_combination(self, n1, n2, n3, remove=False):
        hand_numbers = [card[1] for card in self.hand if card != self.dummy_card]

        # Count the occurrences of each number in the hand
        hand_count = Counter(hand_numbers)
        required_numbers = [n1, n2, n3]
        required_count = Counter(required_numbers)

        for num, count in required_count.items():
            if hand_count[num] < count:
                return False, False  # Not enough of a specific number in the hand

        if n1 != n2 or n2 != n3:
            indices = []
            for num in required_numbers:
                for i, card in enumerate(self.hand):
                    if card[1] == num and i not in indices and card != self.dummy_card:
                        indices.append(i)
                        break

            if len(set([self.hand[i][0] for i in indices])) != 1:
                if remove:
                    self.remove_combination_cards_from_hand(n1, n2, n3)
                return True, False

        if remove:
            self.remove_combination_cards_from_hand(n1, n2, n3)
        return True, True

    def remove_combination_cards_from_hand(self, n1, n2, n3):
        to_remove = [n1, n2, n3]

        for num in to_remove:
            for i, card in enumerate(self.hand):
                if card[1] == num:
                    self.hand[i] = self.dummy_card
                    self.discarded.append(card)
                    break
    
    def check_if_done(self):
        if len(self.deck) == 0:
            if not self.has_valid_combination():
                return True
        if len(self.hand) == 0 or all(card == self.dummy_card for card in self.hand):
            return True
        if not self.has_valid_combination() and len(self.deck) > 0:
            return False

        return False

    def has_valid_combination(self):
        hand_numbers = [card[1] for card in self.hand if card != self.dummy_card]
        number_counts = Counter(hand_numbers)
        for count in number_counts.values():
            if count >= 3:
                return True

        sorted_hand = sorted([card for card in self.hand if card != self.dummy_card], key=lambda x: (x[0], x[1]))
        for i in range(len(sorted_hand) - 2):
            if (
                sorted_hand[i][0] == sorted_hand[i + 1][0] == sorted_hand[i + 2][0] and
                sorted_hand[i][1] + 1 == sorted_hand[i + 1][1] and
                sorted_hand[i + 1][1] + 1 == sorted_hand[i + 2][1]
            ):
                return True

        return False

    def render(self):
        pass
