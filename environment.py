import numpy as np
import random
from collections import Counter

class OkeyEnvironment:
    def __init__(self):
        self.deck = self.initialize_deck()
        self.state_size = 72  # 3 colors * 8 numbers * 3 states (deck, hand, discarded)
        self.action_size = 44  # 20 combination actions + 24 discard actions
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

        # Check for possible same-number combinations
        for action, combination in enumerate([
            (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4),
            (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8)
        ]):
            success, _ = self.try_combination(*combination, remove=False)
            if success:
                valid_actions.append(action)
                self.valid_combinations[action] = (combination, True)

        # Check for possible same-color sequential combinations
        for i, combination in enumerate([
            (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6),
            (5, 6, 7), (6, 7, 8)
        ]):
            success, same_color = self.try_combination(*combination, remove=False)
            if success:
                if same_color:
                    valid_actions.append(8 + i)  # Action numbers 8-13 for same-color sequential combinations
                    self.valid_combinations[8 + i] = (combination, True)
                else:
                    valid_actions.append(14 + i)  # Action numbers 14-19 for different-color sequential combinations
                    self.valid_combinations[14 + i] = (combination, False)

        # Discard actions for specific cards in hand
        for card in self.hand:
            if card != self.dummy_card:
                index = self.get_card_action_index(card)
                valid_actions.append(index)

        return valid_actions

    def get_card_action_index(self, card):
        color, number = card
        return 20 + (color * 8) + (number - 1)  # Action numbers 20-43 for discarding cards



    def step(self, action):
        reward = 0
        hand_before_action = self.hand.copy()
        if action in range(0, 8):  # Same-number combination
            combination = self.valid_combinations[action][0]
            self.remove_combination_cards_from_hand(*combination)
            reward = self.calculate_reward_for_action(action, same_color=True)
            # Replace the removed cards if the deck still has enough cards
            num_cards_to_draw = min(3, len(self.deck))
            for i in range(len(self.hand)):
                if self.hand[i] == self.dummy_card and num_cards_to_draw > 0:
                    new_card = self.deck.pop()
                    self.hand[i] = new_card
                    num_cards_to_draw -= 1
        elif action in range(8, 14):  # Same-color sequential combination
            combination = self.valid_combinations[action][0]
            self.remove_combination_cards_from_hand(*combination)
            reward = self.calculate_reward_for_action(action, same_color=True)
            # Replace the removed cards if the deck still has enough cards
            num_cards_to_draw = min(3, len(self.deck))
            for i in range(len(self.hand)):
                if self.hand[i] == self.dummy_card and num_cards_to_draw > 0:
                    new_card = self.deck.pop()
                    self.hand[i] = new_card
                    num_cards_to_draw -= 1
            
        elif action in range(14, 20):  # Different-color sequential combination
            combination = self.valid_combinations[action][0]
            self.remove_combination_cards_from_hand(*combination)
            reward = self.calculate_reward_for_action(action, same_color=False)
            # Replace the removed cards if the deck still has enough cards
            num_cards_to_draw = min(3, len(self.deck))
            for i in range(len(self.hand)):
                if self.hand[i] == self.dummy_card and num_cards_to_draw > 0:
                    new_card = self.deck.pop()
                    self.hand[i] = new_card
                    num_cards_to_draw -= 1
            
        elif action in range(20, 44):  # Discard action based on specific card
            card_to_discard = self.get_card_from_action_index(action)
            index = self.hand.index(card_to_discard)
            self.hand[index] = self.dummy_card
            self.discarded.append(card_to_discard)
            if len(self.deck) > 0:
                new_card = self.deck.pop()
                self.hand[index] = new_card
            reward = 0  # No points for discarding
        # print(f"Reward: {reward}, Action: {action}, Hand Before: {hand_before_action}, Hand After: {self.hand}")
        done = self.check_if_done()
        new_state = self.get_state()
        return new_state, reward, done

    def get_card_from_action_index(self, action):
        index = action - 20
        color = index // 8
        number = (index % 8) + 1
        return (color, number)



    def calculate_reward_for_action(self, action, same_color):
        if action in range(0, 8):  # For same-number combinations
            return 20 + action * 10
        elif action in range(8, 14):  # For same-color sequential combinations
            return 50 + (action - 8) * 10
        elif action in range(14, 20):  # For different-color sequential combinations
            return 10 + (action - 14) * 10
        else:
            return 0  # No points for discard actions


    def try_combination(self, n1, n2, n3, remove=False):
        hand_numbers = [card[1] for card in self.hand if card != self.dummy_card]

        # Count the occurrences of each number in the hand
        hand_count = Counter(hand_numbers)
        required_numbers = [n1, n2, n3]
        required_count = Counter(required_numbers)

        for num, count in required_count.items():
            if hand_count[num] < count:
                return False, False  # Not enough of a specific number in the hand

        # Check for color match or mismatches
        if n1 != n2 or n2 != n3:
            indices = []
            for num in required_numbers:
                for i, card in enumerate(self.hand):
                    if card[1] == num and i not in indices and card != self.dummy_card:
                        indices.append(i)
                        break

            colors = [self.hand[i][0] for i in indices]
            if len(set(colors)) == 1:  # All cards are the same color
                if remove:
                    self.remove_combination_cards_from_hand(n1, n2, n3)
                return True, True
            else:  # Different colors
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
