import numpy as np
import random
from collections import Counter

class OkeyEnvironment:
    def __init__(self):
        self.deck = self.initialize_deck()
        self.state_size = 48  # Example state size: 3 colors * 8 numbers * 2 (hand + deck)
        self.action_size = 19  # Example action size: 19 possible actions

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
        return self.get_state()

    def get_state(self):
        # Encode the current hand and remaining deck into a state tensor
        state = np.zeros((3, 8, 2))
        for card in self.hand:
            color, number = card
            state[color, number-1, 0] = 1  # Mark card as in hand
        for card in self.deck:
            color, number = card
            state[color, number-1, 1] = 1  # Mark card as in deck
        return state.flatten()

    def step(self, action):
        reward = 0  # Default reward for invalid actions
        valid_combination = self.has_valid_combination()
        if action >= 0 and action <= 4 and action < len(self.hand):
            # Discard the card at position `action` in the hand
            discarded_card = self.hand.pop(action)

            # Draw a new card from the deck, if available
            if len(self.deck) > 0:
                new_card = self.deck.pop()
                self.hand.append(new_card)
            else:
                # If no cards are left in the deck, continue with the remaining hand
                new_card = None
            
            # Set reward for discarding action (if any specific reward is given)
            reward = 0

        elif action >= 5 and action <= 17 and valid_combination:
            # Define a mapping for the combination actions
            if action == 5:
                success, same_color = self.try_combination(1, 1, 1)
                if success: reward = 20
            elif action == 6:
                success, same_color = self.try_combination(2, 2, 2)
                if success: reward = 30
            elif action == 7:
                success, same_color = self.try_combination(3, 3, 3)
                if success: reward = 40
            elif action == 8:
                success, same_color = self.try_combination(4, 4, 4)
                if success: reward = 50
            elif action == 9:
                success, same_color = self.try_combination(5, 5, 5)
                if success: reward = 60
            elif action == 10:
                success, same_color = self.try_combination(6, 6, 6)
                if success: reward = 70
            elif action == 11:
                success, same_color = self.try_combination(7, 7, 7)
                if success: reward = 80
            elif action == 12:
                success, same_color = self.try_combination(8, 8, 8)
                if success: reward = 90
            elif action == 13:
                success, same_color = self.try_combination(1, 2, 3)
                if success and same_color: reward = 50
                elif success: reward = 10
            elif action == 14:
                success, same_color = self.try_combination(2, 3, 4)
                if success and same_color: reward = 60
                elif success: reward = 20
            elif action == 15:
                success, same_color = self.try_combination(3, 4, 5)
                if success and same_color: reward = 70
                elif success: reward = 30
            elif action == 16:
                success, same_color = self.try_combination(4, 5, 6)
                if success and same_color: reward = 80
                elif success: reward = 40
            elif action == 17:
                success, same_color = self.try_combination(5, 6, 7)
                if success and same_color: reward = 90
                elif success: reward = 50
            elif action == 18:
                success, same_color = self.try_combination(6, 7, 8)
                if success and same_color: reward = 100
                elif success: reward = 60
            else:
                success = False
            
            # Only attempt to draw cards if there are cards available in the deck
            if len(self.deck) > 0 and success:
                # Draw up to 3 cards, but only as many as are available
                num_cards_to_draw = min(3, len(self.deck))
                new_cards = [self.deck.pop() for _ in range(num_cards_to_draw)]
                self.hand.extend(new_cards)
            else:
                new_cards = []

        done = self.check_if_done(valid_combination)
        new_state = self.get_state()
        # print(f"Action taken: {action}, Current Hand: {self.hand}, Deck Size: {len(self.deck)}")
        return new_state, reward, done

    def try_combination(self, n1, n2, n3):
        # Extract numbers and their colors from the hand
        hand_numbers = [card[1] for card in self.hand]
        hand_colors = [card[0] for card in self.hand]

        # Count the occurrences of each number in the hand
        hand_count = Counter(hand_numbers)

        # First, check if the hand has the exact required number of each card
        required_numbers = [n1, n2, n3]
        required_count = Counter(required_numbers)

        for num, count in required_count.items():
            if hand_count[num] < count:
                return False, 0  # Not enough of a specific number in the hand

        # If the combination is sequential, check if they are of the same color
        if n1 != n2 or n2 != n3:  # Check if it’s not a triple of the same number
            # Get the indices of the cards that match the combination
            indices = []
            for num in required_numbers:
                for i, card in enumerate(self.hand):
                    if card[1] == num and i not in indices:
                        indices.append(i)
                        break
            
            # Check if all selected cards have the same color
            if len(set([self.hand[i][0] for i in indices])) != 1:
                self.remove_combination_cards_from_hand(n1, n2, n3)
                return True, False  # The cards are not of the same color
        
        # If all checks pass, remove the cards and return success
        self.remove_combination_cards_from_hand(n1, n2, n3)
        return True, True
    
    def remove_combination_cards_from_hand(self, n1, n2, n3):
        # Create a list of the numbers to remove
        to_remove = [n1, n2, n3]

        # Iterate through the numbers and remove them from the hand
        for num in to_remove:
            for card in self.hand:
                if card[1] == num:
                    self.hand.remove(card)
                    break  # Break to ensure only one card is removed for each number
    
    def check_if_done(self, valid_combination):
        # 1. Check if the deck is empty
        if len(self.deck) == 0:
            # If the deck is empty, check if there are any possible combinations left in the hand
            if not valid_combination:
                return True  # Game is done because no more cards can be drawn and no combinations are possible

        # 2. Check if the hand is empty (all cards have been used in combinations)
        if len(self.hand) == 0:
            return True  # Game is done because the player has used all their cards

        # 3. Check if there are no valid moves left (e.g., no possible combinations)
        if not valid_combination and len(self.deck) > 0:
            return False  # Player can still discard and draw new cards

        return False  # Game is not done; player can still make a move

    def has_valid_combination(self):
        # Check for any possible valid combinations in the current hand

        # Check for triples of the same number (e.g., 1-1-1)
        number_counts = Counter([card[1] for card in self.hand])
        for count in number_counts.values():
            if count >= 3:
                return True  # Found a valid combination

        # Check for sequential numbers of the same color (e.g., 1-2-3)
        sorted_hand = sorted(self.hand, key=lambda x: (x[0], x[1]))  # Sort by color, then number
        for i in range(len(sorted_hand) - 2):
            if (
                sorted_hand[i][0] == sorted_hand[i + 1][0] == sorted_hand[i + 2][0] and  # Same color
                sorted_hand[i][1] + 1 == sorted_hand[i + 1][1] and  # Sequential numbers
                sorted_hand[i + 1][1] + 1 == sorted_hand[i + 2][1]
            ):
                return True  # Found a valid combination

        return False  # No valid combinations found

    def render(self):
        # Optional: Print or visualize the current state for debugging
        pass
