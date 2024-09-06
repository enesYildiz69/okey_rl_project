#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

struct Card {
    int number;
    char color;

    bool operator==(const Card& other) const {
        return number == other.number && color == other.color;
    }
};

struct GameState {
    std::vector<Card> hand;     // Cards in the player's hand
    std::vector<Card> deck;     // Cards left in the deck
    int score;                  // Player's score
    bool gameOver;              // Flag to indicate if the game is over
};

// Function to display the player's hand
void displayHand(const std::vector<Card>& hand) {
    std::cout << "Your current hand: ";
    for (const auto& card : hand) {
        std::cout << card.number << card.color << " ";
    }
    std::cout << std::endl;
}

// Function to evaluate the current state (static evaluation)
int evaluateCombination(const std::vector<Card>& selectedCards) {
    if (selectedCards.size() != 3) {
        return 0;  // Invalid combination size
    }

    // Check if all cards have the same number (e.g., 1-1-1, 2-2-2)
    if (selectedCards[0].number == selectedCards[1].number &&
        selectedCards[1].number == selectedCards[2].number) {
        int number = selectedCards[0].number;
        switch (number) {
            case 1: return 20;
            case 2: return 30;
            case 3: return 40;
            case 4: return 50;
            case 5: return 60;
            case 6: return 70;
            case 7: return 80;
            case 8: return 90;
        }
    }

    // Check if all cards are the same color and sequential (e.g., 1-2-3 of same color)
    if (selectedCards[0].color == selectedCards[1].color &&
        selectedCards[1].color == selectedCards[2].color) {
        std::vector<int> numbers = { selectedCards[0].number, selectedCards[1].number, selectedCards[2].number };
        std::sort(numbers.begin(), numbers.end());
        if (numbers[1] == numbers[0] + 1 && numbers[2] == numbers[1] + 1) {
            switch (numbers[0]) {
                case 1: return 50;
                case 2: return 60;
                case 3: return 70;
                case 4: return 80;
                case 5: return 90;
                case 6: return 100;
            }
        }
    }

    // Check if cards are sequential but not necessarily the same color (e.g., 1-2-3 of any color)
    std::vector<int> numbers = { selectedCards[0].number, selectedCards[1].number, selectedCards[2].number };
    std::sort(numbers.begin(), numbers.end());
    if (numbers[1] == numbers[0] + 1 && numbers[2] == numbers[1] + 1) {
        switch (numbers[0]) {
            case 1: return 10;
            case 2: return 20;
            case 3: return 30;
            case 4: return 40;
            case 5: return 50;
            case 6: return 60;
        }
    }

    return 0;  // No valid combination found
}

// Generate all possible moves
std::vector<std::vector<Card>> generateMoves(GameState state) {
    std::vector<std::vector<Card>> moves;

    // Generate all possible discard moves (one card from hand)
    for (const Card& card : state.hand) {
        moves.push_back({card});  // Single card to discard
    }

    // Generate all possible combination moves:
    // 1. Same number combination (e.g., 1-1-1)
    // 2. Same color sequential combination (e.g., 1-2-3 of red)
    // 3. Sequential combination (e.g., 1-2-3 independent of color)

    // Generate all possible 3-card combinations from the hand
    if (state.hand.size() >= 3) {
        // Iterate through all combinations of 3 cards
        for (size_t i = 0; i < state.hand.size(); ++i) {
            for (size_t j = i + 1; j < state.hand.size(); ++j) {
                for (size_t k = j + 1; k < state.hand.size(); ++k) {
                    std::vector<Card> combination = { state.hand[i], state.hand[j], state.hand[k] };

                    // Check if the combination is valid and has points
                    if (evaluateCombination(combination) > 0) {
                        moves.push_back(combination);
                    }
                }
            }
        }
    }

    return moves;
}

// Check if any combinations are possible
bool anyCombinationsPossible(GameState state) {
    std::vector<std::vector<Card>> moves = generateMoves(state);
    for (const auto& move : moves) {
        if (move.size() == 3 && evaluateCombination(move) > 0) {
            return true;
        }
    }
    return false;
}

// Apply the selected move to the game state
GameState makeMove(GameState state, std::vector<Card> selectedCards, bool isCombination) {
    if (isCombination) {
        // Remove the selected cards from the player's hand
        for (const Card& card : selectedCards) {
            auto it = std::find(state.hand.begin(), state.hand.end(), card);
            if (it != state.hand.end()) {
                state.hand.erase(it);
            }
        }
        // Update the score based on the combination made
        state.score += evaluateCombination(selectedCards);
        // Check if the deck is empty after the move
        if (state.deck.empty()) {
            std::cout << "No more cards left in the deck.\n";
        }
    } else {
        // Handle discarding: Remove the selected card from the hand and draw a new one
        Card discardedCard = selectedCards[0];
        auto it = std::find(state.hand.begin(), state.hand.end(), discardedCard);
        if (it != state.hand.end()) {
            state.hand.erase(it);
        }
    }
    
    // Check if the deck is empty or no combinations are possible (end the game)
    if (state.deck.empty() && !anyCombinationsPossible(state)) {
        state.gameOver = true;
    }
    
    return state;
}

int minimax(GameState state, int depth, int alpha, int beta, bool maximizingPlayer) {
    if (depth == 0 || state.gameOver) {
        return state.score;
    }

    std::vector<std::vector<Card>> moves = generateMoves(state);

    if (maximizingPlayer) {
        int maxEval = -9999;
        for (auto move : moves) {
            bool isCombination = (move.size() == 3);
            GameState newState = makeMove(state, move, isCombination);
            int eval = minimax(newState, depth - 1, alpha, beta, false);
            maxEval = std::max(maxEval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha)
                break;  // Beta cutoff
        }
        return maxEval;
    } else {
        int minEval = 9999;
        for (auto move : moves) {
            bool isCombination = (move.size() == 3);
            GameState newState = makeMove(state, move, isCombination);
            int eval = minimax(newState, depth - 1, alpha, beta, true);
            minEval = std::min(minEval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha)
                break;  // Alpha cutoff
        }
        return minEval;
    }
}

// Function to get the best move using minimax
std::vector<Card> getBestMove(GameState state) {
    int bestEval = -9999;
    std::vector<Card> bestMove;
    
    // Generate all possible moves
    std::vector<std::vector<Card>> moves = generateMoves(state);
    
    // Evaluate each move using minimax
    for (auto move : moves) {
        bool isCombination = (move.size() == 3);
        GameState newState = makeMove(state, move, isCombination);
        
        // Evaluate the move using minimax with a depth of 3
        int moveEval = minimax(newState, 3, -9999, 9999, false);
        
        // If this move is better than the current best, store it
        if (moveEval > bestEval) {
            bestEval = moveEval;
            bestMove = move;
        }
    }
    
    return bestMove;  // Return the best move found
}



Card getCardFromInput() {
    int number;
    char color;
    while (true) {
        std::cout << "Enter card number (1-8): ";
        std::cin >> number;
        if (number < 1 || number > 8) {
            std::cout << "Invalid card number. Please try again.\n";
            continue;
        }
        std::cout << "Enter card color (R, Y, B): ";
        std::cin >> color;
        if (color != 'R' && color != 'Y' && color != 'B') {
            std::cout << "Invalid color. Please try again.\n";
            continue;
        }
        break;
    }
    return { number, color };
}


// Function to input a set of cards (e.g., 3 cards for a combination)
std::vector<Card> getCardsFromInput(int numCards) {
    std::vector<Card> cards;
    for (int i = 0; i < numCards; ++i) {
        std::cout << "Enter card " << i + 1 << ":\n";
        cards.push_back(getCardFromInput());
    }
    return cards;
}

std::vector<Card> initializeDeck(const std::vector<Card>& hand) {
    std::vector<Card> deck;
    char colors[] = { 'R', 'Y', 'B' };

    // Generate all 24 cards
    for (int num = 1; num <= 8; ++num) {
        for (char color : colors) {
            Card card = { num, color };
            // Exclude the cards already in the player's hand
            if (std::find(hand.begin(), hand.end(), card) == hand.end()) {
                deck.push_back(card);
            }
        }
    }

    // No need to shuffle the deck here, as a random card will be drawn later
    return deck;
}

// Display cards left in the deck (debug function)
void displayDeck(const std::vector<Card>& deck) {
    std::cout << "Cards left in the deck: ";
    for (const auto& card : deck) {
        std::cout << card.number << card.color << " ";
    }
    std::cout << std::endl;
}

int main() {
    GameState state;
    int deckSize = 24;

    // Prompt the player to input their initial hand
    std::cout << "Enter 5 cards for your initial hand:\n";
    state.hand = getCardsFromInput(5);

    // Initialize the deck based on the remaining cards
    state.deck = initializeDeck(state.hand);

    state.score = 0;
    state.gameOver = false;

    // Game loop
    while (!state.gameOver) {
        // Use the updated minimax (with alpha-beta pruning) to suggest the best move
        int bestMoveEval = minimax(state, 3, -9999, 9999, true);  // Pass alpha and beta values
        std::vector<Card> bestMove = getBestMove(state);  // Get best move from minimax

        if (bestMove.size() == 1) {
            // The best move is to discard a card
            std::cout << "Automatically discarding card: " 
                      << bestMove[0].number << " of color " << bestMove[0].color << std::endl;
            
            // Make the discard move
            state = makeMove(state, bestMove, false);

            // Draw a new card if there are any left in the deck
            if (!state.deck.empty()) {
                std::cout << "Enter the new card you drew:\n";
                Card drawnCard = getCardFromInput();
                state.hand.push_back(drawnCard);

                // Remove the drawn card from the unseen deck to prevent duplicates
                auto it = std::find(state.deck.begin(), state.deck.end(), drawnCard);
                if (it != state.deck.end()) {
                    state.deck.erase(it);
                }
            } else {
                std::cout << "No more cards in the deck to draw.\n";
                state.gameOver = true;  // If no cards left in the deck, end the game
            }
        } else {
            // The best move is to make a combination
            
            std::cout << "Automatically making combination with cards: ";
            for (const auto& card : bestMove) {
                std::cout << card.number << card.color << " ";
            }
            std::cout << std::endl;

            // Make the combination move
            state = makeMove(state, bestMove, true);

            // Ask the player for the new cards, depending on how many were used and how many are left in the deck
            int cardsToDraw = std::min(static_cast<int>(bestMove.size()), static_cast<int>(state.deck.size()));
            for (int i = 0; i < cardsToDraw; ++i) {
                std::cout << "Enter the new card you drew:\n";
                Card drawnCard = getCardFromInput();
                state.hand.push_back(drawnCard);
                // Remove the drawn card from the unseen deck to prevent duplicates
                auto it = std::find(state.deck.begin(), state.deck.end(), drawnCard);
                if (it != state.deck.end()) {
                    state.deck.erase(it);
                }
            }

            if (state.deck.empty() && state.hand.size() < 3) {
                std::cout << "No more cards in the deck to draw.\n";
                state.gameOver = true;  // End the game if no cards left in the deck
            }
        }

        // Check if the game is over (no more cards to draw or no valid combinations/moves)
        if (state.gameOver) {
            std::cout << "Game Over! Final Score: " << state.score << std::endl;
        }

        // Display the current hand
        displayHand(state.hand);
        displayDeck(state.deck);
        // Display score
        std::cout << "Current Score: " << state.score << std::endl;
    }

    return 0;
}
