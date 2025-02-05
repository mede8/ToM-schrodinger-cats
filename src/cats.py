"""
File: cats.py
Author: Andrei Medesan (a.medesan@student.rug.nl)
Description: Implementation of the Schrödinger's Cats card game with
the Evidence-Based variant. The game simulates the game between two
agents: a Zero-Order Agent and a First-Order Agent.
"""
import random
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


root = Path(__file__).resolve().parent.parent
# save details for figures
fig_dir = root / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)

# logs details
logs_dir = root / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)
logs_path = logs_dir / 'schrodingers_cats.log'

logging.basicConfig(filename=logs_path, level=logging.INFO,
                    format='%(message)s')


class GameManager:
    def __init__(self, num_players: int) -> None:
        """
        Initialize the game manager with the given number of players.
        """
        self.deck = {
            'alive_cats': 20,
            'dead_cats': 20,
            'empty_boxes': 8,
            'HUP': 4
        }
        self.players = [
            ZeroOrderAgent(i) if i == 0 else FirstOrderAgent(i)
            for i in range(num_players)
        ]
        self.current_bid = None
        self.current_player = 0
        self.central_pile = []
        self.discarded_cards = []
        self.deal_cards()

    def deal_cards(self) -> None:
        """Deal 6 cards to each player from the shuffled deck."""
        deck_list = []
        for card_type, count in self.deck.items():
            deck_list.extend([card_type] * count)
        random.shuffle(deck_list)

        for player in self.players:
            player.hand = deck_list[:6]
            deck_list = deck_list[6:]
            logging.info(f"Player {player.player_id} was dealt: {player.hand}")

    def make_claim(self, player, claim_type, claim_value, cards_to_reveal):
        if self.current_bid is None or self.is_stronger_claim(claim_type,
                                                              claim_value):
            self.current_bid = (claim_type, claim_value)
            self.current_player = (self.current_player + 1) % len(self.players)
            logging.info(f"Player {player.player_id} claims {claim_value} " +
                         f"{claim_type} and reveals {len(cards_to_reveal)} " +
                         "cards.")
            self.reveal_cards(player, cards_to_reveal)
            return True
        else:
            logging.info("Invalid claim: Must be stronger than the current.")
            return False

    def is_stronger_claim(self, claim_type, claim_value):
        if self.current_bid is None:
            return True
        current_type, current_value = self.current_bid

        type_strength = {'empty_boxes': 2, 'alive_cats': 1, 'dead_cats': 0}
        new_strength = type_strength.get(claim_type, 0) * 10 + claim_value
        current_strength = (
            type_strength.get(current_type, 0) * 10 + current_value)

        return new_strength > current_strength

    def doubt_claim(self, player):
        logging.info(f"Player {player.player_id} doubts the claim.")
        if self.check_claim():
            logging.info("Claim was valid. Challenger loses.")
            return self.players[(self.current_player + 1) % len(self.players)]
        else:
            logging.info("Claim was invalid. Claimer loses.")
            return self.players[self.current_player]

    def check_claim(self):
        claim_type, claim_value = self.current_bid
        total = 0

        for player in self.players:
            total += player.hand.count(claim_type)
            total += player.hand.count('HUP')

        total += self.central_pile.count(claim_type)
        total += self.central_pile.count('HUP')

        return total >= claim_value

    def reveal_cards(self, player, cards):
        for card in cards:
            if card in player.hand:
                player.hand.remove(card)
                self.central_pile.append(card)
                logging.info(f"Player {player.player_id} revealed a {card}.")
            else:
                logging.info(f"Player {player.player_id} does not have " +
                             f"a {card} to reveal.")

    def next_turn(self):
        self.current_player = (self.current_player + 1) % len(self.players)

    def simulate_game(self):
        """Simulate a full game."""
        while True:
            player = self.players[self.current_player]

            # check if player has no cards and wins
            if not player.hand:
                logging.info(f"Player {player.player_id} has" +
                             "no cards and wins!")
                return player.player_id

            action = player.choose_action(self)

            if action == 'claim':
                claim_type, claim_value, cards_to_reveal = player.make_claim()
                if not self.make_claim(player, claim_type,
                                       claim_value, cards_to_reveal):
                    continue
            elif action == 'doubt':
                loser = self.doubt_claim(player)
                return loser.player_id

            self.next_turn()


class ZeroOrderAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.hand = []

    def calculate_probabilities(self, game):
        total_deck = {
            'alive_cats': 20,
            'dead_cats': 20,
            'empty_boxes': 8,
            'HUP': 4
        }
        known_cards = defaultdict(int)
        for card in self.hand:
            known_cards[card] += 1
        for card in game.central_pile:
            known_cards[card] += 1

        opponent_hand_size = 6 * (len(game.players) - 1)
        remaining_cards = {}
        for card_type in total_deck:
            remaining = total_deck[card_type] - known_cards.get(card_type, 0)
            remaining_cards[card_type] = remaining

        total_remaining = sum(remaining_cards.values())
        if total_remaining == 0:
            return {card_type: 0 for card_type in total_deck}

        probabilities = {}
        for card_type in total_deck:
            prob = (
                (remaining_cards[card_type] / total_remaining)
                * opponent_hand_size
            )
            probabilities[card_type] = prob

        return probabilities

    def make_claim(self):
        card_counts = defaultdict(int)
        for card in self.hand:
            card_counts[card] += 1

        claim_type = max(card_counts, key=card_counts.get)
        claim_value = card_counts[claim_type] + 1

        cards_to_reveal = [
            card for card in self.hand if card == claim_type or card == 'HUP']
        return claim_type, claim_value, cards_to_reveal

    def doubt_claim(self, game):
        if game.current_bid is None:
            return 'pass'

        claim_type, claim_value = game.current_bid
        probabilities = self.calculate_probabilities(game)
        expected_cards = (
            probabilities.get(claim_type, 0) + probabilities.get('HUP', 0)
        )

        if claim_value > expected_cards + 2:
            return 'doubt'
        else:
            return 'pass'

    def choose_action(self, game):
        if game.current_bid is None:
            return 'claim'
        else:
            return random.choice(['claim', 'doubt'])


class FirstOrderAgent(ZeroOrderAgent):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.opponent_history = []

    def interpret_opponent_behavior(self, game):
        """Analyze opponent's past claims/doubts to infer their hand."""
        if not self.opponent_history:
            return None

        # extract opponent's typical claim thresholds and doubt patterns
        claim_patterns = []
        doubt_thresholds = []

        for action in self.opponent_history:
            if action[0] == 'claim':
                claim_type, claim_value, _ = action[1]
                claim_patterns.append((claim_type, claim_value))
            elif action[0] == 'doubt':
                doubted_claim = action[1]
                doubt_thresholds.append(doubted_claim[1])

        # calculate averages and frequencies
        self.opponent_claim_profile = defaultdict(list)
        for ctype, cvalue in claim_patterns:
            self.opponent_claim_profile[ctype].append(cvalue)

        self.opponent_doubt_threshold = (
            sum(doubt_thresholds) / len(doubt_thresholds)
            if doubt_thresholds else 0
        )

        logging.info(f"Opponent profile: {self.opponent_claim_profile}, " +
                     f"Doubt threshold: {self.opponent_doubt_threshold}")

    def predict_opponent_reaction(self, game):
        """Predict if opponent will doubt the current claim."""
        if not self.opponent_history:
            return random.choice([True, False])  # random guess

        current_claim = game.current_bid
        if current_claim is None:
            return False

        # check if claim exceeds opponent's historical doubt threshold
        _, claim_value = current_claim
        if claim_value > self.opponent_doubt_threshold * 1.2:
            return True  # likely to doubt
        else:
            return False  # likely to accept

    def make_claim(self):
        """Strategic claim based on opponent's profile."""
        # use zero-order logic as fallback
        base_claim_type, base_claim_value, cards_to_reveal = (
            super().make_claim()
        )

        # adjust claim based on opponent's doubt threshold
        if hasattr(self, 'opponent_doubt_threshold'):
            safe_value = (
                min(base_claim_value, int(self.opponent_doubt_threshold * 0.9))
            )
            return base_claim_type, safe_value, cards_to_reveal
        else:
            return base_claim_type, base_claim_value, cards_to_reveal

    def doubt_claim(self, game):
        """Doubt based on predictive model."""
        self.interpret_opponent_behavior(game)
        if self.predict_opponent_reaction(game):
            return 'doubt'
        else:
            return 'pass'

    def choose_action(self, game):
        """Override action choice with predictive logic."""
        if game.current_bid is None:
            return 'claim'
        else:
            # Update opponent history before deciding
            opponent = game.players[(self.player_id + 1) % len(game.players)]
            last_action = (
                opponent.last_action
                if hasattr(opponent, 'last_action') else None
            )
            if last_action:
                self.opponent_history.append(last_action)

            return 'doubt' if self.doubt_claim(game) == 'doubt' else 'claim'


def evaluate_agents(num_games):
    zero_order_wins = 0
    first_order_wins = 0

    for _ in range(num_games):
        game = GameManager(2)
        winner = game.simulate_game()
        if winner == 0:
            zero_order_wins += 1
        else:
            first_order_wins += 1

    logging.info(f"Zero-order wins: {zero_order_wins}")
    logging.info(f"First-order wins: {first_order_wins}")
    print(f"Zero-order wins: {zero_order_wins}")
    print(f"First-order wins: {first_order_wins}")
    print(f"Success rate of Zero-Order Agent: {zero_order_wins / num_games}")
    print(f"Success rate of First-Order Agent: {first_order_wins / num_games}")

    plt.bar(['Zero-Order Agent', 'First-Order Agent'],
            [zero_order_wins, first_order_wins], color=['cadetblue', 'plum'])
    plt.xlabel('Agent Type')
    plt.ylabel('Number of Wins')
    plt.title('Success Rate of Agents')
    plt.savefig(fig_dir / 'success_rate.png')
    plt.show()


evaluate_agents(1000)
