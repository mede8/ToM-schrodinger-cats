"""
File: cats_experimental.py
Author: Andrei Medesan (a.medesan@student.rug.nl)
Description: Implementation of the Schrödinger's Cats card game with
the Evidence-Based variant. The game simulates the game between two
agents: a Zero-Order Agent and a First-Order Agent. Additionally, the
zero-order agent is capable of learning from its own actions and
the actions of the opponent threough explicit memory integration.
"""
import random
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Union, Optional
from pathlib import Path


# setup directories for figures and logs
root = Path(__file__).resolve().parent.parent
fig_dir = root / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)
logs_dir = root / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)
logs_path = logs_dir / 'schrodingers_cats_experiment.log'

logging.basicConfig(filename=logs_path,
                    level=logging.INFO,
                    format='%(message)s')


ClaimTuple = Optional[tuple[str, int, list[str]]]


class ZeroOrderAgent:
    def __init__(self, player_id: int) -> None:
        """
        Initialize a zero-order theory of mind agent with
        an empty hand and memory.

        :param player_id: the player's ID.
        """
        self.player_id = player_id
        self.hand = []
        # explicit memory stores tuples of
        # (action_type, action_details, outcome)
        self.memory = []

    def update_memory(self, action: tuple) -> None:
        """Add an action and its outcome to the agent's memory."""
        self.memory.append(action)

    def calculate_probabilities(self, game: 'GameManager') -> dict[str, float]:
        """
        Calculate the probabilities of each card type in the opponent's hand.

        :param game: the game manager.
        :param player: the player to calculate the probabilities for.
        :return: a dictionary of card type to probability.
        """
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

    def make_claim(self, current_bid: Optional[tuple[str, int]]
                   ) -> Optional[tuple[str, int, list[str]]]:
        """
        Generate a claim based on the agent's hand and memory,
        ensuring it is stronger than the current bid.

        :param current_bid: the current bid to beat.
        :return: a tuple of claim type, claim value, and cards to reveal.
        """
        card_counts = defaultdict(int)
        valid_claim_types = {'alive_cats', 'dead_cats', 'empty_boxes'}
        for card in self.hand:
            if card in valid_claim_types:
                card_counts[card] += 1

        if not card_counts:
            return 'dead_cats', 1, []

        claim_type = max(card_counts, key=card_counts.get)
        base_value = card_counts[claim_type]

        max_possible = {
            'alive_cats': 24,  # 20 alive + 4 HUP
            'dead_cats': 24,  # 20 dead + 4 HUP
            'empty_boxes': 12  # 8 boxes + 4 HUP
        }[claim_type]

        # adjust claim value based on past outcomes of similar claims
        successful_claims = [
            m[1][1] for m in self.memory
            if m[0] == 'claim' and m[1][0] == claim_type and m[2] is None]
        unsuccessful_claims = [
            m[1][1] for m in self.memory
            if m[0] == 'claim' and m[1][0] == claim_type and m[2] is not None]

        avg_successful = sum(
            successful_claims) / (len(successful_claims) + 1e-6)
        avg_unsuccessful = sum(
            unsuccessful_claims) / (len(unsuccessful_claims) + 1e-6)

        adjustment = 0
        if successful_claims and base_value > avg_successful:
            adjustment += 1
        if unsuccessful_claims and base_value > avg_unsuccessful:
            adjustment -= 1

        adjusted_value = min(base_value + adjustment, max_possible)

        if current_bid:
            current_type, current_value = current_bid
            type_strength = {'empty_boxes': 2, 'alive_cats': 1, 'dead_cats': 0}
            required_strength = (
                type_strength[current_type] * 10 +
                (current_value * 2
                 if current_type == 'empty_boxes' else current_value)
            )
            new_strength = (
                type_strength[claim_type] * 10 +
                (adjusted_value * 2
                 if claim_type == 'empty_boxes' else adjusted_value)
            )

            if new_strength <= required_strength:
                return None

        cards_to_reveal = [
            card for card in self.hand if card == claim_type or card == 'HUP']
        return claim_type, adjusted_value, cards_to_reveal

    def doubt_claim(self, game: 'GameManager') -> str:
        """
        Decide whether to doubt the current claim based on
        probabilities and memory.

        :param game: the game manager.
        :return: 'doubt' if the agent doubts, 'pass' otherwise.
        """
        if game.current_bid is None:
            return 'pass'

        claim_type, claim_value = game.current_bid
        probabilities = self.calculate_probabilities(game)
        expected_cards = (
            probabilities.get(claim_type, 0) + probabilities.get('HUP', 0)
        )

        opponent_claims = [
            m for m in self.memory
            if m[0] == 'claim' and m[1][0] == claim_type]
        successful_opponent_claims = sum(
            1 for claim in opponent_claims if claim[2] is True)
        total_opponent_claims = len(opponent_claims)
        opponent_accuracy = (
            successful_opponent_claims / (total_opponent_claims + 1e-6))

        # adjust doubt threshold based on opponent's past accuracy
        doubt_threshold = expected_cards + 1 + (1 - opponent_accuracy) * 3

        # consider own past success/failure in doubting similar claims
        past_doubts_on_type = [
            m for m in self.memory
            if m[0] == 'doubt' and m[1][0] == claim_type]
        successful_own_doubts = sum(
            1 for doubt in past_doubts_on_type if doubt[2] is False)
        total_own_doubts = len(past_doubts_on_type)
        own_doubt_success_rate = (
            successful_own_doubts / (total_own_doubts + 1e-6))

        # higher success -> lower threshold
        doubt_threshold -= (own_doubt_success_rate - 0.5) * 2

        return 'doubt' if claim_value > doubt_threshold else 'pass'

    def choose_action(self, game: 'GameManager') -> str:
        """
        Choose an action (claim or doubt) based on the game state

        :param game: the game manager.
        :return: 'claim' if the agent makes a claim, 'doubt' otherwise.
        """
        if game.current_bid is None:
            return 'claim'
        else:
            claim_result = self.make_claim(game.current_bid)
            if claim_result is None:
                return 'doubt'
            else:
                return 'claim'


class FirstOrderAgent(ZeroOrderAgent):
    def __init__(self, player_id: int) -> None:
        """
        Initialize a recursive first-order theory of mind agent that
        reasons both about opponent behavior and about hidden card information.
        :param player_id: the player's ID.
        """
        super().__init__(player_id)
        # model of the opponent
        self.opponent_model = {
            'hand_belief': self._initialize_hand_belief(),
            'memory_belief': []
        }

    def _initialize_hand_belief(self) -> defaultdict[str, float]:
        """Initialize a uniform belief over the opponent's possible hand."""
        belief = defaultdict(float)
        possible_cards = (
            ['alive_cats'] * 20 + ['dead_cats'] * 20 +
            ['empty_boxes'] * 8 + ['HUP'] * 4)
        num_unknown_cards = (
            len(possible_cards) - len(self.hand) -
            (len(self.game_ref.central_pile)
             if hasattr(self, 'game_ref') else 0)
        )
        if num_unknown_cards > 0:
            for card in possible_cards:
                if (card not in self.hand and
                    (not hasattr(self, 'game_ref') or
                     card not in self.game_ref.central_pile)):
                    # assuming opponent has 6 cards
                    belief[card] += 1 / num_unknown_cards * 6
        return belief

    def update_opponent_model(self, action: tuple,
                              game: 'GameManager') -> None:
        """
        Update the model of opponent's behavior based on observed actions.

        :param action: the action to update the model with.
        :param game: the game manager.
        """
        self.update_memory(action)
        action_type, action_details = action[0], action[1]

        # update hand belief (interpretative model)
        if action_type == 'claim':
            claim_type, claim_value = action_details
            # if opponent claims many of a certain card,
            # increase belief they have it
            for card_type in self.opponent_model['hand_belief']:
                if card_type == claim_type or card_type == 'HUP':
                    self.opponent_model[
                        'hand_belief'][card_type] *= (1 + 0.1 * claim_value)
                else:
                    self.opponent_model[
                        'hand_belief'][card_type] *= (1 - 0.01 * claim_value)
            # normalize belief
            total_belief = sum(self.opponent_model['hand_belief'].values())
            if total_belief > 0:
                for card_type in self.opponent_model['hand_belief']:
                    self.opponent_model[
                        'hand_belief'][card_type] /= total_belief * 6
        elif action_type == 'doubt':
            # if opponent doubts,
            # it might suggest they have fewer of the claimed card
            if game.current_bid:
                doubted_claim_type, _ = game.current_bid
                for card_type in self.opponent_model['hand_belief']:
                    if card_type == doubted_claim_type:
                        self.opponent_model['hand_belief'][card_type] *= 0.8
                    else:
                        self.opponent_model['hand_belief'][card_type] *= 1.02
                # normalize belief
                total_belief = sum(self.opponent_model['hand_belief'].values())
                if total_belief > 0:
                    for card_type in self.opponent_model['hand_belief']:
                        self.opponent_model[
                            'hand_belief'][card_type] /= total_belief * 6

    def simulate_opponent_doubt_decision(self, game: 'GameManager',
                                         opponent_hand_sample: list[str],
                                         current_bid: tuple[str, int]) -> str:
        """
        Simulate the opponent's doubt decision based on a hypothetical hand.
        This directly uses a simplified version of the ZeroOrderAgent's logic.

        :param game: the game manager.
        :param opponent_hand_sample: a sample of the opponent's hand.
        :param current_bid: the current bid to evaluate.
        :param player: the player making the claim.
        :return: 'doubt' or 'pass' based on the simulated decision.
        """
        if current_bid is None:
            return 'pass'

        claim_type, claim_value = current_bid
        hand_counts = defaultdict(int)
        for card in opponent_hand_sample:
            hand_counts[card] += 1

        expected_cards_in_hand = (
            hand_counts.get(claim_type, 0) + hand_counts.get('HUP', 0))

        return 'doubt' if claim_value > expected_cards_in_hand + 1 else 'pass'

    def predict_opponent_reaction(self, game: 'GameManager',
                                  hypothetical_claim: tuple[str, int]) -> str:
        """
        Predict the opponent's reaction to a hypothetical claim
        by simulating their decision process. This involves sampling
        from our belief about their hand and using their (ToM0) logic.

        :param game: the game manager.
        :param hypothetical_claim: the claim to evaluate.
        :return: 'doubt' or 'claim' based on the predicted reaction.
        """
        doubt_outcomes = []
        num_simulations = 100

        # sample hypothetical hands from our belief distribution
        possible_cards = []
        for card_type, belief in self.opponent_model['hand_belief'].items():
            possible_cards.extend([card_type] * int(belief * 10))

        if not possible_cards:
            return 'claim'

        for _ in range(num_simulations):
            sampled_hand = random.sample(
                possible_cards, min(6, len(possible_cards)))
            reaction = self.simulate_opponent_doubt_decision(
                game, sampled_hand, hypothetical_claim)
            doubt_outcomes.append(reaction == 'doubt')

        # predict 'doubt'
        # if a significant proportion of simulations resulted in doubt
        if sum(doubt_outcomes) / num_simulations > 0.6:
            return 'doubt'
        else:
            return 'claim'

    def make_claim(self,
                   current_bid: Optional[tuple[str, int]]) -> ClaimTuple:
        """
        Make a claim based on both the base hand/memory
        logic and recursive reasoning that considers what
        the opponent likely holds and how they might react.

        :param current_bid: the current bid to beat.
        :return: a tuple of claim type, claim value, and cards to reveal.
        """
        base_result = super().make_claim(current_bid)
        if base_result is None:
            return None
        claim_type, claim_value, cards_to_reveal = base_result

        # predictive model
        predicted_reaction = self.predict_opponent_reaction(
            self.game_ref, (claim_type, claim_value))

        # adjust claim based on predicted reaction
        if predicted_reaction == 'doubt':
            if claim_value > 1:
                weaker_claim = (claim_type, claim_value - 1)
                if current_bid is None or \
                    self.game_ref.is_stronger_claim(
                        weaker_claim[0], weaker_claim[1]):
                    predicted_reaction_weaker = self.predict_opponent_reaction(
                        self.game_ref, weaker_claim)
                    if predicted_reaction_weaker == 'claim':
                        return (
                            weaker_claim[0], weaker_claim[1], cards_to_reveal)

            card_counts = defaultdict(int)
            for card in self.hand:
                if card in {'alive_cats', 'dead_cats', 'empty_boxes'} and \
                   card != claim_type:
                    card_counts[card] += 1
            if card_counts:
                alternative_claim_type = max(card_counts, key=card_counts.get)
                alternative_claim_value = card_counts[alternative_claim_type]
                if current_bid is None or \
                    self.game_ref.is_stronger_claim(
                        alternative_claim_type, alternative_claim_value):
                    return (alternative_claim_type,
                            alternative_claim_value, cards_to_reveal)


Agent = Union[ZeroOrderAgent, FirstOrderAgent]


class GameManager:
    def __init__(self, num_players: int, agent_types: list) -> None:
        """
        Initializes the game manager with players and deals cards.

        :param num_players: number of players in the game.
        :param agent_types: list of agent classes to be used as players.
        """
        self.deck = {
            'alive_cats': 20,
            'dead_cats': 20,
            'empty_boxes': 8,
            'HUP': 4
        }
        self.players = [agent_types[0](0)]
        for i in range(1, num_players):
            self.players.append(agent_types[1](i))
        self.current_bid = None
        self.current_player = 0
        self.central_pile = []
        self.deal_cards()

    def deal_cards(self) -> None:
        """
        Dstributes 6 cards to each player from the deck.
        """
        deck_list = []
        for card_type, count in self.deck.items():
            deck_list.extend([card_type] * count)
        random.shuffle(deck_list)

        for player in self.players:
            player.hand = deck_list[:6]
            deck_list = deck_list[6:]
            logging.info(f"Player {player.player_id} was dealt: {player.hand}")

    def make_claim(self, player: Agent,
                   claim_type: str, claim_value: int,
                   cards_to_reveal: list[str]) -> bool:
        """
        Handles a player making a claim and revealing cards.

        :param player: the player making the claim.
        :param claim_type: the type of claim.
        :param claim_value: the value of the claim.
        :param cards_to_reveal: the cards to reveal.
        :return: True if the claim was valid, False otherwise.
        """
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
            logging.info("Invalid claim: Must be stronger than " +
                         "the current claim.")
            return False

    def is_stronger_claim(self, claim_type: str, claim_value: int) -> bool:
        """
        Check if the new claim is stronger than the current bid.

        :param claim_type: the type of the new claim.
        :param claim_value: the value of the new claim.
        :return: True if the new claim is stronger, False otherwise
        """
        if self.current_bid is None:
            return True
        current_type, current_value = self.current_bid

        # empty_boxes claims count double
        type_strength = {'empty_boxes': 2, 'alive_cats': 1, 'dead_cats': 0}
        current_strength = (
            type_strength[current_type] * 10 +
            (current_value * 2
             if current_type == 'empty_boxes' else current_value)
        )
        new_strength = (
            type_strength[claim_type] * 10 +
            (claim_value * 2 if claim_type == 'empty_boxes' else claim_value))

        return new_strength > current_strength

    def doubt_claim(self, player: Agent) -> Agent:
        """
        Handle a player doubting the current claim and determine the loser.

        :param player: the player doubting the claim.
        :return: the player who loses the round.
        """
        logging.info(f"Player {player.player_id} doubts the claim.")
        if self.check_claim():
            logging.info("Claim was valid. Challenger loses.")
            return self.players[(self.current_player + 1) % len(self.players)]
        else:
            logging.info("Claim was invalid. Claimer loses.")
            return self.players[self.current_player]

    def check_claim(self) -> bool:
        """
        Validate the current claim by counting matching cards in hands
        and central pile.

        :return: True if the claim is valid, False otherwise.
        """
        claim_type, claim_value = self.current_bid
        total = 0

        for player in self.players:
            total += player.hand.count(claim_type)
            total += player.hand.count('HUP')

        total += self.central_pile.count(claim_type)
        total += self.central_pile.count('HUP')

        return total >= claim_value

    def reveal_cards(self, player: Agent, cards: list[str]) -> None:
        """
        Reveal specified cards from the player's hand and add them
        to the central pile.

        :param player: the player revealing the cards.
        :param cards: the cards to reveal.
        """
        for card in cards:
            if card in player.hand:
                player.hand.remove(card)
                self.central_pile.append(card)
                logging.info(f"Player {player.player_id} revealed a {card}.")
            else:
                logging.info(f"Player {player.player_id} does not " +
                             f"have a {card} to reveal.")

    def next_turn(self) -> None:
        """Advance the game to the next player's turn."""
        self.current_player = (self.current_player + 1) % len(self.players)

    def simulate_game(self) -> int:
        """
        Simulate a full game until a player wins by having no cards.
        """
        while True:
            player = self.players[self.current_player]

            # win condition
            if not player.hand:
                return player.player_id

            action = player.choose_action(self)

            if action == 'claim':
                claim_result = player.make_claim(self.current_bid)
                if claim_result is None:
                    # force doubt if claim is impossible
                    action = 'doubt'
                else:
                    claim_type, claim_value, cards_to_reveal = claim_result
                    if not self.make_claim(player, claim_type,
                                           claim_value, cards_to_reveal):
                        self.next_turn()
                        continue
                    # log action in opponent's memory
                    for p in self.players:
                        if p != player:
                            p.update_memory(('claim', (claim_type,
                                                       claim_value)))

            if action == 'doubt':
                loser = self.doubt_claim(player)
                # log doubt in opponent's memory
                for p in self.players:
                    if p != player:
                        p.update_memory(('doubt', self.current_bid))
                return loser.player_id

            self.next_turn()


def evaluate_agents(num_games: int) -> None:
    # track metrics
    metrics = {
        'fo_vs_zo': {'FirstOrder': 0, 'ZeroOrder': 0},
        'zo_vs_zo': {'Player0': 0, 'Player1': 0},
        'learning_curve': {'fo_vs_zo': [], 'zo_vs_zo': []}
    }

    interval = num_games // 10

    for game_num in range(num_games):
        # FO vs ZO game
        game_fo_zo = GameManager(2, [ZeroOrderAgent, FirstOrderAgent])
        winner_fo_zo = game_fo_zo.simulate_game()
        if winner_fo_zo == 0:
            metrics['fo_vs_zo']['ZeroOrder'] += 1
        else:
            metrics['fo_vs_zo']['FirstOrder'] += 1

        # ZO vs ZO game
        game_zo_zo = GameManager(2, [ZeroOrderAgent, ZeroOrderAgent])
        winner_zo_zo = game_zo_zo.simulate_game()
        if winner_zo_zo == 0:
            metrics['zo_vs_zo']['Player0'] += 1
        else:
            metrics['zo_vs_zo']['Player1'] += 1

        # track learning curve at intervals
        if (game_num + 1) % interval == 0:
            # FO vs ZO learning curve
            fo_wins = metrics['fo_vs_zo']['FirstOrder']
            zo_wins = metrics['fo_vs_zo']['ZeroOrder']
            total = fo_wins + zo_wins
            metrics['learning_curve']['fo_vs_zo'].append(
                (game_num + 1, (fo_wins / total) * 100,
                 (zo_wins / total) * 100)
            )

            # ZO vs ZO learning curve
            zo0_wins = metrics['zo_vs_zo']['Player0']
            zo1_wins = metrics['zo_vs_zo']['Player1']
            total_zo = zo0_wins + zo1_wins
            metrics['learning_curve']['zo_vs_zo'].append(
                (game_num + 1, (zo0_wins / total_zo) * 100,
                 (zo1_wins / total_zo) * 100)
            )

    print("\n=== Win Percentages ===")
    print("First-Order vs Zero-Order:")
    print("  First-Order: " +
          f"{(metrics['fo_vs_zo']['FirstOrder'] / num_games) * 100:.1f}%")
    print("  Zero-Order: " +
          f"{(metrics['fo_vs_zo']['ZeroOrder'] / num_games) * 100:.1f}%")
    print("\nZero-Order vs Zero-Order:")
    print("  Player0: " +
          f"{(metrics['zo_vs_zo']['Player0'] / num_games) * 100:.1f}%")
    print("  Player1:" +
          f"{(metrics['zo_vs_zo']['Player1'] / num_games) * 100:.1f}%")

    # plot the win bars
    fig2, (bar_ax1, bar_ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # fo_vs_zo wins
    fo_vs_zo_wins = metrics['fo_vs_zo']
    agents = list(fo_vs_zo_wins.keys())
    wins = list(fo_vs_zo_wins.values())
    bar_ax1.bar(agents, wins, color=['seagreen', 'fuchsia'])
    bar_ax1.set_title('Wins in First-Order vs Zero-Order')
    bar_ax1.set_xlabel('Agent')
    bar_ax1.set_ylabel('Number of Wins')
    for i, win in enumerate(wins):
        bar_ax1.text(i, win / 2, str(win), ha='center', va='center',
                     color='white', fontsize=12)

    # zo_vs_zo wins
    zo_vs_zo_wins = metrics['zo_vs_zo']
    players = list(zo_vs_zo_wins.keys())
    wins_zo = list(zo_vs_zo_wins.values())
    bar_ax2.bar(players, wins_zo, color=['indigo', 'fuchsia'])
    bar_ax2.set_title('Wins in Zero-Order vs Zero-Order')
    bar_ax2.set_xlabel('Player')
    bar_ax2.set_ylabel('Number of Wins')
    for i, win in enumerate(wins_zo):
        bar_ax2.text(i, win / 2, str(win), ha='center', va='center',
                     color='white', fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_dir / 'win_bar_plots.png')
    plt.close()

    # plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # FO vs ZO learning curve
    x_fo_zo = [point[0] for point in metrics['learning_curve']['fo_vs_zo']]
    y_fo = [point[1] for point in metrics['learning_curve']['fo_vs_zo']]
    y_zo = [point[2] for point in metrics['learning_curve']['fo_vs_zo']]
    ax1.plot(x_fo_zo, y_fo, label='First-Order', color='seagreen')
    ax1.plot(x_fo_zo, y_zo, label='Zero-Order', color='fuchsia')
    ax1.set_title('First-Order vs Zero-Order Learning Curve')
    ax1.set_xlabel('Games Played')
    ax1.set_ylabel('Win Rate (%)')
    ax1.legend()

    # ZO vs ZO learning curve
    x_zo_zo = [point[0] for point in metrics['learning_curve']['zo_vs_zo']]
    y_zo0 = [point[1] for point in metrics['learning_curve']['zo_vs_zo']]
    y_zo1 = [point[2] for point in metrics['learning_curve']['zo_vs_zo']]
    ax2.plot(x_zo_zo, y_zo0, label='Player0', color='indigo')
    ax2.plot(x_zo_zo, y_zo1, label='Player1', color='fuchsia')
    ax2.set_title('Zero-Order vs Zero-Order Learning Curve')
    ax2.set_xlabel('Games Played')
    ax2.set_ylabel('Win Rate (%)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'learning_curves.png')
    plt.close()


evaluate_agents(5000)
