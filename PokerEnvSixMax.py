import logging
import random

import numpy as np
from treys import Evaluator, Card

from Player import Player
from PokerMonteCarlo import PokerMonteCarlo

round_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
class PokerEnvSixMax:

    def __init__(self, num_players=6, starting_chips=1000, small_blind=10, big_blind=20):
        self.deck = None
        self.evaluator = Evaluator()
        self.monte_carlo = PokerMonteCarlo(self.evaluator)

        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind_amount = small_blind
        self.big_blind_amount = big_blind
        self.pot_size = 0
        self.action = 0

        self.max_raise_round = 1
        self.actual_raise_round = 0

        self.small_blind_position = 0
        self.big_blind_position = 1

        # Crear instancias de la clase Player
        self.players = [
            Player(name="Player" + str(i), position=i, chips=starting_chips)
            for i in range(num_players)]

        self.community_cards = []
        self.round_history = []
        self.current_bet = 0
        self.current_phase = 'preflop'
        self.player_position = 0
        self.done = False
        self.folded_players = set()

        self._post_blinds()

    def _post_blinds(self):
        self.players[self.small_blind_position].chips -= self.small_blind_amount
        self.players[self.small_blind_position].previous_bet = self.small_blind_amount
        self.pot_size += self.small_blind_amount
        self.players[self.small_blind_position].action_history.append(
            {'round': 0, 'action': 5, 'amount': self.small_blind_amount})

        self.players[self.big_blind_position].chips -= self.big_blind_amount
        self.players[self.big_blind_position].previous_bet = self.big_blind_amount
        self.pot_size += self.big_blind_amount
        self.current_bet = self.big_blind_amount
        self.players[self.big_blind_position].action_history.append(
            {'round': 0, 'action': 6, 'amount': self.big_blind_amount})

    def reset(self):
        self.done = False
        self._rotate_players()
        self.deck = self._create_deck()
        for player in self.players:
            if player not in self.folded_players:
                player.hand = self._deal_cards(2)
                player.previous_bet = 0

        self.community_cards = []
        self.folded_players.clear()
        self.pot_size = 0
        self.current_bet = 0
        self.round_history = []
        self.current_phase = 'preflop'
        self.player_position = (self.big_blind_position + 1) % self.num_players

        self._post_blinds()

        state = self._get_state()

        print(f"State shape after reset: {np.array(state).shape}")

        return self._get_state()

    def _rotate_blinds(self):
        self.small_blind_position = (self.small_blind_position + 1) % self.num_players
        self.big_blind_position = (self.big_blind_position + 1) % self.num_players
        self.player_position = (self.big_blind_position + 1) % self.num_players

    def _rotate_players(self):
        # Rotar la lista de jugadores
        self.players = self.players[1:] + self.players[:1]
        # Actualizar las posiciones de cada jugador
        for i, player in enumerate(self.players):
            player.position = i

    def step(self, action):
        if self.done:
            raise Exception("Game is already done. Reset the environment to start a new game.")

        amount = 0
        self.action = action
        player = self.players[self.player_position]

        if player in self.folded_players:
            action = 0  # Player has folded, no action possible
        else:
            if action == 1:  # Fold
                if self.current_bet == player.previous_bet:
                    self.check_step(player)
                else:
                    self.fold_step(player)
            elif action == 2:  # Call
                if self.current_bet == player.previous_bet:
                    self.check_step(player)
                elif self.current_bet - player.previous_bet > player.chips:
                    self.fold_step(player)
                else:
                    amount = self.call_step(player)
            elif action == 3:  # Raise
                amount = self.calculate_raise_amount(player)
                if self.current_bet - player.previous_bet > player.chips:
                    self.fold_step(player)
                elif self.actual_raise_round > self.max_raise_round:
                    if self.current_bet > 0:
                        amount = self.call_step(player)
                    else:
                        self.check_step(player)
                else:
                    amount = self.raise_step(player, amount)
                    self.current_bet = amount
                    self.actual_raise_round += 1  # Increment raise round
            elif action == 4:  # Check
                if self.current_bet == player.previous_bet:
                    self.check_step(player)
                elif self.current_bet - player.previous_bet > player.chips:
                    self.fold_step(player)
                else:
                    amount = self.call_step(player)

        player.action_history.append({'round': round_mapping.get(self.current_phase, -1),'action': self.action, 'amount': amount})

        if action != 0:
            logging.debug(
                f"Position {self.player_position}, {player.name} takes action: {self.action} with: {amount} chips")

        self.round_history.append({
            'player_position': self.player_position,
            'action': self.action,
            'amount': amount,
            'phase': self.current_phase
        })

        # Move to the next player
        self.player_position = (self.player_position + 1) % self.num_players

        # Update game state
        state = self._get_state()

        # Check if all players have equalized their bets
        if self._all_players_equalized_bet():
            self._next_phase()

        # Check if the game is over
        done = self._is_game_over()

        if done:
            rewards = self._calculate_rewards()
            return state, done, rewards
        else:
            return state, done, {}

    def fold_step(self, player):
        player.reward *= -1
        self.action = 1
        self.folded_players.add(player)

    def call_step(self, player):
        amount = self.current_bet - player.previous_bet
        player.chips -= amount
        self.pot_size += amount
        player.previous_bet = self.current_bet
        player.reward += amount * 0.2
        self.action = 2
        return amount

    def raise_step(self, player, amount):
        player.chips -= amount
        self.pot_size += amount
        self.current_bet = amount
        player.previous_bet = self.current_bet
        player.reward += amount * 0.3
        self.action = 3
        return amount

    def check_step(self, player):
        self.action = 4

    def _all_players_equalized_bet(self):

        for player in self.players:
            if player not in self.folded_players:
                if player.previous_bet != self.current_bet:
                    return False
        return True

    def _calculate_total_bets(self, player):
        return sum(player.bet_history)

    def _calculate_rewards(self):
        rewards = [0] * self.num_players
        winner_position = 0
        if self.current_phase == 'showdown':
            # Evaluar las manos de todos los jugadores activos
            hands = [(player.position, self.evaluator.evaluate(self.community_cards, player.hand))
                     for player in self.players if player.position not in self.folded_players]

            # Encontrar al jugador con la mejor mano
            if hands:
                winner_position = min(hands, key=lambda x: x[1])[0]
                rewards[winner_position] = self.pot_size


        elif len(self.folded_players) == self.num_players - 1:
            # Solo un jugador no se ha retirado
            for player in self.players:
                if player.position not in self.folded_players:
                    rewards[player.position] = self.pot_size
                    winner_position = player.position
        player = self.players[winner_position]
        print(f"The winner is {player.name} with {player.hand} ")

        player = self.players[winner_position];
        print(f"The winner is {player.name}")
        print("\nCards in the player's hand:");
        [print(Card.int_to_str(card)) for card in player.hand]
        print("\nCommunity Cards:");
        [print(Card.int_to_str(card)) for card in self.community_cards]

        for player in self.players:
            if player.position != winner_position:
                if player.reward > 0:
                    player.reward *= -1

        return rewards

    def _is_game_over(self):
        return len(self.folded_players) == self.num_players - 1 or self.current_phase == 'showdown'

    def _next_phase(self):
        if self.current_phase == 'preflop':
            self.community_cards = []
            self.current_phase = 'flop'
            self.community_cards.extend(self._deal_cards(3))
        elif self.current_phase == 'flop':
            self.current_phase = 'turn'
            self.community_cards.append(self._deal_cards(1)[0])
        elif self.current_phase == 'turn':
            self.current_phase = 'river'
            self.community_cards.append(self._deal_cards(1)[0])
        elif self.current_phase == 'river':
            self.current_phase = 'showdown'
        for player in self.players:
            player.previous_bet = 0
        self.actual_raise_round = 0

    def calculate_raise_amount(self, player):
        pot_odds = self.calculate_pot_odds(self.current_bet, self.pot_size)
        equity = self.calculate_equity(player.hand, self.community_cards)

        base_raise_amount = max(20, int(0.05 * player.chips))
        raise_amount = base_raise_amount * (equity / max(pot_odds, 0.01))  # Evita la división por cero

        # Añade un límite máximo para el raise
        max_raise = min(player.chips,
                        self.pot_size + self.current_bet * 3)  # Por ejemplo, el triple de la apuesta actual
        raise_amount = min(max_raise, max(raise_amount, self.current_bet + 1))

        return int(raise_amount)

    def calculate_pot_odds(self, current_bet, pot_size):
        if current_bet == 0:
            return 0.0
        return current_bet / (pot_size + current_bet)

    def calculate_equity(self, player_hand, community_cards):
        # Filtra los jugadores activos basados en el estado actual del juego
        active_players = [i for i in range(self.num_players) if i not in self.folded_players]
        return self.monte_carlo.simulate_hand(player_hand, community_cards, active_players=active_players)

    def _get_state(self):

        pot_odds = self.calculate_pot_odds(self.current_bet, self.pot_size)

        state = {
            'round': round_mapping.get(self.current_phase, -1),
            'community_cards': self.community_cards,
            'pot_info': {
                'pot_size': self.pot_size,
                'current_bet': self.current_bet,
                'pot_odds': pot_odds
            },
            'players': [
                {
                    'position': player.position,
                    'chips': player.chips,
                    'hand': player.hand,
                    'previous_bet': player.previous_bet,
                    'action_history': player.action_history,
                }
                for player in self.players
            ],
            'player_position': self.player_position,
            'win_probabilities': [self.calculate_equity(player.hand, self.community_cards) for player in self.players],
            'stack_sizes': [player.chips for player in self.players],
            'round_history': self.round_history
        }
        print(state)
        return state

    def _create_deck(self):
        deck = [Card.new(f'{rank}{suit}') for rank in '23456789TJQKA' for suit in 'cdhs']
        random.shuffle(deck)
        return deck

    def _deal_cards(self, num):
        return [self.deck.pop() for _ in range(num)]
