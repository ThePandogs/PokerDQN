import random
from itertools import combinations
import numpy as np
from treys import Card, Evaluator
from Player import Player
import sqlite3


class PokerEnvSixMax:
    def __init__(self, num_players=6, starting_chips=1000, small_blind=10, big_blind=20):
        """Inicializa el entorno de Poker con ciegas y otros parámetros."""
        self.evaluator = Evaluator()  # Inicializa el evaluador
        self.num_players = num_players

        self.starting_chips = starting_chips
        self.small_blind_amount = small_blind
        self.big_blind_amount = big_blind
        self.pot_size = 0

        # Posiciones de las ciegas
        self.small_blind_position = 0
        self.big_blind_position = 1

        # Crear jugadores con nombres
        self.players = [Player(f'Player{i}', i, starting_chips) for i in range(num_players)]

        # Inicializar otras variables del juego
        self.community_cards = []
        self.round_history = []
        self.action_history = []
        self.done = False
        self.folded_players = set()
        self.current_bet = 0
        self.current_phase = 'preflop'
        self.game_id = 0

        # Asignar las ciegas
        self._post_blinds()

    ACTION_MAP = {
        'fold': 0,
        'call': 1,
        'raise': 2,
        'check': 3
    }

    def rotate_positions(self):
        # Rota las posiciones de los jugadores
        self.players.append(self.players.pop(0))  # Mueve el primer jugador al final de la lista

    def _create_deck(self):
        """Crea y baraja un mazo de cartas."""
        deck = [Card.new(f'{rank}{suit}') for rank in '23456789TJQKA' for suit in 'cdhs']
        random.shuffle(deck)
        return deck

    def _post_blinds(self):
        """Asigna las apuestas forzadas de las ciegas pequeña y grande."""
        small_blind_player = self.players[self.small_blind_position]
        big_blind_player = self.players[self.big_blind_position]

        # La ciega pequeña apuesta
        small_blind_player.update_chips(-self.small_blind_amount)
        small_blind_player.update_bet(self.small_blind_amount)
        self.pot_size += self.small_blind_amount
        print(f"Player {self.small_blind_position + 1} posts small blind of {self.small_blind_amount}.")

        # La ciega grande apuesta
        big_blind_player.update_chips(-self.big_blind_amount)
        big_blind_player.update_bet(self.big_blind_amount)
        self.pot_size += self.big_blind_amount
        self.current_bet = self.big_blind_amount
        print(f"Player {self.big_blind_position + 1} posts big blind of {self.big_blind_amount}.")

    def _deal_cards(self, num):
        """Reparte un número de cartas del mazo."""
        return [self.deck.pop() for _ in range(num)]

    def create_new_game(self):
        """Inicializa un nuevo juego desde cero."""
        self.deck = self._create_deck()
        self.reset()

    def reset(self):
        """Restablece el estado del entorno al inicio de un nuevo episodio."""
        self.done = False

        # Rotar las ciegas
        self._rotate_blinds()

        # Inicializar el mazo y repartir las cartas
        self.deck = self._create_deck()
        for player in self.players:
            player.update_hand(self._deal_cards(2))  # Reparte 2 cartas a cada jugador
        self.community_cards = []
        self.current_phase = 'preflop'

        self.folded_players.clear()
        self.pot_size = 0
        self.current_bet = 0
        self.round_history = []
        self.action_history = []

        for player in self.players:
            player.update_chips(self.starting_chips)  # Asigna fichas iniciales a cada jugador
            player.update_bet(0)

        # Asignar las ciegas
        self._post_blinds()

        return self._get_state()

    def _rotate_blinds(self):
        """Mueve las posiciones de la ciega pequeña y la ciega grande al siguiente jugador."""
        self.small_blind_position = (self.small_blind_position + 1) % self.num_players
        self.big_blind_position = (self.big_blind_position + 1) % self.num_players
        self.player_position = (self.big_blind_position + 1) % self.num_players

    def _get_state(self):
        """Obtiene el estado actual del juego, accediendo a los historiales de cada jugador."""
        round_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
        pot_odds = self.calculate_pot_odds(self.current_bet, self.pot_size)

        state = {
            'round': round_mapping.get(self.current_phase, -1),
            'community_cards': self.community_cards,
            'pot_size': self.pot_size,
            'current_bet': self.current_bet,
            'pot_odds': pot_odds,
            'player_positions': [player.name for player in self.players],
            'player_hands': [[card for card in player.hand] for player in self.players],
            'player_chips': [player.chips for player in self.players],
            'previous_bets': [player.previous_bet for player in self.players],
            'round_history': self.round_history,
            'player_position': self.player_position,
            'win_probabilities': [self.calculate_equity(player.hand, self.community_cards) for player in self.players],
            'stack_sizes': [player.chips for player in self.players],
            'action_histories': [player.get_action_history() for player in self.players],
            'bet_histories': [player.get_bet_history() for player in self.players]
        }
        print(state)  # Debugging output
        return state

    def _next_phase(self):
        """Avanza a la siguiente fase del juego."""
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
        elif self.current_phase == 'showdown':
            self.done = True

    def step(self, action):
        if self.done:
            raise Exception("Game is already done. Reset the environment to start a new game.")

        player = self.players[self.player_position]
        action_taken = None
        reward = 0

        if action == 0:  # Fold
            total_bet_lost = player.previous_bet
            reward = -total_bet_lost
            self.done = len(self.folded_players) >= self.num_players - 1
            action_taken = 'fold'
            self.player_folds(self.player_position)
            # Registrar la acción y la apuesta en el jugador
            player.set_action('fold', player.previous_bet)

        elif action == 1:  # Call
            call_amount = self.current_bet - player.previous_bet
            player.update_chips(-call_amount)
            self.pot_size += call_amount
            player.update_bet(self.current_bet)
            action_taken = 'call'
            player.set_action('call', call_amount)
            reward = 0

        elif action == 2:  # Raise
            raise_amount = self.calculate_raise_amount(player)
            player.update_chips(-(self.current_bet + raise_amount))
            self.pot_size += (self.current_bet + raise_amount)
            self.current_bet += raise_amount
            player.update_bet(self.current_bet)
            action_taken = 'raise'
            player.set_action('raise', self.current_bet + raise_amount)
            reward = raise_amount * 0.1

        elif action == 3:  # Check
            player.update_bet(self.current_bet)
            action_taken = 'check'
            player.set_action('check')
            reward = 0

        # Registro detallado de apuestas por ronda
        self.round_history.append({
            'player_position': self.player_position,
            'action': action_taken,
            'amount': player.previous_bet
        })

        self._next_phase()
        self.player_position = (self.player_position + 1) % self.num_players

        state = self._get_state()
        done = self._is_game_over()

        if done:
            rewards = self._calculate_rewards()
            return state, rewards[self.player_position], done, {}
        else:
            num_active_players = self.num_players - len(self.folded_players)
            if num_active_players == 1:
                reward += 0.5

            return state, reward, done, {}

    def calculate_raise_amount(self, player):
        """Calcula la cantidad que el jugador debe subir."""
        return max(20, int(0.05 * player.chips))

    def calculate_pot_odds(self, current_bet, pot_size):
        """Calcula las odds del bote."""
        if current_bet == 0:
            return 0.0
        return current_bet / (pot_size + current_bet)

    def calculate_equity(self, player_hand, community_cards):
        """Simula el equity del jugador contra posibles manos oponentes."""
        if not community_cards:
            return 0.5  # Asumir 50% de equity si no hay cartas comunitarias

        hand_rank = self.evaluator.evaluate(player_hand, community_cards)
        return 1 / (1 + hand_rank / 10000)  # Un ejemplo simplificado

    def player_folds(self, player_position):
        """Marca a un jugador como retirado (fold)."""
        self.folded_players.add(player_position)
        self.players[player_position].set_action('fold')

    def _calculate_rewards(self):
        """Calcula las recompensas al final del juego."""
        return [0] * self.num_players

    def _is_game_over(self):
        """Verifica si el juego ha terminado."""
        return self.done
