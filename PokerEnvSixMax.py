import random

import numpy as np
from treys import Card, Evaluator
from itertools import combinations

class PokerEnvSixMax:
    def __init__(self):

        self.evaluator = Evaluator()  # Inicializa el evaluador


        self.num_players = 6
        self.players_hands = []  # Asegúrate de que esto está inicializado correctamente
        self.community_cards = []
        self.player_position = 0

        self.current_games = {}  # Mapa para almacenar el estado de múltiples partidas
        self.game_id = 0  # ID único para cada nueva partida
    def _create_deck(self):
        """Crea y baraja un mazo de cartas."""
        deck = [Card.new(f'{rank}{suit}') for rank in '23456789TJQKA' for suit in 'cdhs']
        random.shuffle(deck)
        return deck

    def _deal_cards(self, num):
        """Reparte un número de cartas del mazo."""
        return [self.deck.pop(random.randrange(len(self.deck))) for _ in range(num)]

    def create_new_game(self):
        """
        Inicializa un nuevo juego desde cero.
        """
        # Configuración del juego: barajar cartas, repartir a jugadores, etc.
        self.deck = self.initialize_deck()
        self.players = self.initialize_players()
        self.current_phase = 'preflop'
        self.current_game_id = 0  # Inicializa el ID del juego
        self.reset()

    def reset(self):
        """
        Restablece el estado del entorno al inicio de un nuevo episodio.

        Returns:
            dict: El estado inicial del entorno.
        """
        self.current_game_id += 1  # Incrementa el ID del juego
        self.done = False

        # Inicializar el mazo y repartir las cartas
        self.deck = self.initialize_deck()
        self.players_hands = [self._deal_cards(2) for _ in range(self.num_players)]  # Reparte 2 cartas a cada jugador
        self.community_cards = []
        self.round_phase = 'preflop'  # Inicializa la fase de la ronda en 'preflop' al comienzo de un nuevo juego

        # Inicializa el tamaño del bote y la apuesta actual
        self.pot_size = 0
        self.current_bet = 0  # Inicializar la apuesta actual

        # Inicializa las apuestas anteriores de cada jugador
        self.previous_bets = [0] * self.num_players  # Inicializa las apuestas de cada jugador en 0

        self.win_probability = 0.5  # Valor temporal o calculado más adelante
        self.pot_odds = 0
        # Inicializa las fichas de cada jugador
        self.player_chips = [1000] * self.num_players  # Asigna 1000 fichas a cada jugador
        self.stack_sizes = [1000] * 6

        # Inicializa el historial de la ronda y las acciones
        self.round_history = []  # Historial vacío al inicio de cada episodio
        self.action_history = []  # Inicializa el historial de acciones como una lista vacía



        # Inicializar el estado
        self.state = {
            'player_hand': self.players_hands[0],  # La mano del jugador actual (posición 0 al inicio)
            'community_cards': [],  # Cartas comunitarias vacías al inicio
            'pot_size': 0,
            'current_bet': self.current_bet,  # Asigna la apuesta inicial al estado
            'player_chips': self.player_chips,  # Las fichas de los jugadores
            'action_history': [],  # Historial de acciones vacío
            'player_position': 0,  # Posición del jugador actual
            'win_probability': 0.5,  # Probabilidad inicial de ganar
            'pot_odds': 1.0,  # Odds del bote inicial
            'stack_sizes': [1000] * 6,  # Tamaño de pila inicial
            'previous_bets': self.previous_bets  # Apuestas anteriores
        }

        return self.state

    def _get_state(self):
        """Obtiene el estado actual del juego."""
        round_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}

        state = {
            'round': round_mapping.get(self.round_phase, -1),
            'player_hand': self.players_hands[self.player_position],
            'community_cards': self.community_cards,
            'pot_size': self.pot_size,
            'current_bet': self.current_bet,
            'player_chips': self.player_chips,
            'action_history': self.action_history,  # Ahora se espera que ya estén en formato numérico
            'player_position': self.player_position,
            'win_probability': self.win_probability,
            'pot_odds': self.pot_odds,
            'stack_sizes': self.stack_sizes,
            'previous_bets': self.previous_bets,
        }
        return state
    def initialize_deck(self):
        """
        Inicializa y baraja una nueva baraja de cartas.
        """
        deck =  self._create_deck()
        np.random.shuffle(deck)
        return deck

    def initialize_players(self):
        """
        Inicializa el estado de los jugadores.
        """
        return {
            'player_hand': [self.draw_card(), self.draw_card()],
            'community_cards': [],
            'pot': 0,
            'bets': [0] * 6,
            'player_position': 0
        }

    def draw_card(self):
        """
        Extrae una carta del mazo.
        """
        return self.deck.pop()
    def calculate_equity(self, player_hand, community_cards):
        """Simula el equity del jugador contra posibles manos oponentes."""
        if not community_cards:
            return 0.5  # Asumir 50% de equity si no hay cartas comunitarias
        hand_rank = self.evaluator.evaluate(player_hand, community_cards)
        return 1 / (1 + hand_rank / 10000)  # Fórmula básica para estimar probabilidad de ganar

    def calculate_pot_odds(self, current_bet, pot_size):
        """Calcula las pot odds."""
        if current_bet == 0:
            return 0
        return current_bet / (pot_size + current_bet)

    def should_bluff(self, equity, community_cards):
        """Decide si hacer farol en función del equity y las cartas comunitarias."""
        if equity < 0.2:  # Farolea si el equity es bajo
            if self.is_failed_draw(community_cards):
                return True
        return False

    def is_failed_draw(self, community_cards):
        """Detecta si el jugador falló un proyecto de color o escalera."""
        # Esta es una simplificación; en la realidad, se necesitaría analizar las cartas.
        return True

    def step(self, action):
        """Ejecuta una acción y actualiza el estado del juego."""
        if self.done:
            raise Exception("Game is already done. Reset the environment to start a new game.")

        # Cálculo de equity y pot odds
        equity = self.calculate_equity(self.players_hands[self.player_position], self.community_cards)
        pot_odds = self.calculate_pot_odds(self.current_bet, self.pot_size)

        # Inicializa el castigo y la acción del jugador actual
        reward = 0
        action_taken = None

        # Lógica para manejar la acción (fold, call, raise, etc.)
        if action == 0:  # Fold
            self.done = True
            reward = -10  # Castigo fijo por hacer fold
            action_taken = 'fold'
            print(f"Player {self.player_position} folds.")

        elif action == 1:  # Call
            if equity > pot_odds:
                call_amount = self.current_bet - self.previous_bets[self.player_position]
                self.player_chips[self.player_position] -= call_amount
                self.pot_size += call_amount
                self.previous_bets[self.player_position] = self.current_bet
                action_taken = 'call'
                print(f"Player {self.player_position} calls {call_amount}.")
            else:
                reward = -10
                self.done = True
                action_taken = 'fold'
                print(f"Player {self.player_position} folds due to insufficient equity.")

        elif action == 2:  # Raise
            if equity > 0.5:
                raise_amount = self.calculate_raise_amount(equity)
            else:
                if self.should_bluff(equity, self.community_cards):
                    raise_amount = self.calculate_bluff_raise_amount()
                else:
                    reward = -self.calculate_bluff_penalty()
                    self.done = True
                    action_taken = 'fold'
                    print(f"Player {self.player_position} folds due to low equity and failed bluff.")
                    return self._get_state(), reward, self.done, {}

            self.player_chips[self.player_position] -= (self.current_bet + raise_amount)
            self.pot_size += (self.current_bet + raise_amount)
            self.current_bet += raise_amount
            self.previous_bets[self.player_position] = self.current_bet
            action_taken = 'raise'
            print(f"Player {self.player_position} raises to {self.current_bet}.")

        elif action == 3:  # Check
            self.previous_bets[self.player_position] = self.current_bet
            action_taken = 'check'
            print(f"Player {self.player_position} checks.")

        # Actualizar el historial de acciones
        self.round_history.append((self.player_position, action_taken))

        # Avanzar a la siguiente fase del juego
        self._next_phase()

        # Mover al siguiente jugador
        self.player_position = (self.player_position + 1) % self.num_players

        # Obtener el nuevo estado y verificar si el juego ha terminado
        state = self._get_state()
        done = self._is_game_over()

        if done:
            rewards = self._calculate_rewards()  # Calcula la recompensa final si el juego termina
            reward = rewards[self.player_position]
            self._determine_winner()  # Determina el ganador y la mano ganadora
            print(f"Game over. Player {self.player_position} reward: {reward}")
            print(f"Winner: Player {self.winner} with hand:{[Card.int_to_str(card) for card in self.winning_hand]} ")
        else:
            # Penaliza el fold solo si el juego termina
            reward = -10 if action == 0 else reward

        return state, reward, done, {}

    def _determine_winner(self):
        """Determina el ganador en el showdown."""
        if self.round_phase == 'showdown':
            hand_ranks = []
            for hand in self.players_hands:
                all_cards = hand + self.community_cards
                best_hand = self._select_best_hand(all_cards)
                rank = self.evaluator.evaluate(best_hand[:2], best_hand[2:])
                hand_ranks.append((rank, best_hand))

            best_rank, best_hand = min(hand_ranks, key=lambda x: x[0])
            self.winner = hand_ranks.index((best_rank, best_hand))
            self.winning_hand = best_hand

    def calculate_bluff_penalty(self):
        """Calcula la penalización en caso de farol."""
        base_penalty = 10
        pot_penalty = 0.1 * self.pot_size  # Penalización basada en el tamaño del bote

        # Penalización adicional en función de la fase del juego
        stage_penalty = {
            "preflop": 0,
            "flop": 5,
            "turn": 10,
            "river": 15
        }.get(self.round_phase, 0)

        return base_penalty + pot_penalty + stage_penalty

    def calculate_raise_amount(self, equity):
        """Calcula el monto del raise basado en la equidad y el tamaño del bote."""
        base_raise = 10
        adjusted_raise = base_raise * (equity / 0.5)
        pot_factor = 0.1 * self.pot_size
        total_raise = adjusted_raise + pot_factor
        raise_amount = min(total_raise, self.player_chips[self.player_position] - self.current_bet)

        # Ajuste según la fase del juego
        stage_multiplier = {
            "preflop": 1.0,
            "flop": 1.2,
            "turn": 1.5,
            "river": 1.8
        }.get(self.round_phase, 1.0)

        return raise_amount * stage_multiplier

    def calculate_bluff_raise_amount(self):
        """Calcula el monto del raise en caso de farol."""
        base_bluff_raise = 20
        pot_factor = 0.2 * self.pot_size
        bluff_raise_amount = base_bluff_raise + pot_factor
        num_opponents = len(self.player_chips) - 1
        bluff_raise_amount *= (num_opponents / 6)
        return min(bluff_raise_amount, self.player_chips[self.player_position])

    def _next_phase(self):
        """Avanza a la siguiente fase del juego."""
        if self.round_phase == 'preflop':
            self.community_cards = []
            self.round_phase = 'flop'
            self.community_cards.extend(self._deal_cards(3))
        elif self.round_phase == 'flop':
            self.round_phase = 'turn'
            self.community_cards.append(self._deal_cards(1)[0])
        elif self.round_phase == 'turn':
            self.round_phase = 'river'
            self.community_cards.append(self._deal_cards(1)[0])
        elif self.round_phase == 'river':
            self.round_phase = 'showdown'
        elif self.round_phase == 'showdown':
            self.done = True

    def _is_game_over(self):
        """Determina si el juego ha terminado."""
        if self.round_phase == 'showdown':
            return True

        # Verifica si algún jugador se ha quedado sin fichas y el juego no está en showdown
        if any(chips <= 0 for chips in self.player_chips):
            return True

        return False

    def _calculate_rewards(self):
        """Calcula las recompensas en el showdown."""
        rewards = [0] * 6
        if not self.community_cards:
            return rewards

        hand_ranks = []
        for hand in self.players_hands:
            all_cards = hand + self.community_cards
            best_hand = self._select_best_hand(all_cards)
            rank = self.evaluator.evaluate(best_hand[:2], best_hand[2:])
            hand_ranks.append(rank)

        best_rank = min(hand_ranks)
        winners = [i for i, rank in enumerate(hand_ranks) if rank == best_rank]

        total_winners = len(winners)
        reward = self.pot_size / total_winners if total_winners > 0 else 0

        for i in range(6):
            if i in winners:
                rewards[i] = reward
            else:
                rewards[i] = -self.pot_size / 6

        return rewards

    def _select_best_hand(self, all_cards):
        """Selecciona la mejor mano de 5 cartas de las cartas dadas."""
        best_hand = None
        best_rank = float('inf')

        for combo in combinations(all_cards, 5):
            hand_cards = list(combo)
            rank = self.evaluator.evaluate(hand_cards[:2], hand_cards[2:])
            if rank < best_rank:
                best_rank = rank
                best_hand = hand_cards

        return best_hand

    def render(self):
        """Muestra el estado actual del juego (opcional)."""
        print(f"Round phase: {self.round_phase}")
        print(f"Community cards: {[Card.int_to_str(card) for card in self.community_cards]}")
        for i, hand in enumerate(self.players_hands):
            print(f"Player {i + 1} hand: {[Card.int_to_str(card) for card in hand]}")
        print(f"Pot size: {self.pot_size}")
        print(f"Current bet: {self.current_bet}")
        print(f"Player chips: {self.player_chips}")
        print(f"Action history: {self.action_history}")
        print(f"Round history: {self.round_history}")
        print(f"Player position: {self.player_position}")
        print(f"Win probability: {self.win_probability}")
        print(f"Pot odds: {self.pot_odds}")
        print(f"Stack sizes: {self.stack_sizes}")
        print(f"Previous bets: {self.previous_bets}")
        if self.round_phase == 'showdown':
            print(f"Winner: Player {self.winner + 1} with hand: {self.winning_hand}")




# Ejemplo de uso
if __name__ == "__main__":
    env = PokerEnvSixMax()
    state = env.reset()
    env.render()
    state, reward, done, _ = env.step(1)  # Ejemplo de llamada a la acción "call"
    env.render()
