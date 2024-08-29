import random

from treys import Evaluator, Card

from Player import Player
from PokerMonteCarlo import PokerMonteCarlo


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
        self.players[self.small_blind_position].action_history.append({'action': 5, 'amount': self.small_blind_amount})

        self.players[self.big_blind_position].chips -= self.big_blind_amount
        self.players[self.big_blind_position].previous_bet = self.big_blind_amount
        self.pot_size += self.big_blind_amount
        self.current_bet = self.big_blind_amount
        self.players[self.big_blind_position].action_history.append({'action': 6, 'amount': self.big_blind_amount})

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

        player = self.players[self.player_position]
        if player in self.folded_players:
            player.action_history.append({'action': -1, 'amount': 0})
        else:

            if action == 0:  # Fold
                player.reward *= -1
                self.folded_players.add(player)
                player.action_history.append({'action': action, 'amount': 0})

            elif action == 1:  # Call
                call_amount = self.current_bet - player.previous_bet
                player.chips -= call_amount
                self.pot_size += call_amount
                player.previous_bet = self.current_bet
                player.action_history.append({'action': action, 'amount': call_amount})
                player.reward = call_amount * 0.2

            elif action == 2:  # Raise
                raise_amount = self.calculate_raise_amount(player)
                player.chips -= raise_amount
                self.pot_size += raise_amount
                self.current_bet += raise_amount
                player.previous_bet = self.current_bet
                player.action_history.append({'action': action, 'amount': raise_amount})
                player.reward = raise_amount * 0.3

            elif action == 3:  # Check
                player.action_history.append({'action': action, 'amount': 0})
                player.reward += 0

        self.round_history.append({
            'player_position': self.player_position,
            'action': action,
            'phase': self.current_phase,
            'amount': player.previous_bet
        })

        self._next_phase()
        self.player_position = (self.player_position + 1) % self.num_players

        state = self._get_state()
        done = self._is_game_over()

        # Verificar si todos los jugadores han igualado la apuesta
        if self._all_players_equalized_bet():
            # Solo cambiamos de fase si todos los jugadores han igualado la apuesta
            self._next_phase()

        if done:
            rewards = self._calculate_rewards()
            return state,  done, {}
        else:
            return state, done, {}

    def _all_players_equalized_bet(self):
        # Verificar si todos los jugadores activos han igualado la apuesta actual
        for player in self.players:
            if player.position not in self.folded_players:
                if player.previous_bet != self.current_bet:
                    return False
        return True

    def _calculate_total_bets(self, player):
        return sum(player.bet_history)

    def _calculate_rewards(self):
        rewards = [0] * self.num_players
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

    def calculate_raise_amount(self, player):
        return max(20, int(0.05 * player.chips))

    def calculate_pot_odds(self, current_bet, pot_size):
        if current_bet == 0:
            return 0.0
        return current_bet / (pot_size + current_bet)

    def calculate_equity(self, player_hand, community_cards):
        # Filtra los jugadores activos basados en el estado actual del juego
        active_players = [i for i in range(self.num_players) if i not in self.folded_players]
        return self.monte_carlo.simulate_hand(player_hand, community_cards, active_players=active_players)

    def _get_state(self):
        round_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3, 'showdown': 4}
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
                    'bet_history': player.bet_history,
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
