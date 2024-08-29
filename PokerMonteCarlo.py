import random
from treys import Card, Evaluator


class PokerMonteCarlo:
    def __init__(self, evaluator):
        self.original_deck = [Card.new(f'{rank}{suit}') for rank in '23456789TJQKA' for suit in 'cdhs']
        self.evaluator = evaluator

    def _create_deck(self):
        return self.original_deck.copy()

    def deal_cards(self, deck, num):
        if num > len(deck):
            raise ValueError("Not enough cards in the deck to deal")
        return [deck.pop() for _ in range(num)]

    def simulate_hand(self, player_hand, community_cards, num_simulations=1000, active_players=None):
        if active_players is None:
            active_players = []

        wins = 0

        for _ in range(num_simulations):
            deck = self._create_deck()  # Crear un nuevo mazo para cada simulación

            # Elimina cartas en las manos del jugador y comunitarias del mazo
            for card in player_hand + community_cards:
                if card in deck:
                    deck.remove(card)

            # Repartir cartas comunitarias simuladas si es necesario
            simulated_community_cards = self.deal_cards(deck, 5 - len(community_cards))
            all_community_cards = community_cards + simulated_community_cards

            # Evaluar la mano del jugador
            player_score = self.evaluator.evaluate(player_hand, all_community_cards)

            # Generar manos para los oponentes activos
            other_hands = self.generate_other_hands(len(active_players), deck)

            better_hands = sum(
                1 for other_hand in other_hands
                if self.evaluator.evaluate(other_hand, all_community_cards) < player_score
            )

            if better_hands == 0:
                wins += 1

        return wins / num_simulations

    def generate_other_hands(self, num_hands, deck):
        hands = []
        for _ in range(num_hands):
            if len(deck) < 2:
                raise ValueError("Not enough cards left in the deck to deal")
            hand = self.deal_cards(deck, 2)
            hands.append(hand)
        return hands
