# common.py

from treys import Card, Evaluator


def evaluate_hand(player_hand, community_cards):
    # Lógica para evaluar la mano (modificar según sea necesario)
    evaluator = Evaluator()
    if len(player_hand) + len(community_cards) >= 5:
        hand_rank = evaluator.evaluate(player_hand, community_cards[:5])
        return hand_rank
    else:
        return None

