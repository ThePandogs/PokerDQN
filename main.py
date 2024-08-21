import numpy as np
from treys import Card, Evaluator
from DQNAgent import DQNAgent
from PokerEnvSixMax import PokerEnvSixMax
import os
import matplotlib.pyplot as plt


def evaluate_hand(player_hand, community_cards):
    evaluator = Evaluator()
    if len(player_hand) + len(community_cards) >= 5:
        hand_rank = evaluator.evaluate(player_hand, community_cards[:5])
        return hand_rank
    else:
        return None


def convert_cards(cards):
    print(f"Converting cards: {[Card.int_to_str(card) for card in cards]}")  # Debug print
    return cards


def process_state(state: dict) -> np.ndarray:
    """
    Processes the raw game state into a format suitable for the DQN model.

    Args:
        state (dict): The raw game state.

    Returns:
        np.ndarray: The processed state as a numpy array.
    """
    # Extract and process relevant state components
    round_stage = state['round']
    player_hand = state['player_hand']
    community_cards = state['community_cards']


    # Evaluate the player's hand rank
    hand_rank = evaluate_hand(player_hand, community_cards)
    hand_rank = hand_rank if hand_rank is not None else -1



    # Convert data to numpy arrays
    pot_size = np.array([state['pot_size']])
    current_bet = np.array([state['current_bet']])
    player_chips = np.array(state['player_chips'])
    action_history = np.array(state['action_history'])
    player_position = np.array([state['player_position']])
    win_probability = np.array(state['win_probability'])
    pot_odds = np.array(state['pot_odds'])
    stack_sizes = np.array(state['stack_sizes'])
    previous_bets = np.array(state['previous_bets'])

    hand_rank = np.array([hand_rank])

    # Concatenate all components to form the state
    processed_state = np.concatenate([
        hand_rank, pot_size, current_bet,
        player_chips, action_history, player_position,
        win_probability, pot_odds, stack_sizes, previous_bets
    ])

    # Print statements for debugging
    print(f"Actual state: {state}")
    print(f"Round stage: {round_stage}")
    print(f"Player {player_position} hand: {[Card.int_to_str(card) for card in player_hand]}")
    print(f"Community hand: {[Card.int_to_str(card) for card in community_cards]}")
    print(f"Rank hand: {hand_rank}")

    return processed_state


def plot_progress(rewards, epsilons):
    # Graficar la recompensa promedio
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Acumulada")
    plt.title("Recompensa a lo Largo del Entrenamiento")

    # Graficar el descenso de epsilon
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.xlabel("Episodios")
    plt.ylabel("Valor de Epsilon")
    plt.title("Descenso de Epsilon durante el Entrenamiento")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    EPISODES = 3500
    env = PokerEnvSixMax()
    state_size = 40  # Ajusta el tamaño del estado según los datos procesados
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    # Cargar modelo si existe
    load_model = True  # Cambia a True si quieres cargar el modelo

    model_filename = "poker_dqn_model.keras"
    if os.path.exists(model_filename):
        print(f"Cargando el modelo desde {model_filename}")
        agent.load(model_filename)
    else:
        print(f"No se encontró el archivo {model_filename}, comenzando entrenamiento desde cero.")

    batch_size = 32

    rewards = []
    epsilons = []

    for e in range(EPISODES):
        state = env.reset()
        state = process_state(state)
        state = np.reshape(state, [1, state_size])

        episode_reward = 0
        while not env.done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            next_state = np.reshape(next_state, [1, state_size])

            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                rewards.append(episode_reward)
                epsilons.append(agent.epsilon)
                print(f"Episode: {e}/{EPISODES}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        if agent.memory.get_size() > batch_size:
            agent.replay(batch_size)

    # Guardar el modelo después de entrenar
    agent.save("poker_dqn_model.keras")

    # Graficar el progreso
    plot_progress(rewards, epsilons)
