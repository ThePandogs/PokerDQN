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
    Procesa el estado crudo del juego en un formato adecuado para el modelo DQN.

    Args:
        state (dict): El estado crudo del juego.

    Returns:
        np.ndarray: El estado procesado como un array numpy.
    """
    # Evaluar el rango de la mano del jugador
    hand_rank = evaluate_hand(state['player_hand'], state['community_cards'])
    hand_rank = hand_rank if hand_rank is not None else -1

    # Convertir datos a arrays numpy
    pot_size = np.array([state['pot_size']])
    current_bet = np.array([state['current_bet']])
    player_chips = np.array(state['player_chips'])
    action_history = np.array(state['action_history'])
    player_position = np.array([state['player_position']])
    win_probability = np.array([state['win_probability']])
    pot_odds = np.array([state['pot_odds']])
    stack_sizes = np.array(state['stack_sizes'])
    previous_bets = np.array(state['previous_bets'])
    # game_id = np.array([state['game_id']])  # Incluye el ID del juego si es necesario

    hand_rank = np.array([hand_rank])

    # Concatenar todos los componentes para formar el estado
    processed_state = np.concatenate([
        hand_rank, pot_size, current_bet,
        player_chips, action_history, player_position,
        win_probability, pot_odds, stack_sizes, previous_bets
    ])

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


def get_state_size(self):
    state = env.reset()  # Obtener el estado inicial
    processed_state = process_state(state)  # Procesar el estado
    return processed_state.size  # Devuelve el tamaño del estado procesado

if __name__ == "__main__":
    EPISODES = 3500
    env = PokerEnvSixMax()
    state_size = 24    # Ajusta el tamaño del estado según los datos procesados
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
    env.create_new_game()
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
