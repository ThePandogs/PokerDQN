import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from treys import Card, Evaluator

from DQNAgent import DQNAgent, process_state
from PokerEnvSixMax import PokerEnvSixMax


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


def get_state_size(env):
    state = env.reset()  # Resetea el entorno para obtener un estado inicial
    processed_state = process_state(state)  # Procesa el estado crudo
    return processed_state.size  # Devuelve el tamaño del estado procesado


if __name__ == "__main__":
    EPISODES = 3
    env = PokerEnvSixMax()
    state_size = get_state_size(env)  # Llama a la función y obtiene el tamaño del estado
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
        state = env.reset()  # Inicia una nueva mano o episodio
        state = process_state(state)
        state_size = state.size  # Usa el tamaño correcto
        state = np.reshape(state, [1, state_size])

        episode_reward = 0
        while not env.done:
            action = agent.act(state)
            next_state, done, _ = env.step(action)
            next_state = process_state(next_state)
            state_size = next_state.size

            next_state = np.reshape(next_state, [1, state_size])


            agent.remember(state, action, next_state,done)


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
