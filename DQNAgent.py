import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.api import optimizers, models, layers

from PokerEnvSixMax import PokerEnvSixMax
from PrioritizedReplayBuffer import PrioritizedReplayBuffer

# Configurar el nivel de logging
logging.basicConfig(level=logging.DEBUG)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(2000)
        self.gamma = 0.95  # Tasa de descuento
        self.epsilon = 1.0  # Tasa de exploración
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        self.model = self._build_model()
        self.target_model = self._build_model()  # Modelo objetivo para el entrenamiento
        self.target_update_freq = 100  # Frecuencia para actualizar el modelo objetivo
        self.player_position = 0  # Suponiendo que tiene acceso a esto
        self.action_history = [0] * 6  # Historial de acciones para 6 jugadores

    def _build_model(self):
        """Construye el modelo de red neuronal con la forma de entrada ajustada."""
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))  # Ajusta la forma de entrada según el tamaño del estado
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        return model

    def act(self, state):
        """Determina la acción a tomar basado en el estado actual."""

        state = np.reshape(state, [1,  state.size])  # Asegurar que el estado tiene las dimensiones correctas
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
            choose_action = "Action Random"
        else:
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])
            choose_action = "Prediction"

        # Logging de la acción tomada
        logging.debug(f"Player {self.player_position} takes action: {action} using {choose_action}")
        logging.debug(f"Action history before update: {self.action_history}")
        return action

    def remember(self, state, action, next_state, done):
        """Almacena la experiencia en la memoria del agente."""
        self.memory.add((state, action, next_state, done), priority=1.0)

    def replay(self, batch_size):
        """Actualiza el modelo basado en una muestra aleatoria de la memoria."""
        if len(self.memory.tree.data) < batch_size:
            return

        minibatch, indices, weights = self.memory.sample(batch_size)

        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Verifica la forma de los estados
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        # Asegúrate de que la forma de `states` sea (batch_size, 144)
        assert states.shape[1:] == (self.state_size,)

        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Entrenamiento del modelo con pesos
        self.model.fit(states, targets, epochs=1, verbose=0, sample_weight=np.array(weights))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Actualizar prioridades de la memoria
        self.memory.update_priorities(indices, np.array(weights) ** (-self.memory.beta))

    def update_target_model(self):
        """Actualiza el modelo objetivo con los pesos del modelo principal."""
        self.target_model.set_weights(self.model.get_weights())

    def update_action_history(self, player_position, action):
        """Actualiza el historial de acciones."""
        self.action_history[player_position] = action
        logging.debug(
            f"Updated action history after action {action} by player {player_position}: {self.action_history}")

    def save(self, filename):
        """Guarda el modelo entrenado a un archivo."""
        self.model.save(filename)

    def load(self, filename):
        """Carga un modelo entrenado desde un archivo."""
        try:
            self.model = models.load_model(filename)
            logging.info(f"Model loaded from {filename}")
            self.update_target_model()  # Sincronizar el modelo objetivo con el cargado
        except FileNotFoundError:
            logging.warning(f"Model file {filename} not found, starting with a new model.")


# Procesar estado (asegúrate de que el tamaño del estado sea el correcto)
def process_state(state):
    round_ = state['round']
    community_cards = np.array(state['community_cards'], dtype=np.int32)
    pot_size = state['pot_info']['pot_size']
    current_bet = state['pot_info']['current_bet']
    pot_odds = state['pot_info']['pot_odds']

    # Players data
    players = state['players']
    player_positions = np.array([player['position'] for player in players], dtype=np.int32)
    player_chips = np.array([player['chips'] for player in players], dtype=np.float32)
    player_hands = np.array([card for player in players for card in player['hand']], dtype=np.int32)
    previous_bets = np.array([player['previous_bet'] for player in players], dtype=np.float32)

    # Flatten action history
    action_histories = []
    for player in players:
        if player['action_history']:
            actions = [ah['action'] for ah in player['action_history']]
            amounts = [ah['amount'] for ah in player['action_history']]
            action_histories.append(np.concatenate([np.array(actions), np.array(amounts)]))
        else:
            action_histories.append(np.zeros(0))  # Or another appropriate default value

    action_histories = np.concatenate(action_histories)

    # Other data
    win_probabilities = np.array(state['win_probabilities'], dtype=np.float32)
    stack_sizes = np.array(state['stack_sizes'], dtype=np.float32)

    # Flatten round history
    round_history = []
    for rh in state['round_history']:
        round_history.extend([
            rh['player_position'],
            rh['action'],
            rh['amount']
        ])
    round_history = np.array(round_history, dtype=np.float32)

    # Concatenate all data into one array
    state_processed = np.concatenate([
        np.array([round_], dtype=np.int32),
        community_cards,
        np.array([pot_size, current_bet, pot_odds], dtype=np.float32),
        player_positions,
        player_chips,
        player_hands,
        previous_bets,
        action_histories,
        win_probabilities,
        stack_sizes,
        round_history
    ])

    return state_processed

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
    state = env.reset()
    processed_state = process_state(state)
    return processed_state.size


if __name__ == "__main__":
    EPISODES = 3500
    env = PokerEnvSixMax()
    state_size = get_state_size(env)  # Ajusta el tamaño del estado según los datos procesados
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
    env.reset()
    for e in range(EPISODES):
        state = env.reset()
        state = process_state(state)
        state_size = get_state_size(env)
        state = np.reshape(state, [1, state_size])

        episode_reward = 0
        while not env.done:
            action = agent.act(state)
            next_state, done, _ = env.step(action)
            next_state = process_state(next_state)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, next_state, done)
            state = next_state

            if done:
                break

        if agent.memory.get_size() > batch_size:
            agent.replay(batch_size)

    # Guardar el modelo después de entrenar
    agent.save("poker_dqn_model.keras")
