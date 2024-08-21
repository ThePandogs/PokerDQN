import logging
from keras.api import optimizers, models, layers
import numpy as np
from treys import Card

from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from common import evaluate_hand

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
        """Construye el modelo de red neuronal."""
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        return model

    def act(self, state):
        """Determina la acción a tomar basado en el estado actual."""
        state = np.reshape(state, [1, self.state_size])  # Asegurar que el estado tiene las dimensiones correctas
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
        else:
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])

        # Logging de la acción tomada
        logging.debug(f"Player {self.player_position} takes action: {action}")
        logging.debug(f"Action history before update: {self.action_history}")
        return action

    def remember(self, state, action, reward, next_state, done):
        """Almacena la experiencia en la memoria del agente."""
        self.memory.add((state, action, reward, next_state, done), priority=1.0)

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

        if len(states.shape) == 3 and states.shape[1] == 1:
            states = np.squeeze(states, axis=1)
        if len(next_states.shape) == 3 and next_states.shape[1] == 1:
            next_states = np.squeeze(next_states, axis=1)

        print(type(states), states.dtype)

        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        logging.debug(f"States shape: {states.shape}")
        logging.debug(f"Targets shape: {targets.shape}")
        logging.debug(f"Weights shape: {np.array(weights).shape}")

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
        logging.debug(f"Updated action history after action {action} by player {player_position}: {self.action_history}")

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

    @staticmethod
    def process_state(state: dict) -> np.ndarray:
        """Procesa el estado para que sea compatible con el modelo."""
        round_stage = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}.get(state['round'], -1)

        player_hand = state['player_hand']
        community_cards = state['community_cards']

        hand_rank = evaluate_hand(player_hand, community_cards)
        hand_rank = hand_rank if hand_rank is not None else -1

        pot_size = np.array([state['pot_size']], dtype=np.float32)
        current_bet = np.array([state['current_bet']], dtype=np.float32)
        player_chips = np.array(state['player_chips'], dtype=np.float32)
        action_history = np.array(state['action_history'], dtype=np.float32)
        player_position = np.array([state['player_position']], dtype=np.float32)
        win_probability = np.array(state['win_probability'], dtype=np.float32)
        pot_odds = np.array(state['pot_odds'], dtype=np.float32)
        stack_sizes = np.array(state['stack_sizes'], dtype=np.float32)
        previous_bets = np.array(state['previous_bets'], dtype=np.float32)

        hand_rank = np.array([hand_rank], dtype=np.float32)

        processed_state = np.concatenate([
            np.array([round_stage], dtype=np.int32), hand_rank, pot_size, current_bet,
            player_chips, action_history, player_position,
            win_probability, pot_odds, stack_sizes, previous_bets
        ])

        logging.debug(f"Processed state shape: {processed_state.shape}")
        return processed_state
