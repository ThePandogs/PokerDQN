class Player:
    def __init__(self, name, position, chips):
        self.name = name
        self.position = position
        self.chips = chips
        self.hand = []
        self.previous_bet = 0
        self.action_history = []  # Historial de acciones del propio jugador
        self.other_players_actions = {}  # Almacenará las acciones de los demás jugadores
        self.reward = 0
        # Para cada jugador, almacenaremos su historial de acciones y apuestas
        self.folded = False  # Para verificar si el jugador ha hecho fold

    def update_chips(self, amount):
        """Actualiza el número de fichas del jugador. `amount` puede ser positivo o negativo."""
        self.chips += amount

    def update_hand(self, hand):
        """Actualiza la mano del jugador."""
        self.hand = hand

    def update_bet(self, bet):
        """Actualiza la última apuesta del jugador."""
        self.previous_bet = bet

    def set_action(self, action, amount=0):
        """Registra la acción tomada por el jugador y la cantidad apostada."""
        if not self.folded:  # Solo registra si el jugador sigue en juego
            self.action_history.append((action, amount))
            self.bet_history.append(amount)

    def record_other_player_action(self, player_position, action, amount):
        """Almacena las acciones y apuestas de los otros jugadores."""
        if player_position not in self.other_players_actions:
            self.other_players_actions[player_position] = []
        self.other_players_actions[player_position].append((action, amount))

    def get_action_history(self):
        """Devuelve el historial de acciones del jugador."""
        return self.action_history

    def get_bet_history(self):
        """Devuelve el historial de apuestas del jugador."""
        return self.bet_history

    def get_other_players_actions(self):
        """Devuelve el historial de acciones de los demás jugadores."""
        return self.other_players_actions

    def player_folds(self):
        """Marca al jugador como retirado (fold) y detiene el registro de acciones."""
        self.folded = True


