class Player:
    def __init__(self, name, position, chips):
        self.name = name
        self.position = position
        self.chips = chips
        self.hand = []
        self.previous_bet = 0
        self.action_history = []
        self.bet_history = []

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
        """Registra la acción tomada por el jugador."""
        self.action_history.append((action, amount))

    def get_action_history(self):
        """Devuelve el historial de acciones del jugador."""
        return self.action_history

    def get_bet_history(self):
        """Devuelve el historial de apuestas del jugador."""
        return self.bet_history
