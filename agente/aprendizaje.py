"""
Módulo del agente de Aprendizaje por Refuerzo.

Implementa Q-learning tabular para la detección de anomalías en una máquina
industrial. El agente mantiene una tabla Q de tamaño (n_estados, n_acciones)
que actualiza mediante la ecuación de Bellman, y selecciona acciones siguiendo
una política epsilon-greedy con decaimiento progresivo de epsilon.
"""

import numpy as np
import random


class QLearning:
    """Agente Q-learning para detección de anomalías.

    Atributos:
        q_table (np.ndarray): Tabla Q de forma (n_estados, n_acciones).
        alpha (float): Tasa de aprendizaje (learning rate).
        gamma (float): Factor de descuento (discount factor).
        epsilon (float): Tasa de exploración actual.
        epsilon_min (float): Tasa de exploración mínima.
        epsilon_decay (float): Factor de decaimiento de epsilon por episodio.
        n_estados (int): Número de estados discretos (625 con 5 niveles por sensor).
        n_acciones (int): Número de acciones posibles.
        historial_recompensas (list): Recompensa acumulada por episodio.
    """

    def __init__(self, n_estados=625, n_acciones=2, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """Inicializa el agente con la tabla Q en ceros y los hiperparámetros.

        Args:
            n_estados: Número de estados discretos (por defecto 625).
            n_acciones: Número de acciones (0=normal, 1=anómalo).
            alpha: Tasa de aprendizaje.
            gamma: Factor de descuento.
            epsilon: Tasa de exploración inicial.
            epsilon_min: Tasa de exploración mínima.
            epsilon_decay: Factor de decaimiento por episodio.
        """
        self.q_table = np.zeros((n_estados, n_acciones))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_estados = n_estados
        self.n_acciones = n_acciones
        self.historial_recompensas = []

    def discretizar_estado(self, temperatura, vibracion, presion, rpm):
        """Convierte las lecturas de los 4 sensores en un índice de estado.

        Cada sensor se discretiza en 5 niveles (0-4) usando truncamiento.
        El estado final se codifica en base 5: t*125 + v*25 + p*5 + r,
        generando hasta 625 estados únicos (0-624).

        Args:
            temperatura: Lectura normalizada de temperatura (0.0-1.0).
            vibracion: Lectura normalizada de vibración (0.0-1.0).
            presion: Lectura normalizada de presión (0.0-1.0).
            rpm: Lectura normalizada de RPM (0.0-1.0).

        Returns:
            int: Índice del estado discretizado (0-624).
        """
        t = min(int(temperatura * 5), 4)
        v = min(int(vibracion * 5), 4)
        p = min(int(presion * 5), 4)
        r = min(int(rpm * 5), 4)
        return t * 125 + v * 25 + p * 5 + r

    def elegir_accion(self, estado):
        """Selecciona una acción usando la política epsilon-greedy.

        Con probabilidad epsilon explora (acción aleatoria); con probabilidad
        (1 - epsilon) explota (acción con mayor valor Q en el estado actual).

        Args:
            estado: Índice del estado actual.

        Returns:
            int: Acción seleccionada (0 = normal, 1 = anómalo).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_acciones - 1)
        return int(np.argmax(self.q_table[estado]))

    def actualizar(self, estado, accion, recompensa, siguiente_estado):
        """Actualiza la tabla Q con la ecuación de actualización de Bellman.

        Formula:
            Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            estado: Índice del estado actual s.
            accion: Acción tomada a.
            recompensa: Recompensa recibida r.
            siguiente_estado: Índice del siguiente estado s'.
        """
        mejor_siguiente = np.max(self.q_table[siguiente_estado])
        objetivo_td = recompensa + self.gamma * mejor_siguiente
        error_td = objetivo_td - self.q_table[estado][accion]
        self.q_table[estado][accion] += self.alpha * error_td

    def decaer_epsilon(self):
        """Reduce epsilon multiplicándolo por epsilon_decay.

        No permite que epsilon sea menor que epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def obtener_estado_para_json(self):
        """Retorna el estado del agente serializable para la API.

        Returns:
            dict: Contiene la tabla Q como lista, epsilon actual,
                  e historial de recompensas.
        """
        return {
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
            "historial_recompensas": self.historial_recompensas
        }

    def reiniciar(self):
        """Reinicia la tabla Q a ceros y restaura los hiperparámetros.

        El epsilon vuelve a 1.0 y el historial de recompensas se limpia.
        """
        self.q_table.fill(0)
        self.epsilon = 1.0
        self.historial_recompensas = []
