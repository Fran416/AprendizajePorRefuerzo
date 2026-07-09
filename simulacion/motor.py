"""
Módulo del motor de simulación que orquesta la interacción entre el agente
Q-learning y el entorno de la máquina industrial.

Permite tanto el entrenamiento por lotes (múltiples episodios) como la
ejecución paso a paso para la simulación interactiva desde el frontend.
Mantiene métricas de rendimiento: recompensa acumulada, precisión global
y matriz de confusión acumulada.
"""

from agente.aprendizaje import QLearning
from simulacion.entorno import EntornoMaquina


class MotorSimulacion:
    """Orquesta el entrenamiento y la simulación del agente RL.

    El motor conecta al agente con el entorno, ejecuta episodios de
    entrenamiento, y mantiene registros de métricas acumuladas.

    Atributos:
        entorno (EntornoMaquina): Entorno de la máquina industrial.
        agente (QLearning): Agente de aprendizaje por refuerzo.
        historial_recompensas (list): Recompensa acumulada por episodio.
        episodio_actual (int): Número de episodios completados.
        aciertos (int): Total de clasificaciones correctas acumuladas.
        total_acciones (int): Total de acciones ejecutadas acumuladas.
        tp (int): Verdaderos positivos acumulados.
        tn (int): Verdaderos negativos acumulados.
        fp (int): Falsos positivos acumulados.
        fn (int): Falsos negativos acumulados.
        estado_actual (int): Índice del estado discretizado actual.
        recompensa_ep_actual (int): Recompensa acumulada del episodio
                                    en curso (modo paso a paso).
    """

    def __init__(self, entorno=None, agente=None):
        """Inicializa el motor con entorno y agente.

        Si no se proporcionan, crea instancias por defecto.

        Args:
            entorno: Instancia de EntornoMaquina (opcional).
            agente: Instancia de QLearning (opcional).
        """
        self.entorno = entorno if entorno else EntornoMaquina()
        self.agente = agente if agente else QLearning()
        self.historial_recompensas = []
        self.episodio_actual = 0
        self.aciertos = 0
        self.total_acciones = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        # Sincronizar estado_actual con el estado inicial del entorno
        self.estado_actual = self.agente.discretizar_estado(
            self.entorno.temperatura,
            self.entorno.vibracion,
            self.entorno.presion,
            self.entorno.rpm
        )
        self.recompensa_ep_actual = 0

    def entrenar(self, episodios):
        """Ejecuta el entrenamiento del agente durante N episodios.

        Cada episodio comienza con un reset del entorno. El agente elige
        acciones, el entorno responde con recompensas, y el agente actualiza
        su tabla Q hasta que el episodio termina. Al final de cada episodio
        se aplica el decaimiento de epsilon.

        Args:
            episodios: Número de episodios a ejecutar.

        Returns:
            list: Recompensa acumulada por cada episodio.
        """
        for _ in range(episodios):
            estado = self._iniciar_episodio()
            recompensa_total = 0
            done = False

            while not done:
                accion = self.agente.elegir_accion(estado)
                (temp, vib, pres, rpm, etiqueta_real,
                 recompensa, done) = self.entorno.step(accion)

                siguiente_estado = self.agente.discretizar_estado(
                    temp, vib, pres, rpm
                )

                self.agente.actualizar(estado, accion, recompensa,
                                       siguiente_estado)
                self._registrar_resultado(accion, etiqueta_real)

                estado = siguiente_estado
                recompensa_total += recompensa

            self._finalizar_episodio(recompensa_total)

        # Sincronizar estado_actual y reiniciar el entorno para una
        # transición limpia al modo paso a paso
        self.estado_actual = self._iniciar_episodio()

        return self.historial_recompensas

    def paso(self):
        """Ejecuta un único paso de simulación.

        Toma el estado actual, elige una acción, la ejecuta en el entorno,
        actualiza la tabla Q y retorna el resultado completo. Si el episodio
        termina (done), aplica decaimiento de epsilon y reinicia el entorno.

        Returns:
            dict: Lecturas de sensores, acción tomada, recompensa obtenida,
                  si terminó el episodio, y métricas actualizadas.
        """
        accion = self.agente.elegir_accion(self.estado_actual)

        # step() ya actualiza los atributos del entorno internamente;
        # capturamos solo los valores que necesitamos
        (temp, vib, pres, rpm,
         etiqueta_real, recompensa, done) = self.entorno.step(accion)

        nuevo_estado = self.agente.discretizar_estado(
            temp, vib, pres, rpm
        )

        self.agente.actualizar(self.estado_actual, accion, recompensa,
                               nuevo_estado)
        self._registrar_resultado(accion, etiqueta_real)
        self.recompensa_ep_actual += recompensa

        self.estado_actual = nuevo_estado

        if done:
            self._finalizar_episodio(self.recompensa_ep_actual)
            self.recompensa_ep_actual = 0
            (temp, vib, pres, rpm,
             _) = self.entorno.reset()
            self.estado_actual = self.agente.discretizar_estado(
                temp, vib, pres, rpm
            )

        return {
            "temperatura": round(self.entorno.temperatura, 4),
            "vibracion": round(self.entorno.vibracion, 4),
            "presion": round(self.entorno.presion, 4),
            "rpm": round(self.entorno.rpm, 4),
            "etiqueta_real": etiqueta_real,
            "accion": accion,
            "accion_texto": "Anómalo" if accion == 1 else "Normal",
            "recompensa": recompensa,
            "done": done,
            "episodio": self.episodio_actual,
            "epsilon": round(self.agente.epsilon, 4),
            "precision": self._calcular_precision(),
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "aciertos": self.aciertos,
            "total_acciones": self.total_acciones
        }

    def _iniciar_episodio(self):
        """Reinicia el entorno y discretiza el primer estado del episodio.

        Returns:
            int: Índice del estado discretizado inicial.
        """
        (temp, vib, pres, rpm, _) = self.entorno.reset()
        return self.agente.discretizar_estado(temp, vib, pres, rpm)

    def _finalizar_episodio(self, recompensa_total):
        """Completa un episodio: aplica decaimiento y registra métricas.

        Args:
            recompensa_total: Recompensa acumulada del episodio que termina.
        """
        self.agente.decaer_epsilon()
        self.agente.historial_recompensas.append(recompensa_total)
        self.historial_recompensas.append(recompensa_total)
        self.episodio_actual += 1

    def _registrar_resultado(self, accion, etiqueta_real):
        """Actualiza los contadores de aciertos y matriz de confusión.

        Estos contadores son acumulativos a lo largo de toda la simulación
        para calcular la precisión global del agente.

        Args:
            accion: Acción tomada por el agente (0 o 1).
            etiqueta_real: Etiqueta real del estado (0 o 1).
        """
        self.total_acciones += 1
        if accion == 1 and etiqueta_real == 1:
            self.tp += 1
            self.aciertos += 1
        elif accion == 0 and etiqueta_real == 0:
            self.tn += 1
            self.aciertos += 1
        elif accion == 1 and etiqueta_real == 0:
            self.fp += 1
        else:
            self.fn += 1

    def _calcular_precision(self):
        """Calcula la precisión global del agente.

        Returns:
            float: Porcentaje de aciertos (0-100), o 0 si no hay acciones.
        """
        if self.total_acciones == 0:
            return 0.0
        return round((self.aciertos / self.total_acciones) * 100, 2)

    def obtener_metricas(self):
        """Retorna las métricas actuales de la simulación.

        Returns:
            dict: Episodio actual, precisión global, epsilon, matriz de
                  confusión acumulada y últimas 20 recompensas.
        """
        return {
            "episodio": self.episodio_actual,
            "precision": self._calcular_precision(),
            "epsilon": round(self.agente.epsilon, 4),
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "aciertos": self.aciertos,
            "total_acciones": self.total_acciones,
            "ultimas_recompensas": self.historial_recompensas[-20:]
        }

    def obtener_historial_completo(self):
        """Retorna la recompensa acumulada por cada episodio.

        Returns:
            list: Recompensa por episodio.
        """
        return self.historial_recompensas

    def reiniciar(self):
        """Reinicia el motor, el agente y el entorno al estado inicial.

        Limpia todas las métricas y contadores. El agente vuelve a tener
        la tabla Q en ceros y epsilon = 1.0. El entorno se reinicia.
        """
        self.agente.reiniciar()
        self.entorno.reiniciar()
        self.historial_recompensas = []
        self.episodio_actual = 0
        self.aciertos = 0
        self.total_acciones = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.recompensa_ep_actual = 0
        # Sincronizar estado_actual con el entorno reiniciado
        self.estado_actual = self.agente.discretizar_estado(
            self.entorno.temperatura,
            self.entorno.vibracion,
            self.entorno.presion,
            self.entorno.rpm
        )
