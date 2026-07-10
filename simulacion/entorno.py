"""
Módulo del entorno de simulación de una máquina industrial.

Define el entorno con 4 sensores (temperatura, vibración, presión, RPM)
que genera lecturas sintéticas mediante distribuciones de probabilidad.
Los estados normales se generan dentro de rangos seguros; los estados
anómalos desvían uno o más sensores fuera de esos rangos.
El agente RL recibe recompensas según la matriz de confusión.
"""

import random
import numpy as np


# Constantes de recompensa según la matriz de confusión
RECOMPENSAS = {
    "tp": 10,   # Anomalía detectada correctamente
    "tn": 1,    # Normal clasificado correctamente
    "fp": -5,   # Falsa alarma (normal clasificado como anómalo)
    "fn": -20   # Anomalía no detectada
}

# Rangos seguros para cada sensor en operación normal (mínimo, máximo)
RANGOS_NORMALES = {
    "temperatura": (0.15, 0.65),
    "vibracion":   (0.10, 0.55),
    "presion":     (0.20, 0.60),
    "rpm":         (0.15, 0.70)
}

# Tipos de anomalía: cada una especifica qué sensor(es) desviar y factor
TIPOS_ANOMALIA = [
    {"nombre": "Sobrecalentamiento",       "sensores": ["temperatura"], "factor_alto": 1.3},
    {"nombre": "Vibración excesiva",       "sensores": ["vibracion"],   "factor_alto": 1.4},
    {"nombre": "Sobrepresión",             "sensores": ["presion"],     "factor_alto": 1.3},
    {"nombre": "Sobrevelocidad",           "sensores": ["rpm"],         "factor_alto": 1.3},
    {"nombre": "Múltiples fallos",         "sensores": ["temperatura", "presion", "rpm"], "factor_alto": 1.2},
    {"nombre": "Vibración + temperatura",  "sensores": ["vibracion", "temperatura"], "factor_alto": 1.25},
]


class EntornoMaquina:
    """Entorno simulado de una máquina industrial con 4 sensores.

    Genera estados normales muestreando cada sensor dentro de su rango
    seguro con distribución normal. Los estados anómalos se crean tomando
    un estado normal base y desviando sensores específicos fuera del rango
    seguro, simulando distintos tipos de fallo mecánico.

    Atributos:
        temperatura (float): Lectura actual de temperatura (0.0-1.0).
        vibracion (float): Lectura actual de vibración (0.0-1.0).
        presion (float): Lectura actual de presión (0.0-1.0).
        rpm (float): Lectura actual de RPM (0.0-1.0).
        etiqueta_real (int): 0 si el estado es normal, 1 si es anómalo.
        tipo_anomalia (str): Nombre del tipo de anomalía actual (si aplica).
        pasos (int): Contador de pasos dentro del episodio actual.
        max_pasos (int): Máximo de pasos por episodio.
    """

    def __init__(self, max_pasos=10):
        """Inicializa el entorno en estado neutro.

        Args:
            max_pasos: Máximo de pasos por episodio (por defecto 10).
        """
        self.temperatura = 0.0
        self.vibracion = 0.0
        self.presion = 0.0
        self.rpm = 0.0
        self.etiqueta_real = 0
        self.tipo_anomalia = None
        self.pasos = 0
        self.max_pasos = max_pasos

    def reset(self):
        """Reinicia el contador de pasos y genera un nuevo estado aleatorio.

        El estado tiene 75% de probabilidad de ser normal y 25% anómalo.

        Returns:
            tuple: (temperatura, vibracion, presion, rpm, etiqueta_real)
        """
        self.pasos = 0
        self._generar_estado()
        return (self.temperatura, self.vibracion, self.presion,
                self.rpm, self.etiqueta_real)

    def reiniciar(self):
        """Reinicia el entorno completamente al estado de fábrica.

        Pone todos los sensores a cero, limpia etiquetas y contadores.
        """
        self.temperatura = 0.0
        self.vibracion = 0.0
        self.presion = 0.0
        self.rpm = 0.0
        self.etiqueta_real = 0
        self.tipo_anomalia = None
        self.pasos = 0

    def _generar_estado(self):
        """Genera un estado muestreando desde distribuciones de probabilidad.

        Si el resultado es normal: cada sensor se muestrea de una normal
        centrada en la media de su rango seguro con std pequeña.

        Si es anómalo: se parte de un estado normal y se desvían sensores
        específicos multiplicando por un factor (>1) y sumando ruido.
        """
        if random.random() < 0.75:
            self._generar_normal()
        else:
            self._generar_anomalia()

    def _generar_normal(self):
        """Genera un estado con todos los sensores dentro de rangos seguros.

        Cada sensor se muestrea de una distribución normal cuya media
        es el punto medio del rango seguro, con desviación que mantiene
        el 99.7% de valores dentro del rango (regla 3-sigma).
        """
        self.etiqueta_real = 0
        self.tipo_anomalia = None

        for sensor in ["temperatura", "vibracion", "presion", "rpm"]:
            min_val, max_val = RANGOS_NORMALES[sensor]
            media = (min_val + max_val) / 2
            # Sigma tal que media ± 3*sigma ≈ cubra el rango
            sigma = (max_val - min_val) / 6
            valor = np.random.normal(media, sigma)
            setattr(self, sensor, np.clip(valor, 0.0, 1.0))

    def _generar_anomalia(self):
        """Genera un estado anómalo desviando sensores fuera del rango seguro.

        Elige un tipo de anomalía al azar y, partiendo de un estado normal
        base, multiplica los sensores afectados por el factor de desviación
        para llevarlos al extremo superior del rango [0,1]. La etiqueta real
        se establece a 1 DESPUÉS de generar la base normal para no ser
        sobrescrita por _generar_normal().
        """
        # Partir de un estado normal base (esto establece etiqueta_real=0)
        self._generar_normal()

        # Elegir tipo de anomalía aleatorio
        tipo = random.choice(TIPOS_ANOMALIA)
        self.tipo_anomalia = tipo["nombre"]

        # Desviar los sensores afectados por esta anomalía
        for sensor in tipo["sensores"]:
            valor_actual = getattr(self, sensor)
            # Desviar hacia arriba con factor + ruido
            valor_desviado = valor_actual * tipo["factor_alto"] + 0.1
            valor_desviado += np.random.normal(0, 0.05)
            setattr(self, sensor, np.clip(valor_desviado, 0.65, 1.0))

        # Establecer etiqueta DESPUÉS de _generar_normal() para que no se
        # sobrescriba
        self.etiqueta_real = 1

    def step(self, accion):
        """Ejecuta una acción del agente y devuelve la transición completa.

        Calcula la recompensa comparando la acción del agente con la
        etiqueta real del estado actual. Luego genera el siguiente estado.
        La etiqueta devuelta es la del estado que se evaluó, no la del
        nuevo estado generado.

        Args:
            accion: Acción del agente (0 = normal, 1 = anómalo).

        Returns:
            tuple: (temperatura, vibracion, presion, rpm, etiqueta_real,
                   recompensa, done)
                   donde etiqueta_real es la del estado evaluado.
        """
        self.pasos += 1

        # Guardar la etiqueta del estado actual ANTES de generar el siguiente
        etiqueta_evaluada = self.etiqueta_real

        # Calcular recompensa según matriz de confusión
        if accion == etiqueta_evaluada:
            if accion == 1:
                recompensa = RECOMPENSAS["tp"]
            else:
                recompensa = RECOMPENSAS["tn"]
        else:
            if accion == 1:
                recompensa = RECOMPENSAS["fp"]
            else:
                recompensa = RECOMPENSAS["fn"]

        # Generar el siguiente estado (o marcar fin de episodio)
        done = self.pasos >= self.max_pasos
        if not done:
            self._generar_estado()

        return (self.temperatura, self.vibracion, self.presion,
                self.rpm, etiqueta_evaluada, recompensa, done)

    def obtener_lecturas(self):
        """Retorna las lecturas actuales de los sensores en formato legible.

        Returns:
            dict: Lecturas actuales con etiqueta real y tipo de anomalía.
        """
        return {
            "temperatura": round(self.temperatura, 4),
            "vibracion": round(self.vibracion, 4),
            "presion": round(self.presion, 4),
            "rpm": round(self.rpm, 4),
            "etiqueta_real": self.etiqueta_real,
            "estado_texto": "Normal" if self.etiqueta_real == 0 else "Anómalo",
            "tipo_anomalia": self.tipo_anomalia or "—"
        }

    def obtener_lecturas_demo(self):
        """Genera lecturas de sensores aleatorias para visualización en vivo.

        Crea un estado demo independiente sin modificar el estado real del
        entorno en el que el agente está entrenando. Útil para el panel de
        monitoreo que se actualiza cada 2 segundos.

        Returns:
            dict: Lecturas de sensores con etiqueta real y estado texto,
                  simulando una máquina industrial en operación.
        """
        demo = EntornoMaquina(max_pasos=self.max_pasos)
        demo._generar_estado()
        return {
            "temperatura": round(demo.temperatura, 4),
            "vibracion": round(demo.vibracion, 4),
            "presion": round(demo.presion, 4),
            "rpm": round(demo.rpm, 4),
            "etiqueta_real": demo.etiqueta_real,
            "estado_texto": "Normal" if demo.etiqueta_real == 0 else "Anómalo",
            "tipo_anomalia": demo.tipo_anomalia or "—"
        }
