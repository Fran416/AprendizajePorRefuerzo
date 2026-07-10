"""
Servidor Flask que integra el backend de Aprendizaje por Refuerzo con la
interfaz web en un único proceso. Las rutas de página sirven el contenido
HTML completo y los endpoints API devuelven fragmentos HTML (partials) que
HTMX actualiza dinámicamente en el frontend sin recargar la página.

El backend RL (MotorSimulacion, QLearning, EntornoMaquina) se importa
directamente como módulo Python y se ejecuta en el mismo proceso que el
servidor web. No hay capa REST separada ni contenedores adicionales.
"""

from flask import Flask, render_template, request, make_response
from simulacion.motor import MotorSimulacion

app = Flask(
    __name__,
    static_folder="web/static",
    template_folder="web/templates"
)
app.secret_key = "rl-app-secret-key-2024"

# Instancia global del motor de simulación. El backend RL vive en el mismo
# proceso que el servidor web, accesible desde cualquier ruta o endpoint.
motor = MotorSimulacion()


# =============================================================================
# Rutas de páginas principales (GET)
# =============================================================================

@app.route("/")
def inicio():
    """Portada del proyecto con introducción al aprendizaje por refuerzo
    aplicado a detección de anomalías en máquinas industriales."""
    return render_template("inicio.html")


@app.route("/conceptos")
def conceptos():
    """Diccionario de conceptos fundamentales de aprendizaje por refuerzo:
    agente, entorno, estado, acción, recompensa, política, Q-learning y
    ecuación de Bellman."""
    return render_template("conceptos.html")

@app.route("/simulacion")
def simulacion():
    """Página de simulación interactiva con controles para entrenar al
    agente paso a paso o por lotes, mostrando métricas en tiempo real."""
    metricas = motor.obtener_metricas()
    lecturas = motor.entorno.obtener_lecturas_demo()
    return render_template("simulacion.html", **metricas, **lecturas)


@app.route("/visualizaciones")
def visualizaciones():
    """Página de gráficos interactivos Plotly.js: heatmap de la tabla Q,
    curva de aprendizaje, timeline de detecciones y gauge de exploración."""
    return render_template("visualizaciones.html")


@app.route("/referencias")
def referencias():
    """Referencias bibliográficas en formato APA, conclusiones del proyecto
    y posibles líneas de mejora futura."""
    return render_template("referencias.html")

# Endpoints API

@app.route("/api/ejecutar-paso", methods=["POST"])
def ejecutar_paso():
    """Ejecuta un paso de simulación y devuelve el partial con métricas.

    El agente selecciona una acción (normal o anómalo) mediante la política
    epsilon-greedy, el entorno responde con una recompensa, y el agente
    actualiza su tabla Q con la ecuación de Bellman. Si el episodio termina,
    se aplica el decaimiento de epsilon y se reinicia el entorno.

    Returns:
        Fragmento HTML (parciales/metricas.html) con las métricas
        actualizadas del episodio, recompensa, epsilon y precisión.
    """
    resultado = motor.paso()
    metricas_html = render_template("parciales/metricas.html", **resultado)
    sensores = motor.entorno.obtener_lecturas()
    sensores_html = render_template("parciales/panel_sensores.html", **sensores)
    response = make_response(metricas_html +
                             f'<div id="panel-sensores" hx-swap-oob="true">{sensores_html}</div>')
    response.headers['HX-Trigger'] = 'actualizar-precision'
    return response


@app.route("/api/entrenar", methods=["POST"])
def entrenar():
    """Entrena al agente durante N episodios y devuelve métricas actualizadas.

    Toma el número de episodios desde el formulario POST. Ejecuta el
    entrenamiento por lotes completo: en cada episodio el agente interactúa
    con el entorno hasta alcanzar el máximo de pasos, actualizando la tabla Q
    y acumulando recompensas. Al finalizar, aplica el decaimiento de epsilon.

    Returns:
        Fragmento HTML (parciales/metricas.html) con las métricas
        resultantes del entrenamiento.
    """
    episodios = int(request.form.get("episodios", 100))
    motor.entrenar(episodios)
    metricas = motor.obtener_metricas()
    metricas_html = render_template("parciales/metricas.html", **metricas)
    sensores = motor.entorno.obtener_lecturas()
    sensores_html = render_template("parciales/panel_sensores.html", **sensores)
    response = make_response(metricas_html +
                             f'<div id="panel-sensores" hx-swap-oob="true">{sensores_html}</div>')
    response.headers['HX-Trigger'] = 'actualizar-precision'
    return response


@app.route("/api/sensores")
def api_sensores():
    """Devuelve el partial con las lecturas actuales de los cuatro sensores.

    HTMX consulta este endpoint periódicamente (típicamente cada 2 segundos)
    para mantener el panel de monitoreo actualizado en tiempo real con las
    últimas lecturas de temperatura, vibración, presión y RPM.

    Returns:
        Fragmento HTML (parciales/panel_sensores.html) con las lecturas
        de sensores y la etiqueta real de estado (normal/anómalo).
    """
    lecturas = motor.entorno.obtener_lecturas_demo()
    return render_template("parciales/panel_sensores.html", **lecturas)


@app.route("/api/matriz-confusion")
def matriz_confusion():
    """Devuelve el partial con la matriz de confusión del agente.

    Muestra los valores TP, TN, FP, FN acumulados en un heatmap 2x2
    con los números visibles en cada celda. Se actualiza dinámicamente
    via HTMX para reflejar el estado actual del agente.

    Returns:
        Fragmento HTML (parciales/matriz_confusion.html) con el heatmap
        y la precisión global calculada.
    """
    metricas = motor.obtener_metricas()
    return render_template("parciales/matriz_confusion.html", **metricas)


@app.route("/api/reiniciar", methods=["POST"])
def reiniciar():
    """Reinicia el agente, el entorno y todas las métricas acumuladas.

    La tabla Q vuelve a ceros, epsilon a 1.0, el historial de recompensas
    y precisión se limpian, y el entorno se reinicia. Es equivalente a
    empezar desde cero.

    Returns:
        Fragmento HTML (parciales/metricas.html) con las métricas en
        estado inicial (todo en 0, epsilon en 1.0).
    """
    motor.reiniciar()
    metricas = motor.obtener_metricas()
    metricas_html = render_template("parciales/metricas.html", **metricas)
    sensores = motor.entorno.obtener_lecturas_demo()
    sensores_html = render_template("parciales/panel_sensores.html", **sensores)
    response = make_response(metricas_html +
                             f'<div id="panel-sensores" hx-swap-oob="true">{sensores_html}</div>')
    response.headers['HX-Trigger'] = 'reiniciado'
    return response


@app.route("/api/grafico-precision")
def grafico_precision():
    """Devuelve el partial con el gráfico de evolución de la precisión.

    Toma el historial de precisión registrado episodio por episodio
    y lo renderiza como un gráfico de líneas Plotly.js.

    Returns:
        Fragmento HTML (parciales/grafico_precision.html) con el gráfico
        de precisión vs episodios.
    """
    return render_template("parciales/grafico_precision.html",
                           historial_precision=motor.historial_precision[-100:],
                           total_acciones=motor.total_acciones)


@app.route("/api/mapa-calor-q")
def mapa_calor_q():
    """Devuelve el partial con los datos de la tabla Q para el heatmap.

    Incluye los valores Q actuales del agente, el valor de epsilon y el
    historial de recompensas, todo empaquetado en un fragmento HTML que
    el frontent usa para renderizar el heatmap con Plotly.js.

    Returns:
        Fragmento HTML (parciales/mapa_calor_q.html) con los datos de
        la tabla Q serializados para el gráfico interactivo.
    """
    datos_q = {
        "q_table": motor.agente.q_table.tolist(),
        "epsilon": motor.agente.epsilon,
        "historial": motor.agente.historial_recompensas[-50:]
    }
    return render_template("parciales/mapa_calor_q.html", **datos_q)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
