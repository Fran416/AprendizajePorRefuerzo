# Aprendizaje por Refuerzo - Detección de Anomalias

Proyecto didactico que demuestra como un agente de Q-learning aprende a detectar anomalias en sensores de una maquina industrial mediante prueba y error, sin necesidad de datos etiquetados previamente.

### Estudiante
- Francisco Lizama | fran416

---

## Tecnologias utilizadas

- **Python 3.12** — Lenguaje principal
- **Flask 3.0** — Servidor web
- **Gunicorn 21.2** — Servidor WSGI para produccion
- **NumPy 1.24** — Tabla Q y operaciones matematicas
- **Pandas 2.0** — Analisis de datos historicos
- **Plotly 5.18** — Graficos interactivos
- **HTMX 1.9** — Actualizaciones sin recargar la pagina
- **Docker** — Contenedorizacion

## Dependencias

```
flask>=3.0
gunicorn>=21.2
numpy>=1.24
pandas>=2.0
plotly>=5.18
```

## Arquitectura

La aplicacion se divide en 3 modulos que funcionan juntos:

```
rl-app/
├── app.py              ← Servidor Flask (rutas y API)
├── agente/
│   └── aprendizaje.py  ← Q-learning (tabla Q, ε-greedy, Bellman)
├── simulacion/
│   ├── entorno.py      ← Maquina industrial con 4 sensores
│   └── motor.py        ← Orquestador de entrenamiento
└── web/
    ├── static/         ← CSS, JavaScript
    └── templates/      ← Paginas HTML con Jinja2
```

El backend RL y el servidor web viven en el mismo proceso. No hay base de datos ni servicios externos.

## Como ejecutar

### Con Docker (recomendado)

```bash
# Construir e iniciar
docker compose up -d
```

Abrir en el navegador: `http://localhost`

### Con Python local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python app.py
```

Abrir en el navegador: `http://localhost:5000`

### Para detener

```bash
# Si esta corriendo con Docker
docker compose down

# Si esta corriendo con Python local
Ctrl + C
```

## Paginas del proyecto

| Ruta | Pagina |
|------|--------|
| `/` | Inicio - Portada del proyecto |
| `/conceptos` | Conceptos - Diccionario visual de RL |
| `/simulacion` | Simulacion - Entrena al agente paso a paso |
| `/visualizaciones` | Visualizaciones - Graficos interactivos |
| `/referencias` | Referencias - Bibliografia y conclusiones |
