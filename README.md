# Aprendizaje por Refuerzo - Detección de Anomalías

Proyecto didáctico que demuestra cómo un agente de Q-learning aprende a detectar anomalías en sensores de una máquina industrial mediante prueba y error, sin necesidad de datos etiquetados previamente.

Nota adicional: La documentación del proyecto es estilo `docstring`, embebido en los archivos lógicos

### Estudiante
- Francisco Lizama | fran416

---

## Tecnologías utilizadas

- **Python 3.12** — Lenguaje principal
- **Flask 3.0** — Servidor web
- **Gunicorn 21.2** — Servidor WSGI para producción
- **NumPy 1.24** — Tabla Q y operaciones matemáticas
- **Pandas 2.0** — Análisis de datos históricos
- **Plotly 5.18** — Gráficos interactivos
- **HTMX 1.9** — Actualizaciones sin recargar la página
- **Docker** — Contenedorización

## Dependencias

```
flask>=3.0
gunicorn>=21.2
numpy>=1.24
pandas>=2.0
plotly>=5.18
```

## Arquitectura

La aplicación se divide en 3 módulos que funcionan juntos:

```
rl-app/
├── app.py              ← Servidor Flask (rutas y API)
├── agente/
│   └── aprendizaje.py  ← Q-learning (tabla Q, ε-greedy, Bellman)
├── simulacion/
│   ├── entorno.py      ← Máquina industrial con 4 sensores
│   └── motor.py        ← Orquestador de entrenamiento
└── web/
    ├── static/         ← CSS, JavaScript
    └── templates/      ← Páginas HTML con Jinja2
```

El backend RL y el servidor web viven en el mismo proceso. No hay base de datos ni servicios externos.

## Cómo ejecutar

El proyecto fue desarrollado y probado en Linux, específicamente `Ubuntu 24.04.4 LTS` por lo que se recomienda encarecidamente su ejecución con Docker para evitar posibles incompatibilidades que puedan surgir en otros sistemas operativos.

### Paso Esencial

Clonar el repositorio y entrar al proyecto 
```
git clone https://github.com/Fran416/AprendizajePorRefuerzo.git

cd AprendizajePorRefuerzo
```

### Con Docker (recomendado)

#### Requisitos técnicos:
- Docker Engine version 29.x o superior
- Docker Compose (V2, integrado de forma nativa en las versiones modernas de Docker)
- Opcional: Docker Desktop o Podman Desktop, esto para inicializar el contenedor mediante una interfaz gráfica de usuario (GUI)

```
# Construir e iniciar
docker compose up -d --build
```

Abrir en el navegador: `http://localhost`

### Con Python local

#### Requisitos técnicos
- Python 3.8 o superior

```
# Crear entorno python y activarlo
# Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python app.py
```

Abrir en el navegador: `http://localhost:5000`

### Para detener

```
# Si está corriendo con Docker
docker compose down

# Si está corriendo con Python local
Ctrl + C
```

## Páginas del proyecto

| Ruta | Página |
|------|--------|
| `/` | Inicio - Portada del proyecto |
| `/conceptos` | Conceptos - Diccionario visual de RL |
| `/simulacion` | Simulación - Entrena al agente paso a paso |
| `/visualizaciones` | Visualizaciones - Gráficos interactivos |
| `/referencias` | Referencias - Bibliografía y conclusiones |
