# ElectricForecast (ELFO) ⚡

Sistema de predicción de demanda eléctrica desarrollado con **Streamlit**, **Prophet** y **XGBoost**.

![Logo](Logo.png)

## 📋 Descripción
ElectricForecast es una herramienta para la ingesta, análisis, entrenamiento y proyección de demanda eléctrica. Permite cargar datos históricos, entrenar modelos de series temporales (Prophet y XGBoost) y generar proyecciones futuras con métricas de evaluación detalladas.

## 🛠️ Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/jlvd/ElectricForecast.git
   cd ElectricForecast
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Ejecución

Para iniciar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador (usualmente en `http://localhost:8501`).

## 📖 Guía de Uso

El flujo de trabajo se divide en 4 pasos secuenciales:

### 1. Ingesta de Datos 📂
- Carga archivos Excel con el formato `rgd_DDMMYY.xlsx`.
- **Formato requerido**:
  - Columna Hora: `A12:A36`
  - Columna Demanda: `L12:L36`
- El sistema procesará, limpiará y almacenará los datos en una base de datos local (`DuckDB`).

### 2. Entrenamiento ⚙️
- Configura el horizonte de predicción (días hacia el futuro).
- Selecciona la fecha de corte para separar conjuntos de entrenamiento y prueba.
- **Modelos disponibles**:
  - **Prophet**: Ideal para capturar estacionalidad y feriados.
  - **XGBoost**: Potente para capturar patrones no lineales. (Opción de optimización Bayesiana disponible).

### 3. Evaluación 📊
- Compara los resultados de los modelos entrenados contra la data de prueba (Backtesting).
- Revisa métricas como RMSE, MAE y MAPE para decidir el mejor modelo.
- Visualiza gráficamente la comparación entre Predicción vs Realidad.

### 4. Proyección Futura 🔮
- Selecciona el modelo ganador.
- Define cuántos días quieres proyectar a futuro.
- Genera y descarga las predicciones en formato CSV.

## 👥 Créditos
Desarrollado como parte del Trabajo de Fin de Máster (TFM).
UIDE - Maestría en Inteligencia Artificial.
