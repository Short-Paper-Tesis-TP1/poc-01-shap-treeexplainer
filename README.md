# PoC 01: SHAP TreeExplainer sobre LightGBM

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20). Autores: Alexander Castillo y Bárbara Quezada.

## Objetivo único
Comprobar que `shap.TreeExplainer` descompone la probabilidad que devuelve un modelo LightGBM en la contribución de cada variable, y que esa descomposición reconstruye exactamente la predicción (aditividad).

No valida el desempeño del modelo ni usa datos reales.

## Datos
Sintéticos: 800 registros (640 de entrenamiento) con las 18 variables del diseño de septiembre de 2026, que fue anterior al diccionario de 9 variables. Los datos solo dan a la herramienta entradas con la forma correcta. El comportamiento de SHAP no depende de cuántas variables tenga el modelo ni de cuáles sean.

## Cómo correrla
```
pip install -r requirements.txt
python poc_shap.py
```

## Resultado
| Medida | Valor |
|---|---|
| Probabilidad del modelo para el caso explicado | 0.9285 |
| Suma en modo por defecto (log-odds) | 2.5638 (sigmoide = 0.9285) |
| Valor base con `model_output="probability"` y background de 200 filas | 0.5784 |
| Suma de contribuciones | +0.3501 |
| Reconstrucción | 0.9285 |
| Diferencia con la probabilidad del modelo | 0.000000 |
| Aditividad verificada | Sí |

## Hallazgos
1. Por defecto, SHAP entrega contribuciones en log-odds y no en probabilidad.
2. Para explicar en escala de probabilidad se requiere un dataset de referencia (background). Sin él, esa escala no está disponible.
3. TreeExplainer recibe el modelo y los datos del caso, no la probabilidad calculada.

## Decisión de diseño
La predicción y la explicación se ejecutan en el mismo proceso, con el modelo y el dataset de referencia cargados en memoria (Motor de Inteligencia en un solo contenedor).

## Archivos
- `poc_shap.py`: script.
- `resultado_poc.json`: salida estructurada.
- `salida_consola.txt`: salida completa de la consola.

## Corrida de referencia
6 de octubre de 2026, Linux, Python 3.12.3, con las versiones de `requirements.txt`. Los resultados son idénticos a la corrida original de septiembre de 2026 (semilla 42).
