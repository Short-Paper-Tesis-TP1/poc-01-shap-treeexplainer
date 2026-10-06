"""
PoC-01 | Objetivo unico: comprobar que shap.TreeExplainer descompone
la probabilidad que devuelve un Booster de LightGBM en la contribucion
de cada una de las 18 variables del perfil.

NO valida el modelo. NO valida DiCE. NO usa datos reales.
Los datos son SINTETICOS: sirven solo para alimentar la herramienta.
"""

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------
# PASO 1 - INPUT: las 18 variables (dataset sintetico, 800 filas)
# ---------------------------------------------------------------
VARIABLES = [
    "ingreso_mensual_promedio",
    "ingreso_mes_peor",
    "coef_variacion_ingreso",
    "gasto_mensual_promedio",
    "ratio_cuota_ingreso_minimo",
    "num_acreedores",
    "deuda_total",
    "cuota_mensual_total",
    "antiguedad_negocio_meses",
    "num_empleados",
    "ahorro_disponible",
    "dias_atraso_max_12m",
    "num_creditos_ultimos_12m",
    "tiene_cuenta_bancaria",
    "usa_credito_informal",
    "carga_familiar",
    "nivel_educativo",
    "formalidad_negocio",
]

N = 800
ingreso = np.random.gamma(4, 400, N) + 500
peor = ingreso * np.random.uniform(0.25, 0.95, N)
cv = (ingreso - peor) / ingreso
cuota = ingreso * np.random.uniform(0.05, 0.75, N)

X = pd.DataFrame({
    "ingreso_mensual_promedio": ingreso,
    "ingreso_mes_peor": peor,
    "coef_variacion_ingreso": cv,
    "gasto_mensual_promedio": ingreso * np.random.uniform(0.3, 0.9, N),
    "ratio_cuota_ingreso_minimo": cuota / peor,
    "num_acreedores": np.random.poisson(2, N),
    "deuda_total": cuota * np.random.uniform(6, 30, N),
    "cuota_mensual_total": cuota,
    "antiguedad_negocio_meses": np.random.randint(1, 180, N),
    "num_empleados": np.random.poisson(1.5, N),
    "ahorro_disponible": np.random.gamma(2, 300, N),
    "dias_atraso_max_12m": np.random.choice([0, 0, 0, 5, 15, 30, 60], N),
    "num_creditos_ultimos_12m": np.random.poisson(1.8, N),
    "tiene_cuenta_bancaria": np.random.binomial(1, 0.6, N),
    "usa_credito_informal": np.random.binomial(1, 0.45, N),
    "carga_familiar": np.random.poisson(2, N),
    "nivel_educativo": np.random.randint(1, 5, N),
    "formalidad_negocio": np.random.binomial(1, 0.35, N),
})[VARIABLES]

# Etiqueta proxy SINTETICA (solo para que el modelo tenga algo que aprender)
riesgo = (
    2.5 * X["ratio_cuota_ingreso_minimo"]
    + 1.8 * X["coef_variacion_ingreso"]
    + 0.35 * X["num_acreedores"]
    + 0.02 * X["dias_atraso_max_12m"]
    + 0.7 * X["usa_credito_informal"]
    - 0.006 * X["antiguedad_negocio_meses"]
    - 0.0009 * X["ahorro_disponible"]
)
y = (riesgo + np.random.normal(0, 0.4, N) > riesgo.median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

print("=" * 70)
print("PASO 1 | INPUTS")
print(f"  Filas de entrenamiento : {len(X_train)}")
print(f"  Variables de entrada   : {X_train.shape[1]}")
print(f"  Balance de la etiqueta : {y_train.mean():.2%} en riesgo")
print("=" * 70)

# ---------------------------------------------------------------
# PASO 2 - IDA: LightGBM devuelve UN numero (la probabilidad)
# ---------------------------------------------------------------
booster = lgb.train(
    params={
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": RANDOM_SEED,
    },
    train_set=lgb.Dataset(X_train, y_train),
    num_boost_round=150,
)

caso = X_test.iloc[[0]]                      # DataFrame de 1 fila x 18 columnas
probabilidad = float(booster.predict(caso)[0])

print("\nPASO 2 | IDA (LightGBM Booster)")
print(f"  Tipo de objeto        : {type(booster).__name__}")
print(f"  Output de booster.predict -> {probabilidad:.4f}")
print("  => Es UN escalar. No dice por que.")

# ---------------------------------------------------------------
# PASO 3 - VUELTA (A): TreeExplainer en espacio log-odds (por defecto)
# ---------------------------------------------------------------
explainer_logodds = shap.TreeExplainer(booster)
sv = explainer_logodds.shap_values(caso)
if isinstance(sv, list):                      # compatibilidad entre versiones
    sv = sv[1]
sv = np.array(sv).reshape(-1)

base = explainer_logodds.expected_value
if isinstance(base, (list, np.ndarray)):
    base = float(np.ravel(base)[-1])

suma_logodds = base + sv.sum()
prob_desde_logodds = 1 / (1 + np.exp(-suma_logodds))

print("\nPASO 3 | VUELTA-A (TreeExplainer, modo por defecto)")
print(f"  expected_value (base)      : {base:.4f}")
print(f"  suma de los 18 shap_values : {sv.sum():.4f}")
print(f"  base + suma                : {suma_logodds:.4f}   <-- NO es 0.xx")
print(f"  sigmoide(base + suma)      : {prob_desde_logodds:.4f}")
print(f"  probabilidad del Booster   : {probabilidad:.4f}")
print(f"  HALLAZGO: los shap_values estan en LOG-ODDS, no en probabilidad.")

# ---------------------------------------------------------------
# PASO 4 - VUELTA (B): TreeExplainer en espacio PROBABILIDAD
#          Requiere el dataset de background en RAM.
# ---------------------------------------------------------------
background = shap.sample(X_train, 200, random_state=RANDOM_SEED)

explainer_prob = shap.TreeExplainer(
    booster,
    data=background,
    model_output="probability",
    feature_perturbation="interventional",
)
sv_p = np.array(explainer_prob.shap_values(caso)).reshape(-1)
base_p = explainer_prob.expected_value
if isinstance(base_p, (list, np.ndarray)):
    base_p = float(np.ravel(base_p)[-1])

reconstruida = base_p + sv_p.sum()

print("\nPASO 4 | VUELTA-B (model_output='probability' + background)")
print(f"  Filas del background        : {len(background)}")
print(f"  base_value (prob. promedio) : {base_p:.4f}")
print(f"  suma de contribuciones      : {sv_p.sum():+.4f}")
print(f"  base + suma                 : {reconstruida:.4f}")
print(f"  probabilidad del Booster    : {probabilidad:.4f}")
print(f"  Diferencia                  : {abs(reconstruida - probabilidad):.6f}")
ok = abs(reconstruida - probabilidad) < 0.01
print(f"  ADITIVIDAD VERIFICADA       : {'SI' if ok else 'NO'}")

# ---------------------------------------------------------------
# PASO 5 - OUTPUT legible: las 18 contribuciones ordenadas
# ---------------------------------------------------------------
tabla = pd.DataFrame({
    "variable": VARIABLES,
    "valor_del_caso": caso.iloc[0].values,
    "contribucion": sv_p,
}).sort_values("contribucion", key=abs, ascending=False)

print("\nPASO 5 | DESCOMPOSICION DEL RESULTADO")
print(f"  Probabilidad de sobreendeudamiento: {probabilidad:.2%}")
print(f"  Promedio poblacional (base)      : {base_p:.2%}")
print("-" * 70)
for _, r in tabla.iterrows():
    signo = "sube" if r["contribucion"] > 0 else "baja"
    print(f"  {r['variable']:<30} {r['valor_del_caso']:>10.2f}   "
          f"{r['contribucion']:+.4f}  ({signo})")
print("-" * 70)

# ---------------------------------------------------------------
# PASO 6 - CONTRATO JSON entre capa 2 y capa 3
# ---------------------------------------------------------------
salida = {
    "poc_id": "PoC-01-TreeExplainer",
    "capa_2_deteccion": {
        "componente": "lightgbm.Booster.predict",
        "output": {"probabilidad_sobreendeudamiento": round(probabilidad, 4)},
    },
    "capa_3_interpretabilidad": {
        "componente": "shap.TreeExplainer",
        "model_output": "probability",
        "background_n": len(background),
        "base_value": round(base_p, 4),
        "suma_contribuciones": round(float(sv_p.sum()), 4),
        "reconstruccion": round(float(reconstruida), 4),
        "aditividad_ok": bool(ok),
        "contribuciones": [
            {
                "variable": r["variable"],
                "valor": round(float(r["valor_del_caso"]), 4),
                "contribucion": round(float(r["contribucion"]), 4),
            }
            for _, r in tabla.iterrows()
        ],
    },
}

with open("resultado_poc.json", "w", encoding="utf-8") as f:
    json.dump(salida, f, indent=2, ensure_ascii=False)

print("\nPASO 6 | Archivo generado: resultado_poc.json")
print("=" * 70)
