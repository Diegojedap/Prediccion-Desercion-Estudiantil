# Predicción de Deserción Estudiantil en Educación Superior

Sistema de machine learning para estimar la probabilidad de que un estudiante abandone sus
estudios, con el fin de habilitar estrategias de intervención temprana y retención.

El proyecto documenta el recorrido completo de un problema real de clasificación
desbalanceada: integración de múltiples fuentes, ingeniería de características, balanceo,
optimización bayesiana de hiperparámetros, ensamblado por *stacking*, calibración del umbral
de decisión e interpretabilidad con SHAP.

---

## Alcance de este repositorio

Este repositorio contiene **únicamente metodología y código**. No incluye datos, modelos
entrenados, identificadores de sistemas ni nombres de esquemas o tablas de ninguna
organización.

- Los `.xlsx` y `.pkl` están excluidos vía [`.gitignore`](.gitignore).
- Los parámetros de conexión se leen de variables de entorno (ver [`.env.example`](.env.example)).
- Los nombres de esquemas y tablas que aparecen en las consultas son **genéricos**,
  sustituidos por [`scripts/sanitizar_notebooks.py`](scripts/sanitizar_notebooks.py).
- Los notebooks se publican sin salidas ejecutadas: las tablas impresas contenían datos
  personales de estudiantes, sujetos a la Ley 1581 de 2012.

---

## El problema

Dado un estudiante en un periodo académico, predecir si desertará (`TARGET_DESERCION = 1`).

La clase positiva es minoritaria, lo que define toda la estrategia de modelado: el énfasis
está en el **recall** de los desertores, porque no detectar a un estudiante en riesgo cuesta
mucho más que emitir una alerta que resulte falsa. Por eso las decisiones clave del proyecto
—SMOTE, `scoring='f1'` en la búsqueda de hiperparámetros y la calibración explícita del
umbral— apuntan a la clase minoritaria y no a la exactitud global.

---

## Arquitectura del flujo

```
6 fuentes relacionales
        │
        ▼
  integración por llave compuesta (estudiante + periodo)
        │
        ▼
  ingeniería de características
        │
        ▼
  construcción del target binario
        │
        ▼
  preprocesamiento (imputación → escalado → codificación)
        │
        ▼
  SMOTE + entrenamiento (XGBoost / LightGBM / Stacking)
        │
        ▼
  búsqueda del umbral que maximiza F1
        │
        ▼
  pipeline serializado  →  scoring masivo
```

---

## Fuentes de datos

Seis tablas de un repositorio relacional sobre SQL Server, integradas con `LEFT JOIN` sobre
la llave compuesta `Identificacion` + `Periodo`:

| Dominio | Aporte |
|---|---|
| Histórico académico | Tabla base: trayectoria del estudiante y perfil sociodemográfico |
| Geolocalización | Ubicación de residencia |
| Becas y descuentos | Beneficios económicos vigentes |
| Aprobación de materias | Materias inscritas, aprobadas y porcentaje de aprobación |
| Cartera | Saldo financiero pendiente |
| Plataforma virtual | Estado de actividad y asistencia |

El acceso se realiza con `SQLAlchemy` + `pyodbc`. Host, puerto y nombre de la base se toman
del entorno.

---

## Variables del modelo

**Identificadores** (excluidos del entrenamiento): `IDENTIFICACION`, `PERIODO`, `AÑO`

**Numéricas:** `SEMESTRE_SINU`, `MATERIAS_INSCRITAS`, `MATERIAS_APROBADAS`,
`PORCENTAJE_APROBACION`, `TOTAL`, `FLAG_NO_APROBO_NADA`

**Categóricas:** `TIPO_SALTO`, `MODALIDAD`, `GENERO`, `ESTADO_PAGO`, `RANGO_EDAD`,
`RANGO_SALARIO`, `ESTA_TRABAJANDO`, `METODO_FINANCIAMIENTO`, `ZONA_RESIDENCIA`,
`REGIMEN_SISTEMA_SALUD`, `ASISTENCIA`

### Ingeniería de características

El caso más interesante del proyecto. Un porcentaje de aprobación igual a cero y un dato
faltante son situaciones opuestas —la primera es la señal de riesgo más fuerte disponible—
pero la imputación por mediana las colapsa en el mismo valor. La solución fue separarlas
antes de imputar:

```python
df['FLAG_NO_APROBO_NADA'] = np.where(df['PORCENTAJE_APROBACION'] == 0, 1, 0)
df['PORCENTAJE_APROBACION'] = df['PORCENTAJE_APROBACION'].replace(0, np.nan)
```

---

## Estructura del repositorio

```
.
├── README.md
├── .gitignore
├── .env.example
│
├── scripts/
│   ├── sanitizar_notebooks.py         Prepara los notebooks para publicación
│   └── mapeo_esquemas.example.json    Plantilla del mapeo de nombres
│
├── Prototipo 1.1.ipynb    Baseline: LogisticRegression, DecisionTree, RandomForest
├── Prototipo 1.2.ipynb    Incorpora SMOTE para balanceo de clases
├── Prototipo 1.3.ipynb    GridSearchCV sobre RandomForest
├── Prototipo 1.4.ipynb    Introduce XGBoost
├── Prototipo 1.5.ipynb    XGBoost con ajuste de hiperparámetros
├── Prototipo 1.7.ipynb    XGBoost + GridSearchCV + SMOTE + umbral óptimo + SHAP
├── Prototipo 1.6.ipynb    ★ Vigente: Stacking XGB+LGBM, optimización bayesiana
├── Untitled-2.ipynb       Consolidación intermedia de XGBoost
│
├── Despliegue 1.7.ipynb   Scoring masivo con el pipeline de XGBoost
└── Despliegue 1.6.ipynb   Scoring masivo con el pipeline de stacking
```

> **Nota sobre la numeración:** `Prototipo 1.6` es **posterior** a `1.7` por fecha de
> modificación. El modelo vigente es el de `1.6`.

---

## Evolución de los prototipos

> La columna de accuracy se incluye solo para trazar la historia del proyecto. **No es el
> criterio de selección** y no debe leerse como medida de calidad: en un problema
> desbalanceado un modelo que predijera "nadie deserta" tendría accuracy alta y sería
> inútil. Las métricas que gobiernan las decisiones son AUC y recall de la clase minoritaria.

| Versión | Enfoque | Accuracy *(no comparable)* |
|---|---|---|
| 1.1 | LogisticRegression + DecisionTree + RandomForest | 0.85 – 0.86 |
| 1.2 | Introduce SMOTE | 0.64 – 0.67 |
| 1.3 | GridSearchCV sobre RandomForest | 0.67 |
| 1.4 | Entra XGBoost | 0.73 – 0.78 |
| 1.5 | XGBoost + tuning | 0.72 – 0.80 |
| Untitled-2 | Consolidación XGBoost, primer `joblib.dump` | 0.74 – 0.80 |
| 1.7 | XGBoost + GridSearchCV + SMOTE + SHAP | 0.84 |
| 1.6 | Stacking XGB + LGBM + BayesSearchCV | 0.86 |

Las cifras de la última columna se midieron sobre particiones y poblaciones distintas, así que
**no son comparables entre sí**: la diferencia entre 0.84 y 0.86 no mide la ganancia del
stacking.

La caída en la versión 1.2 no es un retroceso, y entenderlo fue determinante para el resto del
proyecto: es el efecto esperado de SMOTE, que sacrifica exactitud global a cambio de recall
sobre la clase minoritaria. A partir de esa versión el criterio de selección pasó a ser F1 y
AUC.

---

## Modelo final

```python
StackingClassifier(
    estimators=[
        ('XGB',  XGBClassifier(n_estimators=385, max_depth=9,
                               learning_rate=0.0338, subsample=0.6,
                               colsample_bytree=0.6)),
        ('LGBM', LGBMClassifier(n_estimators=481, num_leaves=32,
                                learning_rate=0.0244, subsample=0.9122)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=StratifiedKFold(n_splits=3),
    passthrough=True,
)
```

Envuelto en un `ImbPipeline`: `preprocessor` → `SMOTE` → `stacking`. Usar el pipeline de
`imbalanced-learn` es deliberado: garantiza que SMOTE se aplique **solo dentro de cada fold**
de entrenamiento y nunca sobre el de validación, que es el error más común al combinar
sobremuestreo con validación cruzada.

Hiperparámetros hallados con `BayesSearchCV` (30 iteraciones, `scoring='f1'`).

### Desempeño sobre el conjunto de prueba

| Métrica | Clase minoritaria (deserta) |
|---|---|
| Precision | 0.37 |
| Recall | 0.57 |
| F1-score | 0.45 |

**AUC-ROC: 0.848** — Umbral de decisión: **0.6162**

### Calibración del umbral

El umbral por defecto de 0.5 es arbitrario y rara vez óptimo en problemas desbalanceados.
Aquí se busca sobre la curva precision-recall y se persiste como atributo del pipeline, de
modo que el despliegue lo consuma sin tener que recalcularlo ni codificarlo a mano:

```python
prec, rec, thr = precision_recall_curve(y_test, y_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-6)
pipeline_final.best_threshold = thr[np.argmax(f1)]
joblib.dump(pipeline_final, "modelo.pkl")
```

### Interpretabilidad

Se emplea **SHAP** (`TreeExplainer`) para explicar las predicciones a nivel global
—importancia agregada de variables— y local, con `waterfall_plot` por estudiante. En un
sistema que emite alertas sobre personas, poder justificar cada predicción individual no es
un extra: es condición para que un comité académico pueda actuar sobre ella.

---

## Requisitos

```
python >= 3.11

pandas          numpy           openpyxl
scikit-learn    xgboost         lightgbm
imbalanced-learn                scikit-optimize
shap            matplotlib      seaborn
sqlalchemy      pyodbc          joblib
unidecode       tqdm
```

```bash
pip install pandas numpy openpyxl scikit-learn xgboost lightgbm \
            imbalanced-learn scikit-optimize shap matplotlib seaborn \
            sqlalchemy pyodbc joblib unidecode tqdm
```

Requiere además el driver **ODBC Driver for SQL Server**. Configura la conexión copiando
`.env.example` a `.env`:

```bash
cp .env.example .env   # luego edita .env con los valores reales
```

> Los notebooks leen `DB_HOST`, `DB_PORT` y `DB_NAME` del entorno. Sin esas variables la
> celda de conexión falla de forma explícita, en lugar de exponer credenciales en el código.

---

## Uso

### Entrenamiento

Ejecutar [`Prototipo 1.6.ipynb`](Prototipo%201.6.ipynb) de principio a fin: extrae los datos,
construye el dataset, entrena el stacking y persiste el pipeline junto con su umbral.

### Scoring

```python
import joblib
import pandas as pd

pipeline = joblib.load('modelo.pkl')

# Las columnas deben coincidir en nombre y orden con las del entrenamiento
X_nuevos = df_nuevos[pipeline.feature_names_in_]

probabilidades = pipeline.predict_proba(X_nuevos)[:, 1]
predicciones = (probabilidades >= pipeline.best_threshold).astype(int)

resultado = pd.DataFrame({
    'IDENTIFICACION': df_nuevos['IDENTIFICACION'],
    'PERIODO': df_nuevos['PERIODO'],
    'PROBABILIDAD_DESERCION': probabilidades,
    'PREDICCION_DESERCION': predicciones,
})
```

---

## Flujo de publicación

El desarrollo ocurre en un directorio de trabajo local, donde residen los datos y donde los
notebooks se ejecutan con sus salidas visibles. Ese directorio **no** es el repositorio.

```bash
python scripts/sanitizar_notebooks.py --origen "RUTA/AL/DIRECTORIO/DE/TRABAJO"
```

El script elimina salidas, contadores de ejecución, estado de widgets y credenciales de
conexión, y sustituye los nombres de esquemas y tablas por equivalentes genéricos según
`scripts/mapeo_esquemas.local.json` (excluido del control de versiones). Luego verifica el
resultado contra una lista de patrones prohibidos y **termina con código de error si alguno
sobrevive**, de modo que el material sensible no pueda publicarse por descuido.

> Ejecuta siempre el script antes de hacer commit. Un dato sensible que entra al historial de
> git permanece allí aunque se borre en un commit posterior.

---

## Limitaciones del enfoque y trabajo pendiente

Documentar lo que un modelo todavía no hace bien es parte del trabajo. Estos son los puntos
identificados durante la revisión, en orden de prioridad.

### Fuga por grupo en la partición *(medido)*

**El problema más grave del proyecto, y el que invalida el AUC reportado.**

El dataset tiene una fila por estudiante y periodo, de modo que un mismo alumno aparece
varias veces: 3,7 filas en promedio, hasta 28, y el **89,7 % de las filas pertenece a
estudiantes con más de un registro**. Al partir train/test de forma aleatoria, las filas de
un mismo estudiante caen a ambos lados.

Medición sobre la partición 70/30 de la versión vigente:

```
Filas de prueba cuyo estudiante también está en entrenamiento:  86,5 %
```

El modelo puede memorizar al individuo en lugar de aprender el fenómeno, y casi todo el
conjunto de prueba deja de ser información nueva. Un indicio adicional lo confirma: **ninguna
variable supera un AUC de 0,673 por sí sola**, y la mejor es `SEMESTRE_SINU`. Que un conjunto
de predictores tan débiles produzca 0,848 se explica mucho mejor por memorización de sujetos
que por interacciones genuinas.

**Corrección:** particionar por estudiante (`GroupKFold` / `GroupShuffleSplit` con
`groups=IDENTIFICACION`), de modo que ningún alumno aparezca en ambos lados. Es previsible
que el AUC caiga de forma considerable; ese número menor será el real.

### Validación temporal

Además de agrupar por estudiante, la división debe ser **temporal**: entrenar con los
periodos antiguos y validar sobre el más reciente, como se ensayó en `Prototipo 1.7`. Una
partición aleatoria sobre datos con estructura temporal produce métricas optimistas incluso
después de corregir la fuga por grupo.

### Horizonte de predicción

Las variables predictoras pertenecen al mismo periodo que el target, de modo que el modelo
**caracteriza el presente en lugar de anticipar el futuro**. La versión útil en producción
requiere desfasar las features un periodo: usar la información de *t* para predecir la
deserción en *t+1*.

### Ajuste del preprocesamiento

El `StandardScaler` y los imputadores se ajustan antes de dividir train/test, lo que permite
que estadísticas del conjunto de prueba influyan en el entrenamiento. El `fit` debe ocurrir
exclusivamente dentro del pipeline, después del split. El `ImbPipeline` de la versión 1.6 ya
resuelve esto para SMOTE; falta extenderlo al resto del preprocesamiento.

### Fuga de target por `TIPO_SALTO`: descartada *(medido)*

`TIPO_SALTO` describe la transición al periodo siguiente y el target se deriva de una lógica
de negocio emparentada, así que era candidata natural a fuga. **La hipótesis se probó y no se
sostiene:** el AUC máximo alcanzable usando únicamente esa variable —puntuando cada fila con
la tasa empírica de su categoría, que es el techo invariante a la codificación— es de
**0,5104**, apenas por encima del azar. Explica el 3 % de la separación del modelo.

Se deja documentado porque la sospecha era razonable y la refutación es parte del resultado:
descartar una causa con una medición vale tanto como confirmarla.

### Variables sin información *(medido)*

Dos de los predictores no aportan absolutamente nada tras la imputación:

| Variable | Situación |
|---|---|
| `ESTADO_PAGO` | **Constante**: un único valor en todo el dataset |
| `ASISTENCIA` | 99,69 % de las filas comparten el mismo valor |

`ASISTENCIA` llegaba con más del 90 % de faltantes y la imputación por moda terminó de
aplanarla. Otras variables presentan la misma patología en menor grado: imputar la mediana
sobre una columna mayoritariamente vacía introduce un valor artificial masivo que aporta
ruido, no señal. Conviene eliminarlas o convertirlas en indicadores binarios de presencia de
registro.

### Consolidación técnica

- Deduplicar las fuentes antes de integrarlas: uno de los `LEFT JOIN` incrementa el número de
  filas por llaves repetidas en la tabla de origen.
- Alinear las columnas del scoring contra `pipeline.feature_names_in_` en lugar de asumir el
  orden, para evitar desajustes silenciosos entre entrenamiento y despliegue.
- Migrar los datasets intermedios de `.xlsx` a **Parquet**, con una reducción de tamaño
  cercana a 10×.

---

## Autor

**Diego Alejandro Ojeda Pinzón**

---

## Licencia

Repositorio publicado con fines de portafolio profesional. El código ilustra la metodología
empleada; no contiene datos, modelos entrenados ni información identificable de ninguna
organización o persona.
