# Predicción de Deserción Estudiantil en Educación Superior

Sistema de machine learning para priorizar estudiantes en riesgo de abandonar sus estudios,
orientado a programas de intervención temprana y retención.

El proyecto documenta el recorrido completo de un problema real de clasificación
desbalanceada — integración de fuentes, ingeniería de características, ensamblado,
interpretabilidad — y, sobre todo, una auditoría que **refutó cuatro de sus propias
conclusiones** al medirlas contra los datos.

---

## Alcance de este repositorio

Contiene **únicamente metodología y código**. No incluye datos, modelos entrenados,
identificadores de sistemas ni nombres de esquemas o tablas de ninguna organización.

- Los `.xlsx` y `.pkl` están excluidos vía [`.gitignore`](.gitignore).
- Los parámetros de conexión se leen de variables de entorno (ver [`.env.example`](.env.example)).
- Los nombres de esquemas y tablas en las consultas son **genéricos**, sustituidos por
  [`scripts/sanitizar_notebooks.py`](scripts/sanitizar_notebooks.py).
- Los notebooks se publican sin salidas ejecutadas: las tablas impresas contenían datos
  personales de estudiantes, sujetos a la Ley 1581 de 2012.

---

## El problema

Dado un estudiante en un periodo académico, estimar su riesgo de deserción. La clase positiva
es minoritaria, así que el énfasis está en el **recall del tramo de mayor riesgo**: no
detectar a un estudiante en riesgo cuesta más que revisar un caso que resulta falso.

---

## Arquitectura del flujo

```
fuentes relacionales
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
  pipeline: imputación → escalado → codificación → LightGBM
        │
        ▼
  ranking de riesgo → priorización por capacidad de atención
```

---

## Fuentes de datos

Tablas de un repositorio relacional sobre SQL Server, integradas con `LEFT JOIN` sobre la
llave compuesta `Identificacion` + `Periodo`. Cada fuente se **deduplica antes** del merge, y
una aserción verifica que el número de filas no cambie: sin ese control, uno de los joins
inflaba el dataset silenciosamente.

| Dominio | Aporte |
|---|---|
| Histórico académico | Trayectoria del estudiante y perfil sociodemográfico |
| Aprobación de materias | Materias inscritas, aprobadas y porcentaje de aprobación |
| Cartera | Saldo financiero pendiente |

Acceso vía `SQLAlchemy` + `pyodbc`. Host, puerto y base se toman del entorno.

---

## Variables

**Numéricas:** `SEMESTRE_SINU`, `MATERIAS_INSCRITAS`, `MATERIAS_APROBADAS`,
`PORCENTAJE_APROBACION`, `TOTAL`, `FLAG_NO_APROBO_NADA`

**Categóricas:** `TIPO_SALTO`, `MODALIDAD`, `GENERO`, `RANGO_EDAD`, `RANGO_SALARIO`,
`ESTA_TRABAJANDO`, `METODO_FINANCIAMIENTO`, `ZONA_RESIDENCIA`, `REGIMEN_SISTEMA_SALUD`

### Ingeniería de características

Un porcentaje de aprobación igual a cero y un dato faltante son situaciones opuestas —la
primera es la señal de riesgo más fuerte disponible— pero la imputación por mediana las
colapsa en el mismo valor. Se separan antes de imputar:

```python
df['FLAG_NO_APROBO_NADA'] = np.where(df['PORCENTAJE_APROBACION'] == 0, 1, 0)
df['PORCENTAJE_APROBACION'] = df['PORCENTAJE_APROBACION'].replace(0, np.nan)
```

### Variables descartadas

`ESTADO_PAGO` resultó **constante** en todo el dataset y `ASISTENCIA` concentraba el 99,69 %
de las filas en un mismo valor tras la imputación. Ninguna aporta información.

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
├── Auditoria.ipynb        Reproduce las mediciones que este README afirma
│
├── Prototipo 1.1.ipynb    Baseline: LogisticRegression, DecisionTree, RandomForest
├── Prototipo 1.2.ipynb    Incorpora SMOTE para balanceo de clases
├── Prototipo 1.3.ipynb    GridSearchCV sobre RandomForest
├── Prototipo 1.4.ipynb    Introduce XGBoost
├── Prototipo 1.5.ipynb    XGBoost con ajuste de hiperparámetros
├── Prototipo 1.7.ipynb    XGBoost + GridSearchCV + SMOTE + umbral óptimo + SHAP
├── Prototipo 1.6.ipynb    Stacking XGB+LGBM, optimización bayesiana
├── Prototipo 1.8.ipynb    ★ Vigente: LightGBM, ranking por capacidad
├── Untitled-2.ipynb       Consolidación intermedia de XGBoost
│
├── Despliegue 1.7.ipynb   Scoring masivo con el pipeline de XGBoost
└── Despliegue 1.6.ipynb   Scoring masivo con el pipeline de stacking
```

---

## Modelo vigente: versión 1.8

```python
Pipeline([
    ('preprocesador', ColumnTransformer([
        ('num', Pipeline([SimpleImputer(strategy='median'), StandardScaler()]), NUMERICAS),
        ('cat', Pipeline([SimpleImputer(strategy='most_frequent'), OrdinalEncoder()]), CATEGORICAS),
    ])),
    ('clasificador', LGBMClassifier(n_estimators=481, num_leaves=32,
                                    learning_rate=0.0244, subsample=0.9122,
                                    scale_pos_weight=peso)),
])
```

Todo el preprocesamiento vive **dentro** del pipeline, de modo que se ajuste solo con los
datos de entrenamiento. Se usa `scale_pos_weight` en lugar de SMOTE: mismo efecto de
balanceo sin sintetizar filas, y sin interpolar entre categorías codificadas como enteros,
donde la operación no tiene sentido.

### Desempeño

Evaluado entrenando con un periodo académico y midiendo sobre el siguiente, que el modelo
nunca vio.

| Tramo priorizado | Precisión | Lift sobre el azar | Recall |
|---|---|---|---|
| **Top 1 %** | **0.890** | **4,6×** | 0.046 |
| Top 2 % | 0.820 | 4,2× | 0.084 |
| Top 5 % | 0.706 | 3,6× | 0.180 |
| Top 10 % | 0.594 | 3,0× | 0.304 |
| Top 15 % | 0.532 | 2,7× | 0.408 |
| Top 20 % | 0.493 | 2,5× | 0.504 |
| Top 30 % | 0.419 | 2,1× | 0.643 |

**AUC-ROC: 0.794**

De cada 100 estudiantes en el tramo de mayor riesgo, 89 desertan efectivamente. Como
clasificador binario sobre toda la población el modelo es discreto; como **priorizador** del
tramo crítico es fuerte — y un programa de permanencia no interviene sobre cien mil personas,
sino sobre mil.

### Qué intervención soporta este modelo, y cuál no

Las variables provienen del **mismo periodo** que el target: rendimiento académico, cartera y
matrícula de ese semestre. El target, a su vez, describe si el estudiante volvió a
matricularse en el periodo siguiente.

Eso ubica la alerta en un momento muy concreto: **al cerrar el periodo, antes de la ventana
de rematrícula.** Que es exactamente cuando opera una campaña de reinscripción, así que el
calendario encaja. Para ese uso el modelo sirve, y bien.

No sirve, en cambio, para intervenir **durante** el periodo — tutorías, apoyo financiero,
acompañamiento académico — porque para entonces sus variables todavía no existen. Eso exige
anticiparse un periodo, y se midió cuánto cuesta:

| | AUC | P@1 % | P@5 % | P@10 % |
|---|---|---|---|---|
| Variables del periodo *t* | 0.785 | 0.834 | 0.592 | 0.492 |
| **Variables del periodo *t−1*** | **0.657** | **0.266** | 0.343 | 0.334 |

Ambos evaluados sobre las mismas filas, de modo que la diferencia sea solo el desfase.

**Anticiparse un periodo hace inviable el modelo.** La precisión del tramo prioritario cae de
0.834 a 0.266 y el lift queda en 1,7×. Un síntoma lo confirma: en el modelo desfasado el
top 1 % rinde *peor* que el top 5 %, es decir, el orden se invierte y sus predicciones más
confiadas dejan de ser las mejores. La cola superior del ranking es ruido.

La conclusión no es que el proyecto falle, sino que **su alcance es más estrecho de lo que
sugiere el nombre**: es un priorizador para campañas de rematrícula, no un sistema de alerta
temprana intra-periodo.

Construir lo segundo requiere variables disponibles **al comienzo** del periodo —actividad en
plataforma durante las primeras semanas, estado de pago temprano, comportamiento de
inscripción, notas parciales— y no las hay en el conjunto actual. La fuente de actividad en
plataforma, que es justamente ese tipo de señal, quedó fuera de esta versión.

### Por qué se opera por capacidad y no por umbral

El umbral de probabilidad **no se transfiere entre periodos**: su óptimo se desplaza de forma
abrupta y con un corte fijo el recall se desploma. El *ranking*, en cambio, es estable.

Así que el sistema no responde *"¿deserta este estudiante?"* sino **"¿cuáles son los N de
mayor riesgo?"**, con N igual a lo que el equipo de retención puede atender. La elección del
tramo es una decisión operativa, no una métrica a maximizar.

### Decisiones de diseño y su evidencia

| Decisión | Alternativa | Evidencia |
|---|---|---|
| LightGBM solo | Stacking XGB + LGBM | AUC **0.777** vs 0.735, y 9× más rápido |
| Operar por capacidad | Umbral de probabilidad | El umbral no se transfiere entre periodos |
| Ventana de 1 periodo | Historia completa | AUC equivalente, 6× menos tiempo |
| Preprocesamiento en el pipeline | `fit` antes del split | Corrección de principio; efecto medido: 0.004 |
| 15 predictores | 17 | Dos eran constantes tras la imputación |

**El stacking es el hallazgo más contraintuitivo:** el ensamblado que había sido la
sofisticación central de la versión anterior, producto de una optimización bayesiana, rinde
**peor que cualquiera de sus dos componentes por separado** y cuesta nueve veces más. Bajo
partición aleatoria parecía mejorar; bajo evaluación honesta no sobrevive.

---

## Auditoría: cuatro conclusiones propias, refutadas

El diagnóstico de este proyecto cambió cuatro veces al medirlo. Se documentan las refutaciones
porque descartar una causa con una medición vale tanto como confirmarla, y evita que alguien
vuelva a perseguirla.

| Hipótesis | Resultado |
|---|---|
| `TIPO_SALTO` filtra el target | ❌ AUC **0.510** usándola sola |
| Los sujetos repetidos inflan la métrica | ❌ Efecto **0.000** |
| El preprocesamiento antes del split infla | ❌ Efecto **−0.004** (el pipeline limpio rinde algo mejor) |
| La partición temporal infla el AUC | ❌ Efecto **−0.006** con datos actuales |

Las tres primeras se refutaron con [`Auditoria.ipynb`](Auditoria.ipynb). La cuarta requirió
volver a la base, y es la más instructiva.

### El hallazgo de fondo: el target no es estable en origen

La cuarta hipótesis **sí parecía confirmada**: medida sobre un export estático, la partición
temporal costaba 0.101 de AUC. Ese número motivó el rediseño completo de la versión 1.8.

Al repetir la medición leyendo directamente de la base meses después, el efecto cayó a
**0.006**. La diferencia no estaba en el método sino en los datos: **la tasa del target había
cambiado en todos los periodos**, incluidos algunos cerrados hacía más de una década, con
desplazamientos de entre 3 y 13 puntos porcentuales. El campo de estado se había recalculado
en el sistema origen.

La consecuencia metodológica es la lección más útil de todo el ejercicio:

> Cuando el target se resuelve o se recalcula con el tiempo, **las métricas medidas sobre un
> export estático no son comparables entre fechas de extracción**, y la inestabilidad del
> dato puede imitar exactamente la firma de una fuga de información. Ninguna validación
> cruzada lo detecta, porque el problema no está en la partición: está en la etiqueta.

Las decisiones de diseño de la 1.8 se sostienen —LightGBM sigue ganando, el ranking sigue
siendo más robusto que el umbral, la ventana corta sigue costando menos— pero la
justificación original de una de ellas resultó ser un artefacto.

---

## Requisitos

```
python >= 3.11

pandas          numpy           openpyxl
scikit-learn    lightgbm        xgboost
imbalanced-learn                shap
sqlalchemy      pyodbc          joblib
matplotlib      seaborn
```

```bash
pip install pandas numpy openpyxl scikit-learn lightgbm xgboost \
            imbalanced-learn shap sqlalchemy pyodbc joblib matplotlib seaborn
```

Requiere el driver **ODBC Driver for SQL Server**. Configura la conexión copiando
`.env.example` a `.env`.

> Los notebooks leen `DB_HOST`, `DB_PORT` y `DB_NAME` del entorno. Sin esas variables la celda
> de conexión falla de forma explícita, en lugar de exponer credenciales en el código.

---

## Uso

### Entrenamiento

Ejecutar [`Prototipo 1.8.ipynb`](Prototipo%201.8.ipynb) de principio a fin.

### Priorización

```python
resultado = priorizar(df_periodo_actual, capacidad=1000)
```

Devuelve el ranking completo con su posición y marca los `capacidad` primeros. **No expone un
umbral de probabilidad**, deliberadamente: sería la vía más fácil de reintroducir el error que
esta versión corrige.

---

## Flujo de publicación

El desarrollo ocurre en un directorio de trabajo local donde residen los datos y donde los
notebooks se ejecutan con sus salidas visibles. Ese directorio **no** es el repositorio.

```bash
python scripts/sanitizar_notebooks.py --origen "RUTA/AL/DIRECTORIO/DE/TRABAJO"
```

El script elimina salidas, contadores de ejecución, estado de widgets y credenciales, y
sustituye los nombres de esquemas y tablas por equivalentes genéricos según
`scripts/mapeo_esquemas.local.json` (excluido del control de versiones). Luego verifica el
resultado contra una lista de patrones prohibidos y **termina con código de error si alguno
sobrevive**.

> Ejecútalo siempre antes de hacer commit. Un dato sensible que entra al historial de git
> permanece allí aunque se borre en un commit posterior.

---

## Limitaciones y trabajo pendiente

### Horizonte de predicción — medido, y es la limitación de fondo

Documentada arriba con números: desfasar las variables un periodo cuesta 0.128 de AUC y hunde
la precisión del tramo prioritario de 0.834 a 0.266.

El pendiente real no es de modelado sino **de datos**: hacen falta variables observables al
comienzo del periodo.

### La actividad en plataforma: prometedora y no verificable *(medido)*

La fuente de actividad en la plataforma virtual —entregas, calificaciones parciales,
participación por curso— es la candidata natural, y agregada por estudiante y periodo resulta
**mucho más predictiva que todo el resto junto**:

| | AUC | P@1 % | Lift |
|---|---|---|---|
| Variables administrativas | 0.614 | 0.386 | 1,3× |
| **Actividad en plataforma sola** | **0.838** | **0.913** | **3,0×** |
| Ambas | 0.854 | 0.913 | 3,0× |

La actividad aporta **+0.24 de AUC** y concentra el 69 % de la importancia del modelo
conjunto. Las variables que manda son nota media, dispersión de notas y volumen de actividad.

El patrón subyacente es nítido: quienes desertan se matriculan con **la misma carga** —igual
número de cursos y de créditos— pero generan **un 25 % menos de actividad registrada**. Es
desenganche, no ausencia de datos: de hecho aparecen en la plataforma con más frecuencia que
quienes se quedan.

**Y aun así el resultado no puede darse por bueno**, por tres razones que conviene dejar
escritas antes de que alguien lo tome como conclusión:

1. **La comparación es contra un baseline incompleto.** Las variables de rendimiento
   académico vienen nulas en la fuente para los periodos con cobertura de plataforma, así que
   el 0.614 corresponde a semestre y perfil sociodemográfico, no al modelo real.
2. **No hay marca temporal dentro del periodo.** Sin la fecha de cada actividad no se puede
   distinguir *"se desenganchó en las primeras semanas"* —señal temprana legítima— de
   *"dejó de participar al irse"*, que sería medir el desenlace en lugar de anticiparlo.
3. **La cobertura histórica es de un solo año**, y sus periodos son flujos paralelos por
   cohorte, no una secuencia que un mismo estudiante recorra. Menos del 1 % de los
   estudiantes aparece en dos periodos consecutivos, así que no se puede construir el desfase
   con esta fuente.

**Requisito para cerrar la pregunta:** marca temporal por actividad. Con ella se puede
truncar el histórico a las primeras semanas del periodo y medir si la señal se sostiene. Sin
ella, este 0.838 es indistinguible de una fuga.

### Estabilidad de las fuentes — el riesgo transversal

Documentado arriba para el target, pero el problema es más amplio y afecta a **toda métrica
que este repositorio reporte**. Durante una sola sesión de medición se observó que:

- El criterio del campo de estado se había **recalculado**, desplazando la tasa base entre 3
  y 13 puntos porcentuales en todos los periodos, incluidos los cerrados hacía una década.
- Una de las tablas de origen estaba **vacía**, de modo que su variable entraba como
  completamente nula y el imputador la descartaba en silencio: el modelo corría con una
  variable menos de las declaradas, sin aviso.
- Otra tenía la columna clave **nula para los periodos recientes** pese a contener filas.
- El conteo de filas de la tabla principal **creció durante la propia sesión**.

Ninguna de esas condiciones produce un error: producen métricas distintas, en silencio.

**Controles mínimos antes de dar por buena cualquier medición futura:**

```python
# 1. Cobertura real de cada variable, no solo que el merge no falle
for c in FEATURES:
    assert df[c].notna().mean() > 0.5, f"{c} tiene cobertura insuficiente"

# 2. Las filas no deben cambiar al integrar fuentes
assert len(df) == filas_iniciales

# 3. Registrar la huella de la extracción junto a las métricas
huella = {'fecha': datetime.now(timezone.utc).isoformat(),
          'filas': len(df), 'tasa_base': float(y.mean())}
```

Sin esa huella, dos métricas de este proyecto no son comparables aunque provengan del mismo
código.

### Fuentes no incorporadas

La versión 1.8 usa tres de las seis fuentes disponibles. Geolocalización, becas y actividad en
plataforma quedaron fuera y no se ha medido si aportan.

### Cobertura de variables

Varias columnas llegan con más del 90 % de faltantes. Imputar la mediana introduce un valor
artificial masivo que aporta ruido; conviene convertirlas en indicadores de presencia de
registro.

---

## Autor

**Diego Alejandro Ojeda Pinzón**

---

## Licencia

Repositorio publicado con fines de portafolio profesional. El código ilustra la metodología
empleada; no contiene datos, modelos entrenados ni información identificable de ninguna
organización o persona.
