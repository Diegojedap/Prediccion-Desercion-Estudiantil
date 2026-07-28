# Model Card — Priorización de riesgo de deserción, versión 1.8

## Identificación

| | |
|---|---|
| **Versión** | 1.8 |
| **Algoritmo** | LightGBM (`LGBMClassifier`), 481 árboles, 32 hojas, learning rate 0.0244 |
| **Tipo** | Clasificación binaria desbalanceada, usada como **ranking de riesgo** |
| **Entrenamiento** | Un periodo académico; evaluación sobre el siguiente |
| **Autor** | Diego Alejandro Ojeda Pinzón |
| **Código** | [`Prototipo 1.8.ipynb`](../Prototipo%201.8.ipynb) |
| **Auditoría** | [`Auditoria.ipynb`](../Auditoria.ipynb), [`Auditoria_base.ipynb`](../Auditoria_base.ipynb) |

---

## Uso previsto

**Para lo que sirve.** Ordenar a los estudiantes de un periodo por riesgo de no rematricularse,
de modo que un programa de permanencia pueda **priorizar a quiénes contactar** con la capacidad
que tenga. La alerta se produce al cerrar el periodo, justo antes de la ventana de rematrícula:
el calendario encaja con una campaña de reinscripción.

**Para lo que no sirve.**

- **No es un sistema de alerta temprana intra-periodo.** Sus variables solo existen al cerrar
  el periodo. Se midió el costo de anticiparse uno: **−0.128 de AUC**, y la precisión del tramo
  prioritario cae de 0.834 a 0.266.
- **No debe operarse con un umbral fijo de probabilidad.** El umbral óptimo no se transfiere
  entre periodos. El modelo se usa por **capacidad**: los N de mayor riesgo, con N igual a lo
  que el equipo puede atender.
- **No debe usarse para decisiones automáticas sobre personas** —sanciones, denegación de
  beneficios, condicionamiento de matrícula. Es una herramienta de priorización de contacto.
- **No es válido indefinidamente.** Ver *Caducidad*.

---

## Datos de entrenamiento

Registros de estudiante × periodo académico de una institución de educación superior, con tres
familias de variables:

| Familia | Variables |
|---|---|
| Trayectoria académica | semestre, materias inscritas, materias aprobadas, porcentaje de aprobación |
| Financiera | saldo de cartera |
| Perfil | modalidad, género, rango de edad, rango salarial, situación laboral, método de financiamiento, zona de residencia, régimen de salud |

**Variable derivada:** `FLAG_NO_APROBO_NADA` separa "aprobó 0 %" de "dato faltante", que la
imputación por mediana colapsaría en el mismo valor pese a ser situaciones opuestas.

**Variables excluidas por decisión:** pertenencia a grupo étnico, condición de discapacidad y
autoidentificación LGBTIQ estaban disponibles en la fuente y **no se incorporaron**. Son
categorías especiales bajo la Ley 1581 de 2012 y su uso para clasificar riesgo individual no
está justificado por el beneficio esperado.

**Variables descartadas por medición:** dos resultaron constantes o casi constantes tras la
imputación y no aportaban información.

---

## Desempeño

Evaluado sobre el periodo siguiente al de entrenamiento, que el modelo nunca vio.

| Tramo priorizado | Precisión | Lift sobre el azar | Recall |
|---|---|---|---|
| **Top 1 %** | **0.890** | **4,6×** | 0.046 |
| Top 5 % | 0.704 | 3,6× | 0.180 |
| Top 10 % | 0.597 | 3,0× | 0.304 |
| Top 20 % | 0.493 | 2,5× | 0.504 |

**AUC-ROC: 0.794**

Como clasificador binario sobre toda la población el modelo es discreto. Como priorizador del
tramo crítico es fuerte, y esa es la forma en que se usa.

### Comparaciones descartadas por medición

| Alternativa | Resultado |
|---|---|
| Stacking XGBoost + LightGBM | AUC 0.735 — **peor** que cualquiera de sus componentes, y 9× más lento |
| Historia completa de entrenamiento | AUC equivalente a un solo periodo, con 7× más cómputo |
| Recalibrar el umbral cada periodo | Recupera 85,8 % del máximo, contra 85,5 % sin recalibrar |

---

## Limitaciones

**Horizonte.** El modelo caracteriza el cierre del periodo, no anticipa dentro de él. Cerrar
esa brecha exige señales observables al comienzo —actividad temprana en plataforma, estado de
pago a la matrícula, comportamiento de inscripción— que no están en el conjunto actual.

**Estabilidad de las fuentes.** Durante una sola sesión de medición se observó que el criterio
del campo de estado se había recalculado en origen, desplazando la tasa base varios puntos en
todos los periodos; que una tabla de origen quedó vacía y volvió a poblarse; y que el conteo de
la tabla principal creció. Ninguna de esas condiciones produce un error: cambian las métricas
en silencio. **Toda métrica de este model card es válida para la extracción con la que se
midió**, cuya fecha consta en [`metadatos.json`](metadatos.json).

**Cobertura de variables.** Varias columnas llegan con alta proporción de faltantes. El
pipeline declara la cobertura de cada una y descarta las vacías en lugar de perderlas en
silencio, pero las que sobreviven con cobertura baja aportan menos de lo que su presencia
sugiere.

**Sesgo por grupo: auditado, y hay disparidad.** Medido en
[`Auditoria_sesgo.ipynb`](../Auditoria_sesgo.ipynb), con el punto de operación en el top 10 %.
La métrica es la **paridad de recall** —un estudiante en riesgo debería tener la misma
probabilidad de ser detectado sea cual sea su grupo— evaluada con la regla del 80 %:

| Atributo | Razón de recall | Estado |
|---|---|---|
| `GENERO` | 0.921 | aceptable |
| `MODALIDAD` | 0.499 | **disparidad** |
| `RANGO_SALARIO` | 0.503 | **disparidad** |
| `ZONA_RESIDENCIA` | 0.374 | **disparidad** |
| `RANGO_EDAD` | **0.099** | **disparidad severa** |

**El modelo detecta sobre todo a estudiantes con ficha incompleta.** En cada atributo, el
grupo con mayor recall es el de valor ausente. No es un artefacto: quienes tienen datos
faltantes desertan más, y el modelo lo explota correctamente. Pero implica que **para la
mayoría, que sí tiene ficha completa, el modelo funciona bastante peor de lo que sugiere su
métrica global**.

**La brecha más preocupante es por edad.** Los estudiantes de 16 a 20 años obtienen un recall
de **0.069** frente a 0.16 en los tramos mayores, pese a una tasa base de deserción similar.
Su precisión es la más alta de todos los grupos, lo que confirma el mecanismo: el modelo solo
los señala cuando está muy seguro y se le escapan casi todos. Son justamente los de primer
ingreso, donde un programa de permanencia tiene más margen.

### Mitigación implementada *(medida)*

La disparidad no viene de que el modelo sea injusto por diseño, sino de cómo se usa: un corte
global asigna menos acompañamiento a los grupos que predice peor, justamente por ser más
difíciles de predecir. Se compararon tres formas de repartir la misma capacidad total:

| Estrategia | Paridad | Desertores captados | Precisión |
|---|---|---|---|
| Capacidad global | **0.099** | 6.815 | 0.597 |
| **Misma tasa por segmento** | **0.850** | 6.564 *(−3,7 %)* | 0.575 |
| Proporcional al riesgo del grupo | 0.657 | 6.768 *(−0,7 %)* | 0.593 |

**Repartir la misma proporción dentro de cada grupo cierra la brecha**: la paridad pasa de
0.099 a 0.850, por encima de la regla del 80 %, y el recall del grupo de 16 a 20 años sube de
**0.069 a 0.313** — cuatro veces y media más detección. El costo es 251 desertores menos
captados sobre 6.815 y dos puntos de precisión.

`priorizar()` reparte por segmento **por defecto**. Aceptar un reparto global exige pasarlo
explícitamente y emite un aviso, porque reintroduce la disparidad.

Conviene notar que la opción teóricamente más elegante —repartir en proporción al riesgo
esperado de cada grupo— **no funciona**: se queda en 0.657, porque la masa de riesgo del grupo
sin datos absorbe la cuota.

**Lo que la mitigación no arregla:** reparte mejor, pero no mejora la señal. El modelo sigue
detectando peor a los estudiantes con ficha completa, y completar el registro sociodemográfico
sigue siendo el pendiente de fondo.

---

## Caducidad

**Reentrenar cada periodo académico.** El modelo se entrena sobre un periodo y evalúa sobre el
siguiente; aplicado a un periodo lejano al de entrenamiento pierde calibración.

Y una advertencia práctica sobre el periodo de evaluación: **no evaluar contra el periodo más
reciente disponible.** El target se resuelve al comprobar que el estudiante no volvió a
matricularse, de modo que los periodos recientes tienen etiquetas a medio resolver. Medido: al
evaluar contra el último periodo disponible el AUC cae a **0.59**, y no por mal desempeño sino
por etiquetas inmaduras. El pipeline exige un margen de maduración explícito.

---

## Contenido del artefacto

| Archivo | Qué es |
|---|---|
| [`modelo_18.txt`](modelo_18.txt) | El modelo entrenado, en formato de texto de LightGBM |
| [`transformadores.json`](transformadores.json) | Imputadores, escalador y codificador, en JSON legible |
| [`metadatos.json`](metadatos.json) | Versión, fecha, hiperparámetros, variables y métricas |

**Por qué no hay un `.pkl`.** Un pickle ejecuta código arbitrario al deserializarse, así que
distribuirlo en un repositorio público invita a que alguien cargue a ciegas un artefacto
ejecutable. El formato de texto de LightGBM es **inspeccionable**, no ejecuta nada al cargarse
y no contiene registros individuales: solo los umbrales aprendidos.

```python
import lightgbm as lgb
booster = lgb.Booster(model_file='modelo/modelo_18.txt')
# Aplicar antes las transformaciones descritas en transformadores.json,
# respetando orden_final_de_columnas
proba = booster.predict(X_transformado)
```

---

## Consideraciones éticas

El modelo emite señales sobre personas. Tres compromisos de diseño se derivan de eso:

1. **Explicabilidad individual.** Cada priorización debe poder justificarse ante quien va a
   actuar sobre ella; el pipeline incorpora SHAP para explicación local.
2. **Exclusión de categorías especiales.** Etnia, discapacidad y condición LGBTIQ estaban
   disponibles y se dejaron fuera deliberadamente.
3. **Uso como priorización de contacto, nunca como decisión automática.** Un falso positivo
   debe costar una llamada, no una consecuencia para el estudiante.
