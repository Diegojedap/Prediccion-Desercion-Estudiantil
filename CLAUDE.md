# Contexto del proyecto

Modelo de priorización de riesgo de deserción estudiantil. La descripción, las métricas y las
limitaciones están en el [README](README.md); aquí solo va lo que hay que saber **antes de
tocar algo**, porque no se deduce leyendo el código.

---

## Reglas que no se negocian

**Este repositorio es público y el proyecto trabaja con datos personales de estudiantes**
sujetos a la Ley 1581 de 2012, además de un compromiso contractual de confidencialidad sobre
información institucional.

1. **Ningún notebook se commitea con salidas ejecutadas.** Las tablas impresas contenían
   identificaciones reales. Los notebooks se preparan con `scripts/sanitizar_notebooks.py`,
   nunca copiándolos a mano.
2. **Ningún nombre de organización, esquema, tabla o IP entra al repositorio.** Se sustituyen
   por genéricos mediante `scripts/mapeo_esquemas.local.json`, que está excluido de control de
   versiones porque sus claves son justamente lo que no debe publicarse.
3. **Nada de `.xlsx`, `.csv` ni `.pkl`.** El artefacto del modelo se publica en formato de
   texto de LightGBM: un pickle ejecuta código arbitrario al deserializarse.
4. **Antes de cualquier commit**, `python scripts/verificar_repo.py`. El hook de pre-commit lo
   hace solo si está activado con `git config core.hooksPath .githooks`.

Un dato sensible que entra al historial de git permanece allí aunque se borre después. Ya
ocurrió una vez en este proyecto y hubo que eliminar el repositorio entero para purgarlo.

---

## Trampas del dominio

Estas costaron mediciones equivocadas. Merecen leerse antes de escribir código de evaluación.

### El periodo más reciente tiene etiquetas inmaduras

El target se resuelve cuando se comprueba que el estudiante **no volvió a matricularse**, así
que los periodos recientes están a medio etiquetar.

```python
anio_test = df['ANIO'].max()                    # MAL: AUC 0.59, etiquetas sin resolver
anio_test = df['ANIO'].max() - MARGEN_MADURACION  # BIEN: AUC 0.79
```

Con `MARGEN_MADURACION = 2`. Este error se cometió dos veces y en ambas produjo métricas que
parecían un mal desempeño del modelo.

### Las fuentes cambian de estado sin avisar

En una sola sesión de medición se observó que el criterio del target se había **recalculado**
—desplazando la tasa base entre 3 y 13 puntos en todos los periodos, incluidos los cerrados
hacía una década—, que una tabla de origen pasó de vacía a poblada, y que la tabla principal
creció mientras se medía.

Consecuencia práctica: **dos métricas de este proyecto no son comparables si provienen de
extracciones distintas**, aunque las produzca el mismo código. Registrar siempre la huella:

```python
huella = {'fecha_utc': datetime.now(timezone.utc).isoformat(),
          'filas': len(df), 'tasa_base': float(y.mean())}
```

### Una tabla vacía no produce ningún error

`SimpleImputer` descarta en silencio las columnas completamente nulas. El modelo entrena con
menos predictores de los declarados y nada lo advierte. Verificar cobertura explícitamente
antes de entrenar, como hace `scripts/entrenar_modelo.py`.

### Los nombres de archivo llevan espacios

`git ls-files` debe leerse con separador nulo (`-z`). Dividir por espacios parte las rutas y
deja archivos sin revisar; un verificador escrito así llegó a comprobar 2 de 14 archivos
creyendo que los revisaba todos.

---

## Hipótesis ya refutadas

No volver a perseguirlas. Están medidas en `Auditoria.ipynb` y `Auditoria_base.ipynb`.

| Hipótesis | Resultado |
|---|---|
| `TIPO_SALTO` filtra el target | AUC **0.510** usándola sola |
| Los sujetos repetidos inflan la métrica | Efecto **0.000** |
| El preprocesamiento antes del split infla | Efecto **−0.004** |
| La partición temporal infla el AUC | Efecto **−0.006** con datos actuales |
| El stacking mejora sobre sus componentes | **Peor** que ambos, y 9× más lento |
| Recalibrar el umbral cada periodo aporta | 85,8 % contra 85,5 % sin recalibrar |

La cuarta parecía confirmada (−0.101) sobre un export estático; al remedir contra la fuente el
efecto desapareció. Era inestabilidad del dato imitando la firma de una fuga.

---

## Convenciones del modelo

- **Se opera por capacidad, nunca por umbral fijo de probabilidad.** El umbral óptimo no se
  transfiere entre periodos; el ranking sí. La función de scoring pide `capacidad` —cuántos
  estudiantes puede atender el programa— y deliberadamente no expone un umbral.
- **Se reentrena cada periodo académico.** Ver caducidad en `modelo/MODEL_CARD.md`.
- **No desplegar con un punto de operación global único**: la auditoría de sesgo encontró
  disparidad de recall por edad, modalidad, zona y rango salarial.
- Variables excluidas por decisión: etnia, discapacidad y condición LGBTIQ estaban disponibles
  y se dejaron fuera. No reincorporarlas sin una justificación explícita.

---

## Comandos

```bash
python scripts/verificar_repo.py                      # fugas: siempre antes de commitear
python scripts/sanitizar_notebooks.py --origen RUTA   # preparar notebooks para publicar
python scripts/entrenar_modelo.py                     # regenerar el artefacto de modelo/
git config core.hooksPath .githooks                   # activar el hook (una vez por clon)
```

Variables de entorno en `.env.example`. Las rutas de esta máquina están en `CLAUDE.local.md`,
que no se versiona.

---

## Estilo

Comentarios y documentación en español, como el resto del proyecto. Los comentarios explican
**por qué**, no qué: casi todos los de este repositorio existen para que nadie repita una
medición equivocada.
