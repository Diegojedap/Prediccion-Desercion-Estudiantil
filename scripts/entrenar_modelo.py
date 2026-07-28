"""
Entrena el modelo 1.8 y exporta el artefacto publicable de `modelo/`.

Existe para que el modelo publicado sea reproducible, no solo sus métricas.

    python scripts/entrenar_modelo.py

Requiere `DB_HOST`, `DB_NAME` y `STATUS_DESERCION` en el entorno (ver
`.env.example`), y los nombres de esquema y tabla de tu instalación en lugar de
los genéricos de las consultas.

Exporta en el formato de texto de LightGBM en vez de pickle: un pickle ejecuta
código arbitrario al deserializarse, mientras que el texto es inspeccionable, no
ejecuta nada al cargarse y no contiene registros individuales, solo los umbrales
aprendidos. Los transformadores se exportan como JSON legible por la misma razón.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sqlalchemy import create_engine, text

DESTINO = Path(__file__).resolve().parent.parent / 'modelo'

# El target se resuelve al comprobar que el estudiante no volvió a matricularse,
# así que los periodos más recientes tienen etiquetas a medio resolver. Evaluar
# contra el último disponible da un AUC de 0.59 que no refleja el desempeño sino
# la inmadurez de las etiquetas.
MARGEN_MADURACION = 2

# Por debajo de esta cobertura una variable no aporta señal utilizable. El
# control es necesario: una tabla de origen vacía no produce ningún error,
# el imputador descarta la columna y el modelo entrena con menos predictores
# de los declarados sin que nada lo advierta.
COBERTURA_MINIMA = 0.05

NUMERICAS = ['SEMESTRE_SINU', 'MATERIAS_INSCRITAS', 'MATERIAS_APROBADAS',
             'PORCENTAJE_APROBACION', 'TOTAL', 'FLAG_NO_APROBO_NADA']
CATEGORICAS = ['TIPO_SALTO', 'MODALIDAD', 'GENERO', 'RANGO_EDAD', 'RANGO_SALARIO',
               'ESTA_TRABAJANDO', 'METODO_FINANCIAMIENTO', 'ZONA_RESIDENCIA',
               'REGIMEN_SISTEMA_SALUD']

CONSULTAS = {
    'historial': '''
        SELECT Identificacion, Periodo, Status, Tipo_Salto, Modalidad, Semestre_SINU,
               año AS ANIO, Genero, RANGO_EDAD, RANGO_SALARIO, ESTA_TRABAJANDO,
               METODO_FINANCIAMIENTO, ZONA_RESIDENCIA, REGIMEN_SISTEMA_SALUD
        FROM academico.historial_academico
    ''',
    'materias': '''
        SELECT IDENTIFICACION AS Identificacion, COD_PERIODO AS Periodo,
               [MATERIAS INSCRITAS] AS MATERIAS_INSCRITAS,
               [MATERIAS APROBADAS] AS MATERIAS_APROBADAS,
               Porcentaje_aprobacion AS PORCENTAJE_APROBACION
        FROM academico.aprobacion_materias
    ''',
    'cartera': 'SELECT Identificacion, Periodo, TOTAL FROM financiera.cartera',
}


def conectar():
    host, puerto, base = os.getenv('DB_HOST'), os.getenv('DB_PORT', '1433'), os.getenv('DB_NAME')
    if not all([host, base]):
        raise RuntimeError('Define DB_HOST y DB_NAME en el entorno. Ver .env.example')
    return create_engine(
        f'mssql+pyodbc://@{host}:{puerto}/{base}?trusted_connection=yes'
        '&driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes',
        connect_args={'timeout': 300})


def cargar(engine) -> pd.DataFrame:
    """Extrae e integra las fuentes, verificando que el merge no infle filas."""
    with engine.connect() as cn:
        datos = {k: pd.read_sql_query(text(sql), cn) for k, sql in CONSULTAS.items()}

    df = datos['historial']
    filas_iniciales = len(df)
    for nombre in ('materias', 'cartera'):
        aux = datos[nombre].drop_duplicates(subset=['Identificacion', 'Periodo'], keep='last')
        df = df.merge(aux, on=['Identificacion', 'Periodo'], how='left')

    if len(df) != filas_iniciales:
        raise RuntimeError(
            f'El merge cambió el número de filas: {filas_iniciales} -> {len(df)}. '
            'Revisa la deduplicación de las fuentes.')

    df.columns = [c.strip().upper() for c in df.columns]
    if df.columns.duplicated().any():
        raise RuntimeError(f'Columnas duplicadas: {df.columns[df.columns.duplicated()].tolist()}')
    return df


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    valor = os.getenv('STATUS_DESERCION', '').strip().lower()
    if not valor:
        raise RuntimeError('Define STATUS_DESERCION en el entorno. Ver .env.example')

    df['TARGET'] = (df['STATUS'].astype(str).str.strip().str.lower() == valor).astype(int)
    if df['TARGET'].sum() == 0:
        presentes = df['STATUS'].astype(str).str.strip().str.lower().unique()[:10]
        raise RuntimeError(
            f'Ninguna fila coincide con STATUS_DESERCION={valor!r}. Valores presentes: {presentes}')

    # Separa "aprobó 0 %" de "no hay dato": la imputación por mediana los
    # colapsaría pese a ser situaciones opuestas.
    df['FLAG_NO_APROBO_NADA'] = np.where(df['PORCENTAJE_APROBACION'] == 0, 1, 0)
    df['PORCENTAJE_APROBACION'] = df['PORCENTAJE_APROBACION'].replace(0, np.nan)

    df['ANIO'] = pd.to_numeric(df['ANIO'], errors='coerce')
    df = df.dropna(subset=['ANIO'])
    df['ANIO'] = df['ANIO'].astype(int)

    for c in NUMERICAS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CATEGORICAS:
        df[c] = df[c].astype(str)
    return df


def main() -> int:
    DESTINO.mkdir(exist_ok=True)
    print('Extrayendo de la base...')
    df = preparar(cargar(conectar()))
    print(f'  {len(df):,} filas integradas')

    cobertura = df[NUMERICAS + CATEGORICAS].notna().mean()
    descartadas = cobertura[cobertura < COBERTURA_MINIMA].index.tolist()
    numericas = [c for c in NUMERICAS if c not in descartadas]
    categoricas = [c for c in CATEGORICAS if c not in descartadas]
    features = numericas + categoricas
    if descartadas:
        print(f'  descartadas por cobertura nula: {descartadas}')
    print(f'  {len(features)} predictores efectivos de {len(NUMERICAS + CATEGORICAS)} declarados')

    anio_test = int(df['ANIO'].max()) - MARGEN_MADURACION
    anio_train = anio_test - 1
    train = df[df['ANIO'] == anio_train]
    test = df[df['ANIO'] == anio_test]
    print(f'  entrena {anio_train}, evalúa {anio_test} '
          f'(se descartan los {MARGEN_MADURACION} más recientes por inmadurez)')

    peso = (train['TARGET'] == 0).sum() / max((train['TARGET'] == 1).sum(), 1)
    modelo = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imputar', SimpleImputer(strategy='median')),
                              ('escalar', StandardScaler())]), numericas),
            ('cat', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                              ('codificar', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                           unknown_value=-1))]), categoricas)])),
        ('clf', LGBMClassifier(n_estimators=481, num_leaves=32, learning_rate=0.0244,
                               subsample=0.9122, random_state=42, n_jobs=-1,
                               verbose=-1, scale_pos_weight=peso))])
    modelo.fit(train[features], train['TARGET'])

    proba = modelo.predict_proba(test[features])[:, 1]
    verdad = test['TARGET'].to_numpy()
    orden = np.argsort(-proba)
    tasa_base = verdad.mean()
    tramos = {f'P@{p}%': round(float(verdad[orden[:int(len(verdad) * p / 100)]].mean()), 4)
              for p in (1, 5, 10, 20)}
    auc = float(roc_auc_score(test['TARGET'], proba))
    print(f'\n  AUC {auc:.4f} | {tramos}')

    # ── Exportación ─────────────────────────────────────────────────────────
    modelo.named_steps['clf'].booster_.save_model(str(DESTINO / 'modelo_18.txt'))

    prep = modelo.named_steps['prep']
    num_pipe = prep.named_transformers_['num']
    cat_pipe = prep.named_transformers_['cat']
    transformadores = {
        '_nota': ('El orden de cada lista de categorías es AUTORITATIVO: el codificador '
                  'ordinal asigna el índice por posición. No reordenar.'),
        'numericas': {
            'orden': numericas,
            'mediana_imputacion': [round(float(x), 6) for x in num_pipe.named_steps['imputar'].statistics_],
            'media': [round(float(x), 6) for x in num_pipe.named_steps['escalar'].mean_],
            'escala': [round(float(x), 6) for x in num_pipe.named_steps['escalar'].scale_],
        },
        'categoricas': {
            'orden': categoricas,
            'moda_imputacion': [str(x) for x in cat_pipe.named_steps['imputar'].statistics_],
            'categorias': {c: [str(x) for x in cats] for c, cats
                           in zip(categoricas, cat_pipe.named_steps['codificar'].categories_)},
            'valor_desconocido': -1,
        },
        'orden_final_de_columnas': features,
    }
    (DESTINO / 'transformadores.json').write_text(
        json.dumps(transformadores, indent=2, ensure_ascii=False), encoding='utf-8')

    metadatos = {
        'version': '1.8',
        'entrenado_utc': datetime.now(timezone.utc).isoformat(),
        'algoritmo': 'LightGBM (LGBMClassifier)',
        'hiperparametros': {'n_estimators': 481, 'num_leaves': 32,
                            'learning_rate': 0.0244, 'subsample': 0.9122},
        'periodo_entrenamiento': anio_train,
        'periodo_evaluacion': anio_test,
        'predictores_declarados': len(NUMERICAS + CATEGORICAS),
        'predictores_efectivos': len(features),
        'descartados_por_cobertura': descartadas,
        'features': features,
        'metricas': {'auc_roc': round(auc, 4), **tramos,
                     'lift_top1pct': round(float(tramos['P@1%'] / tasa_base), 2)},
        'modo_operacion': 'ranking por capacidad; NO usar umbral fijo de probabilidad',
        'caducidad': 'reentrenar cada periodo academico',
        'nota': 'Los conteos de poblacion y la tasa base se omiten deliberadamente.',
    }
    (DESTINO / 'metadatos.json').write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\nArtefacto escrito en {DESTINO.name}/')
    for nombre in ('modelo_18.txt', 'transformadores.json', 'metadatos.json'):
        print(f'  {nombre:<24} {(DESTINO / nombre).stat().st_size / 1024:>8.1f} KB')
    print('\nRevisa modelo/MODEL_CARD.md: sus métricas y la auditoría de sesgo '
          'corresponden a la extracción anterior.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
