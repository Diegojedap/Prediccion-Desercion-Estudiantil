"""
Verifica que el repositorio no contenga datos institucionales ni credenciales.

Diseñado para ejecutarse igual en local y en integración continua. Termina con
código 1 si encuentra algo, de modo que un push que introduzca una fuga falle en
lugar de publicarse.

    python scripts/verificar_repo.py

Comprueba cinco invariantes:

  1. Ningún notebook conserva salidas ejecutadas. Las tablas impresas contenían
     identificaciones de estudiantes; es la vía de fuga más fácil de reintroducir.
  2. No hay archivos de datos ni modelos serializados versionados.
  3. No aparecen credenciales, IPs privadas ni cadenas de conexión.
  4. Los notebooks son JSON válido y su código compila.
  5. Ningún término prohibido adicional está presente.

El punto 5 es opcional y se configura con la variable de entorno
TERMINOS_PROHIBIDOS, una lista separada por comas. Los nombres reales de la
organización, sus esquemas y sus tablas no se escriben aquí por la misma razón
que no se escriben en los notebooks: este archivo es público. En local se toman
de scripts/mapeo_esquemas.local.json; en CI, de un secreto del repositorio.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

EXTENSIONES_PROHIBIDAS = {
    '.xlsx', '.xls', '.csv', '.parquet', '.db', '.sqlite3',
    '.pkl', '.joblib', '.h5', '.onnx',
}

PATRONES = {
    'IP privada': r'\b(?:10|127|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    'credencial en código': (
        r'(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\s*=\s*[\'"][^\'"]{3,}[\'"]'
    ),
    'cadena de conexión con credenciales': r'(?i)(uid|user\s*id|pwd|password)\s*=\s*[^;\'"\s]+;',
}


def archivos_versionados() -> list[Path]:
    """Lista los archivos que git tiene bajo control, con separador nulo.

    Los nombres de este repositorio contienen espacios; dividir por espacios
    partiría las rutas y dejaría archivos sin revisar.
    """
    salida = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=RAIZ, capture_output=True, text=True, check=True
    ).stdout
    return [RAIZ / p for p in salida.split('\0') if p]


def terminos_prohibidos() -> list[str]:
    """Términos adicionales a buscar, del entorno o del mapeo local."""
    del_entorno = os.getenv('TERMINOS_PROHIBIDOS', '')
    if del_entorno.strip():
        return [t.strip() for t in del_entorno.split(',') if t.strip()]

    mapeo = RAIZ / 'scripts' / 'mapeo_esquemas.local.json'
    if mapeo.exists():
        datos = json.loads(mapeo.read_text(encoding='utf-8'))
        return [k for k in datos if not k.startswith('_')]
    return []


def main() -> int:
    archivos = archivos_versionados()
    hallazgos: list[str] = []
    extra = terminos_prohibidos()

    print(f"Revisando {len(archivos)} archivos versionados\n")

    for ruta in archivos:
        rel = ruta.relative_to(RAIZ)

        # 2. Archivos de datos o modelos serializados
        if ruta.suffix.lower() in EXTENSIONES_PROHIBIDAS:
            hallazgos.append(f"{rel}: archivo de datos o modelo serializado versionado")
            continue

        if ruta.suffix == '.ipynb':
            try:
                nb = json.loads(ruta.read_text(encoding='utf-8'))
            except json.JSONDecodeError as e:
                hallazgos.append(f"{rel}: JSON inválido ({e})")
                continue

            for i, celda in enumerate(nb.get('cells', [])):
                # 1. Salidas ejecutadas
                if celda.get('outputs'):
                    hallazgos.append(f"{rel}: la celda {i} conserva salidas ejecutadas")
                if celda.get('execution_count'):
                    hallazgos.append(f"{rel}: la celda {i} conserva contador de ejecución")
                # 4. El código compila
                if celda.get('cell_type') == 'code':
                    fuente = ''.join(celda.get('source', []))
                    if fuente.strip():
                        try:
                            ast.parse(fuente)
                        except SyntaxError as e:
                            hallazgos.append(f"{rel}: la celda {i} no compila ({e.msg})")

        try:
            contenido = ruta.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue

        # 3. Credenciales e IPs
        for nombre, patron in PATRONES.items():
            m = re.search(patron, contenido)
            if m:
                hallazgos.append(f"{rel}: {nombre} -> {m.group(0)[:60]!r}")

        # 5. Términos prohibidos adicionales
        for termino in extra:
            if termino in contenido:
                hallazgos.append(f"{rel}: término prohibido -> {termino[:40]!r}")

    if not extra:
        print("AVISO: no hay términos prohibidos configurados, así que no se")
        print("       verifican nombres de la organización, esquemas ni tablas.")
        print("       Define TERMINOS_PROHIBIDOS o crea scripts/mapeo_esquemas.local.json\n")

    if hallazgos:
        print(f"FALLIDO: {len(hallazgos)} hallazgo(s)\n")
        for h in hallazgos:
            print(f"  [!] {h}")
        print("\nEl repositorio no debe publicarse en este estado.")
        return 1

    print(f"OK: sin salidas ejecutadas, sin archivos de datos, sin credenciales"
          f"{', sin términos prohibidos' if extra else ''}.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
