"""
Sincroniza los notebooks desde el directorio de trabajo hacia este repositorio,
eliminando todo dato institucional antes de publicarlos.

El directorio de trabajo vive en OneDrive junto a los .xlsx y .pkl, y sus
notebooks conservan las salidas ejecutadas. Esas salidas imprimen tablas con
identificaciones reales de estudiantes, género, programa y modalidad: dato
sensible bajo la Ley 1581 de 2012. Este script produce copias publicables.

Qué elimina:
  1. Todas las salidas ejecutadas y sus contadores de ejecución.
  2. El host, puerto y nombre de la base de datos escritos en el código,
     reemplazados por lectura de variables de entorno.
  3. El estado de widgets almacenado en la metadata del notebook.

Uso:
    python scripts/sanitizar_notebooks.py --origen "RUTA/AL/DIRECTORIO/DE/TRABAJO"

Sin argumentos usa ORIGEN_POR_DEFECTO. Falla con código 1 si detecta que algún
dato sensible sobrevivió a la limpieza.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# La ruta del directorio de trabajo se toma del entorno: contiene el nombre de la
# organización y no puede quedar escrita en un archivo público.
_origen_env = os.getenv("NOTEBOOKS_ORIGEN", "").strip()
ORIGEN_POR_DEFECTO = Path(_origen_env) if _origen_env else None

DESTINO = Path(__file__).resolve().parent.parent

# Celda de conexión saneada que sustituye a la que lleva credenciales fijas.
CELDA_CONEXION = """# Configura los detalles de la conexión desde variables de entorno.
# Copia .env.example a .env y define los valores. Nunca escribas credenciales aquí.
import os

host     = os.getenv('DB_HOST')
port     = os.getenv('DB_PORT', '1433')
database = os.getenv('DB_NAME')

if not all([host, database]):
    raise RuntimeError(
        "Faltan variables de entorno: define DB_HOST y DB_NAME. Ver .env.example"
    )

try:
    # Crea la cadena de conexión
    url = (
        'mssql+pyodbc://@{host}:{port}/{db}'
        '?trusted_connection=yes&driver=SQL+Server'
    ).format(host=host, port=port, db=database)

    # Crear conexion con base de datos
    engine = create_engine(url)
    print("Conexion a la base de datos realizada")

except Exception as e:
    print('Error:', e)
"""

# El nombre de la base se lee del entorno y nunca se escribe aquí: este archivo
# es público, así que incluirlo como literal filtraría justo el dato que elimina.
NOMBRE_BD = os.getenv("DB_NAME", "").strip()

# Mapeo de nombres institucionales (esquemas, tablas, variables) a equivalentes
# genéricos. Vive en un archivo aparte excluido por .gitignore, por la misma razón:
# las claves son los nombres reales y no pueden aparecer en un repositorio público.
RUTA_MAPEO = Path(__file__).resolve().parent / "mapeo_esquemas.local.json"


def cargar_mapeo() -> dict[str, str]:
    """Carga el mapeo de nombres institucionales, si está disponible."""
    if not RUTA_MAPEO.exists():
        return {}
    datos = json.loads(RUTA_MAPEO.read_text(encoding="utf-8"))
    # Se sustituyen primero los nombres más largos: un identificador puede estar
    # contenido en otro y el orden inverso lo dejaría a medio reemplazar.
    return {
        clave: valor
        for clave, valor in sorted(datos.items(), key=lambda kv: -len(kv[0]))
        if not clave.startswith("_")
    }


MAPEO = cargar_mapeo()

# Patrones que jamás deben llegar al repositorio público.
PATRONES_PROHIBIDOS = {
    "IP privada": r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "credencial en código": (
        r"(?i)(password|passwd|pwd|secret|api[_-]?key|token)\s*=\s*['\"][^'\"]+['\"]"
    ),
}

if NOMBRE_BD:
    PATRONES_PROHIBIDOS["nombre de la base de datos"] = re.escape(NOMBRE_BD)

for _nombre_real in MAPEO:
    PATRONES_PROHIBIDOS[f"nombre institucional ({_nombre_real[:24]}…)"] = re.escape(
        _nombre_real
    )

RE_IP_CONEXION = re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def genericizar(texto: str) -> str:
    """Sustituye esquemas, tablas y variables institucionales por nombres neutros."""
    for nombre_real, generico in MAPEO.items():
        texto = texto.replace(nombre_real, generico)
    return texto


def sanitizar(nb: dict) -> tuple[dict, int, int]:
    """Devuelve el notebook limpio junto al número de salidas y celdas alteradas."""
    salidas_eliminadas = 0
    celdas_saneadas = 0

    for celda in nb.get("cells", []):
        fuente = "".join(celda.get("source", []))

        if celda.get("cell_type") != "code":
            # Las celdas markdown no ejecutan nada, pero sí citan nombres internos.
            nueva = genericizar(fuente)
            if nueva != fuente:
                celda["source"] = nueva.splitlines(keepends=True)
                celdas_saneadas += 1
            continue

        salidas_eliminadas += len(celda.get("outputs", []))
        celda["outputs"] = []
        celda["execution_count"] = None

        metadata = celda.get("metadata", {})
        for clave in ("execution", "collapsed", "scrolled"):
            metadata.pop(clave, None)
        celda["metadata"] = metadata

        if RE_IP_CONEXION.search(fuente) and "create_engine" in fuente:
            celda["source"] = CELDA_CONEXION.splitlines(keepends=True)
            celdas_saneadas += 1
            continue

        nueva = fuente
        if NOMBRE_BD and f"{NOMBRE_BD}." in nueva:
            # La base ya viene en la cadena de conexión; el prefijo es redundante.
            nueva = nueva.replace(f"{NOMBRE_BD}.", "")
        nueva = genericizar(nueva)

        if nueva != fuente:
            celda["source"] = nueva.splitlines(keepends=True)
            celdas_saneadas += 1

    nb.get("metadata", {}).pop("widgets", None)
    return nb, salidas_eliminadas, celdas_saneadas


def verificar(ruta: Path) -> list[str]:
    """Busca datos sensibles en un notebook ya sanitizado."""
    contenido = ruta.read_text(encoding="utf-8")
    return [
        nombre
        for nombre, patron in PATRONES_PROHIBIDOS.items()
        if re.search(patron, contenido)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origen",
        type=Path,
        default=ORIGEN_POR_DEFECTO,
        help=(
            "Directorio de trabajo con los notebooks ejecutados. "
            "Si se omite, se toma de la variable de entorno NOTEBOOKS_ORIGEN."
        ),
    )
    args = parser.parse_args()

    if args.origen is None:
        print(
            "ERROR: indica el directorio de trabajo con --origen, o define\n"
            "       la variable de entorno NOTEBOOKS_ORIGEN."
        )
        return 1

    if not args.origen.is_dir():
        print(f"ERROR: no existe el directorio de origen: {args.origen}")
        return 1

    notebooks = sorted(args.origen.glob("*.ipynb"))
    if not notebooks:
        print(f"ERROR: no se encontraron notebooks en {args.origen}")
        return 1

    if not NOMBRE_BD:
        print(
            "AVISO: DB_NAME no está definida, así que no se puede detectar ni eliminar\n"
            "       el nombre de la base de datos. Defínela antes de publicar:\n"
            "         export DB_NAME=...   (Linux/macOS)\n"
            "         $env:DB_NAME='...'   (PowerShell)\n"
        )

    if not MAPEO:
        print(
            f"AVISO: no se encontró {RUTA_MAPEO.name}, así que los nombres de esquemas\n"
            "       y tablas internas se publicarán tal cual. Copia\n"
            "       mapeo_esquemas.example.json y complétalo antes de publicar.\n"
        )

    print(f"{'notebook':<26}{'salidas':>9}{'celdas':>8}{'KB':>7}  estado")
    fallos = 0

    for origen in notebooks:
        nb = json.loads(origen.read_text(encoding="utf-8"))
        nb, salidas, celdas = sanitizar(nb)

        destino = DESTINO / origen.name
        with destino.open("w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")

        fugas = verificar(destino)
        if fugas:
            fallos += 1
        estado = "OK" if not fugas else f"FUGA: {', '.join(fugas)}"
        kb = destino.stat().st_size // 1024
        print(f"{origen.name:<26}{salidas:>9}{celdas:>8}{kb:>7}  {estado}")

    if fallos:
        print(f"\nABORTADO: {fallos} notebook(s) conservan datos sensibles. No publicar.")
        return 1

    print(f"\n{len(notebooks)} notebooks sanitizados y listos para publicar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
