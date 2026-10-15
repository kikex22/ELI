"""Inventario de modelos de Ollama: inspeccion y administracion (descargar de VRAM / eliminar)."""
from __future__ import annotations

from . import inspector
from .bench.store import Store
from .ollama import Ollama, OllamaError


class ModelNotFound(Exception):
    pass


class DeleteRefused(Exception):
    pass


def inspect_installed(ollama: Ollama, store: Store | None, hardware: dict, names: list[str] | None = None,
                      extra_stores: tuple = ()) -> list[dict]:
    """Un dict de `inspector.inspect_model` por modelo instalado. Si /api/show falla para uno, se informa sin romper el resto."""
    rows = ollama.models()
    observed = store.observed_vram() if store else {}
    for other in extra_stores:  # p. ej. las evaluaciones de embeddings
        for name, info in other.observed_vram().items():
            if name in observed:
                observed[name]["runs"] += info["runs"]
                for key in ("vram_used_mib", "vram_gpu_peak_mib", "model_vram_bytes"):
                    if info.get(key) is not None and (observed[name][key] is None or info[key] > observed[name][key]):
                        observed[name][key] = info[key]
            else:
                observed[name] = info
    out = []
    for row in rows:
        if names is not None and row["name"] not in names:
            continue
        show, error = None, None
        try:
            show = ollama.show(row["name"])
        except OllamaError as exc:
            error = str(exc)
        info = inspector.inspect_model(show, row, None, observed.get(row["name"]), hardware)
        if error:
            info["inspect_error"] = error
        out.append(info)
    return out


def delete_preview(ollama: Ollama, name: str) -> dict:
    """Que se borraria exactamente y cuanto espacio se recuperaria (sin sumar pesos compartidos)."""
    rows = ollama.models()
    target = next((r for r in rows if r["name"] == name), None)
    if target is None:
        raise ModelNotFound(name)
    digest = target.get("digest")
    sharing = [r["name"] for r in rows if r["name"] != name and digest and r.get("digest") == digest]
    if not digest:
        reclaim = {"kind": "desconocido", "bytes": None, "note": "Ollama no informo el digest; no se puede saber si comparte pesos con otros modelos."}
    elif sharing:
        reclaim = {"kind": "ninguno", "bytes": 0,
                   "note": "Comparte exactamente los mismos pesos con " + ", ".join(sharing)
                           + ": borrar solo este nombre no libera espacio hasta borrar tambien los demas."}
    else:
        reclaim = {"kind": "estimacion", "bytes": target.get("disk_bytes"),
                   "note": "Estimacion: tamano del modelo. Si comparte capas parciales con otro modelo (una base comun), el espacio liberado sera menor; "
                           "la API de Ollama no permite calcularlo con exactitud."}
    return {"name": name, "digest": digest, "disk_bytes": target.get("disk_bytes"), "loaded": target["loaded"],
            "shares_weights_with": sharing, "reclaimable": reclaim,
            "note": "Eliminar borra el modelo del disco. No es lo mismo que descargarlo de la VRAM (eso lo conserva en disco)."}


def delete_model(ollama: Ollama, name: str, confirm: str) -> dict:
    """Elimina UN modelo. Exige que `confirm` sea el nombre exacto y que no este cargado en memoria."""
    if not isinstance(name, str) or not name.strip() or any(c in name for c in "*?,\n"):
        raise DeleteRefused("Nombre de modelo invalido: se elimina un unico modelo por su nombre exacto.")
    if confirm != name:
        raise DeleteRefused("Confirmacion incorrecta: escribe el nombre exacto del modelo para eliminarlo.")
    preview = delete_preview(ollama, name)  # lanza ModelNotFound si no esta instalado
    if preview["loaded"]:
        raise DeleteRefused("El modelo esta cargado en la VRAM. Descargalo primero (eso no lo borra) y vuelve a intentarlo.")
    ollama.delete(name)
    return {"deleted": name, "preview": preview}
