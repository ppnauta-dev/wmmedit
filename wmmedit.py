import xml.etree.ElementTree as ET
import os
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import tempfile
import traceback
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
from PIL import Image, ImageTk

# ======================================================================
#  FFMPEG: localizacion, encoder y ejecucion con progreso
# ======================================================================
_CACHE = {"ffmpeg": None, "qsv": None, "sonda": {}}
ESCALADO = "lanczos"   # filtro de escalado; "bicubic" es bastante mas rapido (en un Atom se nota) con diferencia minima
HILOS_CLIPS = 2      # clips que se procesan a la vez (arranque de ffmpeg/QSV se solapa); 1 = uno por uno
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_RE_PROGRESO = re.compile(
    r"^(frame|fps|stream_\d+_\d+_q|bitrate|total_size|out_time_us|out_time_ms|"
    r"out_time|dup_frames|drop_frames|speed|progress)="
)


def obtener_ffmpeg():
    """Devuelve la ruta de ffmpeg (la descarga con static_ffmpeg si hace falta)."""
    if _CACHE["ffmpeg"]:
        return _CACHE["ffmpeg"]
    exe = None
    try:
        import static_ffmpeg
        try:
            static_ffmpeg.add_paths()
        except Exception:
            pass
        exe = shutil.which("ffmpeg")
        if not exe:
            exe, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except ImportError:
        exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("No se encontró ffmpeg. Instala 'static-ffmpeg' (pip install static-ffmpeg).")
    _CACHE["ffmpeg"] = exe
    return exe


def qsv_disponible(ffmpeg_exe):
    """True si Intel Quick Sync (h264_qsv) funciona en este equipo."""
    if _CACHE["qsv"] is None:
        prueba = [ffmpeg_exe, "-nostdin", "-hide_banner", "-f", "lavfi",
                  "-i", "testsrc=size=1280x720:rate=30", "-t", "1",
                  "-c:v", "h264_qsv", "-f", "null", "-"]
        try:
            r = subprocess.run(prueba, stdin=subprocess.DEVNULL, capture_output=True,
                               creationflags=_NO_WINDOW, timeout=30)
            _CACHE["qsv"] = (r.returncode == 0)
        except Exception:
            _CACHE["qsv"] = False
    return _CACHE["qsv"]


# Tamano objetivo para 2 horas de video: 240p=240 MB, 360p=360, 480p=480, 720p=720, 1080p=1 GB, 1440p=1,4 GB, 2160p=~2,7 GB.
# (alto, kbps de video, kbps de audio); el tamano crece en proporcion a la duracion.
OBJETIVO_BITRATE = [(240, 203, 64), (360, 320, 80), (480, 437, 96),
                    (720, 672, 128), (1080, 983, 128), (1440, 1396, 160), (2160, 2780, 192)]


FACTOR_BITRATE = 1.5     # multiplica el bitrate de VIDEO de la tabla (1.0 = tamanos de arriba); mas = mejor en escenas movidas


def bitrates_objetivo(alto):
    """(kbps de video, kbps de audio) para una altura, interpolando entre las alturas de la tabla."""
    t = OBJETIVO_BITRATE
    if alto <= t[0][0]:
        v, a = t[0][1], t[0][2]
    elif alto > t[-1][0]:
        v, a = t[-1][1], t[-1][2]
    else:
        for (h0, v0, a0), (h1, v1, a1) in zip(t, t[1:]):
            if alto <= h1:
                f = (alto - h0) / (h1 - h0)
                v, a = v0 + (v1 - v0) * f, a0 + (a1 - a0) * f
                break
    return int(round(v * FACTOR_BITRATE)), int(round(a))


# Filtros de imagen (ffmpeg) que se aplican a todo el video, ANTES de poner los captions del proyecto.
FILTROS = {
    "Sin filtro": "",
    "Nitidez suave": "unsharp=5:5:0.6:5:5:0.0",
    "Nitidez media": "unsharp=5:5:1.0:5:5:0.0",
    "Colores vivos": "eq=saturation=1.25:contrast=1.05",
    "Más contraste": "eq=contrast=1.15:gamma=0.95",
    "Anime vivo (color + nitidez)": "eq=saturation=1.2:contrast=1.04,unsharp=5:5:0.5:5:5:0.0",
    "Reducir ruido / grano": "hqdn3d=2:1.5:3:2.5",
    "Blanco y negro": "hue=s=0",
    "Sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:0",
}


# Zona de los subtitulos pegados al video (fraccion del cuadro) para la casilla "Quitar subtitulos"
SUBS_Y = 0.80           # donde empieza la banda (desde arriba)
SUBS_ALTO = 0.16        # alto de la banda
SUBS_MARGEN_X = 0.06    # margen a cada lado
QSV_LOOKAHEAD = True     # Quick Sync mira ~40 cuadros adelante para repartir mejor los bits (False = apagado)


def qsv_lookahead_ok(ffmpeg_exe):
    """True si este Quick Sync acepta look-ahead con bitrate fijo (se prueba con 1 s de video)."""
    if "qsv_la" not in _CACHE:
        prueba = [ffmpeg_exe, "-nostdin", "-hide_banner", "-f", "lavfi",
                  "-i", "testsrc2=size=1280x720:rate=30", "-t", "2", "-pix_fmt", "nv12",
                  "-c:v", "h264_qsv", "-b:v", "1000k", "-maxrate", "2500k", "-bufsize", "2500k",
                  "-look_ahead", "1", "-look_ahead_depth", "40", "-f", "null", "-"]
        try:
            r = subprocess.run(prueba, stdin=subprocess.DEVNULL, capture_output=True,
                               creationflags=_NO_WINDOW, timeout=60)
            _CACHE["qsv_la"] = (r.returncode == 0)
        except Exception:
            _CACHE["qsv_la"] = False
    return _CACHE["qsv_la"]


def obtener_encoder(ffmpeg_exe, nivel="media", kbps=None):
    """Devuelve (argumentos_encoder, formato_de_pixel).
    nivel: 'baja' (borrador/preview), 'media' (720p) o 'alta' (subir a YouTube)."""
    if kbps:       # tamano predecible: bitrate medio fijo (con tope de picos)
        # Bitrate medio fijo con picos moderados (2,5x el promedio, ventana de ~2,5 s). Un buffer
        # mas grande deja que el encoder gaste de mas y despues tenga que comprimir a la fuerza.
        preset = {"baja": "veryfast", "media": "medium", "alta": "medium"}[nivel]
        tope = ["-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 2.5)}k", "-bufsize", f"{int(kbps * 2.5)}k"]
        if qsv_disponible(ffmpeg_exe):
            la = (["-look_ahead", "1", "-look_ahead_depth", "40"]
                  if (QSV_LOOKAHEAD and nivel != "baja" and qsv_lookahead_ok(ffmpeg_exe)) else [])
            return (["-c:v", "h264_qsv", "-preset", preset] + tope + la, "nv12")
        return (["-c:v", "libx264", "-preset", {"baja": "ultrafast", "media": "veryfast",
                                                "alta": "veryfast"}[nivel]] + tope, "yuv420p")
    if qsv_disponible(ffmpeg_exe):
        q, preset = {"baja": ("34", "veryfast"), "media": ("26", "faster"),
                     "alta": ("22", "medium")}[nivel]
        return (["-c:v", "h264_qsv", "-global_quality", q, "-preset", preset], "nv12")
    crf, preset = {"baja": ("32", "ultrafast"), "media": ("23", "superfast"),
                   "alta": ("20", "veryfast")}[nivel]
    return (["-c:v", "libx264", "-preset", preset, "-crf", crf], "yuv420p")


def ejecutar_ffmpeg(cmd, duracion=None, on_frac=None):
    """Ejecuta ffmpeg sin bloquear por stdin, leyendo el progreso en vivo.
    on_frac recibe un valor entre 0.0 y 1.0."""
    cmd = [cmd[0], "-nostdin", "-hide_banner", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats"] + cmd[1:]
    p = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=_NO_WINDOW,
    )
    errores = []
    for linea in p.stdout:
        linea = linea.strip()
        if not linea:
            continue
        if linea.startswith("out_time_us="):
            if duracion and on_frac:
                try:
                    frac = min(int(linea.split("=")[1]) / 1e6 / duracion, 1.0)
                except ValueError:
                    continue  # "N/A" al comienzo
                try:
                    on_frac(frac)
                except BaseException:      # el que llama pide detener (p. ej. no cabe en el disco)
                    p.kill()
                    p.wait()
                    raise
        elif not _RE_PROGRESO.match(linea):
            errores.append(linea)
            errores = errores[-30:]
    p.wait()
    if p.returncode != 0:
        msg = "\n".join(errores)
        print(f"ERROR FFMPEG:\n{msg}")
        if "No space left" in msg:
            raise RuntimeError(
                "Se acabó el espacio en disco mientras se escribían los archivos.\n"
                "Los temporales se crean en la misma carpeta donde guardas el video; "
                "elige una unidad con más espacio libre.\n\n" + msg)
        raise RuntimeError(f"FFmpeg falló:\n{msg}")


def sondear(ffmpeg_exe, ruta):
    """Devuelve {'audio': bool, 'fps': (texto, valor), 'w', 'h', 'dur'} del archivo."""
    cache = _CACHE["sonda"]
    if ruta in cache:
        return cache[ruta]
    r = subprocess.run([ffmpeg_exe, "-nostdin", "-hide_banner", "-i", ruta],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    texto = r.stderr
    audio = "Audio:" in texto
    m = re.search(r"Video:.*?(?<![\dx])(\d{2,5})x(\d{2,5})(?![\dx])", texto)
    ancho, alto = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", texto)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    m = re.search(r"Video:.*?([\d.]+) fps", texto)
    fps = float(m.group(1)) if m else 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    # ffmpeg muestra 23.98 / 29.97 / 59.94 para las tasas NTSC (x/1001) y 24 / 30 / 60 para las exactas
    if 23.95 <= fps <= 23.99:
        fps_info = ("24000/1001", 24000 / 1001)
    elif 29.95 <= fps <= 29.99:
        fps_info = ("30000/1001", 30000 / 1001)
    elif 59.9 <= fps <= 59.98:
        fps_info = ("60000/1001", 60000 / 1001)
    else:
        n = max(1, round(fps))
        fps_info = (str(n), float(n))
    m = re.search(r"SAR (\d+):(\d+)", texto)
    sar = int(m.group(1)) / int(m.group(2)) if m and int(m.group(2)) else 1.0
    cache[ruta] = {"audio": audio, "fps": fps_info, "w": ancho, "h": alto, "dur": dur, "sar": sar}
    return cache[ruta]


# ======================================================================
#  Lectura de proyectos .wlmp
# ======================================================================
def _volumen_clip(elem):
    """Volumen del clip segun el .wlmp (1.0 = normal). 0.0 si esta silenciado.
    Movie Maker lo guarda dentro de <BoundProperties>:
        <BoundPropertyBool  Name="Mute"   Value="false" />
        <BoundPropertyFloat Name="Volume" Value="0.32" />
    (tambien se aceptan atributos sueltos 'volume'/'mute' por compatibilidad)."""
    mute, vol = False, 1.0
    bp = elem.find("BoundProperties")
    if bp is not None:
        for prop in bp:
            nombre = (prop.get("Name") or "").lower()
            valor = (prop.get("Value") or "").strip()
            if nombre == "mute" and valor.lower() in ("true", "1"):
                mute = True
            elif nombre == "volume":
                try:
                    vol = max(0.0, float(valor))
                except ValueError:
                    pass
    for k, v in elem.attrib.items():
        kl = k.lower()
        if "mute" in kl and str(v).strip().lower() in ("true", "1"):
            mute = True
        elif kl in ("volume", "audiovolume", "volumen"):
            try:
                vol = max(0.0, float(v))
            except ValueError:
                pass
    return 0.0 if mute else vol


def _props(bp):
    """<BoundProperties> -> dict {Nombre: valor}. Los conjuntos (...Set) dan una lista."""
    d = {}
    if bp is None:
        return d
    for p in bp:
        n = p.get("Name")
        if n is None:
            continue
        d[n] = [e.get("Value") for e in p] if p.tag.endswith("Set") else p.get("Value")
    return d


def _num(v, defecto=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return defecto


def _leer_titulo(tc):
    """Lee un <TitleClip> (caption / titulo de Movie Maker) a un dict simple."""
    te = tc.find("Effects/TextEffect")
    pe = _props(te.find("BoundProperties")) if te is not None else {}
    pc = _props(tc.find("BoundProperties"))
    color = [_num(v, 1.0) for v in pe.get("color", [1, 1, 1])][:3] or [1.0, 1.0, 1.0]
    borde = [_num(v, 0.0) for v in pe.get("outlineColor", [0, 0, 0])][:3] or [0.0, 0.0, 0.0]
    pos = [_num(v) for v in pe.get("position", [0, 0, 0])]
    largo = [_num(v) for v in pe.get("length", [])]
    t = {
        "gap": _num(tc.get("gapBefore")),
        "duracion": _num(tc.get("duration")),
        "texto": "\n".join(pe.get("string", [])),
        "familia": (pe.get("family") or ["Arial"])[0],
        "estilo": pe.get("style", "Regular") or "Regular",
        "color": color + [1.0] * (3 - len(color)),
        "borde_color": borde + [0.0] * (3 - len(borde)),
        "borde_idx": int(_num(pe.get("outlineSizeIndex"), 0)),
        "tam": _num(pe.get("size"), 0.4),
        "pos": (pos + [0.0, 0.0])[:2],
        "largo": largo,                                  # extension del texto (ancho, alto)
        "max_extent": _num(pe.get("maxExtent"), 5.0) or 5.0,
        "transp_texto": min(1.0, max(0.0, _num(pe.get("transparency"), 0.0))),
        "transp_fondo": min(1.0, max(0.0, _num(pc.get("transparency"), 1.0))),
        "justif": ((pe.get("justify") or ["MIDDLE"])[0] or "MIDDLE").upper(),
        "animado": (te is not None and te.get("TextScriptId") not in (None, "0")),
        "efecto": (te.get("effectTemplateID") or "") if te is not None else "",
    }
    # efectos de entrada que este editor reproduce (de forma aproximada)
    t["spin"] = "SpinIn" in t["efecto"]
    return t


def leer_wlmp(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
        contenido = f.read()

    indice_xml = contenido.find("<")
    if indice_xml != -1:
        contenido = contenido[indice_xml:]

    try:
        root = ET.fromstring(contenido)
    except ET.ParseError as e:
        raise ValueError(f"El archivo XML tiene un error de formato: {e}")

    proyecto = {"archivo": ruta, "nombre": os.path.basename(ruta), "pistas": [], "medios": [],
                "titulos": []}

    medios_dict = {}
    for media in root.findall(".//MediaItem"):
        m_id = media.get("id")
        ruta_media = media.get("filePath", "")
        try:
            dur_medio = float(media.get("duration", 0) or 0)
        except ValueError:
            dur_medio = 0.0
        medios_dict[m_id] = {"ruta": ruta_media, "duracion": dur_medio,
                             "tipo": "Video" if media.get("mediaItemType") == "1" else "Audio"}
        if ruta_media not in proyecto["medios"]:
            proyecto["medios"].append(ruta_media)

    # Relacion de aspecto del proyecto (Movie Maker: 1.777 = 16:9, 1.333 = 4:3)
    proyecto["aspecto"] = 16 / 9
    for conj in root.findall("./BoundProperties/BoundPropertyFloatSet"):
        if conj.get("Name") == "AspectRatio":
            el = conj.find("BoundPropertyFloatElement")
            try:
                proyecto["aspecto"] = float(el.get("Value"))
            except (AttributeError, TypeError, ValueError):
                pass

    # Lo que este editor todavia NO aplica (para avisarlo en pantalla)
    n_trans = sum(len(c.findall("Transitions/*")) for c in root.iter()
                  if c.tag in ("VideoClip", "AudioClip"))
    n_fade = sum(1 for c in root.iter() if c.tag in ("VideoClip", "AudioClip")
                 for e in c.findall("Effects/AudioEffect"))
    n_otros = sum(1 for c in root.iter() if c.tag in ("VideoClip", "AudioClip")
                  for e in c.findall("Effects/*") if e.tag != "AudioEffect")
    proyecto["no_soportado"] = [t for t in (
        f"{n_trans} transición(es)" if n_trans else "",
        f"{n_fade} fundido(s) de audio" if n_fade else "",
        f"{n_otros} efecto(s) de video" if n_otros else "") if t]

    extents_dict = {}
    proyecto["debug"] = []
    for tag in ["VideoClip", "AudioClip"]:
        valores = {}   # atributo -> valores distintos que toma, para diagnosticar
        for n, clip in enumerate(root.findall(f".//{tag}")):
            e_id = clip.get("extentID")
            m_id = clip.get("mediaItemID")
            t_in = float(clip.get("inTime", 0))
            t_out = float(clip.get("outTime", 0))
            # Movie Maker guarda outTime="0" cuando el clip NO esta recortado:
            # significa "hasta el final del archivo".
            dur_medio = medios_dict.get(m_id, {}).get("duracion", 0.0)
            if t_out <= t_in and dur_medio > t_in:
                t_out = dur_medio
            extents_dict[e_id] = {
                "tipo": "Video" if tag == "VideoClip" else "Audio",
                "ruta": medios_dict.get(m_id, {}).get("ruta", ""),
                "gapBefore": float(clip.get("gapBefore", 0)),
                "inTime": t_in,
                "outTime": t_out,
                "speed": float(clip.get("speed", 1)),
                "volumen": _volumen_clip(clip),
            }
            for k, v in clip.attrib.items():
                if k not in ("extentID", "mediaItemID", "gapBefore", "inTime", "outTime"):
                    valores.setdefault(k, set()).add(v)
            bp = clip.find("BoundProperties")
            if bp is not None:
                for prop in bp:
                    nom, val = prop.get("Name"), prop.get("Value", "")
                    try:
                        val = f"{float(val):.2f}"
                    except ValueError:
                        pass
                    if nom:
                        valores.setdefault(nom, set()).add(val)
        if valores:
            resumen = ", ".join(f"{k}={{{', '.join(sorted(vs)[:4])}}}" for k, vs in valores.items())
            proyecto["debug"].append(f"{tag} atributos: {resumen}")

    titulos_dict = {tc.get("extentID"): _leer_titulo(tc) for tc in root.findall(".//TitleClip")}
    n_pantalla = n_animados = 0

    for selector in root.findall(".//ExtentSelector"):
        es_principal = selector.get("primaryTrack") == "true"
        pista = {"id": selector.get("extentID"),
                 "tipo": "Principal" if es_principal else "Secundaria",
                 "clips": []}
        cursor_t = 0.0   # linea de tiempo de la pista: inicio = fin del anterior + gapBefore
        for ref in selector.findall(".//ExtentRef"):
            ref_id = ref.get("id")
            if ref_id in titulos_dict:
                if es_principal:
                    n_pantalla += 1      # titulo a pantalla completa dentro de la pista de video
                    continue
                t = dict(titulos_dict[ref_id])
                t["inicio"] = cursor_t + t["gap"]
                t["fin"] = t["inicio"] + t["duracion"]
                cursor_t = t["fin"]
                n_animados += 1 if (t["animado"] and not t["spin"]) else 0   # sin soporte
                if t["texto"].strip() and t["duracion"] > 0:
                    proyecto["titulos"].append(t)
                continue
            if ref_id in extents_dict:
                datos = extents_dict[ref_id]
                duracion = (datos["outTime"] - datos["inTime"]) / datos["speed"]
                pista["clips"].append({
                    "gap": datos["gapBefore"], "in": datos["inTime"], "out": datos["outTime"],
                    "duracion": duracion, "ruta": datos["ruta"], "speed": datos["speed"],
                    "volumen": datos["volumen"],
                })
        if pista["clips"]:
            proyecto["pistas"].append(pista)
    if n_pantalla:
        proyecto["no_soportado"].append(f"{n_pantalla} título(s) a pantalla completa en la pista de video")
    if n_animados:
        proyecto["no_soportado"].append(f"animación de {n_animados} título(s) (se muestran estáticos)")
    return proyecto


# ======================================================================
#  Exportacion con cortes exactos
#
#  - Cada clip se recodifica (cortes al cuadro, no al keyframe).
#  - La duracion de cada clip se redondea a un numero entero de cuadros,
#    y el audio usa EXACTAMENTE esa misma duracion: video y audio no
#    se desfasan entre clips.
#  - El audio de cada clip se guarda como WAV (PCM): sin huecos de AAC.
#    Se une todo en WAV y se codifica a AAC una sola vez al final.
# ======================================================================
def _escapar_concat(ruta):
    return ruta.replace("\\", "/").replace("'", "'\\''")


def _lista_concat(archivos, carpeta_temp, prefijo):
    """Escribe la lista para el demuxer 'concat' de ffmpeg y devuelve su ruta."""
    lista_txt = os.path.join(carpeta_temp, f"lista_{prefijo}.txt")
    with open(lista_txt, "w", encoding="utf-8") as f:
        for arch in archivos:
            f.write(f"file '{_escapar_concat(arch)}'\n")
    return lista_txt


def _unir(ffmpeg_exe, archivos, salida, carpeta_temp, prefijo):
    if len(archivos) == 1:
        return archivos[0]
    lista_txt = _lista_concat(archivos, carpeta_temp, prefijo)
    ejecutar_ffmpeg([ffmpeg_exe, "-y", "-f", "concat", "-safe", "0",
                     "-i", lista_txt, "-c", "copy", salida])
    return salida


# ----------------------------------------------------------------------
#  Titulos / captions (filtro drawtext de ffmpeg)
# ----------------------------------------------------------------------
# familia (minusculas) -> archivos (normal, negrita, cursiva, negrita+cursiva) en C:\Windows\Fonts
_FUENTES_WIN = {
    "arial": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
    "arial black": ("ariblk.ttf",) * 4,
    "times new roman": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
    "verdana": ("verdana.ttf", "verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"),
    "tahoma": ("tahoma.ttf", "tahomabd.ttf", "tahoma.ttf", "tahomabd.ttf"),
    "segoe ui": ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
    "calibri": ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
    "cambria": ("cambria.ttc", "cambriab.ttf", "cambriai.ttf", "cambriaz.ttf"),
    "comic sans ms": ("comic.ttf", "comicbd.ttf", "comici.ttf", "comicz.ttf"),
    "georgia": ("georgia.ttf", "georgiab.ttf", "georgiai.ttf", "georgiaz.ttf"),
    "courier new": ("cour.ttf", "courbd.ttf", "couri.ttf", "courbi.ttf"),
    "trebuchet ms": ("trebuc.ttf", "trebucbd.ttf", "trebucit.ttf", "trebucbi.ttf"),
    "consolas": ("consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"),
    "impact": ("impact.ttf",) * 4,
    "lucida console": ("lucon.ttf",) * 4,
    "palatino linotype": ("pala.ttf", "palab.ttf", "palai.ttf", "palabi.ttf"),
}
_DIRS_FUENTES = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
    "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
]


def buscar_fuente(familia, estilo="Regular"):
    """Ruta del archivo de fuente para una familia/estilo de Movie Maker, o None."""
    est = (estilo or "").lower()
    idx = (1 if "bold" in est else 0) + (2 if "italic" in est or "oblique" in est else 0)
    candidatos = []
    nombres = _FUENTES_WIN.get((familia or "").strip().lower())
    if nombres:
        candidatos.append(nombres[idx] if idx < len(nombres) else nombres[0])
    # respaldo: fuentes habituales
    candidatos += [("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")[idx], "arial.ttf",
                   ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf")[idx],
                   ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
                    "DejaVuSans-BoldOblique.ttf")[idx], "DejaVuSans.ttf"]
    for nombre in candidatos:
        for d in _DIRS_FUENTES:
            ruta = os.path.join(d, nombre)
            if d and os.path.isfile(ruta):
                return ruta
    return None


def _drawtext_ok(ffmpeg_exe):
    """(drawtext disponible, soporta text_align)."""
    if "drawtext" not in _CACHE:
        try:
            r = subprocess.run([ffmpeg_exe, "-nostdin", "-hide_banner", "-h", "filter=drawtext"],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
            _CACHE["drawtext"] = (r.returncode == 0 and "fontfile" in r.stdout,
                                  "text_align" in r.stdout)
        except Exception:
            _CACHE["drawtext"] = (False, False)
    return _CACHE["drawtext"]


def _esc_ruta_filtro(ruta):
    """Ruta para usar entre comillas simples dentro de un filtro de ffmpeg (Windows incluido)."""
    return ruta.replace("\\", "/").replace(":", "\\:")


def _hex(rgb):
    return "0x%02X%02X%02X" % tuple(max(0, min(255, int(round(c * 255)))) for c in rgb[:3])


# Aproximacion de las coordenadas de Movie Maker: la escena mide "max_extent" unidades
# de ancho (y ese ancho / aspecto de alto); (0,0) es el centro y +y es hacia arriba.
def _geometria_titulo(t, fuente, ancho, alto, aspecto):
    escena_w = t["max_extent"]
    escena_h = escena_w / aspecto
    cx = min(0.95, max(0.05, 0.5 + t["pos"][0] / escena_w))
    cy = min(0.95, max(0.05, 0.5 - t["pos"][1] / escena_h))
    # ancho del texto: "length[0]" es su extension en unidades de la escena
    frac = (t["largo"][0] / escena_w) if t["largo"] and t["largo"][0] > 0 else 0.0
    frac = min(0.95, max(0.05, frac)) if frac else min(0.9, 0.04 * max(1, len(t["texto"])))
    objetivo = frac * ancho
    tam = alto * 0.08 * (t["tam"] / 0.4)           # tamano de respaldo
    try:
        from PIL import ImageFont
        f100 = ImageFont.truetype(fuente, 100)
        w100 = max(f100.getlength(l) for l in t["texto"].split("\n")) if t["texto"] else 0
        if w100 > 0:
            tam = 100 * objetivo / w100
    except Exception:
        pass
    tam = min(alto * 0.4, max(alto * 0.02, tam))
    return cx, cy, int(round(tam))


def _filtros_titulos(ffmpeg_exe, titulos, t0, dur, ancho, alto, aspecto, carpeta_txt, tag):
    """Filtros drawtext de los titulos que se solapan con el tramo [t0, t0+dur) de la linea de tiempo."""
    ok, con_align = _drawtext_ok(ffmpeg_exe)
    filtros = []
    if not ok:
        return filtros
    for k, t in enumerate(titulos):
        a, b = max(t["inicio"], t0), min(t["fin"], t0 + dur)
        if b - a <= 0.001:
            continue
        fuente = buscar_fuente(t["familia"], t["estilo"])
        if not fuente:
            continue
        cx, cy, tam = _geometria_titulo(t, fuente, ancho, alto, aspecto)
        txt = os.path.join(carpeta_txt, f"titulo_{tag}_{k}.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write(t["texto"])
        borde = t["borde_idx"] * max(1, alto // 360)
        f_txt = (f"drawtext=fontfile='{_esc_ruta_filtro(fuente)}':textfile='{_esc_ruta_filtro(txt)}'"
                 f":expansion=none:fontsize={tam}:fontcolor={_hex(t['color'])}@{1 - t['transp_texto']:.3f}"
                 # el recuadro del texto SIEMPRE queda dentro del cuadro (margen de seguridad 3 %),
        # aunque la posicion de Movie Maker caiga en el borde (p. ej. subtitulos al pie)
        f":x='max(w*0.03,min(w*{cx:.4f}-text_w/2,w*0.97-text_w))'"
        f":y='max(h*0.03,min(h*{cy:.4f}-text_h/2,h*0.95-text_h))'"
                 f":borderw={borde}:bordercolor={_hex(t['borde_color'])}"
                 f"{':text_align=center' if (con_align and t['justif'] == 'MIDDLE') else ''}"
                 f":enable='between(t,{a - t0:.4f},{b - t0:.4f})'")
        filtros.append(f_txt)
    return filtros


# ----------------------------------------------------------------------
#  Efecto "Spin In" (giro de entrada) - aproximacion
#  El texto se dibuja a una imagen PNG transparente y se superpone con
#  rotacion + escala + aparicion progresiva durante ANIM_SEG segundos;
#  despues queda fijo. (Movie Maker no documenta su animacion exacta.)
# ----------------------------------------------------------------------
ANIM_SEG = 0.6          # duracion de la entrada (segundos): muy rapida
ANIM_VUELTAS = 4.0      # vueltas completas que da, como una helice
ANIM_HORARIO = True     # sentido del giro (False = antihorario)
ANIM_MAX_GRADOS_CUADRO = 170   # mas que esto por cuadro se ve girando al reves (efecto rueda de carreta)
ANIM_ESCALA_INI = 0.04  # tamano inicial (fraccion del final): arranca lejos y "se acerca"
ANIM_APARECE = 0.25     # fraccion de la entrada que tarda en hacerse totalmente visible


def _png_titulo(t, fuente, tam, destino):
    """Dibuja el texto del titulo en un PNG transparente. Devuelve (ancho, alto) o None."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(fuente, tam)
        borde = t["borde_idx"] * max(1, tam // 25)
        rgb = tuple(int(round(c * 255)) for c in t["color"][:3])
        rgb_b = tuple(int(round(c * 255)) for c in t["borde_color"][:3])
        alfa = int(round((1 - t["transp_texto"]) * 255))
        d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        bb = d.multiline_textbbox((0, 0), t["texto"], font=font, align="center", stroke_width=borde)
        pad = max(4, tam // 6)
        w, h = int(bb[2] - bb[0]) + 2 * pad, int(bb[3] - bb[1]) + 2 * pad
        w, h = w + w % 2, h + h % 2
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).multiline_text(
            (pad - bb[0], pad - bb[1]), t["texto"], font=font, fill=rgb + (alfa,), align="center",
            stroke_width=borde, stroke_fill=rgb_b + (alfa,))
        img.save(destino)
        return w, h
    except Exception:
        return None


def _capas_spin(t, png, w, h, t0, dur, ancho, alto, aspecto):
    """Tramos (de este clip) en que el titulo 'Spin In' esta activo.
    Cada tramo: dict(tipo='anim'|'fija', ini, fin) con tiempos locales al clip."""
    ini_anim, fin_anim = t["inicio"], min(t["fin"], t["inicio"] + ANIM_SEG)
    capas = []
    for tipo, a, b in (("anim", ini_anim, fin_anim), ("fija", fin_anim, t["fin"])):
        a, b = max(a, t0), min(b, t0 + dur)
        if b - a > 0.001:
            capas.append({"tipo": tipo, "png": png, "w": w, "h": h, "ini": a - t0, "fin": b - t0})
    return capas


def _grafo_spin(capas_por_titulo, base, filtros_fijos, pix_fmt, fps_txt, t0, ancho, alto, aspecto,
                entrada="[0:v]", idx_ini=1):
    """Construye (entradas_extra, filter_complex) para un clip con titulos 'Spin In'."""
    entradas, grafo = [], []
    # El giro empieza rapido y frena (cuadratico). Si gira mas de ~170 grados entre un cuadro
    # y el siguiente, el ojo lo ve al reves; se limita el numero de vueltas para evitarlo.
    try:
        a, _, b = fps_txt.partition("/")
        fps_n = float(a) / float(b or 1)
    except ValueError:
        fps_n = 30.0
    vueltas = min(ANIM_VUELTAS, ANIM_MAX_GRADOS_CUADRO * ANIM_SEG * fps_n / 720.0)
    signo = 1 if ANIM_HORARIO else -1
    cadena = f"{entrada}{base + ',' if base else ''}format=yuv420p" + ("," + ",".join(filtros_fijos) if filtros_fijos else "")
    grafo.append(cadena + "[b0]")
    prev, idx, n = "b0", idx_ini, 0
    for t, capas, w, h in capas_por_titulo:
        cx = min(0.95, max(0.05, 0.5 + t["pos"][0] / t["max_extent"]))
        cy = min(0.95, max(0.05, 0.5 - t["pos"][1] / (t["max_extent"] / aspecto)))
        # centro en pixeles, con el recuadro del texto siempre dentro del cuadro
        cxp = min(ancho * 0.97 - w / 2, max(ancho * 0.03 + w / 2, ancho * cx))
        cyp = min(alto * 0.95 - h / 2, max(alto * 0.03 + h / 2, alto * cy))
        for c in capas:
            dur_c = c["fin"] - c["ini"]
            extra = 2.0 / fps_n   # margen: evita un cuadro sin texto al pasar a la fase fija o al final del clip
            entradas += ["-loop", "1", "-framerate", fps_txt, "-t", f"{dur_c + extra:.4f}", "-i", c["png"]]
            etiqueta, salida, n = f"tx{n}", f"b{n + 1}", n + 1
            if c["tipo"] == "anim":
                p = f"clip((t+{t0:.4f}-{t['inicio']:.4f})/{ANIM_SEG:.3f},0,1)"
                e = f"(1-pow(1-{p},3))"                         # progreso con frenado suave 0 -> 1
                grafo.append(
                    f"[{idx}:v]format=rgba,setpts=PTS+{c['ini']:.4f}/TB,"
                    # aparicion progresiva segun el tiempo GLOBAL (sigue bien si la entrada cruza
                    # un corte entre clips); se aplica al texto aun sin rotar, que es chico
                    f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)'"
                    f":a='alpha(X,Y)*clip((T+{t0:.4f}-{t['inicio']:.4f})/{ANIM_SEG * ANIM_APARECE:.4f},0,1)',"
                    # el angulo llega a 0 DESDE el lado contrario al del giro: asi el final avanza
                    # en el mismo sentido (no parece que gire de vuelta al asentarse)
                    f"rotate=a='{-signo * 2 * 3.14159265 * vueltas:.5f}*pow(1-{p},2)'"
                    f":ow='hypot(iw,ih)':oh='hypot(iw,ih)':c=0x00000000,"
                    f"scale=w='max(2,iw*({ANIM_ESCALA_INI}+{1 - ANIM_ESCALA_INI}*{e}))'"
                    f":h='max(2,ih*({ANIM_ESCALA_INI}+{1 - ANIM_ESCALA_INI}*{e}))':eval=frame[{etiqueta}]")
                pos = f"x='{cxp:.2f}-w/2':y='{cyp:.2f}-h/2'"
            else:
                grafo.append(f"[{idx}:v]format=rgba,setpts=PTS+{c['ini']:.4f}/TB[{etiqueta}]")
                pos = f"x={cxp - w / 2:.2f}:y={cyp - h / 2:.2f}"
            if c["tipo"] == "anim":
                habil = f"gte(t,{c['ini']:.4f})*lt(t,{c['fin']:.4f})"
            else:
                habil = f"between(t,{c['ini']:.4f},{c['fin']:.4f})"
            grafo.append(f"[{prev}][{etiqueta}]overlay={pos}:eof_action=pass:format=auto"
                         f":enable='{habil}'[{salida}]")
            prev, idx = salida, idx + 1
    grafo.append(f"[{prev}]format={pix_fmt}[v]")
    return entradas, ";".join(grafo)


def _capas_musica(pistas_a, total_dur):
    """Pistas de audio agregadas (musica) ubicadas en la linea de tiempo como en Movie Maker:
    inicio = fin del clip anterior + gapBefore (gapBefore negativo = se superpone).
    Devuelve [(ruta, entrada, duracion, inicio, volumen)]; descarta lo que empieza despues del final
    del video y los clips con volumen 0 / Mute."""
    capas = []
    for pista_a in pistas_a:          # cada pista tiene su propia linea de tiempo
        cursor = 0.0
        for clip in pista_a["clips"]:
            dur_c = clip["out"] - clip["in"]
            inicio = cursor + clip.get("gap", 0.0)
            cursor = inicio + dur_c
            vol_c = clip.get("volumen", 1.0)
            if vol_c <= 0.001:
                continue
            entrada = clip["in"]
            if inicio < 0:            # empieza antes de 0: recortar el comienzo
                entrada, dur_c, inicio = entrada - inicio, dur_c + inicio, 0.0
            if dur_c <= 0.01 or inicio >= total_dur:
                continue
            capas.append((clip["ruta"], entrada, min(dur_c, total_dur - inicio), inicio, vol_c))
    return capas


def _grafo_quitar_subs(entrada, salida, ancho, alto, pref=""):
    """Fragmento de filtro (filter_complex) que borra el subtitulo pegado SOLO donde hay letras.
    1) Detecta el texto: relleno claro junto a un contorno oscuro (umbrales a resolucion completa,
       luego a media resolucion), engordado para cubrir el contorno y el suavizado de bordes.
    2) Rellena esos pixeles con el promedio de los vecinos que NO son texto (convolucion
       normalizada a media resolucion, en dos escalas: fina donde hay vecinos cerca y gruesa si no).
    3) Pega el relleno solo sobre las letras, con el borde suavizado. El resto de la banda no se toca.
    Todo se calcula solo en la banda de subtitulos (parte baja del cuadro)."""
    par = lambda v: max(2, int(v) // 2 * 2)
    bx, by = par(ancho * SUBS_MARGEN_X), par(alto * SUBS_Y)
    bw, bh = par(ancho - 2 * bx), par(min(alto * SUBS_ALTO, alto - by))
    bw, bh = max(4, bw // 4 * 4), max(4, bh // 4 * 4)
    hw, hh = bw // 2, bh // 2
    n_dk = max(1, int(round(alto / 720)))               # cercania al contorno oscuro (a media resolucion)
    n_exp = max(2, int(round(alto / 480)))              # cuanto se engorda la mascara (a media resolucion)
    s1, s2 = max(1.2, alto / 360) / 2 * 1.4, max(3.0, alto / 120) / 2
    dil = lambda n: "," + ",".join(["dilation"] * n) if n > 0 else ""
    L = lambda e: f"lutrgb=r='{e}':g='{e}':b='{e}'"
    g = (
        f"{entrada}format=yuv420p,split=2[sf][sb0];"
        f"[sb0]crop={bw}:{bh}:{bx}:{by},split=4[bi][bwt][bd][bh];"
        f"[bwt]extractplanes=y,lutyuv=y='if(gt(val,190),255,0)',scale={hw}:{hh}:flags=area,"
        f"lutyuv=y='if(gt(val,60),255,0)'[w0];"
        f"[bd]extractplanes=y,lutyuv=y='if(lt(val,70),255,0)',scale={hw}:{hh}:flags=area,"
        f"lutyuv=y='if(gt(val,60),255,0)'{dil(n_dk)}[dk];"
        f"[w0][dk]blend=all_mode=darken{dil(n_exp)},split=3[m1][mfa][mfb];"
        f"[m1]format=gbrp,{L('255-val')},split=3[nmA][nmB][nmC];"
        f"[bh]scale={hw}:{hh}:flags=fast_bilinear,format=gbrp[hg];"
        f"[hg][nmA]blend=all_mode=multiply,split=2[aA][aB];"
        f"[aA]gblur=sigma={s1:.1f}[ba1];[nmB]gblur=sigma={s1:.1f},split=2[bn1a][bn1b];"
        f"[bn1a]{L('max(val,2)')}[bn1c];[ba1][bn1c]blend=all_mode=divide[f1];"
        f"[aB]gblur=sigma={s2:.1f}[ba2];[nmC]gblur=sigma={s2:.1f},{L('max(val,2)')}[bn2];"
        f"[ba2][bn2]blend=all_mode=divide[f2];"
        f"[bn1b]{L('if(gt(val,110),255,0)')}[sel];"
        f"[f2][f1][sel]maskedmerge,format=yuv420p,scale={bw}:{bh}:flags=bilinear[fill];"
        # mascara final en yuv420p: Y a la resolucion de la banda, U y V a la mitad
        f"[mfa]format=gray,scale={bw}:{bh}:flags=bilinear,gblur=sigma=1,lutyuv=y='clip(val*2,0,255)'[mY];"
        f"[mfb]format=gray,lutyuv=y='clip(val*2,0,255)',split=2[mU][mV];"
        f"[mY][mU][mV]mergeplanes=0x001020:yuv420p[mk];"
        f"[bi][fill][mk]maskedmerge[rr];"
        f"[sf][rr]overlay={bx}:{by}{salida}"
    )
    # etiquetas internas con prefijo propio (el grafo completo tiene una copia por clip)
    return re.sub(r"\[(sf|sb0|bi|bwt|bd|bh|w0|dk|m1|mfa|mfb|nmA|nmB|nmC|hg|aA|aB|ba1|bn1a|bn1b|bn1c|f1|ba2|bn2|f2|sel|"
                  r"fill|mY|mU|mV|mk|rr)\]", lambda m: f"[{pref}{m.group(1)}]", g)


def crear_video_final(proyecto, ruta_salida, altura=720, con_audio=True, agrandar=True,
                      progreso_callback=None, estado_callback=None, limite_seg=None, quitar_subs=False, filtro_video=""):
    pista_video = next((p for p in proyecto["pistas"] if p["tipo"] == "Principal"), None)
    pistas_a = [p for p in proyecto["pistas"] if p["tipo"] == "Secundaria"] if con_audio else []
    if not pista_video:
        raise ValueError("No hay pista de video principal.")

    def decir(t):
        if estado_callback:
            estado_callback(t)

    decir("Preparando FFmpeg...")
    ffmpeg_exe = obtener_ffmpeg()
    # altura=None -> la mayor resolucion presente en el proyecto (para subir a YouTube)
    # La compresion "baja" (borrador) solo se usa en la previsualizacion; todas las exportaciones
    # de 720p hacia abajo usan la misma compresion media.
    nivel = "alta" if (altura is None or altura >= 1080) else ("media" if agrandar else "baja")
    args_enc, pix_fmt = obtener_encoder(ffmpeg_exe, nivel)

    clips_v = pista_video["clips"]
    n_audio = sum(len(p["clips"]) for p in pistas_a)

    # Todos los archivos deben existir antes de empezar
    faltan = sorted({c["ruta"] for c in clips_v + [c for p in pistas_a for c in p["clips"]]
                     if not os.path.isfile(c["ruta"])})
    if faltan:
        raise FileNotFoundError("No se encuentran estos archivos del proyecto:\n" + "\n".join(faltan))

    # Clips "sin recortar" (out <= in): duran hasta el final del archivo
    for c in clips_v + [c for p in pistas_a for c in p["clips"]]:
        if c["out"] <= c["in"]:
            c["out"] = sondear(ffmpeg_exe, c["ruta"])["dur"]
        if c["out"] <= c["in"]:
            raise ValueError(f"No se pudo determinar la duración de: {c['ruta']}")

    # Modo prueba: solo los primeros 'limite_seg' segundos del video (titulos y musica se recortan solos)
    if limite_seg:
        restante, recortados = float(limite_seg), []
        for c in clips_v:
            d = c["out"] - c["in"]
            if restante <= 0.01:
                break
            c2 = dict(c)
            if d > restante:
                c2["out"] = c2["in"] + restante
                d = restante
            recortados.append(c2)
            restante -= d
        clips_v = recortados

    # --- Formato comun de salida ---
    # Los clips pueden venir de archivos con distinta resolucion y fps (p. ej. 1080p a
    # 30 fps junto con 480p a 23,976 fps). Para poder unirlos sin recodificar otra vez,
    # TODOS se convierten al mismo tamano, al mismo fps y con el mismo codec.
    info_v = [sondear(ffmpeg_exe, c["ruta"]) for c in clips_v]
    alto_max = max((i["h"] for i in info_v), default=0) or 720
    # Si eliges una altura (p. ej. 1080 o 1440) el video se ajusta a ella aunque las fuentes
    # sean mas chicas: se agranda con lanczos. Solo la previsualizacion no agranda.
    if altura is None:
        alto = alto_max
    else:
        alto = altura if agrandar else min(altura, alto_max)
    alto = max(2, alto - alto % 2)
    ancho = max(2, int(round(alto * proyecto.get("aspecto", 16 / 9) / 2)) * 2)
    etiqueta = f"{ancho}x{alto}"
    kbps_audio = 192
    if agrandar:      # exportacion: tamano objetivo segun la resolucion
        kbps_video, kbps_audio = bitrates_objetivo(alto)
        args_enc, pix_fmt = obtener_encoder(ffmpeg_exe, nivel, kbps=kbps_video)
    peso_fps = {}
    for c, i in zip(clips_v, info_v):      # fps con mas tiempo de pantalla
        peso_fps[i["fps"]] = peso_fps.get(i["fps"], 0.0) + (c["out"] - c["in"])
    fps_txt, fps_val = max(peso_fps, key=peso_fps.get)
    aspecto_p = proyecto.get("aspecto", 16 / 9)
    if agrandar:      # un cuadro clave cada 2 s: reproduccion fluida y busqueda rapida en cualquier reproductor
        args_enc = args_enc + ["-g", str(max(12, int(round(fps_val * 2))))]

    def filtro_base(info, delogo=False):
        """Escala de un clip al tamano comun (el formato de pixel se agrega al final)."""
        ar = (info["w"] * info["sar"] / info["h"]) if info["h"] else aspecto_p
        if abs(ar / aspecto_p - 1) < 0.02:
            # casi 16:9 (p. ej. 848x480 = 1.767): se ajusta al cuadro completo; la deformacion
            # (<2 %) no se nota y evita franjas negras finas a los lados
            esc = f"scale={ancho}:{alto}:flags={ESCALADO},setsar=1"
        else:
            # otra proporcion (4:3, etc.): se respeta y se rellena con barras negras.
            # Antes se pasa a pixeles cuadrados (fuentes anamorficas).
            esc = (f"scale=trunc(iw*sar/2)*2:ih,setsar=1,"
                   f"scale={ancho}:{alto}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags={ESCALADO},"
                   f"pad={ancho}:{alto}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
        extra = ""
        if delogo:            # (modo por clips) rellena la banda de subtitulos copiando/interpolando los pixeles vecinos
            sx = max(1, int(ancho * SUBS_MARGEN_X))
            sy = max(1, int(alto * SUBS_Y))
            sw = max(8, ancho - 2 * sx - 2)
            sh = max(8, min(int(alto * SUBS_ALTO), alto - sy - 3))
            extra = f",format=yuv420p,delogo=x={sx}:y={sy}:w={sw}:h={sh}"
        if filtro_video:      # filtro de imagen elegido (nitidez, color, B/N...)
            extra += f",format=yuv420p,{filtro_video}"
        return f"{esc},fps={fps_txt}{extra}"
    todos_titulos = proyecto.get("titulos", [])
    tit_spin = [t for t in todos_titulos if t["spin"]]          # entran girando (PNG + overlay)
    titulos = [t for t in todos_titulos if not t["spin"]]       # texto fijo (drawtext)
    advertencias = []
    if titulos and not _drawtext_ok(ffmpeg_exe)[0]:
        advertencias.append("Tu FFmpeg no incluye el filtro 'drawtext': los títulos no se dibujaron.")
        titulos = []
    sin_fuente = {t["familia"] for t in todos_titulos if not buscar_fuente(t["familia"], t["estilo"])}
    if sin_fuente:
        advertencias.append("No se encontró ninguna fuente para: " + ", ".join(sorted(sin_fuente))
                            + ". Esos títulos no se dibujaron.")
        tit_spin = [t for t in tit_spin if t["familia"] not in sin_fuente]

    PESO_V, PESO_A, PESO_FIN = 1.0, 0.1, 0.5
    total = (len(clips_v) * PESO_V + (len(clips_v) * PESO_A if con_audio else 0)
             + n_audio * PESO_A + PESO_FIN)
    acum = {"p": 0.0}

    lock_p = threading.Lock()
    en_curso_p = {}

    def _emitir():
        if progreso_callback:
            progreso_callback(int(min((acum["p"] + sum(en_curso_p.values())) / total, 1.0) * 100))

    def reportar(peso, frac=0.0, clave=None):
        with lock_p:
            en_curso_p[clave] = peso * frac
            _emitir()

    def avanzar(peso, clave=None):
        with lock_p:
            acum["p"] += peso
            en_curso_p.pop(clave, None)
            _emitir()

    # Los temporales van junto al archivo de salida (misma unidad que eliges para
    # guardar), NO en la carpeta temporal de Windows, que suele estar en C: y llenarse.
    carpeta_destino = os.path.dirname(os.path.abspath(ruta_salida))
    seg_total = sum(c["out"] - c["in"] for c in clips_v)
    try:
        carpeta_temp = tempfile.mkdtemp(prefix="wmpedit_", dir=carpeta_destino)
    except OSError:
        carpeta_temp = tempfile.mkdtemp(prefix="wmpedit_")
    # Archivos de texto de los titulos (diminutos): un ' en la ruta romperia el filtro
    carpeta_txt = carpeta_temp if "'" not in carpeta_temp else tempfile.mkdtemp(prefix="wmpedit_txt_")
    try:
        # Titulos con "Spin In": se dibuja el texto una vez en un PNG transparente
        spin_info = []
        for k, t in enumerate(tit_spin):
            fuente = buscar_fuente(t["familia"], t["estilo"])
            _, _, tam = _geometria_titulo(t, fuente, ancho, alto, aspecto_p)
            png = os.path.join(carpeta_txt, f"spin_{k}.png")
            dims = _png_titulo(t, fuente, tam, png)
            if dims:
                spin_info.append((t, png, dims[0], dims[1]))
            else:    # sin Pillow: se muestra fijo
                titulos.append(t)
                advertencias.append("No se pudo animar el título «" + t["texto"][:30] + "»; se muestra fijo.")

        # --- PASO 1: video + audio nativo (FLAC, sin perdida) de cada clip, en UN solo proceso ---
        n_clips = len(clips_v)
        jobs, cum_raw, cum_frames = [], 0.0, 0
        for i, clip in enumerate(clips_v):
            # Los cuadros se reparten sobre la linea de tiempo ACUMULADA (no clip por clip):
            # asi el redondeo no se acumula y la musica y los titulos no se corren
            # aunque haya cientos de cortes.
            cum_raw += clip["out"] - clip["in"]
            frames = max(1, round(cum_raw * fps_val) - cum_frames)
            t0 = cum_frames / fps_val          # inicio de este clip en el video final
            cum_frames += frames
            jobs.append({"i": i, "clip": clip, "info": info_v[i], "frames": frames,
                         "t0": t0, "dur": frames / fps_val})
        durs = [j["dur"] for j in jobs]
        videos = [os.path.join(carpeta_temp, f"v_{i:04d}.mp4") for i in range(n_clips)]
        audios = [os.path.join(carpeta_temp, f"a_{i:04d}.flac") for i in range(n_clips)] if con_audio else []
        est = {"titulos": titulos, "spin": spin_info, "hechos": 0}
        lock = threading.Lock()

        # --- MODO DIRECTO (preferido): UNA sola pasada, sin archivos temporales de video/audio ---
        # Un solo ffmpeg abre cada clip (-ss/-t), los recorta cuadro a cuadro, los une con el filtro
        # 'concat', pone titulos y musica sobre la linea de tiempo global y codifica UNA vez al
        # archivo final. Usa solo el espacio del archivo final y no hay que volver a copiarlo.
        def directo(titulos_d, spin_d):
            total_dur = sum(durs)
            capas_m = _capas_musica(pistas_a, total_dur) if n_audio else []
            entradas_c, g = [], []
            hilos_dec = 3 if n_clips <= 20 else (2 if n_clips <= 60 else 1)
            for j in jobs:
                i, clip, info = j["i"], j["clip"], j["info"]
                crudo = clip["out"] - clip["in"]
                entradas_c += ["-threads", str(hilos_dec), "-ss", f"{clip['in']:.6f}",
                               "-t", f"{crudo + 0.3:.6f}", "-i", clip["ruta"]]
                # tpad repite el ultimo cuadro si el archivo termina unos cuadros antes de lo previsto
                ent_c = f"[{i}:v:0]"
                if quitar_subs and info["w"] >= 64 and info["h"] >= 64:
                    # el subtitulo se borra a la resolucion NATIVA del archivo (mucho mas barato que
                    # a la de salida, y el parche se disimula al agrandar despues)
                    g.append(_grafo_quitar_subs(ent_c, f"[q{i}]", info["w"], info["h"], pref=f"c{i}_"))
                    ent_c = f"[q{i}]"
                g.append(f"{ent_c}{filtro_base(info)},format=yuv420p,tpad=stop=4:stop_mode=clone,"
                         f"trim=end_frame={j['frames']},setpts=PTS-STARTPTS[v{i}]")
                if con_audio:
                    vol = clip.get("volumen", 1.0)
                    if info["audio"] and vol > 0.001:
                        if abs(vol - 1) < 1e-3:
                            fv = ""
                        elif vol > 1:
                            fv = f"volume={vol:.3f},alimiter=limit=0.97,"
                        else:
                            fv = f"volume={vol:.3f},"
                        g.append(f"[{i}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                                 f"{fv}apad,atrim=duration={j['dur']:.6f},asetpts=PTS-STARTPTS[a{i}]")
                    else:       # clip silenciado en Movie Maker (o sin audio): silencio de la misma duracion
                        g.append(f"anullsrc=r=48000:cl=stereo,aformat=sample_fmts=fltp,"
                                 f"atrim=duration={j['dur']:.6f},asetpts=PTS-STARTPTS[a{i}]")
            if con_audio:
                g.append("".join(f"[v{j['i']}][a{j['i']}]" for j in jobs)
                         + f"concat=n={n_clips}:v=1:a=1[vc][ac]")
            else:
                g.append("".join(f"[v{j['i']}]" for j in jobs) + f"concat=n={n_clips}:v=1:a=0[vc]")

            # musica agregada
            for k, (ruta_m, entrada, dur_c, inicio, vol_c) in enumerate(capas_m):
                entradas_c += ["-ss", f"{entrada:.6f}", "-t", f"{dur_c:.6f}", "-i", ruta_m]
                ms = int(round(inicio * 1000))
                g.append(f"[{n_clips + k}:a]aresample=48000,aformat=channel_layouts=stereo,"
                         f"volume={vol_c:.4f},adelay={ms}|{ms}[m{k}]")
            etiqueta_a = "ac"
            if capas_m:
                if len(capas_m) > 1:
                    g.append("".join(f"[m{k}]" for k in range(len(capas_m)))
                             + f"amix=inputs={len(capas_m)}:duration=longest:dropout_transition=0:normalize=0[mx]")
                    ultima = "[mx]"
                else:
                    ultima = "[m0]"
                g.append(f"{ultima}apad=whole_dur={total_dur:.6f}[mf]")
                g.append("[ac][mf]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]")
                etiqueta_a = "a"

            # titulos sobre la linea de tiempo global (t0 = 0)
            tf = _filtros_titulos(ffmpeg_exe, titulos_d, 0.0, total_dur, ancho, alto,
                                  aspecto_p, carpeta_txt, "g")
            capas_t = [(t, c, w, h) for (t, png, w, h) in spin_d
                       for c in [_capas_spin(t, png, w, h, 0.0, total_dur, ancho, alto, aspecto_p)] if c]
            entradas_png = []
            ent_v = "[vc]"
            if capas_t:
                entradas_png, grafo_v = _grafo_spin(capas_t, "", tf, pix_fmt, fps_txt, 0.0, ancho, alto,
                                                    aspecto_p, entrada=ent_v,
                                                    idx_ini=n_clips + len(capas_m))
                g.append(grafo_v)
            else:
                g.append(f"{ent_v}format=yuv420p" + ("," + ",".join(tf) if tf else "") + f",format={pix_fmt}[v]")

            script = os.path.join(carpeta_temp, "grafo.txt")
            with open(script, "w", encoding="utf-8") as f:
                f.write(";\n".join(g))
            salida_cmd = ["-map", "[v]"] + (["-map", f"[{etiqueta_a}]"] if con_audio else ["-an"])
            salida_cmd += args_enc + (["-c:a", "aac", "-b:a", f"{kbps_audio}k"] if con_audio else [])
            salida_cmd += ["-movflags", "+faststart", ruta_salida]
            ultimo = None
            for opcion in (["-filter_complex_script", script], ["-/filter_complex", script]):
                cmd = [ffmpeg_exe, "-y"] + entradas_c + entradas_png + opcion + salida_cmd
                if len(subprocess.list2cmdline(cmd)) > 30000:
                    raise RuntimeError("orden demasiado larga")
                try:
                    def vigilar(f):
                        if progreso_callback:
                            progreso_callback(int(min(f, 1.0) * 99))
                    ejecutar_ffmpeg(cmd, duracion=total_dur, on_frac=vigilar)
                    return capas_t
                except RuntimeError as e:
                    ultimo = e
                    if "No space left" in str(e) or "espacio" in str(e).lower():
                        raise
                    if "filter_complex" not in str(e):     # no era un problema de la opcion
                        break
            raise ultimo

        decir(f"Codificando el video final en {etiqueta} (una sola pasada)...")
        if os.path.exists(ruta_salida):
            os.remove(ruta_salida)
        try:
            try:
                directo(titulos, spin_info)
            except RuntimeError as e:
                if "espacio" in str(e).lower() or "orden demasiado" in str(e):
                    raise
                if spin_info:    # quiza este FFmpeg no soporta algun filtro de la animacion
                    advertencias.append("Tu FFmpeg no pudo animar el efecto de giro; "
                                        "esos títulos se muestran fijos.")
                    directo(titulos + [x[0] for x in spin_info], [])
                else:
                    raise
            if progreso_callback:
                progreso_callback(100)
            return advertencias
        except RuntimeError as e:
            if os.path.exists(ruta_salida):          # no dejar un MP4 incompleto ocupando disco
                os.remove(ruta_salida)
            if "espacio" in str(e).lower() and "orden demasiado" not in str(e):
                raise
            # Respaldo: modo por clips
            print("Modo directo no disponible, se usa el modo por clips:", str(e)[:300])
            advertencias[:] = [a for a in advertencias if "giro" not in a]
            advertencias.append("La exportación directa falló (" + str(e).strip().splitlines()[-1][:150]
                                + "); se usó el modo por clips.")
        decir(f"Procesando clips en {etiqueta} (modo por clips)...")

        def procesar(j):
            i, clip, info = j["i"], j["clip"], j["info"]
            frames, t0, dur = j["frames"], j["t0"], j["dur"]
            with lock:
                titulos_l, spin_l = list(est["titulos"]), list(est["spin"])
                est["hechos"] += 1
                decir(f"Procesando clips ({est['hechos']}/{n_clips}) en {etiqueta}...")
            vf = filtro_base(info, delogo=quitar_subs)
            tf = _filtros_titulos(ffmpeg_exe, titulos_l, t0, dur, ancho, alto,
                                  aspecto_p, carpeta_txt, i)
            capas_t = [(t, c, w, h) for (t, png, w, h) in spin_l
                       for c in [_capas_spin(t, png, w, h, t0, dur, ancho, alto, aspecto_p)] if c]
            video_i, audio_i = videos[i], (audios[i] if con_audio else None)

            # Audio del clip: se saca del MISMO ffmpeg que hace el video (el archivo se abre una
            # sola vez) y se guarda en FLAC (sin perdida, pesa la mitad que un WAV).
            vol = clip.get("volumen", 1.0)
            audio_en_proceso = bool(con_audio and info["audio"] and vol > 0.001)
            args_audio = []
            if audio_en_proceso:
                if abs(vol - 1) < 1e-3:
                    filtro_a = "apad"
                elif vol > 1:     # Movie Maker permite hasta 2x: limitar para no distorsionar
                    filtro_a = f"volume={vol:.3f},alimiter=limit=0.97,apad"
                else:
                    filtro_a = f"volume={vol:.3f},apad"
                # apad + -t => duracion exacta (rellena con silencio si el audio es corto)
                args_audio = ["-map", "0:a:0", "-af", filtro_a, "-ac", "2", "-ar", "48000",
                              "-sample_fmt", "s16", "-c:a", "flac", "-t", f"{dur:.6f}", audio_i]

            def cmd_fijo(extra_titulos=()):
                """Comando con los titulos como texto fijo (drawtext)."""
                tf2 = tf + _filtros_titulos(ffmpeg_exe, list(extra_titulos), t0, dur, ancho, alto,
                                            aspecto_p, carpeta_txt, f"{i}s")
                v2 = vf + (",format=yuv420p," + ",".join(tf2) if tf2 else "") + f",format={pix_fmt}"
                return ([ffmpeg_exe, "-y", "-ss", str(clip["in"]), "-i", clip["ruta"],
                         "-an", "-frames:v", str(frames), "-vf", v2] + args_enc + [video_i] + args_audio)

            def en_curso(f):
                reportar(PESO_V, f, i)

            if capas_t:      # este clip contiene un titulo que entra girando
                entradas, grafo = _grafo_spin(capas_t, vf, tf, pix_fmt, fps_txt, t0, ancho, alto, aspecto_p)
                cmd_v = ([ffmpeg_exe, "-y", "-ss", str(clip["in"]), "-i", clip["ruta"]] + entradas
                         + ["-filter_complex", grafo, "-map", "[v]", "-an", "-frames:v", str(frames)]
                         + args_enc + [video_i] + args_audio)
                try:
                    ejecutar_ffmpeg(cmd_v, duracion=dur, on_frac=en_curso)
                except RuntimeError:
                    # Este FFmpeg no soporta algun filtro de la animacion: mostrar el texto fijo
                    with lock:
                        if est["spin"]:
                            advertencias.append("Tu FFmpeg no pudo animar el efecto de giro; "
                                                "esos títulos se muestran fijos.")
                            est["titulos"] = est["titulos"] + [x[0] for x in est["spin"]]
                            est["spin"] = []
                    ejecutar_ffmpeg(cmd_fijo([t for (t, _c, _w, _h) in capas_t]),
                                    duracion=dur, on_frac=en_curso)
            else:
                ejecutar_ffmpeg(cmd_fijo(), duracion=dur, on_frac=en_curso)

            if con_audio and not audio_en_proceso:
                # Clip silenciado en Movie Maker (o sin audio) -> silencio de la misma duracion
                ejecutar_ffmpeg([ffmpeg_exe, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                                 "-ac", "2", "-ar", "48000", "-sample_fmt", "s16", "-c:a", "flac",
                                 "-t", f"{dur:.6f}", audio_i])
            avanzar(PESO_V + (PESO_A if con_audio else 0), i)

        hilos = max(1, min(HILOS_CLIPS, n_clips))
        if hilos == 1:
            for j in jobs:
                procesar(j)
        else:
            with ThreadPoolExecutor(max_workers=hilos) as ex:
                futs = [ex.submit(procesar, j) for j in jobs]
                try:
                    for f in futs:
                        f.result()
                except BaseException:
                    for f in futs:
                        f.cancel()
                    raise

        # --- PASO 2: pista de audio agregada (musica) ---
        # Cada clip se ubica en la linea de tiempo como en Movie Maker:
        #   inicio = fin del clip anterior + gapBefore   (gapBefore negativo = se superpone)
        # Los clips que empiezan despues del final del video se descartan, y los
        # clips con volumen 0 / Mute no suenan. Se mezcla directamente en el paso final
        # (sin archivo intermedio).
        capas = []
        if n_audio:
            capas = _capas_musica(pistas_a, sum(durs))
            avanzar(n_audio * PESO_A)

        # --- PASO 3: unir y mezclar ---
        decir("Uniendo los clips...")
        if os.path.exists(ruta_salida):
            os.remove(ruta_salida)

        if not con_audio:
            video_unido = _unir(ffmpeg_exe, videos,
                                os.path.join(carpeta_temp, "video_unido.mp4"), carpeta_temp, "v")
            shutil.move(video_unido, ruta_salida)
        else:
            # Un solo paso: ffmpeg lee las listas de clips (y la musica) directamente.
            # Sin archivos intermedios gigantes. El video se copia; el audio se codifica
            # en AAC una sola vez.
            decir("Uniendo y mezclando audio y video final...")
            cmd_final = [ffmpeg_exe, "-y",
                         "-f", "concat", "-safe", "0", "-i", _lista_concat(videos, carpeta_temp, "v"),
                         "-f", "concat", "-safe", "0", "-i", _lista_concat(audios, carpeta_temp, "a")]
            if capas:
                total_dur = sum(durs)
                filtros = []
                for k, (ruta_m, entrada, dur_c, inicio, vol_c) in enumerate(capas):
                    cmd_final += ["-ss", f"{entrada:.6f}", "-t", f"{dur_c:.6f}", "-i", ruta_m]
                    ms = int(round(inicio * 1000))
                    filtros.append(f"[{k + 2}:a]aresample=48000,aformat=channel_layouts=stereo,"
                                   f"volume={vol_c:.4f},adelay={ms}|{ms}[m{k}]")
                if len(capas) > 1:
                    filtros.append("".join(f"[m{k}]" for k in range(len(capas)))
                                   + f"amix=inputs={len(capas)}:duration=longest:"
                                     f"dropout_transition=0:normalize=0[mx]")
                    ultima = "[mx]"
                else:
                    ultima = "[m0]"
                filtros.append(f"{ultima}apad=whole_dur={total_dur:.6f}[mf]")
                filtros.append("[1:a][mf]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0[a]")
                cmd_final += ["-filter_complex", ";".join(filtros),
                              "-map", "0:v:0", "-map", "[a]"]
            else:
                cmd_final += ["-map", "0:v:0", "-map", "1:a:0"]
            cmd_final += ["-c:v", "copy", "-c:a", "aac", "-b:a", f"{kbps_audio}k",
                          "-movflags", "+faststart", ruta_salida]
            ejecutar_ffmpeg(cmd_final)

        if progreso_callback:
            progreso_callback(100)
        return advertencias        # lista de avisos (vacia si todo salio bien)

    finally:
        shutil.rmtree(carpeta_temp, ignore_errors=True)
        if carpeta_txt != carpeta_temp:
            shutil.rmtree(carpeta_txt, ignore_errors=True)


# ======================================================================
#  Interfaz grafica
# ======================================================================
MODOS = {
    "2160p / 4K (agranda si hace falta)": 2160,
    "1440p (agranda si hace falta)": 1440,          # opcion predeterminada
    "1080p (agranda si hace falta)": 1080,
    "Máxima calidad (resolución mayor del proyecto)": None,
    "Calidad 720p": 720,
    "480p": 480,
    "360p": 360,
    "240p": 240,
}
ALTURA_PREVIEW = 360


class LectorWLMP(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Editor WLMP Ligero")
        self.geometry("1100x700")
        self.proyecto = None
        self.video_cap = None
        self.reproduciendo = False
        self.ocupado = False
        self._intervalo = 33
        self._construir_ui()

        # Preparar ffmpeg y detectar encoder SIN bloquear la ventana
        self.estado.config(text="Preparando FFmpeg...")
        threading.Thread(target=self._preparar_ffmpeg, daemon=True).start()

    def _preparar_ffmpeg(self):
        try:
            exe = obtener_ffmpeg()
            gpu = qsv_disponible(exe)
            texto = "Listo. Encoder: " + ("Intel Quick Sync (GPU)" if gpu else "CPU (libx264)")
        except Exception as e:
            texto = f"No se pudo preparar FFmpeg: {e}"
        self.after(0, lambda: self.estado.config(text=texto))

    def _construir_ui(self):
        barra = ttk.Frame(self, padding=6)
        barra.pack(fill="x")

        ttk.Button(barra, text="Abrir .wlmp…", command=self.abrir).pack(side="left")

        self.btn_play = ttk.Button(barra, text="▶ Previsualizar", command=self.previsualizar, state="disabled")
        self.btn_play.pack(side="left", padx=10)

        self.combo_modo = ttk.Combobox(barra, values=list(MODOS), state="readonly", width=28)
        self.combo_modo.current(list(MODOS).index("1440p (agranda si hace falta)"))
        self.combo_modo.pack(side="left")

        ttk.Label(barra, text="Filtro:").pack(side="left", padx=(8, 2))
        self.combo_filtro = ttk.Combobox(barra, values=list(FILTROS), state="readonly", width=24)
        self.combo_filtro.current(0)
        self.combo_filtro.pack(side="left")

        self.var_prueba = tk.BooleanVar(value=False)
        ttk.Checkbutton(barra, text="Solo prueba:", variable=self.var_prueba).pack(side="left", padx=(8, 0))
        self.var_minutos = tk.StringVar(value="10")
        ttk.Spinbox(barra, from_=1, to=600, increment=1, width=5, textvariable=self.var_minutos).pack(side="left")
        ttk.Label(barra, text="min").pack(side="left", padx=(2, 0))

        self.var_subs = tk.BooleanVar(value=False)
        ttk.Checkbutton(barra, text="Quitar subtítulo", variable=self.var_subs).pack(side="left", padx=(8, 0))

        self.btn_export = ttk.Button(barra, text="💾 Exportar MP4 Final", command=self.exportar_final, state="disabled")
        self.btn_export.pack(side="left", padx=10)

        self.btn_pausa = ttk.Button(barra, text="⏸ Pausar", command=self.pausar, state="disabled")
        self.btn_pausa.pack(side="left")

        self.lbl_archivo = ttk.Label(barra, text="(ningún archivo)")
        self.lbl_archivo.pack(side="left", padx=10)

        panel = ttk.PanedWindow(self, orient="horizontal")
        panel.pack(fill="both", expand=True, padx=6, pady=6)

        izq = ttk.Frame(panel)
        panel.add(izq, weight=1)

        ttk.Label(izq, text="Medios referenciados").pack(anchor="w")
        self.lista_medios = tk.Listbox(izq, height=5)
        self.lista_medios.pack(fill="both", expand=True, pady=(0, 6))

        ttk.Label(izq, text="Pistas / Clips").pack(anchor="w")
        cols = ("pista", "tipo", "clip", "inicio", "fin", "duracion", "ruta")
        self.tabla = ttk.Treeview(izq, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (100, 70, 50, 80, 80, 80, 260)):
            self.tabla.heading(c, text=c.capitalize())
            self.tabla.column(c, width=w, anchor="w")
        self.tabla.pack(fill="both", expand=True)

        der = ttk.Frame(panel)
        panel.add(der, weight=2)

        ttk.Label(der, text="Previsualización").pack(anchor="w")
        self.canvas_video = tk.Canvas(der, bg="black")
        self.canvas_video.pack(fill="both", expand=True)

        ttk.Label(der, text="Detalle del proyecto").pack(anchor="w", pady=(10, 0))
        self.txt_detalle = tk.Text(der, wrap="word", height=8)
        self.txt_detalle.pack(fill="x")

        self.barra_progreso = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.barra_progreso.pack(fill="x", side="bottom", padx=6, pady=2)

        self.estado = ttk.Label(self, text="Listo.", relief="sunken", anchor="w")
        self.estado.pack(fill="x", side="bottom")

    # ---------- utilidades de UI ----------
    def _bloquear(self, ocupado):
        self.ocupado = ocupado
        st = "disabled" if ocupado else "normal"
        self.btn_play.config(state=st)
        self.btn_export.config(state=st)
        self.combo_modo.config(state="disabled" if ocupado else "readonly")
        self.combo_filtro.config(state="disabled" if ocupado else "readonly")

    def _actualizar_barra(self, porcentaje):
        self.barra_progreso.configure(value=porcentaje)

    def _actualizar_estado(self, texto):
        self.estado.config(text=texto)

    def _lanzar_trabajo(self, altura, destino, al_terminar, con_audio=True, agrandar=True, limite_seg=None,
                        quitar_subs=False, filtro_video=""):
        """Ejecuta crear_video_final en un hilo y reactiva la UI al terminar."""
        self._bloquear(True)
        self.barra_progreso["value"] = 0

        def tarea():
            try:
                self.advertencias = crear_video_final(
                    self.proyecto, destino, altura=altura, con_audio=con_audio, agrandar=agrandar,
                    progreso_callback=lambda p: self.after(0, self._actualizar_barra, p),
                    estado_callback=lambda t: self.after(0, self._actualizar_estado, t),
                    limite_seg=limite_seg, quitar_subs=quitar_subs, filtro_video=filtro_video,
                ) or []
                self.after(0, al_terminar)
            except Exception as e:
                self.after(0, lambda msg=str(e): self._error_trabajo(msg))

        threading.Thread(target=tarea, daemon=True).start()

    def _error_trabajo(self, msg):
        self.barra_progreso["value"] = 0
        self.estado.config(text="Error.")
        self._bloquear(False)
        messagebox.showerror("Error", msg)

    # ---------- abrir proyecto ----------
    def abrir(self):
        ruta = filedialog.askopenfilename(title="Abrir proyecto", filetypes=[("Movie Maker", "*.wlmp")])
        if not ruta:
            return
        try:
            self.proyecto = leer_wlmp(ruta)
            self._cargar_en_ui()
            self.estado.config(text=f"Cargado: {ruta}")
            if not self.ocupado:
                self.btn_play.config(state="normal")
                self.btn_export.config(state="normal")
        except Exception:
            messagebox.showerror("Error", traceback.format_exc())

    def _cargar_en_ui(self):
        p = self.proyecto
        self.lbl_archivo.config(text=p["nombre"])
        self.lista_medios.delete(0, tk.END)
        for m in p["medios"]:
            self.lista_medios.insert(tk.END, f"{'✅' if os.path.exists(m) else '❌'} {os.path.basename(m)}")

        for fila in self.tabla.get_children():
            self.tabla.delete(fila)
        for pista in p["pistas"]:
            for j, clip in enumerate(pista["clips"], 1):
                self.tabla.insert("", "end", values=(
                    f"Pista {pista['id']} ({pista['tipo']})", "-", j,
                    f"{clip['in']:.1f}s", f"{clip['out']:.1f}s", f"{clip['duracion']:.1f}s",
                    os.path.basename(clip["ruta"])))

        # Avisos de lo que este editor aun no replica
        # (los gaps de la pista de audio ya se aplican; solo avisamos los de la pista de video)
        con_gap = sum(1 for pi in p["pistas"] if pi["tipo"] == "Principal"
                      for c in pi["clips"] if c["gap"] > 0)
        con_speed = sum(1 for pi in p["pistas"] for c in pi["clips"] if abs(c.get("speed", 1) - 1) > 1e-6)
        self.txt_detalle.delete("1.0", tk.END)
        self.txt_detalle.insert(tk.END, f"Pistas: {len(p['pistas'])} | Medios: {len(p['medios'])}\n")
        for linea in p.get("debug", []):
            self.txt_detalle.insert(tk.END, linea + "\n")
        for t in p.get("titulos", []):
            self.txt_detalle.insert(
                tk.END, f"Caption: «{t['texto']}» de {t['inicio']:.1f}s a {t['fin']:.1f}s"
                        f" ({t['familia']}{', con efecto' if t['animado'] else ''})\n")
        if p.get("no_soportado"):
            self.txt_detalle.insert(
                tk.END, "AVISO: este editor todavía NO aplica: " + ", ".join(p["no_soportado"]) + ".\n")
        if con_gap or con_speed:
            self.txt_detalle.insert(
                tk.END,
                f"AVISO: {con_gap} clips con espacio vacío (gap) y {con_speed} con velocidad "
                f"distinta de 1x. Este editor los ignora, el resultado diferirá de Movie Maker.\n")

    # ---------- exportar ----------
    def exportar_final(self):
        if not self.proyecto or self.ocupado:
            return
        limite = None
        if self.var_prueba.get():          # prueba: solo los primeros N minutos
            try:
                minutos = float(self.var_minutos.get().replace(",", "."))
                if minutos <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "En «Solo prueba» escribe un número de minutos mayor que 0 (por ejemplo 10).")
                return
            limite = minutos * 60
        ruta_salida = filedialog.asksaveasfilename(
            title="Guardar video final como...", defaultextension=".mp4",
            filetypes=[("Video MP4", "*.mp4")])
        if not ruta_salida:
            return
        altura = MODOS[self.combo_modo.get()]
        self.estado.config(text="Iniciando exportación...")
        self._lanzar_trabajo(altura, ruta_salida, lambda: self._finalizar_exportacion(ruta_salida),
                             limite_seg=limite, quitar_subs=self.var_subs.get(),
                             filtro_video=FILTROS[self.combo_filtro.get()])

    def _finalizar_exportacion(self, ruta_salida):
        self.barra_progreso["value"] = 100
        self.estado.config(text="¡Video exportado con éxito!")
        self._bloquear(False)
        extra = ("\n\nAvisos:\n- " + "\n- ".join(self.advertencias)) if getattr(self, "advertencias", None) else ""
        messagebox.showinfo("Éxito", f"El video se ha guardado en:\n{ruta_salida}{extra}")
        try:
            os.startfile(os.path.dirname(ruta_salida))
        except Exception:
            pass

    # ---------- previsualizar ----------
    def previsualizar(self):
        if not self.proyecto or self.ocupado:
            return
        # Liberar el archivo antes de que ffmpeg lo sobrescriba
        self.reproduciendo = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.btn_pausa.config(state="disabled")

        archivo_temp = os.path.join(tempfile.gettempdir(), "preview_wlmp.mp4")
        self.estado.config(text="Generando previsualización...")
        # Preview: 360p, solo video (sin audio, porque OpenCV no lo reproduce)
        self._lanzar_trabajo(ALTURA_PREVIEW, archivo_temp,
                             lambda: self._preview_lista(archivo_temp), con_audio=False, agrandar=False,
                             filtro_video=FILTROS[self.combo_filtro.get()])

    def _preview_lista(self, archivo):
        self._bloquear(False)
        self._reproducir_video(archivo)

    def _reproducir_video(self, archivo):
        self.estado.config(text="Reproduciendo previsualización (sin audio)...")
        if self.video_cap is not None:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(archivo)
        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
        self._intervalo = max(1, int(1000 / fps))
        self.reproduciendo = True
        self.btn_pausa.config(state="normal")
        self._actualizar_frame()

    def _actualizar_frame(self):
        if not self.reproduciendo or self.video_cap is None:
            return

        ret, frame = self.video_cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            canvas_w = self.canvas_video.winfo_width()
            canvas_h = self.canvas_video.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                h, w, _ = frame_rgb.shape
                aspect = w / h
                if canvas_w / canvas_h > aspect:
                    new_h = canvas_h
                    new_w = int(new_h * aspect)
                else:
                    new_w = canvas_w
                    new_h = int(new_w / aspect)
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

            img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas_video.delete("all")
            self.canvas_video.create_image(canvas_w // 2, canvas_h // 2, image=img, anchor="center")
            self.canvas_video.image = img

            self.after(self._intervalo, self._actualizar_frame)
        else:
            self.reproduciendo = False
            self.btn_pausa.config(state="disabled")
            self.estado.config(text="Reproducción finalizada.")

    def pausar(self):
        self.reproduciendo = False
        self.btn_pausa.config(state="disabled")
        self.estado.config(text="Pausado (dale a Previsualizar para reiniciar).")


if __name__ == "__main__":
    app = LectorWLMP()
    app.mainloop()
