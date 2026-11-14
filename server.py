"""
SPHEREx Sky Blinker — бекенд.

Дані:
  * SPHEREx Quick Release (QR2/QR3...) — NASA/IPAC IRSA
      - пошук знімків: TAP (таблиці spherex.artifact + spherex.plane)
      - вирізки (cutouts): IRSA cutout-сервіс (?center=RA,DEC&size=deg до URL FITS-файлу)
  * Gaia DR3 — ESA Gaia TAP (резерв: VizieR TAP)

Ендпоінти (під фронтенд index.html):
  GET  /api/cutouts?ra=&dec=&radius_deg=&invert=&band=&max_epochs=&lam_min=&lam_max=
  GET  /api/gaia?ra=&dec=&radius_deg=
  GET  /api/wise?ra=&dec=&radius_deg=
  GET  /api/sed?ra=&dec=&max=
  GET  /api/discoveries
  POST /api/discoveries

Запуск:  pip install -r requirements.txt && python server.py
"""
import base64
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import requests
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from scipy.ndimage import map_coordinates

# --------------------------------------------------------------------------
# Налаштування
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
DISCOVERIES_FILE = os.path.join(BASE_DIR, "discoveries.json")

IRSA_BASE = "https://irsa.ipac.caltech.edu/"
IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP/sync"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

IMG_SIZE = 256            # розмір вихідних PNG (px)
DEFAULT_MAX_EPOCHS = 8    # скільки кадрів віддавати в блінк
SED_CUTOUT_DEG = 0.02     # розмір вирізки для спектру (~10 пікселів SPHEREx)
SED_DEFAULT_MAX = 48      # максимум знімків для одного спектру (кожен ~ кілька МБ)
HTTP_TIMEOUT = 180
WORKERS = 6
MAX_EPOCHS_LIMIT = 32            # верхня межа max_epochs
SED_MAX_LIMIT = 96               # верхня межа max для /api/sed
MAX_RADIUS_DEG = 0.5             # верхня межа radius_deg
CACHE_MAX_BYTES = 2 * 1024 ** 3  # ~2 ГБ дискового кешу

os.makedirs(CACHE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spherex")

app = Flask(__name__, static_folder=None)
http = requests.Session()
http.headers["User-Agent"] = "SPHEREx-Sky-Blinker/1.0 (hackathon)"


# --------------------------------------------------------------------------
# Допоміжне: TAP, кеш, час
# --------------------------------------------------------------------------
def tap_query(url, adql, timeout=HTTP_TIMEOUT):
    """Синхронний TAP-запит, повертає список словників (CSV)."""
    r = http.post(
        url,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
        timeout=timeout,
    )
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


_cache_lock = threading.Lock()
_cache_writes = 0


def _looks_like_fits(data):
    return data.startswith((b"SIMPLE", b"\x1f\x8b"))  # FITS або gzip-FITS


def _prune_cache():
    """Видаляє найстаріші файли кешу, якщо він перевищив CACHE_MAX_BYTES."""
    try:
        files = []
        for e in os.scandir(CACHE_DIR):
            if e.is_file() and e.name.endswith(".fits"):
                st = e.stat()
                files.append((st.st_mtime, st.st_size, e.path))
        total = sum(f[1] for f in files)
        for _, size, p in sorted(files):
            if total <= CACHE_MAX_BYTES:
                break
            try:
                os.remove(p)
                total -= size
            except OSError:
                pass
    except OSError as e:
        log.debug("cache prune failed: %s", e)


def cached_get(url, params=None):
    """Завантажує FITS (з дисковим кешем) і повертає bytes. HTML-помилки з кодом 200 не кешуються."""
    global _cache_writes
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".fits")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = f.read()
            if _looks_like_fits(data):
                return data
        except OSError:
            pass
        try:  # пошкоджений запис кешу — видаляємо й качаємо знову
            os.remove(path)
        except OSError:
            pass
    r = http.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.content
    if not _looks_like_fits(data):
        raise ValueError(f"відповідь не є FITS ({r.headers.get('Content-Type', '?')}): {data[:80]!r}")
    # унікальне ім'я tmp: паралельні потоки/процеси не псують один одного
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    with _cache_lock:
        _cache_writes += 1
        do_prune = _cache_writes % 20 == 0
    if do_prune:
        _prune_cache()
    return data


def mjd_to_iso(mjd):
    return (datetime(1858, 11, 17) + timedelta(days=float(mjd))).strftime("%Y-%m-%d %H:%M")


def f_or_none(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


BAND_RE = re.compile(r"[A-Za-z0-9_.\-]{1,40}")


def parse_band(value):
    """Назва смуги йде в ADQL — пропускаємо лише безпечні символи."""
    if not value:
        return None
    if not BAND_RE.fullmatch(value):
        raise ValueError("bad band name")
    return value


def parse_coords(args):
    """(ra, dec) у градусах; KeyError/ValueError → 400."""
    ra, dec = float(args["ra"]), float(args["dec"])
    if not (math.isfinite(ra) and math.isfinite(dec)):
        raise ValueError("ra/dec must be finite numbers")
    if not -90.0 <= dec <= 90.0:
        raise ValueError("dec must be within [-90, 90]")
    return ra % 360.0, dec


def parse_radius(args, default=0.06):
    r = float(args.get("radius_deg", default))
    if not math.isfinite(r) or r <= 0:
        raise ValueError("radius_deg must be a positive number")
    return min(r, MAX_RADIUS_DEG)


def parse_int(args, name, default, lo, hi):
    return max(lo, min(int(args.get(name, default)), hi))


def _version_key(v):
    """'10.1' > '9.3' (числове, а не лексикографічне порівняння версій)."""
    v = v or ""
    return (tuple(int(n) for n in re.findall(r"\d+", v)), v)


# --------------------------------------------------------------------------
# SPHEREx: пошук знімків, що покривають точку
# --------------------------------------------------------------------------
def find_spherex_images(ra, dec, band=None):
    """
    Усі спектральні зображення SPHEREx, що покривають (ra, dec), за часом.
    Дублі (та сама експозиція в різних версіях обробки/релізах) відсіюються —
    лишається найновіша версія.
    """
    band = parse_band(band)
    band_clause = f"AND p.energy_bandpassname = '{band}'" if band else ""
    adql = f"""
        SELECT a.uri AS uri,
               p.time_bounds_lower AS mjd,
               p.energy_bandpassname AS band,
               p.provenance_version AS version
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {ra:.7f}, {dec:.7f}), p.poly)
        {band_clause}
        ORDER BY p.time_bounds_lower
    """
    rows = tap_query(IRSA_TAP, adql)
    best = {}
    for r in rows:
        mjd = f_or_none(r.get("mjd"))
        if mjd is None or not r.get("uri"):
            continue
        key = (round(mjd, 4), r["band"])
        if key not in best or _version_key(r.get("version")) > _version_key(best[key].get("version")):
            best[key] = {
                "url": IRSA_BASE + r["uri"].lstrip("/"),
                "mjd": mjd,
                "band": r["band"],
                "version": r.get("version"),
            }
    return sorted(best.values(), key=lambda x: x["mjd"])


def fetch_cutout(url, ra, dec, size_deg):
    data = cached_get(url, {"center": f"{ra:.7f},{dec:.7f}", "size": f"{size_deg:.5f}"})
    return fits.open(io.BytesIO(data), memmap=False)


# --------------------------------------------------------------------------
# Перепроєкція вирізки на північ-вгору сітку (щоб епохи співпадали з Gaia)
# --------------------------------------------------------------------------
def to_north_up(data, wcs, ra0, dec0, size_deg, n=IMG_SIZE, order=1):
    """
    Ресемплінг у тангенціальну проєкцію: північ вгору, схід (RA↑) ліворуч —
    рівно так, як фронтенд малює Gaia-оверлей і рахує координати кліку.
    """
    idx = (np.arange(n) + 0.5) / n
    xi = (0.5 - idx)[None, :] * np.ones((n, 1)) * math.radians(size_deg)   # на схід
    eta = (0.5 - idx)[:, None] * np.ones((1, n)) * math.radians(size_deg)  # на північ
    d0, a0 = math.radians(dec0), math.radians(ra0)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc, cosc = np.sin(c), np.cos(c)
        dec = np.arcsin(np.where(rho == 0, np.sin(d0),
                                 cosc * np.sin(d0) + eta * sinc * np.cos(d0) / rho))
        ra = a0 + np.arctan2(xi * sinc, rho * np.cos(d0) * cosc - eta * np.sin(d0) * sinc)
    ra_deg, dec_deg = np.degrees(ra) % 360.0, np.degrees(dec)

    h, w = data.shape
    if not np.isfinite(data).any():
        return np.full((n, n), np.nan), 0.0
    x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1) & np.isfinite(x) & np.isfinite(y)

    clean = np.where(np.isfinite(data), data, np.nanmedian(data))
    out = map_coordinates(clean, [np.clip(np.nan_to_num(y), 0, h - 1),
                                  np.clip(np.nan_to_num(x), 0, w - 1)],
                          order=order, mode="nearest")
    out[~valid] = np.nan
    return out, float(valid.mean())


def render_png(img, invert):
    """asinh-стретч навколо фону → PNG data URL."""
    ok = np.isfinite(img)
    if ok.sum() < 10:
        return None
    vals = img[ok]
    med = np.median(vals)
    sig = 1.4826 * np.median(np.abs(vals - med)) or (np.std(vals) or 1.0)
    z = np.arcsinh((img - med) / sig)
    z = np.clip((z + 1.0) / (np.arcsinh(12.0) + 1.0), 0, 1)
    z[~ok] = 0
    arr = (z * 255).astype(np.uint8)
    if invert:
        arr = 255 - arr
        arr[~ok] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pick_evenly(items, k):
    if len(items) <= k:
        return items
    pos = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(pos))]


# --------------------------------------------------------------------------
# Метадані епохи та прапорці якості (розширення FLAGS у SPHEREx L2)
# Номери бітів — за IRSA-ноутбуком "Introduction to SPHEREx Spectral Images".
# --------------------------------------------------------------------------
FLAG_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7,
    "MISSING_DATA": 9, "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "CROSSTALK": 20, "SOURCE": 21,
    "GHOST": 22, "GHOST_FPA": 23, "GHOST_EXT": 24, "STREAK": 25, "BLOOM": 26,
    "SNOWBALL": 27, "HALO": 28,
}
# Дефекти детектора/обробки (червоний) та артефакти-забруднення (жовтий).
# SOURCE / DICHROIC / FULLSAMPLE — лише інформаційні, на маску не йдуть.
DEFECT_FLAGS = ["OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD",
                "PHANMISS", "NONLINEAR"]
ARTIFACT_FLAGS = ["TRANSIENT", "PERSIST", "OUTLIER", "CROSSTALK", "GHOST", "GHOST_FPA",
                  "GHOST_EXT", "STREAK", "BLOOM", "SNOWBALL", "HALO"]


def _bitmask(names):
    return sum(1 << FLAG_BITS[n] for n in names)


DEFECT_MASK = _bitmask(DEFECT_FLAGS)
ARTIFACT_MASK = _bitmask(ARTIFACT_FLAGS)


def hdr_get(hdul, key):
    """Значення ключа з заголовка IMAGE або PRIMARY (або None)."""
    for h in (hdul["IMAGE"].header, hdul[0].header):
        v = h.get(key)
        if v not in (None, ""):
            return v
    return None


def epoch_meta(hdul, wcs, ra, dec):
    """Метадані кадру: дата, детектор, OBSID, довжина хвилі в цілі, рівень зодіаку."""
    meta = {}
    # DATE за стандартом FITS — дата створення файлу; час спостереження — DATE-OBS.
    for key, name in (("DATE-OBS", "date_obs"), ("DATE", "date_created"),
                      ("DETECTOR", "detector"), ("OBSID", "obsid"), ("PSF_FWHM", "psf_fwhm")):
        try:
            v = hdr_get(hdul, key)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                v = f_or_none(v)  # NaN/inf дали б невалідний JSON
                if v is None:
                    continue
            else:
                v = str(v)
            meta[name] = v
        except Exception:
            pass
    try:
        x, y = wcs.all_world2pix(ra, dec, 0)
        lam = f_or_none(wavelength_um(hdul, hdul["IMAGE"].header, x, y))
        if lam is not None:
            meta["lambda_um"] = round(lam, 4)
    except Exception as e:
        log.debug("wavelength at target failed: %s", e)
    try:
        if "ZODI" in hdul:
            z = f_or_none(np.nanmedian(np.array(hdul["ZODI"].data, float)))
            if z is not None:
                meta["zodi_mjysr"] = round(z, 3)
    except Exception:
        pass
    return meta


def flags_info(hdul, wcs, data_shape, ra, dec, size, n=IMG_SIZE):
    """
    Перепроєктує FLAGS на ту саму сітку, що й зображення, і повертає:
      image  — RGBA-PNG маска (червоне — дефекти, жовте — артефакти),
      pct    — % площі кадру з кожним прапорцем,
      center — прапорці в околі цілі (центр кадру).
    """
    if "FLAGS" not in hdul:
        return None
    fl = np.array(hdul["FLAGS"].data, dtype=float)
    if fl.shape != data_shape:
        return None
    raw, _ = to_north_up(fl, wcs, ra, dec, size, n=n, order=0)
    valid = np.isfinite(raw)
    if valid.sum() == 0:
        return None
    grid = np.nan_to_num(raw).astype(np.int64)

    pct = {}
    for name, bit in FLAG_BITS.items():
        frac = float(((grid >> bit) & 1)[valid].mean())
        if frac > 0:
            pct[name] = round(frac * 100, 1)

    c, r = n // 2, max(3, n // 40)
    patch = grid[c - r:c + r + 1, c - r:c + r + 1]
    center = [name for name, bit in FLAG_BITS.items()
              if name != "FULLSAMPLE" and np.any((patch >> bit) & 1)]

    rgba = np.zeros((n, n, 4), np.uint8)
    rgba[(grid & ARTIFACT_MASK) != 0] = (251, 191, 36, 150)
    rgba[(grid & DEFECT_MASK) != 0] = (239, 68, 68, 190)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return {
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "pct": pct,
        "center": center,
        # внутрішнє (не для JSON — api_cutouts його забирає): піксельна маска для стека
        "_bad": (grid & (DEFECT_MASK | ARTIFACT_MASK)) != 0,
    }


# --------------------------------------------------------------------------
# /api/cutouts — кадри для блінк-компаратора
# --------------------------------------------------------------------------
@app.get("/api/cutouts")
def api_cutouts():
    try:
        ra, dec = parse_coords(request.args)
        fov = parse_radius(request.args)
        invert = request.args.get("invert", "true").lower() in ("1", "true", "yes")
        band = parse_band(request.args.get("band"))  # напр. SPHEREx-D2
        max_epochs = parse_int(request.args, "max_epochs", DEFAULT_MAX_EPOCHS, 1, MAX_EPOCHS_LIMIT)
        # Необов'язковий фільтр за довжиною хвилі в цілі (µm). У межах одного детектора λ
        # змінюється від візиту до візиту, тож без фільтра кадри блінку можуть бути на різних λ.
        lam_min = f_or_none(request.args.get("lam_min"))
        lam_max = f_or_none(request.args.get("lam_max"))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"bad parameters: {e}"), 400

    size = max(0.01, min(2 * fov, 0.5))  # cutout-сервіс: 0.01–0.5°
    try:
        images = find_spherex_images(ra, dec, band)
    except Exception as e:
        log.exception("TAP failed")
        return jsonify(error=f"IRSA TAP error: {e}"), 502

    if not images:
        return jsonify(epochs=[], size_deg=size,
                       note="SPHEREx не покриває цю точку (або дані ще не завантажені)")

    # Для блінку беремо ОДИН детектор/смугу — із найбільшою кількістю повторних візитів,
    # інакше різні довжини хвилі дають хибні "зміни яскравості".
    if not band:
        counts = {}
        for im in images:
            counts[im["band"]] = counts.get(im["band"], 0) + 1
        band = max(counts, key=counts.get)
        images = [im for im in images if im["band"] == band]

    lam_filter = lam_min is not None or lam_max is not None
    # запас на невдалі/крайові кадри (і на кадри, відсіяні фільтром за λ)
    candidates = pick_evenly(images, 3 * max_epochs + 4 if lam_filter else max_epochs + 4)

    def work(im):
        try:
            with fetch_cutout(im["url"], ra, dec, size) as hdul:
                hdu = hdul["IMAGE"]
                data = np.array(hdu.data, dtype=float)
                wcs = WCS(hdu.header)
                # метадані й прапорці читаємо, поки файл відкритий
                meta = epoch_meta(hdul, wcs, ra, dec)
                try:
                    flags = flags_info(hdul, wcs, data.shape, ra, dec, size)
                except Exception as e:
                    log.warning("flags failed %s: %s", im["url"], e)
                    flags = None
            bad = flags.pop("_bad") if flags else None
            if lam_filter:
                lam = meta.get("lambda_um")
                if lam is None or (lam_min is not None and lam < lam_min) \
                        or (lam_max is not None and lam > lam_max):
                    return None
            img, cover = to_north_up(data, wcs, ra, dec, size)
            if cover < 0.5:
                return None
            png = render_png(img, invert)
            meta["version"] = im.get("version")
            return ({"image": png, "mjd": im["mjd"], "survey": im["band"], "arr": img,
                     "bad": bad, "meta": meta, "flags": flags}
                    if png else None)
        except Exception as e:
            log.warning("cutout failed %s: %s", im["url"], e)
            return None

    with ThreadPoolExecutor(WORKERS) as ex:
        results = [r for r in ex.map(work, candidates) if r]

    # Середнє всіх вдалих кадрів: кожен кадр спершу очищається від фону (медіана) і від
    # пікселів з прапорцями дефектів/артефактів, потім усереднюється — зростає S/N для слабких
    # статичних джерел. Рухомі об'єкти при цьому розмазуються (видно як «хвіст» між епохами).
    stack_png, stack_n = None, 0
    if results:
        frames = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for r in results:
                a = r["arr"].copy()
                if r["bad"] is not None:
                    a[r["bad"]] = np.nan
                frames.append(a - np.nanmedian(a))
            stacked = np.nanmean(np.stack(frames), axis=0)
        stack_png = render_png(stacked, invert)
        stack_n = len(results)

    results = pick_evenly(results, max_epochs)
    epochs = [
        {"epoch": i + 1, "image": r["image"], "survey": r["survey"],
         "date": f"{mjd_to_iso(r['mjd'])} (MJD {r['mjd']:.3f})",
         "meta": r["meta"], "flags": r["flags"]}
        for i, r in enumerate(results)
    ]
    # size_deg — ФАКТИЧНИЙ розмір кадру (після обрізання radius_deg до 0.01–0.5°):
    # фронтенд має малювати Gaia/WISE-оверлеї саме за ним.
    return jsonify(epochs=epochs, band=band, size_deg=size, total_visits=len(images),
                   stack={"image": stack_png, "n_frames": stack_n} if stack_png else None)


# --------------------------------------------------------------------------
# /api/gaia — Gaia DR3 у конусі
# --------------------------------------------------------------------------
@app.get("/api/gaia")
def api_gaia():
    try:
        ra, dec = parse_coords(request.args)
        r = parse_radius(request.args) * 1.5  # з запасом на кути кадру
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e), stars=[]), 400

    def to_star(row, k_id, k_ra, k_dec, k_pmra, k_pmdec, k_mag):
        mag = f_or_none(row[k_mag])
        return {
            "source_id": row[k_id],
            "ra": float(row[k_ra]), "dec": float(row[k_dec]),
            # None = у Gaia немає власного руху (2-параметричне рішення), а не «рух = 0»
            "pmra": f_or_none(row[k_pmra]), "pmdec": f_or_none(row[k_pmdec]),
            "mag": 99.0 if mag is None else mag,
        }

    try:
        adql = f"""
            SELECT TOP 800 source_id, ra, dec, pmra, pmdec, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f}))
            ORDER BY phot_g_mean_mag ASC
        """
        # список збирається повністю або не збирається взагалі — без дублів при фолбеку
        stars = [to_star(row, "source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag")
                 for row in tap_query(GAIA_TAP, adql, timeout=60)]
    except Exception as e:
        log.warning("ESA Gaia failed (%s), trying VizieR", e)
        try:
            adql = f"""
                SELECT TOP 800 Source, RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag
                FROM "I/355/gaiadr3"
                WHERE 1 = CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f}))
                ORDER BY Gmag ASC
            """
            stars = [to_star(row, "Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag")
                     for row in tap_query(VIZIER_TAP, adql, timeout=60)]
        except Exception as e2:
            return jsonify(error=f"Gaia unavailable: {e2}", stars=[]), 502
    return jsonify(stars=stars)


# --------------------------------------------------------------------------
# /api/wise — джерела AllWISE у конусі (оверлей)
# --------------------------------------------------------------------------
@app.get("/api/wise")
def api_wise():
    try:
        ra, dec = parse_coords(request.args)
        r = parse_radius(request.args) * 1.5
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e), sources=[]), 400

    # AllWISE Source Catalog в IRSA; власні рухи (pmra/pmdec) у mas/yr.
    def build(cols):
        return f"""
            SELECT TOP 800 {cols}
            FROM allwise_p3as_psd
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f}))
            ORDER BY w2mpro ASC
        """

    base_cols = "designation, ra, dec, w1mpro, w2mpro, pmra, pmdec"
    try:
        # sigpmra/sigpmdec — похибки власного руху: без них вектори PM ненадійні
        rows = tap_query(IRSA_TAP, build(base_cols + ", sigpmra, sigpmdec"), timeout=60)
    except Exception:
        try:
            rows = tap_query(IRSA_TAP, build(base_cols), timeout=60)
        except Exception as e:
            log.warning("WISE query failed: %s", e)
            return jsonify(error=f"WISE unavailable: {e}", sources=[]), 502

    sources = []
    for row in rows:
        try:
            sources.append({
                "designation": row["designation"],
                "ra": float(row["ra"]), "dec": float(row["dec"]),
                "w1": f_or_none(row.get("w1mpro")),
                "w2": f_or_none(row.get("w2mpro")),
                "pmra": f_or_none(row.get("pmra")),    # None = власний рух не виміряно
                "pmdec": f_or_none(row.get("pmdec")),
                "sigpmra": f_or_none(row.get("sigpmra")),
                "sigpmdec": f_or_none(row.get("sigpmdec")),
            })
        except (KeyError, ValueError):
            continue
    return jsonify(sources=sources)


# --------------------------------------------------------------------------
# /api/sed — спектр SPHEREx у точці (спектрофотометрія по кожному знімку)
# --------------------------------------------------------------------------
def wavelength_um(hdul, header, x, y):
    """Довжина хвилі в µm у піксельній позиції (лінійно-змінний фільтр, WCS key 'W')."""
    swcs = WCS(header, fobj=hdul, key="W")
    swcs.sip = None
    res = swcs.pixel_to_world(x, y)
    if isinstance(res, (tuple, list)):
        res = res[0]
    try:
        return float(res.to(u.um).value)
    except Exception:
        return float(getattr(res, "value", res))


def photometry_point(url, ra, dec):
    """
    Потік (mJy) у точці: сума 3×3 пікселі мінус фон, × тілесний кут пікселя.
    Фон — медіана вирізки ПОЗА апертурою; пікселі з прапорцями дефектів/артефактів
    маскуються (їх кількість у апертурі повертається). Апертурної корекції немає,
    тож потік занижений.  Повертає (λ_µm, flux_mJy, n_flagged) або None.
    """
    with fetch_cutout(url, ra, dec, SED_CUTOUT_DEG) as hdul:
        hdu = hdul["IMAGE"]
        data = np.array(hdu.data, dtype=float)  # MJy/sr
        wcs = WCS(hdu.header)
        x, y = wcs.all_world2pix(ra, dec, 0)
        h, w = data.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        bad = np.zeros(data.shape, bool)
        if "FLAGS" in hdul:
            fl = np.array(hdul["FLAGS"].data, dtype=float)
            if fl.shape == data.shape:
                bad = (np.nan_to_num(fl).astype(np.int64) & (DEFECT_MASK | ARTIFACT_MASK)) != 0
        xi, yi = int(round(float(x))), int(round(float(y)))
        y0, y1, x0, x1 = max(yi - 1, 0), min(yi + 2, h), max(xi - 1, 0), min(xi + 2, w)
        patch = data[y0:y1, x0:x1]
        good = np.isfinite(patch) & ~bad[y0:y1, x0:x1]
        if not good.any():
            return None
        outside = np.isfinite(data) & ~bad
        outside[y0:y1, x0:x1] = False
        if not outside.any():
            return None
        bkg = float(np.median(data[outside]))
        scales = proj_plane_pixel_scales(wcs)  # deg/pix
        omega = (scales[0] * scales[1]) * (math.pi / 180.0) ** 2  # sr/pix
        flux_mjy = float(np.sum(patch[good] - bkg)) * omega * 1e9  # MJy → mJy
        n_flagged = int(good.size - good.sum())
        return wavelength_um(hdul, hdu.header, x, y), flux_mjy, n_flagged


@app.get("/api/sed")
def api_sed():
    try:
        ra, dec = parse_coords(request.args)
        max_n = parse_int(request.args, "max", SED_DEFAULT_MAX, 1, SED_MAX_LIMIT)
    except (KeyError, ValueError) as e:
        return jsonify(found=False, error=str(e)), 400

    try:
        images = find_spherex_images(ra, dec)
    except Exception as e:
        log.exception("TAP failed")
        return jsonify(found=False, error=str(e)), 502
    if not images:
        return jsonify(found=False, bands=[])

    chosen = pick_evenly(images, max_n)

    def work(im):
        try:
            return photometry_point(im["url"], ra, dec)
        except Exception as e:
            log.warning("sed point failed: %s", e)
            return None

    with ThreadPoolExecutor(WORKERS) as ex:
        pts = [p for p in ex.map(work, chosen)
               if p and f_or_none(p[0]) is not None and f_or_none(p[1]) is not None]

    pts.sort(key=lambda p: p[0])
    return jsonify(
        found=bool(pts),
        designation=f"SPHEREx @ {ra:.5f}, {dec:+.5f}",
        n_images_covering=len(images),
        method="3x3 px box; background = median outside aperture; flagged pixels masked; "
               "no aperture correction (flux is a lower limit)",
        bands=[{"wavelength_um": round(lam, 4), "flux_mjy": f, "n_flagged_px": nf}
               for lam, f, nf in pts],
    )


# --------------------------------------------------------------------------
# /api/discoveries — простий JSON-журнал
# --------------------------------------------------------------------------
_lock = threading.Lock()  # захищає лише потоки одного процесу → запускайте один процес


def _read_discoveries():
    if not os.path.exists(DISCOVERIES_FILE):
        return []
    try:
        with open(DISCOVERIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("очікувався JSON-масив")
        return data
    except ValueError as e:  # JSONDecodeError та UnicodeDecodeError — підкласи ValueError
        backup = f"{DISCOVERIES_FILE}.corrupt-{int(time.time())}"
        log.error("discoveries.json пошкоджено (%s) — збережено як %s", e, backup)
        try:
            os.replace(DISCOVERIES_FILE, backup)
        except OSError:
            pass
        return []


def _write_discoveries(data):
    """Атомарний запис: збій посеред нього не псує існуючий файл."""
    tmp = f"{DISCOVERIES_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DISCOVERIES_FILE)


@app.get("/api/discoveries")
def get_discoveries():
    with _lock:
        return jsonify(list(reversed(_read_discoveries())))


@app.post("/api/discoveries")
def post_discovery():
    item = request.get_json(silent=True)
    if not isinstance(item, dict):
        return jsonify(error="JSON object expected"), 400
    required = ("observer", "type", "coords")
    if not all(item.get(k) for k in required):
        return jsonify(error=f"required: {required}"), 400
    keep = ("id", "observer", "type", "coords", "displacement", "distance", "date")
    clean = {k: str(item.get(k, ""))[:200] for k in keep}
    with _lock:
        data = _read_discoveries()
        data.append(clean)
        _write_discoveries(data)
    return jsonify(ok=True), 201


# --------------------------------------------------------------------------
# Фронтенд
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)),
            threaded=True)
