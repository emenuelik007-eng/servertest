"""
SPHEREx Sky Blinker — бекенд.

Дані:
  * SPHEREx Quick Release (QR2/QR3...) — NASA/IPAC IRSA
      - пошук знімків: TAP (таблиці spherex.artifact + spherex.plane)
      - вирізки (cutouts): IRSA cutout-сервіс (?center=RA,DEC&size=deg до URL FITS-файлу)
  * Gaia DR3 — ESA Gaia TAP (резерв: VizieR TAP)

Ендпоінти (під фронтенд index.html):
  GET  /api/cutouts?ra=&dec=&radius_deg=&invert=&band=&max_epochs=
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
import threading
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


def cached_get(url, params=None):
    """Завантажує файл (з дисковим кешем) і повертає bytes."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".fits")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    r = http.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, path)
    return r.content


def mjd_to_iso(mjd):
    return (datetime(1858, 11, 17) + timedelta(days=float(mjd))).strftime("%Y-%m-%d %H:%M")


def f_or_none(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# SPHEREx: пошук знімків, що покривають точку
# --------------------------------------------------------------------------
def find_spherex_images(ra, dec, band=None):
    """
    Усі спектральні зображення SPHEREx, що покривають (ra, dec), за часом.
    Дублі (та сама експозиція в різних версіях обробки/релізах) відсіюються —
    лишається найновіша версія.
    """
    band_clause = f"AND p.energy_bandpassname = '{band}'" if band else ""
    adql = f"""
        SELECT a.uri AS uri,
               p.time_bounds_lower AS mjd,
               p.energy_bandpassname AS band,
               p.provenance_version AS version
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
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
        if key not in best or (r.get("version") or "") > (best[key].get("version") or ""):
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

    x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
    h, w = data.shape
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1) & np.isfinite(x) & np.isfinite(y)

    clean = np.where(np.isfinite(data), data, np.nanmedian(data))
    out = map_coordinates(clean, [np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)],
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
    Image.fromarray(arr, "L").save(buf, format="PNG", optimize=True)
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
    for key, name in (("DATE", "date_obs"), ("DETECTOR", "detector"),
                      ("OBSID", "obsid"), ("PSF_FWHM", "psf_fwhm")):
        try:
            v = hdr_get(hdul, key)
            if v is not None:
                meta[name] = v if isinstance(v, (int, float)) else str(v)
        except Exception:
            pass
    try:
        x, y = wcs.all_world2pix(ra, dec, 0)
        meta["lambda_um"] = round(wavelength_um(hdul, hdul["IMAGE"].header, x, y), 4)
    except Exception as e:
        log.debug("wavelength at target failed: %s", e)
    try:
        if "ZODI" in hdul:
            meta["zodi_mjysr"] = round(float(np.nanmedian(np.array(hdul["ZODI"].data, float))), 3)
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
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return {
        "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "pct": pct,
        "center": center,
    }


# --------------------------------------------------------------------------
# /api/cutouts — кадри для блінк-компаратора
# --------------------------------------------------------------------------
@app.get("/api/cutouts")
def api_cutouts():
    try:
        ra = float(request.args["ra"])
        dec = float(request.args["dec"])
        fov = float(request.args.get("radius_deg", 0.06))
        invert = request.args.get("invert", "true").lower() == "true"
        band = request.args.get("band")  # напр. SPHEREx-D2
        max_epochs = int(request.args.get("max_epochs", DEFAULT_MAX_EPOCHS))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"bad parameters: {e}"), 400

    size = max(0.01, min(2 * fov, 0.5))  # cutout-сервіс: 0.01–0.5°
    try:
        images = find_spherex_images(ra, dec, band)
    except Exception as e:
        log.exception("TAP failed")
        return jsonify(error=f"IRSA TAP error: {e}"), 502

    if not images:
        return jsonify(epochs=[], note="SPHEREx не покриває цю точку (або дані ще не завантажені)")

    # Для блінку беремо ОДИН детектор/смугу — із найбільшою кількістю повторних візитів,
    # інакше різні довжини хвилі дають хибні "зміни яскравості".
    if not band:
        counts = {}
        for im in images:
            counts[im["band"]] = counts.get(im["band"], 0) + 1
        band = max(counts, key=counts.get)
        images = [im for im in images if im["band"] == band]

    candidates = pick_evenly(images, max_epochs + 4)  # запас на невдалі/крайові кадри

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
            img, cover = to_north_up(data, wcs, ra, dec, size)
            if cover < 0.5:
                return None
            png = render_png(img, invert)
            meta["version"] = im.get("version")
            return ({"image": png, "mjd": im["mjd"], "survey": im["band"], "arr": img,
                     "meta": meta, "flags": flags}
                    if png else None)
        except Exception as e:
            log.warning("cutout failed %s: %s", im["url"], e)
            return None

    with ThreadPoolExecutor(WORKERS) as ex:
        results = [r for r in ex.map(work, candidates) if r]

    # Сума (середнє) усіх вдалих кадрів: кожен кадр спершу очищається від фону
    # (медіана), потім усереднюється — зростає S/N для слабких статичних джерел.
    # Рухомі об'єкти при цьому розмазуються (видно як «хвіст» між епохами).
    stack_png, stack_n = None, 0
    if results:
        frames = np.stack([r["arr"] - np.nanmedian(r["arr"]) for r in results])
        with np.errstate(all="ignore"):
            stacked = np.nanmean(frames, axis=0)
        stack_png = render_png(stacked, invert)
        stack_n = len(results)

    results = pick_evenly(results, max_epochs)
    epochs = [
        {"epoch": i + 1, "image": r["image"], "survey": r["survey"],
         "date": f"{mjd_to_iso(r['mjd'])} (MJD {r['mjd']:.3f})",
         "meta": r["meta"], "flags": r["flags"]}
        for i, r in enumerate(results)
    ]
    return jsonify(epochs=epochs, band=band, total_visits=len(images),
                   stack={"image": stack_png, "n_frames": stack_n} if stack_png else None)


# --------------------------------------------------------------------------
# /api/gaia — Gaia DR3 у конусі
# --------------------------------------------------------------------------
@app.get("/api/gaia")
def api_gaia():
    try:
        ra = float(request.args["ra"])
        dec = float(request.args["dec"])
        r = float(request.args.get("radius_deg", 0.06)) * 1.5  # з запасом на кути кадру
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e)), 400

    stars = []
    try:
        adql = f"""
            SELECT TOP 800 source_id, ra, dec, pmra, pmdec, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r}))
            ORDER BY phot_g_mean_mag ASC
        """
        for row in tap_query(GAIA_TAP, adql, timeout=60):
            stars.append({
                "source_id": row["source_id"],
                "ra": float(row["ra"]), "dec": float(row["dec"]),
                "pmra": f_or_none(row["pmra"]) or 0.0,
                "pmdec": f_or_none(row["pmdec"]) or 0.0,
                "mag": f_or_none(row["phot_g_mean_mag"]) or 99.0,
            })
    except Exception as e:
        log.warning("ESA Gaia failed (%s), trying VizieR", e)
        try:
            adql = f"""
                SELECT TOP 800 Source, RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag
                FROM "I/355/gaiadr3"
                WHERE 1 = CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), CIRCLE('ICRS', {ra}, {dec}, {r}))
                ORDER BY Gmag ASC
            """
            for row in tap_query(VIZIER_TAP, adql, timeout=60):
                stars.append({
                    "source_id": row["Source"],
                    "ra": float(row["RA_ICRS"]), "dec": float(row["DE_ICRS"]),
                    "pmra": f_or_none(row["pmRA"]) or 0.0,
                    "pmdec": f_or_none(row["pmDE"]) or 0.0,
                    "mag": f_or_none(row["Gmag"]) or 99.0,
                })
        except Exception as e2:
            return jsonify(error=f"Gaia unavailable: {e2}", stars=[]), 502
    return jsonify(stars=stars)


# --------------------------------------------------------------------------
# /api/wise — джерела AllWISE у конусі (оверлей)
# --------------------------------------------------------------------------
@app.get("/api/wise")
def api_wise():
    try:
        ra = float(request.args["ra"])
        dec = float(request.args["dec"])
        r = float(request.args.get("radius_deg", 0.06)) * 1.5
    except (KeyError, ValueError) as e:
        return jsonify(error=str(e), sources=[]), 400

    # AllWISE Source Catalog в IRSA; власні рухи (pmra/pmdec) у mas/yr.
    def build(cols):
        return f"""
            SELECT TOP 800 {cols}
            FROM allwise_p3as_psd
            WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r}))
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
                "pmra": f_or_none(row.get("pmra")) or 0.0,
                "pmdec": f_or_none(row.get("pmdec")) or 0.0,
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
    """Потік (mJy) у точці: сума 3×3 мінус фон, × тілесний кут пікселя."""
    with fetch_cutout(url, ra, dec, SED_CUTOUT_DEG) as hdul:
        hdu = hdul["IMAGE"]
        data = np.array(hdu.data, dtype=float)  # MJy/sr
        wcs = WCS(hdu.header)
        x, y = wcs.all_world2pix(ra, dec, 0)
        h, w = data.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        xi, yi = int(round(float(x))), int(round(float(y)))
        y0, y1, x0, x1 = max(yi - 1, 0), min(yi + 2, h), max(xi - 1, 0), min(xi + 2, w)
        patch = data[y0:y1, x0:x1]
        if not np.isfinite(patch).any():
            return None
        bkg = np.nanmedian(data)
        scales = proj_plane_pixel_scales(wcs)  # deg/pix
        omega = (scales[0] * scales[1]) * (math.pi / 180.0) ** 2  # sr/pix
        flux_mjy = float(np.nansum(patch - bkg)) * omega * 1e9  # MJy → mJy
        return wavelength_um(hdul, hdu.header, x, y), flux_mjy


@app.get("/api/sed")
def api_sed():
    try:
        ra = float(request.args["ra"])
        dec = float(request.args["dec"])
        max_n = int(request.args.get("max", SED_DEFAULT_MAX))
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
        pts = [p for p in ex.map(work, chosen) if p and math.isfinite(p[0])]

    pts.sort(key=lambda p: p[0])
    return jsonify(
        found=bool(pts),
        designation=f"SPHEREx @ {ra:.5f}, {dec:+.5f}",
        n_images_covering=len(images),
        bands=[{"wavelength_um": round(w, 4), "flux_mjy": f} for w, f in pts],
    )


# --------------------------------------------------------------------------
# /api/discoveries — простий JSON-журнал
# --------------------------------------------------------------------------
_lock = threading.Lock()


def _read_discoveries():
    if not os.path.exists(DISCOVERIES_FILE):
        return []
    with open(DISCOVERIES_FILE, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/discoveries")
def get_discoveries():
    with _lock:
        return jsonify(list(reversed(_read_discoveries())))


@app.post("/api/discoveries")
def post_discovery():
    item = request.get_json(silent=True) or {}
    required = ("observer", "type", "coords")
    if not all(item.get(k) for k in required):
        return jsonify(error=f"required: {required}"), 400
    keep = ("id", "observer", "type", "coords", "displacement", "distance", "date")
    clean = {k: str(item.get(k, ""))[:200] for k in keep}
    with _lock:
        data = _read_discoveries()
        data.append(clean)
        with open(DISCOVERIES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify(ok=True), 201


# --------------------------------------------------------------------------
# Фронтенд
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
