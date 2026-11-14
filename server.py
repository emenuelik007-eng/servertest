import io
import json
import base64
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from astroquery.skyview import SkyView
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="SPHEREx DR3 & Planet X Hunter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DISCOVERIES_FILE = BASE_DIR / "discoveries.json"

# URL сервісів NASA/IPAC IRSA для даних SPHEREx
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"


class DiscoveryItem(BaseModel):
    id: str
    observer: str
    type: str
    coords: str
    displacement: str
    distance: str
    date: str


def array_to_base64_png(data: np.ndarray, invert: bool = True) -> str:
    """Конвертує 2D-масив пікселів FITS у Base64 PNG з нормалізацією ZScale."""
    clean_data = np.nan_to_num(data, nan=np.nanmedian(data))
    norm = ImageNormalize(clean_data, interval=ZScaleInterval(), stretch=AsinhStretch())

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    cmap = "gray_r" if invert else "gray"
    ax.imshow(clean_data, norm=norm, cmap=cmap, origin="lower")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


# 1. Віддача головної HTML-сторінки
@app.get("/")
async def serve_frontend():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        html_path = BASE_DIR / "index(2).html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML файл не знайдено поруч із server.py")
    return FileResponse(html_path)


# 2. Завантаження спектральних/епохальних вирізок SPHEREx через NASA IRSA SIA v2
@app.get("/api/cutouts")
def get_sky_cutouts(
    ra: float = Query(..., description="Пряме сходження (градуси)"),
    dec: float = Query(..., description="Схилення (градуси)"),
    radius_deg: float = Query(0.05, description="Радіус вирізки у градусах"),
    invert: bool = Query(True, description="Інверсія (FITS негатив)")
):
    cache_key = f"spherex_cutouts_{ra:.4f}_{dec:.4f}_{radius_deg:.3f}_{int(invert)}.json"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        return JSONResponse(content=json.loads(cache_path.read_text(encoding="utf-8")))

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    epochs_data = []

    # Спроба отримати з IRSA SIA SPHEREx
    try:
        params = {
            "COLLECTION": "spherex",
            "POS": f"CIRCLE {ra} {dec} {radius_deg}",
            "FORMAT": "image/fits"
        }
        resp = requests.get(IRSA_SIA_URL, params=params, timeout=15)
        if resp.status_code == 200:
            votable = Table.read(io.BytesIO(resp.content), format="votable")
            if len(votable) > 0:
                if "t_min" in votable.colnames:
                    votable.sort("t_min")
                indices = np.linspace(0, len(votable) - 1, min(4, len(votable)), dtype=int)
                for idx_num, row_idx in enumerate(indices, start=1):
                    row = votable[row_idx]
                    fits_url = str(row["access_url"])
                    obs_date = str(row["t_min"]) if "t_min" in row.colnames else f"Frame #{idx_num}"
                    with fits.open(fits_url, cache=False) as hdul:
                        img_hdu = hdul[1] if (len(hdul) > 1 and hdul[1].data is not None) else hdul[0]
                        wcs = WCS(img_hdu.header)
                        size = u.Quantity((radius_deg * 2, radius_deg * 2), u.deg)
                        cutout = Cutout2D(img_hdu.data, position=coord, size=size, wcs=wcs, mode="partial")
                        b64_img = array_to_base64_png(cutout.data, invert=invert)
                        epochs_data.append({"epoch": idx_num, "survey": "SPHEREx IR", "date": obs_date, "image": b64_img})
    except Exception:
        pass

    # КАДРИ-ЗАМІНИ (Fallback): якщо IRSA нічого не повернув, використовуємо DSS SkyView для демонстрації блінк-компаратора
    if not epochs_data:
        fallback_surveys = [
            {"survey": "DSS1 Blue", "date": "1950-1990 (E1)"},
            {"survey": "DSS1 Red",  "date": "1950-1990 (E2)"},
            {"survey": "DSS2 Blue", "date": "1990-2000 (E3)"},
            {"survey": "DSS2 Red",  "date": "1990-2000 (E4)"},
        ]
        for idx, item in enumerate(fallback_surveys, start=1):
            try:
                images = SkyView.get_images(position=coord, survey=[item["survey"]], radius=radius_deg * u.deg, pixels=300)
                if images and len(images) > 0:
                    img_array = images[0][0].data
                    b64_img = array_to_base64_png(img_array, invert=invert)
                    epochs_data.append({"epoch": idx, "survey": f"SPHEREx-Sim ({item['survey']})", "date": item["date"], "image": b64_img})
            except Exception:
                pass

    response_payload = {"ra": ra, "dec": dec, "epochs": epochs_data}
    if epochs_data:
        cache_path.write_text(json.dumps(response_payload), encoding="utf-8")
    return response_payload

# 3. Отримання реальних зірок з Gaia DR3 для накладання на карту
@app.get("/api/gaia")
def get_gaia_sources(
    ra: float = Query(...),
    dec: float = Query(...),
    radius_deg: float = Query(0.05)
):
    query = f"""
    SELECT TOP 150 source_id, ra, dec, phot_g_mean_mag, pmra, pmdec, parallax
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
    """
    try:
        job = Gaia.launch_job(query)
        res = job.get_data()
        stars = []
        for row in res:
            stars.append({
                "source_id": str(row["source_id"]),
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "mag": float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] else 20.0,
                "pmra": float(row["pmra"]) if row["pmra"] else 0.0,
                "pmdec": float(row["pmdec"]) if row["pmdec"] else 0.0,
            })
        return {"count": len(stars), "stars": stars}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Помилка Gaia TAP: {str(e)}")


# 4. Отримання спектрофотометричного профілю (SED) з NASA IRSA (SPHEREx TAP)
@app.get("/api/sed")
def get_irsa_sed(ra: float = Query(...), dec: float = Query(...)):
    # Пошук найближчого спектрального джерела SPHEREx у радіусі 5 кутових секунд (0.0014 deg)
    radius_deg = 5.0 / 3600.0

    try:
        # Спочатку визначаємо точну назву таблиці каталогу SPHEREx у IRSA TAP
        tables_q = "SELECT table_name FROM TAP_SCHEMA.tables WHERE table_name LIKE '%spherex%'"
        tables_res = Irsa.query_tap(tables_q).to_table()

        if len(tables_res) == 0:
            raise HTTPException(status_code=502, detail="Таблиці SPHEREx не знайдено в IRSA TAP")

        # Обираємо основну фотометричну/спектральну таблицю джерел (зазвичай містить 'source' або 'spec' або 'psc')
        available_tables = [str(t) for t in tables_res["table_name"]]
        target_table = next(
            (t for t in available_tables if any(k in t.lower() for k in ("spec", "source", "phot", "cat"))),
            available_tables[0]
        )

        # Запит спектральних вимірювань навколо заданих координат
        adql = f"""
        SELECT TOP 200 *
        FROM {target_table}
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        """
        spec_table = Irsa.query_tap(adql).to_table()

        if spec_table is None or len(spec_table) == 0:
            return {"found": False, "table": target_table, "bands": []}

        # Визначаємо колонки довжини хвилі та потоку в схемі SPHEREx
        cols = {c.lower(): c for c in spec_table.colnames}
        wl_col = next((cols[k] for k in ("wavelength", "wave", "lambda", "wl") if k in cols), None)
        flux_col = next((cols[k] for k in ("flux", "fnu", "flux_mjy", "psf_flux") if k in cols), None)
        id_col = next((cols[k] for k in ("source_id", "designation", "objid", "id") if k in cols), None)

        if not wl_col or not flux_col:
            return {
                "found": True,
                "table": target_table,
                "raw_columns": spec_table.colnames,
                "bands": []
            }

        # Сортуємо за зростанням довжини хвилі (0.75 -> 5.0 мкм)
        spec_table.sort(wl_col)
        designation = str(spec_table[0][id_col]) if id_col else f"SPHEREx J{ra:.4f}{dec:+.4f}"

        bands = []
        for row in spec_table:
            wl = float(row[wl_col])
            flux = float(row[flux_col])
            if np.isfinite(wl) and np.isfinite(flux) and flux > 0:
                # Конвертація потоку у зоряну величину AB (якщо потік у мЯн: m_AB = 16.4 - 2.5 * log10(F_mJy))
                mag_ab = 16.4 - 2.5 * np.log10(flux)
                bands.append({
                    "wavelength_um": round(wl, 4),
                    "mag": round(float(mag_ab), 3),
                    "flux_mjy": round(flux, 4)
                })

        return {
            "found": True,
            "table": target_table,
            "designation": designation,
            "bands": bands
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Помилка NASA IRSA SPHEREx TAP: {str(e)}")


# 5. Журнал відкриттів (Збереження та читання знахідок)
@app.get("/api/discoveries")
def list_discoveries():
    if not DISCOVERIES_FILE.exists():
        return []
    return json.loads(DISCOVERIES_FILE.read_text(encoding="utf-8"))


@app.post("/api/discoveries")
def add_discovery(item: DiscoveryItem):
    items = []
    if DISCOVERIES_FILE.exists():
        items = json.loads(DISCOVERIES_FILE.read_text(encoding="utf-8"))
    items.insert(0, item.model_dump())
    DISCOVERIES_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "saved", "total": len(items)}
