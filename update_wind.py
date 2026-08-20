import io, re, zipfile, math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
import xarray as xr

LAT = 49.0169
LON = 21.7008
TZ = ZoneInfo("Europe/Bratislava")
BASE = "https://odp.met.hu/weather/nwp/WRF/nc"
GRID_URL = "https://odp.met.hu/weather/nwp/latlon/lonlat-WRF.nc"
RUN_HOURS = (0, 6, 12, 18)
VARS = ("U10", "V10", "WGUST", "T2")
USER_AGENT = "Domasza-WRF-1.5km/1.0"

def list_dir(hour):
    url = f"{BASE}/{hour:02d}/"
    r = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text

def find_latest_run():
    now = datetime.now(timezone.utc)
    candidates = []

    for h in RUN_HOURS:
        html = list_dir(h)

        # Szukamy przebiegów, dla których istnieją wszystkie trzy
        # potrzebne pola do +36 h: U10, V10, WGUST i T2.
        date_sets = []
        for var in VARS:
            pat = re.compile(
                rf"WRF-{var}-(\d{{8}})_{h:02d}00\+03600\.nc\.zip"
            )
            date_sets.append(set(pat.findall(html)))

        common_dates = set.intersection(*date_sets) if date_sets else set()

        for d in common_dates:
            run = datetime.strptime(
                d + f"{h:02d}00", "%Y%m%d%H%M"
            ).replace(tzinfo=timezone.utc)

            if run <= now + timedelta(hours=1):
                candidates.append(run)

    if not candidates:
        raise RuntimeError(
            "Nie znaleziono kompletnego przebiegu WRF "
            "z U10, V10, WGUST i T2 do +36 h."
        )

    return max(candidates)

def file_url(var, run, lead):
    return (
        f"{BASE}/{run.hour:02d}/"
        f"WRF-{var}-{run:%Y%m%d}_{run:%H}00+{lead:03d}00.nc.zip"
    )

def download_zip_nc(url):
    r = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = [n for n in z.namelist() if n.lower().endswith(".nc")]
    if not names:
        raise RuntimeError(f"Brak pliku .nc w {url}")
    return z.read(names[0])

def open_nc(data):
    # netCDF4 engine obsługuje typowe pliki HungaroMet.
    bio = io.BytesIO(data)
    try:
        return xr.open_dataset(bio, engine="h5netcdf")
    except Exception:
        bio.seek(0)
        return xr.open_dataset(bio, engine="scipy")

def pick_data_var(ds, wanted):
    preferred = [wanted, wanted.lower(), wanted.upper()]
    for name in preferred:
        if name in ds.data_vars:
            return ds[name]
    # wybór największej zmiennej numerycznej
    candidates = []
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.number) and da.size > 100:
            candidates.append((da.size, name))
    if not candidates:
        raise RuntimeError(f"Nie znaleziono pola danych {wanted}.")
    return ds[max(candidates)[1]]

def latlon_arrays(ds, da):
    lat_names = ("lat","latitude","LAT","XLAT","xlat")
    lon_names = ("lon","longitude","LON","XLONG","xlong")
    lat = next((ds[n] for n in lat_names if n in ds.variables), None)
    lon = next((ds[n] for n in lon_names if n in ds.variables), None)

    if lat is not None and lon is not None:
        return np.asarray(lat), np.asarray(lon)

    # Czasem współrzędne są 1D na wymiarach y/x.
    dims = da.dims[-2:]
    y = ds.coords.get(dims[0])
    x = ds.coords.get(dims[1])
    if y is not None and x is not None:
        yy = np.asarray(y)
        xx = np.asarray(x)
        if yy.ndim == 1 and xx.ndim == 1:
            lon2, lat2 = np.meshgrid(xx, yy)
            return lat2, lon2

    raise RuntimeError("Nie znaleziono współrzędnych siatki w pliku NetCDF.")

def load_official_grid():
    """
    Pobiera oficjalny plik współrzędnych HungaroMet dla WRF.
    Dzięki temu nie traktujemy indeksów x/y jak stopni geograficznych.
    """
    r = requests.get(
        GRID_URL,
        timeout=120,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()

    ds = open_nc(r.content)
    try:
        names = {name.lower(): name for name in ds.variables}

        lat_name = next(
            (names[k] for k in ("lat", "latitude") if k in names),
            None
        )
        lon_name = next(
            (names[k] for k in ("lon", "longitude") if k in names),
            None
        )

        if lat_name is None:
            lat_name = next(
                (n for n in ds.variables if "lat" in n.lower()),
                None
            )
        if lon_name is None:
            lon_name = next(
                (n for n in ds.variables if "lon" in n.lower()),
                None
            )

        if lat_name is None or lon_name is None:
            raise RuntimeError(
                "Nie znaleziono LAT/LON w lonlat-WRF.nc."
            )

        lat = np.asarray(ds[lat_name]).squeeze()
        lon = np.asarray(ds[lon_name]).squeeze()

        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lat2d, lon2d

        if lat.ndim == 2 and lon.ndim == 2:
            return lat, lon

        raise RuntimeError(
            f"Nieoczekiwany kształt siatki: "
            f"LAT {lat.shape}, LON {lon.shape}"
        )
    finally:
        ds.close()


OFFICIAL_LAT, OFFICIAL_LON = load_official_grid()

COSLAT = math.cos(math.radians(LAT))
GRID_D2 = (
    (OFFICIAL_LAT - LAT) ** 2
    + ((OFFICIAL_LON - LON) * COSLAT) ** 2
)
GRID_INDEX = np.unravel_index(
    np.nanargmin(GRID_D2),
    GRID_D2.shape
)
GRID_LAT = float(OFFICIAL_LAT[GRID_INDEX])
GRID_LON = float(OFFICIAL_LON[GRID_INDEX])


def value_from_official_grid(data, wanted):
    ds = open_nc(data)
    try:
        da = pick_data_var(ds, wanted).squeeze()
        vals = np.asarray(da, dtype=float)

        while vals.ndim > 2:
            vals = vals[0]

        y, x = GRID_INDEX

        if vals.shape == OFFICIAL_LAT.shape:
            return float(vals[y, x])

        if vals.T.shape == OFFICIAL_LAT.shape:
            return float(vals[x, y])

        raise RuntimeError(
            f"Pole {wanted} ma rozmiar {vals.shape}, "
            f"a siatka {OFFICIAL_LAT.shape}."
        )
    finally:
        ds.close()


def nearest_value(data, wanted):
    value = value_from_official_grid(data, wanted)
    return value, GRID_LAT, GRID_LON


def ms_to_kn(x): return x * 1.943844492

def k_to_c(x):
    # HungaroMet WRF T2 jest publikowane w kelwinach.
    return x - 273.15 if x > 170 else x

def temp_class(x):
    if x < 5: return "t-vcold"
    if x < 10: return "t-cold"
    if x < 15: return "t-cool"
    if x < 20: return "t-mild"
    if x < 25: return "t-warm"
    if x < 30: return "t-hot"
    return "t-vhot"

def wind_dir(u, v):
    deg = (math.degrees(math.atan2(-u, -v)) + 360) % 360
    labs = ("N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW")
    return deg, labs[int((deg + 11.25)//22.5) % 16]

def speed_class(x):
    if x < 8: return "calm"
    if x < 12: return "light"
    if x < 16: return "good"
    if x < 20: return "strong"
    if x < 24: return "vstrong"
    return "hard"

def gust_class(x):
    if x < 12:
        return "g-neutral"
    if x < 16:
        return "g-light"
    if x < 20:
        return "g-strong"
    if x < 24:
        return "g-vstrong"
    if x < 28:
        return "g-hard"
    return "g-extreme"

def day_name(dt, today):
    if dt.date() == today: return "Dzisiaj"
    if dt.date() == today + timedelta(days=1): return "Jutro"
    if dt.date() == today + timedelta(days=2): return "Pojutrze"
    return ("Poniedziałek","Wtorek","Środa","Czwartek","Piątek","Sobota","Niedziela")[dt.weekday()]

def main():
    run = find_latest_run()
    now = datetime.now(TZ)
    rows = []
    point_lat = point_lon = None

    for lead in range(37):
        u, la, lo = nearest_value(download_zip_nc(file_url("U10", run, lead)), "U10")
        v, _, _ = nearest_value(download_zip_nc(file_url("V10", run, lead)), "V10")
        gust, _, _ = nearest_value(download_zip_nc(file_url("WGUST", run, lead)), "WGUST")
        t2, _, _ = nearest_value(download_zip_nc(file_url("T2", run, lead)), "T2")
        if point_lat is None:
            point_lat, point_lon = la, lo

        speed = ms_to_kn(math.hypot(u, v))
        gust_kn = ms_to_kn(gust)
        temp_c = k_to_c(t2)
        deg, direction = wind_dir(u, v)
        valid = (run + timedelta(hours=lead)).astimezone(TZ)
        if valid >= now.replace(minute=0, second=0, microsecond=0):
            rows.append(dict(time=valid, speed=speed, gust=gust_kn, temp=temp_c, deg=deg, direction=direction))

    if not rows:
        raise RuntimeError("Brak przyszłych godzin prognozy.")

    groups = {}
    for r in rows:
        groups.setdefault(r["time"].date(), []).append(r)

    def rows_html(rs):
        out = []
        for r in rs:
            delta = max(0, r["gust"] - r["speed"])
            out.append(f"""
<tr class="{speed_class(r['speed'])}">
<td class="time">{r['time']:%H:%M}</td>
<td><b>{r['speed']:.1f}</b> <small>kn</small></td>
<td class="gust {gust_class(r['gust'])}"><b>{r['gust']:.1f}</b> <small>kn</small><span class="delta">+{delta:.1f}</span></td>
<td class="temp {temp_class(r['temp'])}"><b>{r['temp']:.1f}</b> <small>°C</small></td>
<td><b>{r['direction']}</b> <span class="deg">{r['deg']:.0f}°</span></td>
</tr>""")
        return "".join(out)

    sections = []
    for day in sorted(groups):
        rs = groups[day]
        avg = sum(r["speed"] for r in rs)/len(rs)
        mx = max(rs, key=lambda r:r["speed"])
        mg = max(rs, key=lambda r:r["gust"])
        label = day_name(rs[0]["time"], now.date())
        sections.append(f"""
<section class="card">
<div class="dayhead"><h2>{label}</h2><div>{rs[0]['time']:%d.%m.%Y}</div></div>
<div class="stats">
<div><span>Średnio</span><b>{avg:.1f} kn</b></div>
<div><span>Maks. wiatr</span><b>{mx['speed']:.1f} kn · {mx['time']:%H:%M}</b></div>
<div><span>Maks. poryw</span><b>{mg['gust']:.1f} kn · {mg['time']:%H:%M}</b></div>
</div>
<div class="wrap"><table><thead><tr><th>Godz.</th><th>Wiatr</th><th>Porywy</th><th>Temp.</th><th>Kierunek</th></tr></thead>
<tbody>{rows_html(rs)}</tbody></table></div>
</section>""")

    mx_all = max(rows, key=lambda r:r["speed"])
    mg_all = max(rows, key=lambda r:r["gust"])
    run_local = run.astimezone(TZ)

    html = f"""<!doctype html>
<html lang="pl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Domasza – WRF 1,5 km</title>
<style>
:root{{--blue:#0b65a8;--bg:#edf2f6;--text:#17202a;--calm:#eef2f5;--light:#e5f5ef;--good:#dff4cf;--strong:#fff0b8;--vstrong:#ffd79a;--hard:#ffb4ad}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:var(--text)}}
header{{background:linear-gradient(135deg,#074b7d,var(--blue));color:#fff;padding:20px 14px}}.inner,main{{max-width:920px;margin:auto}}
h1{{margin:0;font-size:27px}}header p{{margin:6px 0 0}}main{{padding:10px}}
.hero,.card{{background:#fff;border-radius:16px;margin-bottom:14px;box-shadow:0 2px 12px #0001;overflow:hidden}}
.hero{{padding:14px}}.heroGrid,.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px}}
.heroGrid div,.stats div{{padding:10px;background:#f8fafb}}span,small,.deg{{color:#61707c}}.heroGrid span,.stats span{{display:block;font-size:11px;margin-bottom:3px}}
.meta{{font-size:13px;color:#61707c;line-height:1.55;margin-top:10px}}.legend{{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}}.legend i{{font-style:normal;padding:5px 8px;border-radius:999px;font-size:12px;font-weight:700}}
.dayhead{{padding:14px;border-bottom:1px solid #e5e9ed}}.dayhead h2{{margin:0;font-size:23px}}.wrap{{overflow-x:auto}}table{{width:100%;min-width:650px;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th{{background:var(--blue);color:#fff;padding:10px 8px;text-align:right}}th:first-child,td.time{{text-align:left}}td{{padding:10px 8px;text-align:right;border-bottom:1px solid #0001}}
td b{{font-size:17px}}.delta{{display:inline-block;margin-left:5px;padding:2px 5px;background:#fff9;border-radius:999px;font-size:11px;font-weight:800;color:#4c5964}}
.calm{{background:var(--calm)}}.light{{background:var(--light)}}.good{{background:var(--good)}}.strong{{background:var(--strong)}}.vstrong{{background:var(--vstrong)}}.hard{{background:var(--hard)}}
.g-neutral{{background:#f3f5f7!important}}
.g-light{{background:#dff4cf!important}}
.g-strong{{background:#ffe68a!important}}
.g-vstrong{{background:#ffbd66!important}}
.g-hard{{background:#ff7f73!important}}
.g-extreme{{background:#b83232!important;color:#fff!important}}
.g-extreme small,.g-extreme .delta{{color:#fff!important}}
.g-extreme .delta{{background:rgba(255,255,255,.22)!important}}
.temp{{font-variant-numeric:tabular-nums}}
.t-vcold{{background:#9dc9ff!important}}
.t-cold{{background:#b9dcff!important}}
.t-cool{{background:#bfeee9!important}}
.t-mild{{background:#d8f2cf!important}}
.t-warm{{background:#ffe89a!important}}
.t-hot{{background:#ffc36d!important}}
.t-vhot{{background:#ff8b7f!important}}
footer{{text-align:center;color:#61707c;font-size:12px;padding:8px 12px 24px}}
@media(max-width:680px){{.heroGrid,.stats{{grid-template-columns:1fr}}main{{padding:7px}}}}
</style></head><body>
<header><div class="inner"><h1>🌬️ Domasza – WRF 1,5 km</h1><p>Wielka Domasza · środkowa część</p></div></header>
<main>
<div class="hero">
<div class="heroGrid">
<div><span>Najsilniejszy wiatr</span><b>{mx_all['speed']:.1f} kn · {mx_all['time']:%H:%M}</b></div>
<div><span>Najsilniejszy poryw</span><b>{mg_all['gust']:.1f} kn · {mg_all['time']:%H:%M}</b></div>
<div><span>Aktualizacja</span><b>{now:%H:%M}</b></div>
</div>
<div class="legend"><i style="background:var(--calm)">wiatr 0–8</i><i style="background:var(--light)">8–12</i><i style="background:var(--good)">12–16</i><i style="background:var(--strong)">16–20</i><i style="background:var(--vstrong)">20–24</i><i style="background:var(--hard)">24+ kn</i></div>
<div class="legend"><i style="background:#f3f5f7">poryw &lt;12</i><i style="background:#dff4cf">12–16</i><i style="background:#ffe68a">16–20</i><i style="background:#ffbd66">20–24</i><i style="background:#ff7f73">24–28</i><i style="background:#b83232;color:#fff">28+ kn</i></div>
<div class="legend"><i style="background:#9dc9ff">&lt;5°C</i><i style="background:#b9dcff">5–10°C</i><i style="background:#bfeee9">10–15°C</i><i style="background:#d8f2cf">15–20°C</i><i style="background:#ffe89a">20–25°C</i><i style="background:#ffc36d">25–30°C</i><i style="background:#ff8b7f">30+°C</i></div>
<div class="meta"><b>Model:</b> HungaroMet WRF · przebieg {run_local:%d.%m.%Y %H:%M}<br>
<b>Punkt docelowy:</b> {LAT:.4f}°N, {LON:.4f}°E<br>
<b>Punkt siatki:</b> {point_lat:.4f}°N, {point_lon:.4f}°E</div>
</div>
{''.join(sections)}
</main><footer>Dane: HungaroMet ODP · U10/V10/WGUST/T2</footer>
</body></html>"""

    with open("index.html","w",encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
