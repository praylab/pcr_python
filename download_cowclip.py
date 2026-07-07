import requests
import os
from itertools import product

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_OPENDAP = (
    "https://data-cbr.csiro.au/thredds/dodsC/catch_all/oa-cmip6-wave/"
    "UniMelb-CSIRO_CMIP6_projections/historical/ACCESS-CM2/CDFAC108/ww3_ounf_glout"
)
BASE_HTTP = (
    "https://data-cbr.csiro.au/thredds/fileServer/catch_all/oa-cmip6-wave/"
    "UniMelb-CSIRO_CMIP6_projections/historical/ACCESS-CM2/CDFAC108/ww3_ounf_glout"
)

VARIABLES   = ["hs", "fp", "dir"]
YEARS       = range(1985, 2015)   # historical run — adjust as needed
MONTHS      = range(1, 13)
OUTPUT_DIR  = "./data/cowclip_wave_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Option A: subset & download via xarray / OPeNDAP ──────────────────────────
import xarray as xr

def download_via_opendap(years=YEARS, months=MONTHS, variables=VARIABLES):
    """Open each file remotely, select only the needed variable, save locally."""
    for year, month, var in product(years, months, variables):
        filename  = f"ww3.{year}{month:02d}_{var}.nc"
        opendap_url = f"{BASE_OPENDAP}/{filename}"
        out_path    = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(out_path):
            print(f"[skip] {filename}")
            continue

        try:
            print(f"[fetch] {filename}")
            ds = xr.open_dataset(opendap_url, engine="netcdf4")
            # Files are already split by variable, so just save the whole file
            ds.to_netcdf(out_path)
            ds.close()
        except Exception as e:
            print(f"[error] {filename}: {e}")


# ── Option B: direct HTTP download (faster for full files) ────────────────────
def download_via_http(years=YEARS, months=MONTHS, variables=VARIABLES):
    """Download raw .nc files directly — no subsetting, faster."""
    session = requests.Session()

    for year, month, var in product(years, months, variables):
        filename = f"ww3.{year}{month:02d}_{var}.nc"
        url      = f"{BASE_HTTP}/{filename}"
        out_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(out_path):
            print(f"[skip] {filename}")
            continue

        try:
            print(f"[download] {filename}")
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except requests.HTTPError as e:
            print(f"[missing] {filename}: {e}")   # file may not exist for that month
        except Exception as e:
            print(f"[error]   {filename}: {e}")


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use HTTP for speed (recommended for full-file downloads)
    download_via_http()

    # OR use OPeNDAP if you want to subset spatially/temporally first:
    # download_via_opendap()