
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Literal
import hashlib
import json
import math
import secrets
import time

import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from shapely.geometry import Polygon
from shapely import contains_xy

PRODUCT_ID = "MEDSEA_MULTIYEAR_WAV_006_012"
DATASET_ID = "cmems_mod_med_wav_my_4.2km_PT1H-i"

# Sessions stay in RAM only. If Render restarts the service, users simply log in again.
SESSION_TTL_HOURS = 8
SESSIONS = {}

CACHE_DIR = Path(".wave_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_VERSION = "gateway-v1"

app = FastAPI(title="Copernicus Offshore Weather Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict this to your HTML host/domain if you later host the HTML online.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


class LoginRequest(BaseModel):
    username: str
    password: str


class Vertex(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class AnalysisRequest(BaseModel):
    polygon: List[Vertex]
    start_datetime: datetime
    end_datetime: datetime
    hs_limit: float = Field(default=1.5, gt=0)
    survey_days: float = Field(default=30, gt=0)
    minimum_window_hours: int = Field(default=24, ge=0, le=168)
    spatial_rule: Literal["max", "p90", "mean"] = "max"
    planning_percentile: Literal[50, 80, 90] = 80


class PointRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    start_datetime: datetime
    end_datetime: datetime
    hs_limit: float = Field(default=1.5, gt=0)


def clean_sessions():
    now = time.time()
    expired = [t for t, v in SESSIONS.items() if v["expires"] < now]
    for t in expired:
        SESSIONS.pop(t, None)


def get_session(authorization: str | None):
    clean_sessions()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing session token.")
    token = authorization.split(" ", 1)[1].strip()
    session = SESSIONS.get(token)
    if not session:
        raise HTTPException(401, "Session expired or invalid.")
    session["expires"] = time.time() + SESSION_TTL_HOURS * 3600
    return session


@app.get("/health")
def health():
    return {"ok": True, "dataset_id": DATASET_ID}


@app.post("/api/session/login")
def session_login(req: LoginRequest):
    username = req.username.strip()
    password = req.password
    if not username or not password:
        raise HTTPException(400, "Username and password are required.")

    # Validate credentials with a very small lazy Copernicus request.
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            username=username,
            password=password,
            variables=["VHM0"],
            minimum_longitude=24.0,
            maximum_longitude=24.1,
            minimum_latitude=38.0,
            maximum_latitude=38.1,
            start_datetime="2025-01-15T00:00:00",
            end_datetime="2025-01-15T00:00:00",
        )
        # Force one value to be read so invalid credentials fail now, not later.
        _ = ds["VHM0"].isel(time=0, latitude=0, longitude=0).compute().item()
        try:
            ds.close()
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(401, f"Copernicus login failed: {type(e).__name__}: {e}")

    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "username": username,
        "password": password,
        "expires": time.time() + SESSION_TTL_HOURS * 3600,
    }

    if "@" in username:
        a, b = username.split("@", 1)
        display = (a[:1] + "*" * max(2, len(a) - 1) + "@" + b)
    else:
        display = username[:1] + "*" * max(3, len(username) - 1)

    return {"session_token": token, "display_name": display}


@app.post("/api/session/logout")
def session_logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        SESSIONS.pop(token, None)
    return {"ok": True}


def iso_naive(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(timespec="seconds")


def year_chunks(start: datetime, end: datetime):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)
    cur = s
    while cur <= e:
        ce = min(e, pd.Timestamp(year=cur.year, month=12, day=31, hour=23))
        yield cur.to_pydatetime(), ce.to_pydatetime()
        cur = pd.Timestamp(year=cur.year + 1, month=1, day=1)


def build_polygon(vertices: List[Vertex]) -> Polygon:
    if len(vertices) < 3:
        raise HTTPException(400, "At least three polygon vertices are required.")
    p = Polygon([(v.lon, v.lat) for v in vertices])
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or not p.is_valid:
        raise HTTPException(400, "Invalid polygon.")
    return p


def normalise(ds: xr.Dataset):
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds.sortby("latitude").sortby("longitude")


def open_ds(session, *, start, end, min_lon, max_lon, min_lat, max_lat, variables=("VHM0", "VMDR")):
    try:
        return normalise(copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            username=session["username"],
            password=session["password"],
            variables=list(variables),
            minimum_longitude=float(min_lon),
            maximum_longitude=float(max_lon),
            minimum_latitude=float(min_lat),
            maximum_latitude=float(max_lat),
            start_datetime=iso_naive(start),
            end_datetime=iso_naive(end),
        ))
    except Exception as e:
        raise HTTPException(502, f"Copernicus data access failed: {type(e).__name__}: {e}")


def cache_key(poly, start, end):
    raw = json.dumps({
        "v": CACHE_VERSION,
        "dataset": DATASET_ID,
        "poly": list(poly.exterior.coords),
        "start": iso_naive(start),
        "end": iso_naive(end),
    }, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def process_polygon(session, poly, start, end):
    cache = CACHE_DIR / f"{cache_key(poly,start,end)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files} | {"cache_hit": True}

    min_lon, min_lat, max_lon, max_lat = poly.bounds
    margin = 0.07

    all_t, all_max, all_mean, all_p90 = [], [], [], []
    lats_ref = lons_ref = mask_np = None
    hs_sum = hs_count = hs_max = None
    dir_sin = dir_cos = dir_count = None
    # For map P90, accumulate yearly values and quantile across all hourly records
    # only if grid is modest. We keep per-year chunks then concatenate flattened time.
    hs_map_chunks = []

    for ys, ye in year_chunks(start, end):
        ds = open_ds(
            session, start=ys, end=ye,
            min_lon=min_lon-margin, max_lon=max_lon+margin,
            min_lat=min_lat-margin, max_lat=max_lat+margin,
        )

        lats = np.asarray(ds.latitude.values, float)
        lons = np.asarray(ds.longitude.values, float)

        if lats_ref is None:
            lats_ref, lons_ref = lats, lons
            xx, yy = np.meshgrid(lons_ref, lats_ref)
            mask_np = contains_xy(poly.buffer(1e-10), xx, yy)
            if not mask_np.any():
                c = poly.centroid
                mask_np[np.abs(lats_ref-c.y).argmin(), np.abs(lons_ref-c.x).argmin()] = True
            shp = (len(lats_ref), len(lons_ref))
            hs_sum = np.zeros(shp, float)
            hs_count = np.zeros(shp, float)
            hs_max = np.full(shp, np.nan)
            dir_sin = np.zeros(shp, float)
            dir_cos = np.zeros(shp, float)
            dir_count = np.zeros(shp, float)

        mask = xr.DataArray(mask_np, dims=("latitude","longitude"),
                            coords={"latitude": ds.latitude, "longitude": ds.longitude})

        h = ds["VHM0"]
        d = ds["VMDR"]
        hp = h.where(mask)
        rr = np.deg2rad(d)

        red = xr.Dataset({
            "pmax": hp.max(("latitude","longitude"), skipna=True),
            "pmean": hp.mean(("latitude","longitude"), skipna=True),
            "pp90": hp.quantile(.90, dim=("latitude","longitude"), skipna=True),
            "hs_sum": h.sum("time", skipna=True),
            "hs_count": h.count("time"),
            "hs_max": h.max("time", skipna=True),
            "sin_sum": np.sin(rr).sum("time", skipna=True),
            "cos_sum": np.cos(rr).sum("time", skipna=True),
            "dir_count": d.count("time"),
        }).compute()

        times = pd.to_datetime(ds.time.values).view("int64")
        vmax = np.asarray(red.pmax.values, np.float32)
        vmean = np.asarray(red.pmean.values, np.float32)
        vp90 = np.asarray(red.pp90.squeeze(drop=True).values, np.float32)
        valid = np.isfinite(vmax) & np.isfinite(vmean) & np.isfinite(vp90)

        all_t.append(times[valid]); all_max.append(vmax[valid]); all_mean.append(vmean[valid]); all_p90.append(vp90[valid])

        hs_sum += np.nan_to_num(red.hs_sum.values)
        hs_count += red.hs_count.values
        hs_max = np.fmax(hs_max, red.hs_max.values)
        dir_sin += np.nan_to_num(red.sin_sum.values)
        dir_cos += np.nan_to_num(red.cos_sum.values)
        dir_count += red.dir_count.values

        # Compute map P90 for this year, then combine approximately by collecting yearly P90s.
        # This is intentionally lightweight for browser mapping.
        y_p90 = h.quantile(.90, dim="time", skipna=True).compute()
        hs_map_chunks.append(np.asarray(y_p90.squeeze(drop=True).values, np.float32))

        try: ds.close()
        except Exception: pass

    t = np.concatenate(all_t)
    order = np.argsort(t)
    pmax = np.concatenate(all_max)[order]
    pmean = np.concatenate(all_mean)[order]
    pp90 = np.concatenate(all_p90)[order]
    t = t[order]

    map_mean = np.divide(hs_sum, hs_count, out=np.full_like(hs_sum,np.nan), where=hs_count>0)
    map_p90 = np.nanmedian(np.stack(hs_map_chunks, axis=0), axis=0)
    map_dir = (np.rad2deg(np.arctan2(dir_sin, dir_cos))+360)%360
    map_dir[dir_count<=0] = np.nan

    np.savez_compressed(cache,
        times_ns=t, hs_poly_max=pmax, hs_poly_mean=pmean, hs_poly_p90=pp90,
        lats=lats_ref, lons=lons_ref, map_hs_mean=map_mean.astype(np.float32),
        map_hs_p90=map_p90.astype(np.float32), map_hs_max=hs_max.astype(np.float32),
        map_dir_mean=map_dir.astype(np.float32), mask=mask_np.astype(np.uint8)
    )

    return {
        "times_ns":t,"hs_poly_max":pmax,"hs_poly_mean":pmean,"hs_poly_p90":pp90,
        "lats":lats_ref,"lons":lons_ref,"map_hs_mean":map_mean,
        "map_hs_p90":map_p90,"map_hs_max":hs_max,"map_dir_mean":map_dir,
        "mask":mask_np.astype(np.uint8),"cache_hit":False
    }


def q(a, p):
    a = np.asarray(a,float); a=a[np.isfinite(a)]
    return float(np.quantile(a,p)) if len(a) else float("nan")


def apply_min_window(good, min_hours):
    """Only count workable runs whose total duration meets min_hours."""
    good = np.asarray(good, bool)
    if min_hours <= 1:
        return good
    out = np.zeros_like(good)
    i = 0
    n = len(good)
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < n and good[j]:
            j += 1
        if j-i >= min_hours:
            out[i:j] = True
        i = j
    return out


def workability(times_ns, hs, limit, min_window):
    idx = pd.to_datetime(times_ns, utc=True)
    hs = np.asarray(hs,float)
    valid = np.isfinite(hs)
    idx = idx[valid]; hs=hs[valid]
    raw_good = hs <= limit
    good = apply_min_window(raw_good, min_window)

    f = pd.DataFrame({"hs":hs,"good":good}, index=idx)
    monthly=[]
    for m in range(1,13):
        x=f[f.index.month==m]
        monthly.append(float(100*x.good.mean()) if len(x) else None)

    daily=f.resample("1D").agg(n=("hs","count"), mx=("hs","max"))
    full=int(((daily.n>=24)&(daily.mx<=limit)).sum())

    return {
        "valid_hours":int(len(hs)),
        "workable_hours":int(good.sum()),
        "workable_pct":float(100*good.mean()) if len(good) else 0,
        "equivalent_total_days":len(hs)/24,
        "equivalent_workable_days":good.sum()/24,
        "fully_workable_days":full,
        "monthly_pct":monthly
    }, good


def downtime(times_ns, usable, survey_days, planning_p):
    times_ns=np.asarray(times_ns,np.int64)
    usable=np.asarray(usable,bool)
    prefix=np.zeros(len(usable)+1,np.int64); prefix[1:]=np.cumsum(usable)
    required=int(math.ceil(survey_days*24))
    years=sorted(set(pd.to_datetime(times_ns,utc=True).year.tolist()))
    ref=pd.date_range("2001-01-01","2001-12-31",freq="D",tz="UTC")
    curve=[]

    for di, rd in enumerate(ref):
        vals=[]
        for y in years:
            st=pd.Timestamp(year=y,month=rd.month,day=rd.day,tz="UTC").value
            i=int(np.searchsorted(times_ns,st))
            if i>=len(times_ns) or abs(int(times_ns[i])-int(st))>2*3600*1e9: continue
            target=int(prefix[i])+required
            jp=int(np.searchsorted(prefix,target))
            if jp<=i or jp>len(usable): continue
            j=jp-1
            elapsed=(times_ns[j]-times_ns[i])/3.6e12+1
            vals.append(max(0,(elapsed-required)/24))
        if len(vals)>=3:
            curve.append({"day":di,"label":rd.strftime("%d %b"),
                          "p50":q(vals,.5),"p80":q(vals,.8),"p90":q(vals,.9),
                          "samples":len(vals)})
    if not curve:
        raise HTTPException(400,"Not enough repeated historical years for P50/P80/P90.")

    key={50:"p50",80:"p80",90:"p90"}[planning_p]
    ranked=sorted(curve,key=lambda x:x[key])
    best=ranked[0]
    picks=[]
    for x in ranked:
        if all(min(abs(x["day"]-p["day"]),365-abs(x["day"]-p["day"]))>=14 for p in picks):
            picks.append(x)
        if len(picks)>=5: break
    return {"best":best,"curve":curve,"top_windows":picks}


def map_payload(data, max_points=25000):
    n=len(data["lats"])*len(data["lons"])
    step=1 if n<=max_points else int(math.ceil(math.sqrt(n/max_points)))
    s=slice(None,None,step)
    return {
        "lats":np.asarray(data["lats"][s],float).tolist(),
        "lons":np.asarray(data["lons"][s],float).tolist(),
        "hs_mean":np.asarray(data["map_hs_mean"][s,s],float).tolist(),
        "hs_p90":np.asarray(data["map_hs_p90"][s,s],float).tolist(),
        "hs_max":np.asarray(data["map_hs_max"][s,s],float).tolist(),
        "dir_mean":np.asarray(data["map_dir_mean"][s,s],float).tolist(),
    }


@app.post("/api/analyse")
def analyse(req: AnalysisRequest, authorization: str | None = Header(default=None)):
    session=get_session(authorization)
    if req.start_datetime>=req.end_datetime:
        raise HTTPException(400,"Start must be before end.")
    poly=build_polygon(req.polygon)
    data=process_polygon(session,poly,req.start_datetime,req.end_datetime)

    hs = {
        "max":data["hs_poly_max"],
        "p90":data["hs_poly_p90"],
        "mean":data["hs_poly_mean"],
    }[req.spatial_rule]

    w, usable = workability(data["times_ns"],hs,req.hs_limit,req.minimum_window_hours)
    d = downtime(data["times_ns"],usable,req.survey_days,req.planning_percentile)

    return {
        "metadata":{
            "product_id":PRODUCT_ID,
            "dataset_id":DATASET_ID,
            "start_datetime":iso_naive(req.start_datetime),
            "end_datetime":iso_naive(req.end_datetime),
            "hs_limit":req.hs_limit,
            "survey_days":req.survey_days,
            "minimum_window_hours":req.minimum_window_hours,
            "spatial_rule":req.spatial_rule,
            "planning_percentile":req.planning_percentile,
            "valid_hours":w["valid_hours"],
            "polygon_grid_cells":int(np.asarray(data["mask"]).sum()),
            "cache_hit":bool(data.get("cache_hit",False)),
        },
        "map":map_payload(data),
        "workability":w,
        "downtime":d,
    }


@app.post("/api/point")
def point(req: PointRequest, authorization: str | None = Header(default=None)):
    session=get_session(authorization)
    hs_all=[]; dir_all=[]; model_lat=model_lon=None
    pad=.08
    for ys,ye in year_chunks(req.start_datetime,req.end_datetime):
        ds=open_ds(session,start=ys,end=ye,
                   min_lon=req.lon-pad,max_lon=req.lon+pad,
                   min_lat=req.lat-pad,max_lat=req.lat+pad)
        p=ds.sel(latitude=req.lat,longitude=req.lon,method="nearest")
        model_lat=float(p.latitude.values); model_lon=float(p.longitude.values)
        z=p[["VHM0","VMDR"]].compute()
        hs_all.append(np.asarray(z.VHM0.values,float).reshape(-1))
        dir_all.append(np.asarray(z.VMDR.values,float).reshape(-1))
        try: ds.close()
        except Exception: pass

    hs=np.concatenate(hs_all); hs=hs[np.isfinite(hs)]
    dr=np.concatenate(dir_all); dr=dr[np.isfinite(dr)]
    if not len(hs): raise HTTPException(404,"No valid data at this point.")
    rr=np.deg2rad(dr)
    dm=float((np.rad2deg(np.arctan2(np.sin(rr).mean(),np.cos(rr).mean()))+360)%360) if len(dr) else float("nan")

    return {
        "model_lat":model_lat,"model_lon":model_lon,
        "hs_mean":float(np.mean(hs)),"hs_p50":q(hs,.5),"hs_p80":q(hs,.8),
        "hs_p90":q(hs,.9),"hs_max":float(np.max(hs)),
        "dir_mean":dm,"workable_pct":float(100*np.mean(hs<=req.hs_limit)),
        "valid_hours":int(len(hs))
    }
