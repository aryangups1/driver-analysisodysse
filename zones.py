import re
import json
import os
import pandas as pd
from shapely.geometry import Point, shape

# ── Load GeoJSON zone polygons once at import time ────────────────────────────
_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "london_zones.geojson")

with open(_GEOJSON_PATH, encoding="utf-8") as f:
    _geojson = json.load(f)

# Build sorted list [(zone_number, shapely_polygon), ...] innermost first
ZONE_POLYGONS = sorted(
    [(feat["properties"]["zone"], shape(feat["geometry"])) for feat in _geojson["features"]],
    key=lambda x: x[0],
)

GEOJSON_DATA = _geojson  # expose for map rendering

# Charing Cross
CENTER_LAT = 51.5081
CENTER_LON = -0.1247

# "London region" bounding box — includes Heathrow, Gatwick, Stansted, Luton
# but excludes Cambridge, Southampton, Milton Keynes, etc.
LONDON_BOUNDS = {
    "lat_min": 51.05,
    "lat_max": 51.95,
    "lon_min": -0.65,
    "lon_max":  0.55,
}

# Hard UK sanity bounds — anything outside this is a clear GPS error
UK_BOUNDS = {
    "lat_min": 50.0,
    "lat_max": 53.0,
    "lon_min": -2.0,
    "lon_max":  1.8,
}


def is_valid_london_trip(plat, plon, dlat, dlon):
    """Return True if both endpoints are real UK coordinates and the dropoff
    is within the Greater London region (airports included).
    Excludes bogus GPS readings and long-haul trips to Cambridge / Southampton etc.
    """
    if any(v is None for v in (plat, plon, dlat, dlon)):
        return False
    # Pickup must be a real UK coordinate
    if not (UK_BOUNDS["lat_min"] <= plat <= UK_BOUNDS["lat_max"] and
            UK_BOUNDS["lon_min"] <= plon <= UK_BOUNDS["lon_max"]):
        return False
    # Dropoff must be within London region
    if not (LONDON_BOUNDS["lat_min"] <= dlat <= LONDON_BOUNDS["lat_max"] and
            LONDON_BOUNDS["lon_min"] <= dlon <= LONDON_BOUNDS["lon_max"]):
        return False
    return True


# ── Coordinate parsing ────────────────────────────────────────────────────────

def parse_dms(coord_str):
    """Parse various coordinate formats -> (lat, lon) in decimal degrees.

    Handles:
      - DMS with symbols:  '51° 36' 39.4" N,0° 2' 36.5" E'
      - DMS no-seconds:    '51° 42' 11 N,0° 1' 32 W'
      - Decimal:           '51.5067702,-0.0416209'
    """
    if not coord_str or not isinstance(coord_str, str) or not coord_str.strip():
        return None, None

    s = coord_str.strip()

    # Decimal format: two bare numbers separated by comma
    dec_m = re.match(r"^(-?[\d.]+)\s*,\s*(-?[\d.]+)$", s)
    if dec_m:
        lat, lon = float(dec_m.group(1)), float(dec_m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
        return None, None

    # Find the comma that splits lat from lon (after N/S direction letter)
    split_m = re.search(r"([NS])\s*,\s*", s, re.IGNORECASE)
    if not split_m:
        return None, None
    lat_str = s[: split_m.end()].rstrip(",").strip()
    lon_str = s[split_m.end() :].strip()

    def dms_to_decimal(part):
        # Full DMS: degrees ° minutes ' seconds " direction
        m = re.match(
            r"(\d+)\s*[°d]\s*(\d+)\s*['′m]\s*([\d.]+)?\s*[\"″s]?\s*([NSEW])",
            part.strip(),
            re.IGNORECASE,
        )
        if m:
            deg = float(m.group(1))
            mins = float(m.group(2))
            sec = float(m.group(3)) if m.group(3) else 0.0
            direc = m.group(4).upper()
            val = deg + mins / 60 + sec / 3600
            if direc in ("S", "W"):
                val = -val
            return val
        # Degrees + decimal minutes fallback
        m2 = re.match(r"(\d+)\s*[°d]\s*([\d.]+)\s*([NSEW])", part.strip(), re.IGNORECASE)
        if m2:
            val = float(m2.group(1)) + float(m2.group(2)) / 60
            if m2.group(3).upper() in ("S", "W"):
                val = -val
            return val
        return None

    lat = dms_to_decimal(lat_str)
    lon = dms_to_decimal(lon_str)
    return lat, lon


# ── Zone assignment (GeoJSON polygon-based) ───────────────────────────────────

def assign_zone(lat, lon):
    """Return TfL zone (1-6) for a lat/lon using the GeoJSON polygons.
    Checks innermost zone first; returns 6 if outside all polygons.
    Returns None if coordinates are invalid.
    """
    if lat is None or lon is None:
        return None
    pt = Point(lon, lat)  # shapely uses (x=lon, y=lat)
    for zone_num, polygon in ZONE_POLYGONS:
        if polygon.contains(pt):
            return zone_num
    return 6  # outside all defined polygons → treat as Zone 6+


# ── DataFrame enrichment ──────────────────────────────────────────────────────

def enrich_zones(df):
    """Add pickup_zone, dropoff_zone, zone_pair, trip_type columns.
    Rows with bogus GPS or out-of-London-region dropoffs are dropped."""
    df = df.copy()

    pickup_coords = df["pickup_lat_long"].apply(parse_dms)
    dropoff_coords = df["dropoff_latlong"].apply(parse_dms)

    df["pickup_lat"] = [c[0] for c in pickup_coords]
    df["pickup_lon"] = [c[1] for c in pickup_coords]
    df["dropoff_lat"] = [c[0] for c in dropoff_coords]
    df["dropoff_lon"] = [c[1] for c in dropoff_coords]

    # Drop outlier coordinates
    valid_mask = df.apply(
        lambda r: is_valid_london_trip(r.pickup_lat, r.pickup_lon, r.dropoff_lat, r.dropoff_lon),
        axis=1,
    )
    df = df[valid_mask].copy()

    df["pickup_zone"] = df.apply(lambda r: assign_zone(r.pickup_lat, r.pickup_lon), axis=1)
    df["dropoff_zone"] = df.apply(lambda r: assign_zone(r.dropoff_lat, r.dropoff_lon), axis=1)

    df["zone_pair"] = df.apply(
        lambda r: f"Z{int(r.pickup_zone)}→Z{int(r.dropoff_zone)}"
        if pd.notna(r.pickup_zone) and pd.notna(r.dropoff_zone)
        else None,
        axis=1,
    )
    df["trip_type"] = df.apply(lambda r: _classify_trip(r.pickup_zone, r.dropoff_zone), axis=1)
    return df


def calc_true_rph(df):
    """Add gap_mins and true_rph columns to a zone-enriched trip DataFrame.

    Full-cycle definition (matches actual earnings / online-hours within ~5%):
      true_rph = fare / ((inter_trip_gap + pickup_duration + ride_duration) / 60)

    inter_trip_gap: time from previous dropoff to this pickup, capped at 90 min.
    pickup_duration: time driving to passenger + waiting for board, capped at 30 min.
    """
    df = df.copy().sort_values(["dim_driver_id", "pickedup_trip_datetime"])
    df["pickedup_trip_datetime"] = pd.to_datetime(df["pickedup_trip_datetime"])
    df["dropoff_trip_datetime"]  = pd.to_datetime(df["dropoff_trip_datetime"])

    df["prev_dropoff"] = df.groupby("dim_driver_id")["dropoff_trip_datetime"].shift(1)

    df["gap_mins"] = (
        (df["pickedup_trip_datetime"] - df["prev_dropoff"])
        .dt.total_seconds()
        .div(60)
        .clip(lower=0, upper=90)
        .fillna(0)
    )

    ride   = df["pob_duration_in_min"].fillna(0).clip(lower=1)
    pickup = df["pickup_duration_in_min"].fillna(0).clip(lower=0, upper=30)
    df["true_total_mins"] = df["gap_mins"] + pickup + ride
    df["true_rph"] = (df["trip_price_in_pound"] / (df["true_total_mins"] / 60)).round(2)

    return df


def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line distance in miles between two coordinate pairs."""
    import math
    R = 3958.8
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def estimate_ping(plat, plon, dlat, dlon):
    """Estimate road distance (miles) and fare (£) for a declined ping from coords.
    Road distance = straight-line * 1.3. Fare uses Bolt London pricing:
    £2.50 base + £1.25/mile + £0.15/min at 10 mph avg speed.
    Returns (road_miles, fare) or (None, None) if coords are invalid.
    """
    import math
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in [plat, plon, dlat, dlon]):
        return None, None
    straight   = haversine_miles(plat, plon, dlat, dlon)
    road_miles = straight * 1.3
    est_mins   = road_miles / 10 * 60          # 10 mph avg London speed
    fare       = 2.50 + 1.25 * road_miles + 0.15 * est_mins
    return round(road_miles, 1), round(fare, 2)


def _classify_trip(pz, dz):
    if pd.isna(pz) or pd.isna(dz):
        return "Unknown"
    pz, dz = int(pz), int(dz)
    if pz == 1 and dz == 1:
        return "Zone 1 local"
    elif pz == 1:
        return "Zone 1 out"
    elif dz == 1:
        return "Zone 1 in"
    elif pz == dz:
        return f"Zone {pz} local"
    elif dz > pz:
        return "Outbound"
    else:
        return "Inbound"
