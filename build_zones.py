"""
Build accurate TfL zone polygons from live station data via the TfL API.
Strategy: fetch all tube/overground/DLR/Elizabeth line stations with zone data,
buffer each station by 2.2km, then union per cumulative zone to produce nested
outer-boundary polygons (same structure our code already expects).
"""
import json
import urllib.request
import urllib.parse
from shapely.geometry import Point, mapping
from shapely.ops import unary_union
import shapely.affinity

LINES = [
    "central", "bakerloo", "circle", "district",
    "hammersmith-city", "jubilee", "metropolitan",
    "northern", "piccadilly", "victoria", "waterloo-city",
    "dlr", "elizabeth", "overground",
]

# ── Fetch stations ─────────────────────────────────────────────────────────────
def fetch_stations(line):
    url = f"https://api.tfl.gov.uk/Line/{line}/StopPoints"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  Warning: could not fetch {line}: {e}")
        return []

print("Fetching stations from TfL API...")
all_stations = {}  # naptan_id -> {lat, lon, zone}

for line in LINES:
    print(f"  {line}...", end=" ", flush=True)
    stations = fetch_stations(line)
    added = 0
    for s in stations:
        sid = s.get("id") or s.get("naptanId")
        lat = s.get("lat")
        lon = s.get("lon")
        if not sid or not lat or not lon:
            continue
        # Zone is in additionalProperties
        zone = None
        for prop in s.get("additionalProperties", []):
            if prop.get("key") == "Zone":
                raw = prop.get("value", "")
                # Zones can be "1", "2", "2/3" (boundary stations) — take the lower
                parts = str(raw).replace("+", "/").split("/")
                try:
                    zone = min(int(p.strip()) for p in parts if p.strip().isdigit())
                except ValueError:
                    pass
                break
        if zone and 1 <= zone <= 6 and sid not in all_stations:
            all_stations[sid] = {"lat": lat, "lon": lon, "zone": zone}
            added += 1
    print(f"{added} stations")

print(f"\nTotal unique stations with zone data: {len(all_stations)}")

by_zone = {}
for sid, s in all_stations.items():
    z = s["zone"]
    by_zone.setdefault(z, []).append((s["lon"], s["lat"]))

for z in sorted(by_zone):
    print(f"  Zone {z}: {len(by_zone[z])} stations")

# ── Build cumulative zone polygons ────────────────────────────────────────────
# Zone N outer boundary = union of all stations in zones 1..N, each buffered.
# Buffer in degrees: 2.2km ≈ 0.020° at London latitude
BUFFER_DEG = 0.022

print("\nBuilding zone polygons...")
features = []
for zone_num in range(1, 7):
    # Collect all stations up to and including this zone
    pts = []
    for z in range(1, zone_num + 1):
        pts.extend(by_zone.get(z, []))

    if not pts:
        print(f"  Zone {zone_num}: no stations, skipping")
        continue

    # Buffer each station point and union
    buffered = [Point(lon, lat).buffer(BUFFER_DEG, resolution=16) for lon, lat in pts]
    union = unary_union(buffered)

    # Simplify slightly to reduce polygon vertex count
    simplified = union.simplify(0.003, preserve_topology=True)

    # If it's a MultiPolygon, take the largest piece
    if simplified.geom_type == "MultiPolygon":
        simplified = max(simplified.geoms, key=lambda g: g.area)

    geojson_geom = mapping(simplified)
    features.append({
        "type": "Feature",
        "properties": {
            "zone": zone_num,
            "description": f"TfL Zone {zone_num} outer boundary (built from {len(pts)} station buffers)",
            "station_count": len(pts),
        },
        "geometry": geojson_geom,
    })
    verts = len(list(simplified.exterior.coords))
    print(f"  Zone {zone_num}: {len(pts)} stations -> polygon with {verts} vertices")

output = {
    "type": "FeatureCollection",
    "name": "TfL_Fare_Zones_1_to_6",
    "source": "Built from TfL Open Data API station coordinates",
    "features": features,
}

out_path = "london_zones.geojson"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {out_path}")
print("Done — reload the dashboard to use the new zone boundaries.")
