import psycopg2
import pandas as pd
import streamlit as st
from config import DB_CONFIG, TOP_DRIVER_IDS, BAD_DRIVER_IDS

@st.cache_resource
def get_conn():
    return psycopg2.connect(**DB_CONFIG)

def query(sql, params=None):
    conn = get_conn()
    return pd.read_sql(sql, conn, params=params)

IDS = TOP_DRIVER_IDS

# Max trip distance used as a SQL pre-filter across all trip-level queries.
# Removes long-haul outliers (Cambridge, Southampton, Milton Keynes etc.)
# before they hit any aggregation. Coordinate-level filtering (bogus GPS)
# is handled in Python via zones.is_valid_london_trip().
MAX_TRIP_MILES = 60

@st.cache_data(ttl=3600)
def load_overview():
    return query("""
        SELECT
            dim_driver_id,
            MAX(driver_full_name) as driver_name,
            COUNT(DISTINCT driver_performance_date) as active_days,
            SUM(number_of_finished_rides) as total_rides,
            ROUND(SUM(online_time_in_hrs)::numeric, 1) as total_online_hrs,
            ROUND(SUM(revenue)::numeric, 2) as total_revenue,
            ROUND(SUM(earnings)::numeric, 2) as total_earnings,
            ROUND((SUM(revenue) / NULLIF(SUM(online_time_in_hrs),0))::numeric, 2) as rph,
            ROUND(AVG(utilisation_percent)::numeric, 1) as avg_util,
            ROUND(AVG(NULLIF(average_driver_rating,0))::numeric, 2) as avg_rating,
            ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric, 1) as avg_acceptance,
            ROUND((SUM(revenue) / NULLIF(SUM(number_of_finished_rides),0))::numeric, 2) as rev_per_trip
        FROM rep_fact_driver_performance
        WHERE dim_driver_id = ANY(%s) AND online_time_in_hrs > 0
        GROUP BY dim_driver_id
        ORDER BY rph DESC
    """, (IDS,))

@st.cache_data(ttl=3600)
def load_daily_performance():
    return query("""
        SELECT
            dim_driver_id,
            MAX(driver_full_name) as driver_name,
            driver_performance_date,
            SUM(number_of_finished_rides) as rides,
            ROUND(SUM(online_time_in_hrs)::numeric, 2) as online_hrs,
            ROUND(SUM(revenue)::numeric, 2) as revenue,
            ROUND(SUM(earnings)::numeric, 2) as earnings,
            ROUND((SUM(revenue) / NULLIF(SUM(online_time_in_hrs),0))::numeric, 2) as rph,
            ROUND(AVG(utilisation_percent)::numeric, 1) as utilisation,
            ROUND(AVG(NULLIF(average_driver_rating,0))::numeric, 2) as rating
        FROM rep_fact_driver_performance
        WHERE dim_driver_id = ANY(%s) AND online_time_in_hrs > 0
        GROUP BY dim_driver_id, driver_performance_date
        ORDER BY dim_driver_id, driver_performance_date
    """, (IDS,))

@st.cache_data(ttl=3600)
def load_hourly_trips():
    return query("""
        SELECT
            dim_driver_id,
            driver_full_name,
            trips_hr as hour_of_day,
            EXTRACT(DOW FROM pickedup_trip_datetime) as day_of_week,
            COUNT(*) as trips,
            ROUND(AVG(trip_price_in_pound)::numeric, 2) as avg_fare,
            ROUND(AVG(distance_in_miles)::numeric, 2) as avg_distance,
            ROUND(AVG(pob_duration_in_min)::numeric, 1) as avg_ride_mins,
            ROUND(AVG(pickup_duration_in_min)::numeric, 1) as avg_pickup_mins
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed', 'Finished')
          AND trips_hr IS NOT NULL
          AND distance_in_miles <= %s
        GROUP BY dim_driver_id, driver_full_name, trips_hr,
                 EXTRACT(DOW FROM pickedup_trip_datetime)
        ORDER BY dim_driver_id, hour_of_day
    """, (IDS, MAX_TRIP_MILES))

@st.cache_data(ttl=3600)
def load_trip_economics():
    return query("""
        SELECT
            dim_driver_id,
            driver_full_name,
            source,
            COUNT(*) as trips,
            ROUND(AVG(trip_price_in_pound)::numeric, 2) as avg_fare,
            ROUND(AVG(distance_in_miles)::numeric, 2) as avg_distance,
            ROUND(AVG(pob_duration_in_min)::numeric, 1) as avg_ride_mins,
            ROUND(AVG(pickup_duration_in_min)::numeric, 1) as avg_pickup_mins,
            ROUND(AVG(rider_tips_in_pound)::numeric, 2) as avg_tip,
            ROUND(SUM(trip_price_in_pound)::numeric, 2) as total_revenue
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed', 'Finished')
          AND distance_in_miles <= %s
        GROUP BY dim_driver_id, driver_full_name, source
        ORDER BY dim_driver_id, trips DESC
    """, (IDS, MAX_TRIP_MILES))

@st.cache_data(ttl=3600)
def load_map_trips(driver_id, limit=500):
    """Trips for a single driver for map rendering, most recent first.
    SQL-filters on distance; coordinate-level filter applied in app layer."""
    return query("""
        SELECT
            dim_driver_id,
            driver_full_name,
            pickedup_trip_datetime,
            pickup_lat_long,
            dropoff_latlong,
            pickup_address,
            dropoff_address,
            trip_price_in_pound,
            distance_in_miles,
            pob_duration_in_min,
            pickup_duration_in_min,
            source,
            trips_hr
        FROM rep_fact_trips
        WHERE dim_driver_id = %s
          AND status IN ('completed', 'Finished')
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
          AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
          AND distance_in_miles <= %s
        ORDER BY pickedup_trip_datetime DESC
        LIMIT %s
    """, (driver_id, MAX_TRIP_MILES, limit))

@st.cache_data(ttl=3600)
def load_zone_trips():
    """Raw trip coords for zone enrichment — includes timestamps for true-RPH gap calc."""
    return query("""
        SELECT
            dim_driver_id,
            driver_full_name,
            pickup_lat_long,
            dropoff_latlong,
            trip_price_in_pound,
            distance_in_miles,
            pob_duration_in_min,
            pickup_duration_in_min,
            trips_hr,
            source,
            pickedup_trip_datetime,
            dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed', 'Finished')
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
          AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
          AND distance_in_miles <= %s
        ORDER BY dim_driver_id, pickedup_trip_datetime
    """, (IDS, MAX_TRIP_MILES))

@st.cache_data(ttl=3600)
def load_day_patterns():
    """Individual trip records needed for day-pattern analysis — size, timing, sequence."""
    return query("""
        SELECT
            dim_driver_id,
            driver_full_name,
            DATE(pickedup_trip_datetime)            AS trip_date,
            pickedup_trip_datetime,
            trips_hr                               AS hour,
            trip_price_in_pound                    AS fare,
            distance_in_miles                      AS distance,
            pob_duration_in_min                    AS ride_mins,
            pickup_duration_in_min                 AS pickup_mins,
            source,
            EXTRACT(DOW FROM pickedup_trip_datetime) AS dow
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed', 'Finished')
          AND distance_in_miles <= %s
          AND trips_hr IS NOT NULL
          AND trip_price_in_pound > 0
        ORDER BY dim_driver_id, pickedup_trip_datetime
    """, (IDS, MAX_TRIP_MILES))

@st.cache_data(ttl=3600)
def load_flow_with_address():
    """All trips (accepted + declined) with pickup_address for Zone 1 area breakdown.
    Separate from flow_data.parquet because that pre-built file has no address strings."""
    return query("""
        SELECT dim_driver_id, status,
               pickup_lat_long, dropoff_latlong,
               pickup_address,
               trip_price_in_pound, distance_in_miles,
               trips_hr
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed','Finished','Driver did not respond','Driver rejected')
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
          AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
          AND distance_in_miles <= %s
    """, (IDS, MAX_TRIP_MILES))

@st.cache_data(ttl=3600)
def load_all_driver_coords(sample_per_driver=60, days_back=30):
    """Sample pickup coords for every active driver — used for fleet positioning map."""
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    return query("""
        SELECT dim_driver_id, driver_full_name, pickup_lat_long
        FROM (
            SELECT dim_driver_id, driver_full_name, pickup_lat_long,
                   ROW_NUMBER() OVER (
                       PARTITION BY dim_driver_id
                       ORDER BY pickedup_trip_datetime DESC
                   ) AS rn
            FROM rep_fact_trips
            WHERE status IN ('completed','Finished')
              AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
              AND distance_in_miles <= 60
              AND pickedup_trip_datetime >= %s
        ) sub
        WHERE rn <= %s
    """, (from_date, sample_per_driver))

@st.cache_data(ttl=3600)
def load_comparison_trips(driver_ids):
    """All accepted trips for a given set of driver IDs — used for good vs bad comparison."""
    return query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               pob_duration_in_min, pickup_duration_in_min,
               trips_hr, source,
               pickedup_trip_datetime, dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed','Finished')
          AND pickup_lat_long  IS NOT NULL AND pickup_lat_long  != ''
          AND dropoff_latlong  IS NOT NULL AND dropoff_latlong  != ''
          AND distance_in_miles <= 60
        ORDER BY dim_driver_id, pickedup_trip_datetime
    """, (list(driver_ids),))

@st.cache_data(ttl=3600)
def load_comparison_flow(driver_ids):
    """Accepted + declined trips for zone flow comparison."""
    return query("""
        SELECT dim_driver_id,
               status,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               trips_hr
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed','Finished','Driver did not respond','Driver rejected')
          AND pickup_lat_long  IS NOT NULL AND pickup_lat_long  != ''
          AND dropoff_latlong  IS NOT NULL AND dropoff_latlong  != ''
          AND distance_in_miles <= 60
    """, (list(driver_ids),))

@st.cache_data(ttl=3600)
def load_comparison_performance(driver_ids):
    """Aggregated performance metrics for a given set of driver IDs."""
    return query("""
        SELECT dim_driver_id,
               MAX(driver_full_name)                                          AS driver_name,
               SUM(number_of_finished_rides)                                  AS total_rides,
               ROUND(SUM(online_time_in_hrs)::numeric, 1)                    AS online_hrs,
               ROUND(SUM(revenue)::numeric, 2)                               AS total_revenue,
               ROUND((SUM(revenue)/NULLIF(SUM(online_time_in_hrs),0))::numeric,2) AS rph,
               ROUND(AVG(utilisation_percent)::numeric,1)                    AS utilisation,
               ROUND(AVG(NULLIF(total_acceptance_rate_percent,0))::numeric,1) AS acceptance,
               ROUND((SUM(revenue)/NULLIF(SUM(number_of_finished_rides),0))::numeric,2) AS avg_fare
        FROM rep_fact_driver_performance
        WHERE dim_driver_id = ANY(%s) AND online_time_in_hrs > 0
        GROUP BY dim_driver_id
        ORDER BY rph DESC
    """, (list(driver_ids),))

@st.cache_data(ttl=3600)
def load_gap_accepted(driver_ids, days_back=14):
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    return query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               pob_duration_in_min, pickup_duration_in_min,
               trips_hr, source,
               pickedup_trip_datetime, dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed','Finished')
          AND pickup_lat_long  IS NOT NULL AND pickup_lat_long  != ''
          AND dropoff_latlong  IS NOT NULL AND dropoff_latlong  != ''
          AND distance_in_miles <= 60
          AND pickedup_trip_datetime >= %s
        ORDER BY dim_driver_id, pickedup_trip_datetime
    """, (list(driver_ids), from_date))

@st.cache_data(ttl=3600)
def load_gap_declined(driver_ids, days_back=14):
    from_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    return query("""
        SELECT dim_driver_id,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               trip_booking_datetime, source
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('Driver did not respond','Driver rejected')
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
          AND trip_booking_datetime >= %s
    """, (list(driver_ids), from_date))

@st.cache_data(ttl=3600)
def load_driver_declined_day(driver_id, date_str):
    """Declined pings for one driver on one date, with pickup coords."""
    return query("""
        SELECT pickup_lat_long,
               trip_booking_datetime,
               distance_in_miles,
               source,
               trip_price_in_pound
        FROM rep_fact_trips
        WHERE dim_driver_id = %s
          AND status IN ('Driver did not respond', 'Driver rejected')
          AND DATE(trip_booking_datetime) = %s::date
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
    """, (driver_id, date_str))

@st.cache_data(ttl=3600)
def load_any_driver_trips(driver_id):
    """All completed trips for any single driver — used by Driver Day for non-top-10 drivers."""
    return query("""
        SELECT dim_driver_id, driver_full_name,
               pickup_lat_long, dropoff_latlong,
               trip_price_in_pound, distance_in_miles,
               pob_duration_in_min, pickup_duration_in_min,
               trips_hr, source,
               pickedup_trip_datetime, dropoff_trip_datetime
        FROM rep_fact_trips
        WHERE dim_driver_id = %s
          AND status IN ('completed','Finished')
          AND pickup_lat_long IS NOT NULL AND pickup_lat_long != ''
          AND dropoff_latlong IS NOT NULL AND dropoff_latlong != ''
          AND distance_in_miles <= %s
        ORDER BY pickedup_trip_datetime
    """, (driver_id, MAX_TRIP_MILES))

@st.cache_data(ttl=3600)
def search_drivers(name_fragment):
    """Return (dim_driver_id, driver_full_name) rows matching a name fragment."""
    return query("""
        SELECT DISTINCT dim_driver_id, MAX(driver_full_name) AS driver_full_name
        FROM rep_fact_driver_performance
        WHERE UPPER(driver_full_name) LIKE UPPER(%s)
          AND online_time_in_hrs > 0
        GROUP BY dim_driver_id
        ORDER BY driver_full_name
        LIMIT 20
    """, (f"%{name_fragment}%",))

@st.cache_data(ttl=3600)
def load_shift_patterns():
    """First and last trip hour per driver per day to understand shift structure."""
    return query("""
        SELECT
            dim_driver_id,
            driver_full_name,
            DATE(pickedup_trip_datetime) as shift_date,
            MIN(trips_hr) as shift_start_hr,
            MAX(trips_hr) as shift_end_hr,
            COUNT(*) as trips_in_shift,
            MAX(trips_hr) - MIN(trips_hr) as shift_span_hrs,
            ROUND(SUM(trip_price_in_pound)::numeric, 2) as shift_revenue
        FROM rep_fact_trips
        WHERE dim_driver_id = ANY(%s)
          AND status IN ('completed', 'Finished')
          AND trips_hr IS NOT NULL
          AND distance_in_miles <= %s
        GROUP BY dim_driver_id, driver_full_name, DATE(pickedup_trip_datetime)
        HAVING COUNT(*) >= 3
        ORDER BY dim_driver_id, shift_date
    """, (IDS, MAX_TRIP_MILES))
