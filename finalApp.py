from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import numpy as np
import joblib
import requests
from datetime import date

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("AviationData_cleaned_decimal.csv")
    df['Event.Date'] = pd.to_datetime(df['Event.Date'], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    return df.copy()

df = load_data()

coords = df[['Latitude', 'Longitude']]
kmeans_model = KMeans(n_clusters=5, random_state=42).fit(coords)
df['Cluster'] = kmeans_model.labels_
cluster_counts = df['Cluster'].value_counts(normalize=True).to_dict()

try:
    ml_model = joblib.load("model_xgb.pkl")
    use_ml_model = True
except FileNotFoundError:
    use_ml_model = False
    
def fetch_weather_features(lat, lon, target_date):
    """Fetch daily weather data (temp, precip, windspeed) for given lat/lon and date using Open-Meteo."""
    date_str = target_date.strftime("%Y-%m-%d")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        "&timezone=UTC"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        daily = data.get("daily", {})
        wx = {
            "temp_max": daily.get("temperature_2m_max", [None])[0],
            "temp_min": daily.get("temperature_2m_min", [None])[0],
            "precip": daily.get("precipitation_sum", [None])[0],
            "windspeed": daily.get("windspeed_10m_max", [None])[0],
        }
    except Exception as e:
        st.warning(f"Weather data unavailable ({e}). Using defaults.")
        wx = {"temp_max": None, "temp_min": None, "precip": None, "windspeed": None}
    return wx

def heuristic_weather_factor(wx):
    """Simple heuristic to adjust risk based on weather."""
    factor = 1.0
    if wx["precip"] is not None and wx["precip"] > 5:
        factor += 0.2
    if wx["windspeed"] is not None and wx["windspeed"] > 10:
        factor += 0.2
    if wx["temp_max"] is not None and wx["temp_max"] > 35:
        factor += 0.1
    if wx["temp_min"] is not None and wx["temp_min"] < -5:
        factor += 0.1
    return factor

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Year-wise Analysis", "Risk Prediction" , "Crash Path Simulator"])

if page == "Homepage":
    st.title("✈️ Airplane Crash Risk Analysis Dashboard")
    st.markdown("### Overview: Crash Frequency and Clustering (2008 - 2012)")

    st.subheader("📊 Crash Frequency Over Years")
    fig1, ax1 = plt.subplots()
    df['Event.Date'].dt.year.value_counts().sort_index().plot(kind='bar', ax=ax1)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Crashes")
    st.pyplot(fig1)

    st.subheader("🗺️ Crash Clustering Map")
    map_ = folium.Map(location=[coords['Latitude'].mean(), coords['Longitude'].mean()], zoom_start=2)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=4,
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(map_)
    folium_static(map_)

elif page == "Year-wise Analysis":
    st.title("📅 Year-wise Airplane Crash Data")
    year_options = sorted(df['Event.Date'].dt.year.dropna().unique())
    selected_year = st.selectbox("Select Year", year_options)
    filtered_df = df[df['Event.Date'].dt.year == selected_year]

    st.subheader(f"📋 Crash Data for {selected_year}")
    st.dataframe(filtered_df)

    st.subheader(f"📈 Crash Frequency in {selected_year}")
    daily_counts = filtered_df['Event.Date'].value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    daily_counts.plot(kind='line', marker='o', ax=ax2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Crashes")
    st.pyplot(fig2)

elif page == "Risk Prediction":
    st.title("🛡️ Crash Risk Prediction")
    st.markdown("Enter a location and date to estimate crash risk.")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            location_name = st.text_input("Enter Location (City or Airport Name)", "New York")
            geolocator = Nominatim(user_agent="risk_predictor")
            location = geolocator.geocode(location_name)

            if location:
                lat, lon = location.latitude, location.longitude
            else:
                st.error("Could not geolocate the entered location.")

        with col2:
            pred_date = st.date_input("Date", value=date.today())
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        year = pred_date.year
        month = pred_date.month
        dayofyear = pred_date.timetuple().tm_yday

        wx = fetch_weather_features(lat, lon, pred_date)

        if use_ml_model:
            features = pd.DataFrame({
                'Latitude': [lat],
                'Longitude': [lon],
                'Year': [year],
                'Month': [month],
                'DayOfYear': [dayofyear],
                'TempMax': [wx['temp_max']],
                'TempMin': [wx['temp_min']],
                'Precip': [wx['precip']],
                'WindSpeed': [wx['windspeed']]
            })
            try:
                risk_prob = ml_model.predict_proba(features)[:, 1][0]
                st.metric("Predicted Risk Score", f"{risk_prob*100:.2f} %")
            except Exception as e:
                st.error(f"ML model error: {e}")
        else:
            cluster_id = int(kmeans_model.predict(np.array([[lat, lon]]))[0])
            base_risk = cluster_counts.get(cluster_id, 0)
            month_counts = df[df['Event.Date'].dt.month == month].shape[0]
            overall_counts = df.shape[0]
            seasonal_factor = month_counts / overall_counts if overall_counts else 0
            heuristic_risk = (0.6 * base_risk) + (0.4 * seasonal_factor)
            weather_factor = heuristic_weather_factor(wx)
            total_risk = heuristic_risk * weather_factor
            st.metric("Estimated Risk Score", f"{total_risk*100:.2f} %")
            st.caption("Heuristic combines cluster density, seasonal crash frequency, and weather conditions.")


elif page == "Crash Path Simulator":

    @st.cache
    def geocode_location(location_name):
        geolocator = Nominatim(user_agent="crash_simulator")
        return geolocator.geocode(location_name)

    @st.cache
    def get_hotspots():
        df = pd.read_csv("AviationData_cleaned_decimal.csv")
        coords = df[['Latitude', 'Longitude']].dropna()
        kmeans = KMeans(n_clusters=40, random_state=42).fit(coords)
        return kmeans.cluster_centers_

    def haversine_distance(coord1, coord2):
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 6371 * 2 * np.arcsin(np.sqrt(a))  # in km

    st.title("✈️ Crash Path Simulator")

    st.write("Enter Start and End Locations:")
    start_location = st.text_input("Start Location (City or Airport Name)", "New York")
    end_location = st.text_input("End Location (City or Airport Name)", "Chicago")

    start = geocode_location(start_location)
    end = geocode_location(end_location)

    if start and end:
        start_coords = (start.latitude, start.longitude)
        end_coords = (end.latitude, end.longitude)

        num_points = 100
        lats = np.linspace(start.latitude, end.latitude, num_points)
        longs = np.linspace(start.longitude, end.longitude, num_points)
        route = np.array(list(zip(lats, longs)))

        hotspots = get_hotspots()

        threshold_km = 50
        intersecting_hotspots = []
        for hotspot in hotspots:
            for point in route:
                if haversine_distance(point, hotspot) <= threshold_km:
                    intersecting_hotspots.append(tuple(hotspot))
                    break

        risk_percentage = (len(intersecting_hotspots) / len(hotspots)) * 100

        m = folium.Map(location=start_coords, zoom_start=5)
        folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(end_coords, tooltip="End", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine(route, color="blue", weight=2.5, opacity=0.7).add_to(m)

        for hotspot in hotspots:
            folium.Circle(
                location=hotspot,
                radius=50000,
                color="crimson",
                fill=True,
                fill_opacity=0.3
            ).add_to(m)

        st_folium(m, width=700)

        st.metric("Crash Risk Along Route", f"{risk_percentage:.2f}%")
    else:
        st.error("Could not geolocate one or both locations.")


