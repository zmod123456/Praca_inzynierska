import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
import os
import xgboost as xgb

st.set_page_config(page_title="Prognoza PM2.5", page_icon="🌫️", layout="wide")
st.title("Prognoza jakości powietrza (PM2.5) dla miasta Poznania")
st.caption("Prognoza jest wykonywana rekurencyjnie, dlatego przewidywane wartości mogą różnić się od rzeczywistych pomiarów.")

STACJA_GIOS_SENSOR_ID = 3497   
LAT, LON = 52.4064, 16.9252    


import os

@st.cache_resource
def wczytaj_model():
    katalog = os.path.dirname(os.path.abspath(__file__))
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(katalog, "model_finalny.json"))
    features = joblib.load(os.path.join(katalog, "features_lista.pkl"))
    return model, features

model, features_lista = wczytaj_model()

# pobieranie danych pogodowych
@st.cache_data(ttl=3600)  
def pobierz_prognoze_pogody(lat, lon, dni_wstecz=2, dni_wprzod=7):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,winddirection_10m,"
                   "surface_pressure,pressure_msl,relative_humidity_2m,precipitation",
        "past_days": dni_wstecz,
        "forecast_days": dni_wprzod,
        "timezone": "Europe/Warsaw"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    dane = r.json()["hourly"]
    df = pd.DataFrame(dane)
    df["Data"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "temperature_2m": "temp_powietrza",
        "windspeed_10m": "fwr_predkosc_wiatru",
        "winddirection_10m": "krwr_kierunek_wiatru",
        "surface_pressure": "ppps_cisnienie_stacja",
        "pressure_msl": "pppm_cisnienie_morze",
        "relative_humidity_2m": "wlgw_wilgotnosc",
        "precipitation": "wo6g_opad_6h",
    })
    return df

# pobieranie danych z gios
@st.cache_data(ttl=3600)
def pobierz_ostatnie_pm25(sensor_id):
    url = f"https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/{sensor_id}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    dane = r.json()
    wartosci = dane.get("Lista danych pomiarowych", dane.get("values", []))
    df = pd.DataFrame(wartosci)

  
    kol_data = "Data" if "Data" in df.columns else "date"
    kol_wart = "Wartość" if "Wartość" in df.columns else "value"

    df["Data"] = pd.to_datetime(df[kol_data])
    df["pm25"] = pd.to_numeric(df[kol_wart], errors="coerce")
    df = df.dropna(subset=["pm25"]).sort_values("Data")
    return df[["Data", "pm25"]]


# cechy z trenowania modelu

def przygotuj_cechy(df_pogoda):
    d = df_pogoda.copy().sort_values("Data").reset_index(drop=True)

    d["wiatr_kier_sin"] = np.sin(np.radians(d["krwr_kierunek_wiatru"]))
    d["wiatr_kier_cos"] = np.cos(np.radians(d["krwr_kierunek_wiatru"]))

    d["fwr_roll6"]   = d["fwr_predkosc_wiatru"].rolling(6, min_periods=1).mean()
    d["fwr_roll24"]  = d["fwr_predkosc_wiatru"].rolling(24, min_periods=1).mean()
    d["temp_roll24"] = d["temp_powietrza"].rolling(24, min_periods=1).mean()
    d["cisza_24h"]   = (d["fwr_predkosc_wiatru"] < 2).rolling(24, min_periods=1).sum()
    
    d["Rok"] = d["Data"].dt.year
    d["Miesiac"] = d["Data"].dt.month
    d["Dzien"] = d["Data"].dt.day
    d["Godzina"] = d["Data"].dt.hour
    d["Dzien_Tygodnia"] = d["Data"].dt.dayofweek
    d["Dzien_Roku"] = d["Data"].dt.dayofyear
    d["Weekend"] = (d["Dzien_Tygodnia"] >= 5).astype(int)
    d["Godzina_sin"] = np.sin(2 * np.pi * d["Godzina"] / 24)
    d["Godzina_cos"] = np.cos(2 * np.pi * d["Godzina"] / 24)
    d["Miesiac_sin"] = np.sin(2 * np.pi * d["Miesiac"] / 12)
    d["Miesiac_cos"] = np.cos(2 * np.pi * d["Miesiac"] / 12)
    d["DzienTyg_sin"] = np.sin(2 * np.pi * d["Dzien_Tygodnia"] / 7)
    d["DzienTyg_cos"] = np.cos(2 * np.pi * d["Dzien_Tygodnia"] / 7)
    return d

# prognoza za pomoca rekurencji

def prognozuj(model, features_lista, df_cechy, df_pm25_historyczne):
    historia_pm25 = df_pm25_historyczne.set_index("Data")["pm25"].to_dict()

    wyniki = []
    for _, wiersz in df_cechy.iterrows():
        czas_lag = wiersz["Data"] - pd.Timedelta(hours=24)
        lag_val = historia_pm25.get(czas_lag, np.nan)

        wejscie = wiersz.copy()
        wejscie["PM25_lag24"] = lag_val

        X = pd.DataFrame([wejscie[features_lista].astype(float)])
        pred = float(model.predict(X)[0])
        pred = max(pred, 0)  # PM2.5 nie może być ujemne

        wyniki.append({"Data": wiersz["Data"], "PM25_pred": pred})
        historia_pm25[wiersz["Data"]] = pred

    return pd.DataFrame(wyniki)

# sidebar z ustawieniami i odswiezaniem

with st.sidebar:
    st.header("Ustawienia")
    dni = st.slider("Ilość dni", 1, 7, 7)
    st.markdown("---")
    odswiez = st.button("Odśwież dane")

if odswiez:
    st.cache_data.clear()

# pobieranie danych

with st.spinner("Pobieranie danych pogodowych..."):
    df_pogoda = pobierz_prognoze_pogody(LAT, LON, dni_wprzod=dni)

with st.spinner("Pobieranie ostatnich pomiarów PM2.5..."):
    try:
        df_pm25_hist = pobierz_ostatnie_pm25(STACJA_GIOS_SENSOR_ID)
        pm25_ok = len(df_pm25_hist) > 0
    except Exception as e:
        st.warning(f"Nie udało się pobrać danych PM2.5 z GIOŚ: {e}")
        df_pm25_hist = pd.DataFrame(columns=["Data", "pm25"])
        pm25_ok = False

if not pm25_ok:
    st.error(
        "Brak świeżych danych PM2.5 do policzenia cechy lag24. "
        "Prognoza może być mniej dokładna dla pierwszych 24h."
    )

df_cechy = przygotuj_cechy(df_pogoda)
teraz = pd.Timestamp.now().floor("h")
df_cechy_przyszlosc = df_cechy[df_cechy["Data"] >= teraz].reset_index(drop=True)

df_prognoza = prognozuj(model, features_lista, df_cechy_przyszlosc, df_pm25_hist)

# agregacja do wartości dobowych
df_prognoza["Dzien"] = df_prognoza["Data"].dt.date
prognoza_dobowa = df_prognoza.groupby("Dzien")["PM25_pred"].mean().reset_index()
prognoza_dobowa.columns = ["Data", "PM25_srednia"]

# wyswietlanie wynikow

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prognoza godzinowa PM2.5")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prognoza["Data"], y=df_prognoza["PM25_pred"],
        mode="lines", name="Prognoza PM2.5", line=dict(color="#1f77b4")
    ))
    if pm25_ok:
        fig.add_trace(go.Scatter(
            x=df_pm25_hist["Data"], y=df_pm25_hist["pm25"],
            mode="lines", name="Dane rzeczywiste (ostatnie dni)",
            line=dict(color="black", dash="dash")
        ))
    fig.update_layout(
        xaxis_title="Data", yaxis_title="PM2.5 (µg/m³)",
        height=450, hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Średnie dobowe")
    st.dataframe(
        prognoza_dobowa.style.format({"PM25_srednia": "{:.1f}"}),
        use_container_width=True, hide_index=True
    )



with st.expander("Zobacz surowe dane wejściowe (pogoda + PM2.5)"):
    tabela_pogoda = df_cechy[["Data", "temp_powietrza", "fwr_predkosc_wiatru",
                                "ppps_cisnienie_stacja", "wlgw_wilgotnosc"]]

    if pm25_ok:
        tabela = tabela_pogoda.merge(df_pm25_hist, on="Data", how="outer")
    else:
        tabela = tabela_pogoda.copy()
        tabela["pm25"] = np.nan

    tabela_pelna = tabela.dropna()

    st.dataframe(
        tabela_pelna.sort_values("Data").round(2),
        use_container_width=True, hide_index=True
    )