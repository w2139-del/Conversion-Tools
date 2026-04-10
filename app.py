# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import os
import time

# ==========================================
# 📍 デフォルト位置設定
# ==========================================
DEFAULT_LAT = 37.76162301842977
DEFAULT_LON = 140.47599584578793

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="緯度経度変換ツール", layout="wide")
st.title("緯度経度変換ツール")

# セッション状態の初期化
if "result" not in st.session_state:
    st.session_state.result = pd.DataFrame(columns=["点名", "X", "Y", "標高H", "緯度", "経度", "ジオイド高", "楕円体高", "適用モデル"])
if "registered_marker_coords" not in st.session_state:
    st.session_state.registered_marker_coords = []
if "drawn_data" not in st.session_state:
    st.session_state.drawn_data = {}

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    try:
        d = int(deg)
        m = int((deg - d) * 60)
        s = (deg - d - m / 60) * 3600
        return f"{d}°{m:02d}'{s:07.4f}\""
    except:
        return str(deg)

@st.cache_resource
def load_geoid_data():
    data = {}
    if os.path.exists("geoid2024.npz"):
        data["2024"] = np.load("geoid2024.npz")["grid"]
    if os.path.exists("geoid2011.npz"):
        loader = np.load("geoid2011.npz")
        data["2011"] = loader["grid"]
        data["2011_h"] = loader["header"]
    return data

geoid_db = load_geoid_data()

def get_geoid_height(lat, lon, model_name):
    if model_name == "使用しない" or not geoid_db:
        return 0.0
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get("2024")
            r, c = (50.0 - lat) * 60.0, (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "日本のジオイド2011":
            g, h = geoid_db.get("2011"), geoid_db.get("2011_h")
            r, c = (lat - h[0]) / h[2], (lon - h[1]) / h[3]
        else:
            return 0.0
        
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        if r1 >= g.shape[0] or c1 >= g.shape[1] or r0 < 0 or c0 < 0:
            return 0.0
        v = [g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]]
        if any(x > 900 for x in v): return 0.0
        dr, dc = r - r0, c - c0
        return round((1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3], 4)
    except:
        return 0.0

def read_csv_auto(uploaded_file):
    uploaded_file.seek(0)
    first_line = uploaded_file.readline().decode("shift-jis", errors="replace").strip()
    uploaded_file.seek(0)
    cols = first_line.split(",")
    try:
        float(cols[1])
        float(cols[2])
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=None)
    except:
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=0)
    
    if df.shape[1] < 4:
        raise ValueError("CSVは最低4列（点名, X, Y, H）必要です。")
    df = df.iloc[:, :4]
    df.columns = ["点名", "X", "Y", "H"]
    for c in ["X", "Y", "H"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["X", "Y", "H"])

def build_kml(res_data, kml_export_type, drawn_data):
    placemarks = []
    # ポイント出力
    if "ポイント" in kml_export_type and res_data is not None and not res_data.empty:
        for _, r in res_data.iterrows():
            pname = str(r["点名"]).replace("&", "&amp;").replace("<", "&lt;")
            placemarks.append(f"""    <Placemark>
      <name>{pname}</name>
      <Point><coordinates>{r['経度']},{r['緯度']},{r['楕円体高']}</coordinates></Point>
    </Placemark>""")

    # 図形（ポリゴン・ライン）出力
    if "図形" in kml_export_type or "ポリゴン" in kml_export_type:
        features = drawn_data.get("all_drawings", []) if isinstance(drawn_data, dict) else []
        for i, feat in enumerate(features):
            geom = feat.get("geometry", {})
            g_type = geom.get("type")
            coords = geom.get("coordinates", [])
            if g_type == "Polygon":
                c_list = coords[0]
                if c_list[0] != c_list[-1]: c_list.append(c_list[0])
                c_str = " ".join([f"{c[0]},{c[1]},0" for c in c_list])
                placemarks.append(f"    <Placemark><name>Polygon_{i+1}</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{c_str}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
            elif g_type == "LineString":
                c_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                placemarks.append(f"    <Placemark><name>Line_{i+1}</name><LineString><coordinates>{c_str}</coordinates></LineString></Placemark>")

    return (f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f'{"".join(placemarks)}</Document></kml>').encode("utf-8")

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
latlon_fmt = st.sidebar.radio("緯度経度の表示形式", ["10進法 (DD)", "60進法 (DMS)"], index=0)
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"], index=0)

# 座標変換器の準備
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
transformer_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)

st.sidebar.markdown("---")
st.sidebar.header("💾 成果品保存")

# CSV/KMLダウンロードボタン（サイドバー）
if not st.session_state.result.empty:
    # CSV用データ作成
    disp_csv = st.session_state.result.copy()
    if latlon_fmt == "60進法 (DMS)":
        disp_csv["緯度"] = disp_csv["緯度"].apply(decimal_to_dms)
        disp_csv["経度"] = disp_csv["経度"].apply(decimal_to_dms)
    
    st.sidebar.download_button("📊 CSVを保存", disp_csv.to_csv(index=False).encode("utf-8-sig"), f"result_{int(time.time())}.csv", "text/csv", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 KML保存設定")
    kml_target = st.sidebar.selectbox("KML出力対象", ["ポイントと図形の両方", "ポイントのみ", "図形のみ"], index=0)
    kml_bytes = build_kml(st.session_state.result, kml_target, st.session_state.drawn_data)
    st.sidebar.download_button("🌍 KMLを保存", kml_bytes, f"spatial_data_{int(time.time())}.kml", "application/vnd.google-earth.kml+xml", use_container_width=True)
    
    if st.sidebar.button("🗑 リストをクリア", use_container_width=True):
        st.session_state.result = pd.DataFrame(columns=["点名", "X", "Y", "標高H", "緯度", "経度", "ジオイド高", "楕円体高", "適用モデル"])
        st.session_state.registered_marker_coords = []
        st.rerun()
else:
    st.sidebar.info("計算を実行すると保存ボタンが表示されます。")

# --- 4. メインコンテンツ ---
tab1, tab2, tab3 = st.tabs(["📝 1点手入力", "📂 ファイル一括", "📖 マニュアル"])

def update_results(input_df):
    if input_df.empty:
