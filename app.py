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

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="緯度経度変換ツール", layout="wide")
st.title("緯度経度変換ツール")

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    """10進法度をDMS形式(度分秒)の文字列に変換"""
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:07.4f}\""

@st.cache_resource
def load_geoid_data():
    """ジオイドモデルの読み込み"""
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
    """指定した位置のジオイド高をバイリニア補間"""
    if model_name == "使用しない" or not geoid_db:
        return 0.0
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get("2024")
            r = (50.0 - lat) * 60.0
            c = (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "日本のジオイド2011":
            g = geoid_db.get("2011")
            h = geoid_db.get("2011_h")
            r = (lat - h[0]) / h[2]
            c = (lon - h[1]) / h[3]
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
    """CSV読み込み（ヘッダー有無を自動判定）"""
    uploaded_file.seek(0)
    first_line = uploaded_file.readline().decode("shift-jis", errors="replace").strip()
    uploaded_file.seek(0)
    
    cols = first_line.split(",")
    try:
        float(cols[1])
        float(cols[2])
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=None)
    except (ValueError, IndexError):
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=0)
    
    if df.shape[1] < 4:
        raise ValueError("CSVは最低4列（点名, X, Y, H）必要です。")
    
    df = df.iloc[:, :4]
    df.columns = ["点名", "X", "Y", "H"]
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    return df.dropna(subset=["X", "Y", "H"])

def build_kml(res_data, kml_export_type, non_point_features):
    """Google Earth対応KML生成（修正済み）"""
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("  <Document>")
    lines.append("    <name>spatial_data</name>")
    
    # スタイル定義
    lines += [
        '    <Style id="poly_style"><LineStyle><color>ffff0000</color><width>2</width></LineStyle><PolyStyle><color>6400ffff</color></PolyStyle></Style>',
        '    <Style id="line_style"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>',
        '    <Style id="point_style"><IconStyle><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>',
    ]

    # ポイント出力
    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            lines += [
                "    <Placemark>",
                f"      <name>{r['点名']}</name>",
                "      <styleUrl>#point_style</styleUrl>",
                "      <Point>",
                f"        <coordinates>{r['経度']},{r['緯度']},{r['楕円体高']}</coordinates>",
                "      </Point>",
                "    </Placemark>",
            ]

    # 図形出力（修正箇所：判定を「図形」に変更し、座標形式を調整）
    if "図形" in kml_export_type and non_point_features:
        for i, feat in enumerate(non_point_features):
            geom = feat.get("geometry", {})
            g_type = geom.get("type")
            coords = geom.get("coordinates", [])
            
            if g_type == "Polygon":
                ring = coords[0]
                if ring[0] != ring[-1]: ring.append(ring[0])
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in ring])
                lines += [
                    "    <Placemark>",
                    f"      <name>Polygon_{i+1}</name>",
                    "      <styleUrl>#poly_style</styleUrl>",
                    "      <Polygon><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><outerBoundaryIs><LinearRing><coordinates>",
                    f"        {coord_str}",
                    "      </coordinates></LinearRing></outerBoundaryIs></Polygon>",
                    "    </Placemark>",
                ]
            elif g_type == "LineString":
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                lines += [
                    "    <Placemark>",
                    f"      <name>Line_{i+1}</name>",
                    "      <styleUrl>#line_style</styleUrl>",
                    "      <LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><coordinates>",
                    f"        {coord_str}",
                    "      </coordinates></LineString>",
                    "    </Placemark>",
                ]

    lines.append("  </Document>")
    lines.append("</kml>")
    return "\n".join(lines).encode("utf-8")

# --- 3. セッション初期化 ---
if "saved_polygons" not in st.session_state:
    st.session_state.saved_polygons = []
if "map_markers" not in st.session_state:
    st.session_state.map_markers = []
if "registered_marker_coords" not in st.session_state:
    st.session_state.registered_marker_coords = []

# --- 4. サイドバー ---
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

if "result" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("💾 成果品保存")
    latlon_fmt = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"])
    
    df_save = st.session_state.result.copy()
    if latlon_fmt == "60進法 (DMS)":
        df_save["緯度"] = df_save["緯度"].apply(decimal_to_dms)
        df_save["経度"] = df_save["経度"].apply(decimal_to_dms)
    
    st.sidebar.download_button("📊 CSVを保存", df_save.to_csv(index=False).encode("utf-8-sig"), f"result_{int(time.time())}.csv", "text/csv", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 KML保存")
    kml_type = st.sidebar.selectbox("KML出力対象", ["ポイントと図形の両方", "ポイントのみ", "図形のみ"])
    kml_data = build_kml(st.session_state.result, kml_type, st.session_state.saved_polygons)
    st.sidebar.download_button("🌍 KMLを保存", kml_data, f"map_data_{int(time.time())}.kml", "application/vnd.google-earth.kml+xml", use_container_width=True)

# --- 5. メイン画面 ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)

tab1, tab2, tab3 = st.tabs(["📝 1点手入力", "📂 ファイル一括", "📖 ガイド"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    p_name = c1.text_input("点名", "P-1")
    p_x = c2.number_input("X座標", value=0.0, format="%.4f")
    p_y = c3.number_input("Y座標", value=0.0, format="%.4f")
    p_h = c4.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行", type="primary"):
        lon, lat = transformer.transform(p_y, p_x)
        gh = get_geoid_height(lat, lon, use_geoid)
        st.session_state.result = pd.DataFrame([{
            "点名": p_name, "X": p_x, "Y": p_y, "標高H": p_h, "緯度": lat, "経度": lon,
            "ジオイド高": gh, "楕円体高": round(p_h + gh + offset_val, 4), "適用モデル": use_geoid
        }])

with tab2:
    up = st.file_uploader("CSV / SIMA", type=["csv", "sim"])
    if up and st.button("一括計算開始", type="primary"):
        if up.name.lower().endswith(".sim"):
            pts = []
            for line in up.read().decode("shift-jis", errors="replace").splitlines():
                p = line.split(",")
                if len(p) >= 6 and p[0] in ["A01", "C00", "C01"]:
                    pts.append({"点名": p[1] if p[0].startswith("C") else p[2], "X": float(p[3]), "Y": float(p[4]), "H": float(p[5])})
            df_in = pd.DataFrame(pts)
        else:
            df_in = read_csv_auto(up)
        
        lons, lats = transformer.transform(df_in["Y"].values, df_in["X"].values)
        ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
        df_in["緯度"], df_in["経度"], df_in["ジオイド高"] = lats, lons, ghs
        df_in["楕円体高"] = df_in["H"] + np.array(ghs) + offset_val
        df_in["適用モデル"] = use_geoid
        st.session_state.result = df_in.rename(columns={"H": "標高H"})

# 結果表示
if "result" in st.session_state:
    st.divider()
    st.dataframe(st.session_state.result, use_container_width=True)
    c_lat, c_lon = st.session_state.result["緯度"].mean(), st.session_state.result["経度"].mean()
else:
    c_lat, c_lon = 35.6812, 139.7671

# マップ表示
st.subheader("🗺 マッププレビュー")
t_url = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg" if map_type=="航空写真" else "OpenStreetMap"
m = folium.Map(location=[c_lat, c_lon], zoom_start=15, tiles=t_url, attr="GSI/OSM")

if st.session_state.saved_polygons:
    folium.GeoJson({"type": "FeatureCollection", "features": st.session_state.saved_polygons}).add_to(m)
if "result" in st.session_state:
    for _, r in st.session_state.result.iterrows():
        folium.Marker([r["緯度"], r["経度"]], popup=r["点名"]).add_to(m)

Draw(draw_options={"circle":False, "rectangle":True, "marker":True}).add_to(m)
out = st_folium(m, width=1200, height=600)

# 描画データの取得
if out and out.get("last_active_drawing"):
    last = out["last_active_drawing"]
    if last not in st.session_state.saved_polygons:
        st.session_state.saved_polygons.append(last)
        st.rerun()
