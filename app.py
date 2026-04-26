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
    """10進法度をDMS形式(度分秒)に変換"""
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:07.4f}\""

@st.cache_resource
def load_geoid_data():
    """ジオイドモデルの読み込み"""
    data = {}
    if os.path.exists("geoid2024.npz"):
        try:
            data["2024"] = np.load("geoid2024.npz")["grid"]
        except: pass
    if os.path.exists("geoid2011.npz"):
        try:
            loader = np.load("geoid2011.npz")
            data["2011"] = loader["grid"]
            data["2011_h"] = loader["header"]
        except: pass
    return data

geoid_db = load_geoid_data()

def get_geoid_height(lat, lon, model_name):
    """ジオイド高の計算"""
    if model_name == "使用しない" or not geoid_db:
        return 0.0
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get("2024")
            if g is None: return 0.0
            r, c = (50.0 - lat) * 60.0, (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "日本のジオイド2011":
            g, h = geoid_db.get("2011"), geoid_db.get("2011_h")
            if g is None or h is None: return 0.0
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
    """CSV読み込み"""
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
    """【修正版】Googleマイマップ互換性を最大化したKML生成"""
    def escape_xml(text):
        return str(text).replace("&", "&").replace("<", "<").replace(">", ">")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        f'    <name>spatial_data_{int(time.time())}</name>',
        '    <Style id="point_style"><IconStyle><Icon><href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href></Icon></IconStyle></Style>',
        '    <Style id="poly_style"><LineStyle><color>ffff0000</color><width>2</width></LineStyle><PolyStyle><color>6400ffff</color><fill>1</fill><outline>1</outline></PolyStyle></Style>',
        '    <Style id="line_style"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>',
        '    <Folder><name>Points</name>'
    ]

    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            lines += [
                "      <Placemark>",
                f"        <name>{escape_xml(r['点名'])}</name>",
                "        <styleUrl>#point_style</styleUrl>",
                f"        <Point><coordinates>{r['経度']},{r['緯度']},{r['楕円体高']}</coordinates></Point>",
                "      </Placemark>"
            ]
    
    lines.append('    </Folder>')
    
    if "図形" in kml_export_type and non_point_features:
        lines.append('    <Folder><name>Shapes</name>')
        for i, feat in enumerate(non_point_features):
            geom = feat.get("geometry", {})
            g_type, coords = geom.get("type"), geom.get("coordinates", [])
            if g_type == "Polygon":
                ring = list(coords[0])
                if ring[0] != ring[-1]: ring.append(ring[0])
                c_str = " ".join(f"{c[0]},{c[1]},0" for c in ring)
                lines += [f"<Placemark><name>Poly_{i}</name><styleUrl>#poly_style</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{c_str}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"]
            elif g_type == "LineString":
                c_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                lines += [f"<Placemark><name>Line_{i}</name><styleUrl>#line_style</styleUrl><LineString><coordinates>{c_str}</coordinates></LineString></Placemark>"]
        lines.append('    </Folder>')

    lines += ['  </Document>', '</kml>']
    return "\n".join(lines).encode("utf-8")

# --- セッション初期化 ---
for key in ["saved_polygons", "map_markers", "registered_marker_coords"]:
    if key not in st.session_state: st.session_state[key] = []

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

st.sidebar.markdown("---")
st.sidebar.header("💾 成果品保存")
latlon_format = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"], index=0)

if "result" in st.session_state:
    res_data = st.session_state.result
    disp_csv = res_data.copy()
    if latlon_format == "60進法 (DMS)":
        disp_csv["緯度"] = disp_csv["緯度"].map(decimal_to_dms)
        disp_csv["経度"] = disp_csv["経度"].map(decimal_to_dms)
    else:
        # 【修正】Excelでの表示を考慮し、あえて下9桁で出力して精度を確保
        disp_csv["緯度"] = disp_csv["緯度"].map(lambda x: f"{x:.9f}")
        disp_csv["経度"] = disp_csv["経度"].map(lambda x: f"{x:.9f}")
    
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]: 
        disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")
    
    st.sidebar.download_button(label="📊 CSVを保存", data=disp_csv.to_csv(index=False).encode("utf-8-sig"), file_name=f"result_{int(time.time())}.csv", mime="text/csv", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 KML保存")
kml_export_type = st.sidebar.selectbox("KML出力対象", ["ポイントと図形の両方", "ポイントのみ", "図形のみ"], index=0)
kml_bytes = build_kml(st.session_state.get("result"), kml_export_type, st.session_state.saved_polygons)
st.sidebar.download_button(label="🌍 KMLを保存", data=kml_bytes, file_name=f"spatial_data_{int(time.time())}.kml", mime="application/vnd.google-earth.kml+xml", use_container_width=True)

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
tab1, tab2, tab3 = st.tabs(["📝 手入力", "📂 一括変換", "📖 ヘルプ"])

def run_calc(input_df):
    if input_df.empty: return
    lons, lats = transformer.transform(input_df["Y"].values, input_df["X"].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    st.session_state.result = pd.DataFrame({"点名": input_df["点名"], "X": input_df["X"], "Y": input_df["Y"], "標高H": input_df["H"], "緯度": lats, "経度": lons, "ジオイド高": ghs, "楕円体高": input_df["H"].values + np.array(ghs) + offset_val, "適用モデル": use_geoid})
    st.rerun()

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    p_n, p_x, p_y, p_h = c1.text_input("点名", "P-1"), c2.number_input("X", value=0.0, format="%.4f"), c3.number_input("Y", value=0.0, format="%.4f"), c4.number_input("H", value=0.0, format="%.4f")
    if st.button("計算実行", type="primary"): run_calc(pd.DataFrame([{"点名": p_n, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up = st.file_uploader("CSV/SIMA", type=["csv", "sim"])
    if up and st.button("一括変換開始", type="primary"):
        try:
            if up.name.lower().endswith(".sim"):
                pts = []
                content = up.read().decode("shift-jis", errors="replace")
                for line in content.splitlines():
                    p = line.split(",")
                    if len(p) >= 6 and p[0] in ["A01", "C00", "C01"]: pts.append({"点名": p[1] if p[0].startswith("C") else p[2], "X": float(p[3]), "Y": float(p[4]), "H": float(p[5])})
                run_calc(pd.DataFrame(pts))
            else: run_calc(read_csv_auto(up))
        except Exception as e: st.error(f"エラー: {e}")

if "result" in st.session_state:
    st.divider()
    res_disp = st.session_state.result.copy()
    # 画面表示も下9桁に統一
    res_disp["緯度"] = res_disp["緯度"].map(lambda x: f"{x:.9f}")
    res_disp["経度"] = res_disp["経度"].map(lambda x: f"{x:.9f}")
    st.dataframe(res_disp, use_container_width=True)
    c_lat, c_lon = st.session_state.result["緯度"].mean(), st.session_state.result["経度"].mean()
else:
    c_lat, c_lon = 37.7616, 140.4760

# --- 5. マップ表示 ---
m = folium.Map(location=[c_lat, c_lon], zoom_start=17, tiles="OpenStreetMap" if map_type=="標準地図" else "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg", attr="GSI")
if "result" in st.session_state:
    for _, r in st.session_state.result.iterrows():
        folium.Marker([r["緯度"], r["経度"]], tooltip=folium.Tooltip(str(r["点名"]), permanent=True)).add_to(m)
if st.session_state.saved_polygons:
    folium.GeoJson({"type": "FeatureCollection", "features": st.session_state.saved_polygons}).add_to(m)

Draw(export=False).add_to(m)
output = st_folium(m, width=1200, height=500)

# 描画データの保存
last_draw = output.get("last_active_drawing")
if last_draw:
    if last_draw.get("geometry", {}).get("type") in ["Polygon", "LineString"]:
        if last_draw not in st.session_state.saved_polygons:
            st.session_state.saved_polygons.append(last_draw)
            st.rerun()
