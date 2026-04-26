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
    """ジオイド高の計算（バイリニア補間）"""
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
    """CSV読み込み（ヘッダー自動判定）"""
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
    """【完全修正版】Googleマイマップ対応KML生成"""
    def escape_xml(text):
        """XML特殊文字を確実にエスケープする"""
        return str(text).replace("&", "&").replace("<", "<").replace(">", ">")

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("  <Document>")
    lines.append(f"    <name>spatial_data_{int(time.time())}</name>")
    
    # スタイル定義 (Google標準のプッシュピンURLを使用)
    lines += [
        '    <Style id="poly_style"><LineStyle><color>ffff0000</color><width>2</width></LineStyle><PolyStyle><color>6400ffff</color><fill>1</fill><outline>1</outline></PolyStyle></Style>',
        '    <Style id="line_style"><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>',
        '    <Style id="point_style"><IconStyle><Icon><href>https://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href></Icon></IconStyle></Style>',
    ]

    lines.append("    <Folder><name>Converted Data</name>")

    # ポイント出力
    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            safe_name = escape_xml(r["点名"])
            lines += [
                "      <Placemark>",
                f"        <name>{safe_name}</name>",
                "        <styleUrl>#point_style</styleUrl>",
                "        <Point><coordinates>" + f"{r['経度']},{r['緯度']},{r['楕円体高']}" + "</coordinates></Point>",
                "      </Placemark>",
            ]

    # 図形出力
    if "図形" in kml_export_type and non_point_features:
        poly_count, line_count = 1, 1
        for feat in non_point_features:
            geom = feat.get("geometry", {})
            g_type, coords = geom.get("type"), geom.get("coordinates", [])
            if g_type == "Polygon":
                ring = list(coords[0])
                if len(ring) < 3: continue
                if ring[0] != ring[-1]: ring.append(ring[0])
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in ring)
                lines += [
                    "      <Placemark>",
                    f"        <name>Polygon_{poly_count}</name>",
                    "        <styleUrl>#poly_style</styleUrl>",
                    "        <Polygon><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><outerBoundaryIs><LinearRing><coordinates>"+coord_str+"</coordinates></LinearRing></outerBoundaryIs></Polygon>",
                    "      </Placemark>"
                ]
                poly_count += 1
            elif g_type == "LineString":
                if len(coords) < 2: continue
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                lines += [
                    "      <Placemark>",
                    f"        <name>Line_{line_count}</name>",
                    "        <styleUrl>#line_style</styleUrl>",
                    "        <LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode><coordinates>"+coord_str+"</coordinates></LineString>",
                    "      </Placemark>"
                ]
                line_count += 1

    lines.append("    </Folder>")
    lines.append("  </Document>")
    lines.append("</kml>")
    return "\n".join(lines).encode("utf-8")

# =============================================================================
# 初期化
# =============================================================================
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
latlon_format = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"], index=0, key="fmt_radio")

if "result" in st.session_state:
    res_data = st.session_state.result
    disp_csv = res_data.copy()
    if latlon_format == "60進法 (DMS)":
        disp_csv["緯度"] = disp_csv["緯度"].map(decimal_to_dms)
        disp_csv["経度"] = disp_csv["経度"].map(decimal_to_dms)
    else:
        # 【修正箇所】Excelでの表示切れを防ぐため、精度を .9f (小数点以下9桁) に変更
        for c in ["緯度", "経度"]: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.9f}")
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")
    st.sidebar.download_button(label="📊 CSVを保存", data=disp_csv.to_csv(index=False).encode("utf-8-sig"), file_name=f"result_{int(time.time())}.csv", mime="text/csv", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 KML保存")
kml_export_type = st.sidebar.selectbox("KML出力対象", ["ポイントと図形の両方", "ポイントのみ", "図形のみ"], index=0)
kml_bytes = build_kml(st.session_state.get("result"), kml_export_type, st.session_state.saved_polygons)
st.sidebar.download_button(label="🌍 KMLを保存", data=kml_bytes, file_name=f"spatial_data_{int(time.time())}.kml", mime="application/vnd.google-earth.kml+xml", use_container_width=True)

# マーカー追加
st.sidebar.markdown("---")
st.sidebar.subheader("📍 マーカー追加")
_new_m = [m for m in st.session_state.map_markers if (round(m[0], 8), round(m[1], 8)) not in st.session_state.registered_marker_coords]
if _new_m and st.sidebar.button("➕ マーカーを変換リストに追加", type="primary"):
    _t_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)
    _ext = st.session_state.get("result", pd.DataFrame(columns=["点名", "X", "Y", "標高H", "緯度", "経度", "ジオイド高", "楕円体高", "適用モデル"]))
    _rows = []
    for la, lo in _new_m:
        y_v, x_v = _t_inv.transform(lo, la)
        gh = get_geoid_height(la, lo, use_geoid)
        _rows.append({"点名": f"Click_{len(_ext)+len(_rows)+1}", "X": round(x_v, 4), "Y": round(y_v, 4), "標高H": 0.0, "緯度": la, "経度": lo, "ジオイド高": gh, "楕円体高": round(0.0 + gh + offset_val, 4), "適用モデル": use_geoid})
        st.session_state.registered_marker_coords.append((round(la, 8), round(lo, 8)))
    st.session_state.result = pd.concat([_ext, pd.DataFrame(_rows)], ignore_index=True)
    st.rerun()

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
tab1, tab2, tab3 = st.tabs(["📝 1点手入力変換", "📂 ファイル一括変換", "📖 操作マニュアル"])

def run_calc(input_df):
    if input_df.empty: return
    lons, lats = transformer.transform(input_df["Y"].values, input_df["X"].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    st.session_state.result = pd.DataFrame({"点名": input_df["点名"], "X": input_df["X"], "Y": input_df["Y"], "標高H": input_df["H"], "緯度": lats, "経度": lons, "ジオイド高": ghs, "楕円体高": input_df["H"].values + np.array(ghs) + offset_val, "適用モデル": use_geoid})
    st.rerun()

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    p_n, p_x, p_y, p_h = c1.text_input("点名", "Point_1"), c2.number_input("X座標", value=0.0, format="%.4f"), c3.number_input("Y座標", value=0.0, format="%.4f"), c4.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行 (1点)", type="primary"): run_calc(pd.DataFrame([{"点名": p_n, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up = st.file_uploader("CSV/SIMAアップロード", type=["csv", "sim"])
    if up and st.button("一括計算開始 🚀", type="primary"):
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

with tab3: st.markdown("### 📖 操作ガイド\n1. 設定を確認し計算を実行します。")

if "result" in st.session_state:
    st.divider()
    res_disp = st.session_state.result.copy()
    if latlon_format == "60進法 (DMS)":
        res_disp["緯度"], res_disp["経度"] = res_disp["緯度"].map(decimal_to_dms), res_disp["経度"].map(decimal_to_dms)
    else:
        # 【修正箇所】画面表示も一貫性を持たせるため .9f に変更
        res_disp["緯度"], res_disp["経度"] = res_disp["緯度"].map(lambda x: f"{x:.9f}"), res_disp["経度"].map(lambda x: f"{x:.9f}")
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]: res_disp[c] = res_disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(res_disp, use_container_width=True)
    c_lat, c_lon, zoom = st.session_state.result["緯度"].mean(), st.session_state.result["経度"].mean(), 18
else:
    c_lat, c_lon, zoom = 37.7616, 140.4760, 14

# --- 5. マップ表示 ---
st.subheader("🗺 マッププレビュー")
t_url = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg" if map_type=="航空写真" else "OpenStreetMap"
m = folium.Map(location=[c_lat, c_lon], zoom_start=zoom, tiles=t_url, attr="GSI/OSM")

if st.session_state.saved_polygons:
    folium.GeoJson({"type": "FeatureCollection", "features": st.session_state.saved_polygons}).add_to(m)

if "result" in st.session_state:
    for _, r in st.session_state.result.iterrows():
        folium.Marker(
            [r["緯度"], r["経度"]], 
            popup=str(r["点名"]),
            tooltip=folium.Tooltip(str(r["点名"]), permanent=True) 
        ).add_to(m)

Draw(export=False, draw_options={"circle": False, "rectangle": True}).add_to(m)
output = st_folium(m, width=1200, height=600, key="folium_map")

# マップ更新処理
last_active = output.get("last_active_drawing")
if last_active:
    g_type = last_active.get("geometry", {}).get("type", "")
    if g_type in ("Polygon", "LineString"):
        coords = last_active.get("geometry", {}).get("coordinates")
        if coords not in [f.get("geometry", {}).get("coordinates") for f in st.session_state.saved_polygons]:
            st.session_state.saved_polygons.append(last_active)
            st.rerun()
    elif g_type == "Point":
        coords = last_active.get("geometry", {}).get("coordinates", [])
        if len(coords) >= 2:
            p_key = (coords[1], coords[0])
            if p_key not in st.session_state.map_markers:
                st.session_state.map_markers.append(p_key)
                st.rerun()
