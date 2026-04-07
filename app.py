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
# 📍 デフォルト位置設定（福島県福島市北五老内町５−２５）
# ==========================================
DEFAULT_LAT = 37.754483
DEFAULT_LON = 140.473454
# ==========================================

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="緯度経度変換ツール", layout="wide")
st.title("緯度経度変換ツール")

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:07.4f}\""

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
        if any(x > 900 for x in v):
            return 0.0
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
    except (ValueError, IndexError):
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=0)
    if df.shape[1] < 4:
        raise ValueError("CSVは最低4列（点名, X, Y, H）必要です。")
    df = df.iloc[:, :4]
    df.columns = ["点名", "X", "Y", "H"]
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    df = df.dropna(subset=["X", "Y", "H"])
    return df

def build_kml(res_data, kml_export_type, drawn_data):
    placemarks = []
    # ポイント出力
    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            pname = str(r["点名"]).replace("&", "&").replace("<", "<").replace(">", ">")
            placemark = (
                "    <Placemark>\n"
                "      <name>" + pname + "</name>\n"
                "      <Style><IconStyle><Icon>\n"
                "        <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>\n"
                "      </Icon></IconStyle></Style>\n"
                "      <Point>\n"
                "        <coordinates>" + str(r["経度"]) + "," + str(r["緯度"]) + "," + str(r["楕円体高"]) + "</coordinates>\n"
                "      </Point>\n"
                "    </Placemark>"
            )
            placemarks.append(placemark)

    # 図形出力
    if ("図形" in kml_export_type or "ポリゴン" in kml_export_type) and drawn_data:
        features = drawn_data.get("all_drawings", [])
        poly_count = 1
        line_count = 1
        for feat in features:
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                if len(coords) < 3: continue
                if coords[0] != coords[-1]: coords = coords + [coords[0]]
                coord_str = " ".join(str(c[0]) + "," + str(c[1]) + ",0" for c in coords)
                placemark = (
                    "    <Placemark>\n"
                    "      <name>Polygon_" + str(poly_count) + "</name>\n"
                    "      <Style>\n"
                    "        <LineStyle><color>ffff0000</color><width>2</width></LineStyle>\n"
                    "        <PolyStyle><color>6400ffff</color><fill>1</fill><outline>1</outline></PolyStyle>\n"
                    "      </Style>\n"
                    "      <Polygon>\n"
                    "        <outerBoundaryIs><LinearRing>\n"
                    "          <coordinates>" + coord_str + "</coordinates>\n"
                    "        </LinearRing></outerBoundaryIs>\n"
                    "      </Polygon>\n"
                    "    </Placemark>"
                )
                placemarks.append(placemark)
                poly_count += 1
            elif geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) < 2: continue
                coord_str = " ".join(str(c[0]) + "," + str(c[1]) + ",0" for c in coords)
                placemark = (
                    "    <Placemark>\n"
                    "      <name>Line_" + str(line_count) + "</name>\n"
                    "      <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>\n"
                    "      <LineString>\n"
                    "        <coordinates>" + coord_str + "</coordinates>\n"
                    "      </LineString>\n"
                    "    </Placemark>"
                )
                placemarks.append(placemark)
                line_count += 1

    body = "\n".join(placemarks)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        '    <name>spatial_data</name>\n'
        + body + "\n"
        '  </Document>\n'
        '</kml>'
    ).encode("utf-8")

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("💾 成果品保存")

# --- CSV保存セクション ---
if "result" in st.session_state:
    res_data = st.session_state.result
    latlon_fmt = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"], index=0)
    disp_csv = res_data.copy()
    if latlon_fmt == "60進法 (DMS)":
        disp_csv["緯度"] = disp_csv["緯度"].map(decimal_to_dms); disp_csv["経度"] = disp_csv["経度"].map(decimal_to_dms)
    else:
        for c in ["緯度", "経度"]: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")

    st.sidebar.download_button(
        label="📊 CSVを保存",
        data=disp_csv.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"result_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("計算を実行するとCSV保存が可能です。")

st.sidebar.markdown("---")

# --- KML保存セクション ---
st.sidebar.subheader("🌍 KML保存設定")
kml_type = st.sidebar.selectbox("KML出力対象", ["ポイントと図形の両方", "ポイントのみ", "図形のみ"], index=0)

# KML用データの準備
current_drawn = st.session_state.get("drawn_data", {})
res_for_kml = st.session_state.get("result")
kml_bytes = build_kml(res_for_kml, kml_type, current_drawn)

# 図形の件数表示
all_draws = current_drawn.get("all_drawings") or []
non_p_count = sum(1 for f in all_draws if f.get("geometry", {}).get("type") != "Point")
if non_p_count > 0:
    st.sidebar.caption(f"✏️ 描画済み図形: {non_p_count} 件")

st.sidebar.download_button(
    label="🌍 KMLを保存",
    data=kml_bytes,
    file_name=f"spatial_data_{int(time.time())}.kml",
    mime="application/vnd.google-earth.kml+xml",
    use_container_width=True
)

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
transformer_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)

tab1, tab2, tab3 = st.tabs(["📝 1点手入力変換", "📂 ファイル一括変換", "📖 操作マニュアル"])

def run_calculation_process(input_df):
    if input_df.empty: return
    lons, lats = transformer.transform(input_df["Y"].values, input_df["X"].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    st.session_state.result = pd.DataFrame({
        "点名": input_df["点名"], "X": input_df["X"], "Y": input_df["Y"], "標高H": input_df["H"],
        "緯度": lats, "経度": lons, "ジオイド高": ghs,
        "楕円体高": input_df["H"].values + np.array(ghs) + offset_val, "適用モデル": use_geoid
    })
    if "map_key" not in st.session_state: st.session_state.map_key = "folium_map_fixed"
    if "registered_marker_coords" not in st.session_state: st.session_state.registered_marker_coords = []
    st.rerun()

with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: p_name = st.text_input("点名", "Point_1")
    with col_b: p_x = st.number_input("X座標", value=0.0, format="%.4f")
    with col_c: p_y = st.number_input("Y座標", value=0.0, format="%.4f")
    with col_d: p_h = st.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行 (1点)", type="primary"): run_calculation_process(pd.DataFrame([{"点名": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up_file = st.file_uploader("CSV/SIMAアップロード", type=["csv", "sim"])
    if up_file and st.button("一括計算開始 🚀", type="primary"):
        try:
            if up_file.name.lower().endswith(".sim"):
                pts = []
                content_sim = up_file.read().decode("shift-jis", errors="replace")
                for line in content_sim.splitlines():
                    p = line.split(",")
                    if len(p) >= 6 and p[0] in ["A01", "C00", "C01"]:
                        pts.append({"点名": p[1] if p[0].startswith("C") else p[2], "X": float(p[3]), "Y": float(p[4]), "H": float(p[5])})
                df_input = pd.DataFrame(pts)
            else: df_input = read_csv_auto(up_file)
            run_calculation_process(df_input)
        except Exception as e: st.error(f"エラー: {e}")

# --- 5. 結果表示 & マップ描画（常時表示セクション） ---
st.divider()

if "result" in st.session_state:
    res = st.session_state.result
    st.dataframe(res, use_container_width=True)

st.subheader("🗺 マッププレビュー & 描画ツール")

# 中心座標の決定
if "result" in st.session_state and not st.session_state.result.empty:
    avg_lat = st.session_state.result["緯度"].mean()
    avg_lon = st.session_state.result["経度"].mean()
    zoom_start = 18
else:
    avg_lat, avg_lon = DEFAULT_LAT, DEFAULT_LON
    zoom_start = 17

# タイル設定
if map_type == "航空写真":
    tiles_url, attr = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg", "GSI"
else:
    tiles_url, attr = "OpenStreetMap", "OpenStreetMap"

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_start, tiles=tiles_url, attr=attr)

# 登録済みマーカー（計算結果）の描画
if "result" in st.session_state:
    fg = folium.FeatureGroup(name="Markers")
    for _, row in st.session_state.result.iterrows():
        folium.Marker([row["緯度"], row["経度"]], popup=str(row["点名"]), tooltip=str(row["点名"])).add_to(fg)
        folium.Marker([row["緯度"], row["経度"]], icon=folium.DivIcon(icon_size=(150, 30), icon_anchor=(7, 25),
            html='<div style="font-size:11pt;color:red;font-weight:bold;text-shadow:2px 2px 2px #fff;">' + str(row["点名"]) + '</div>')).add_to(fg)
    fg.add_to(m)

# 描画コントロール
Draw(export=False, draw_options={"marker":True, "polyline":True, "rectangle":True, "polygon":True, "circle":False, "circlemarker":False},
     edit_options={"edit": {}, "remove": {}}).add_to(m)

# マップ表示
output = st_folium(m, width=1200, height=600, key="main_folium_map")

# 描画データ管理（サイドバーのKMLボタンに反映される）
if output.get("all_drawings") is not None:
    if st.session_state.get("drawn_data") != output:
        st.session_state.drawn_data = output
        st.rerun()

# マーカー追加UI（マップ直下）
placed_markers = [f for f in (current_drawn.get("all_drawings") or []) if f.get("geometry", {}).get("type") == "Point"]
if placed_markers:
    st.divider()
    if "registered_marker_coords" not in st.session_state: st.session_state.registered_marker_coords = []
    registered = st.session_state.registered_marker_coords
    new_m = []
    for feat in placed_markers:
        coords = feat["geometry"]["coordinates"]
        k = (round(coords[1], 8), round(coords[0], 8))
        if k not in registered: new_m.append((coords[1], coords[0]))

    if new_m:
        st.info(f"📍 未登録のマーカーが {len(new_m)} 件あります。")
        if st.button("➕ マーカーを変換リストに追加", type="primary"):
            if "result" not in st.session_state:
                st.session_state.result = pd.DataFrame(columns=["点名", "X", "Y", "標高H", "緯度", "経度", "ジオイド高", "楕円体高", "適用モデル"])
            
            existing_click_nums = []
            for name in st.session_state.result["点名"].astype(str):
                if name.startswith("Click_"):
                    try: existing_click_nums.append(int(name.split("_")[1]))
                    except: pass
            next_num = max(existing_click_nums, default=0) + 1

            new_rows = []
            new_reg = []
            for m_lat, m_lon in new_m:
                y_v, x_v = transformer_inv.transform(m_lon, m_lat)
                gh = get_geoid_height(m_lat, m_lon, use_geoid)
                new_rows.append({"点名": f"Click_{next_num}", "X": round(x_v, 4), "Y": round(y_v, 4), "標高H": 0.0,
                                 "緯度": m_lat, "経度": m_lon, "ジオイド高": gh,
                                 "楕円体高": round(0.0 + gh + offset_val, 4), "適用モデル": use_geoid})
                new_reg.append((round(m_lat, 8), round(m_lon, 8)))
                next_num += 1

            st.session_state.result = pd.concat([st.session_state.result, pd.DataFrame(new_rows)], ignore_index=True)
            st.session_state.registered_marker_coords.extend(new_reg)
            st.rerun()
