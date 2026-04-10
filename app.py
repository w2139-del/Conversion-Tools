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

def build_kml(res_data, kml_export_type, non_point_features):
    """Google Earth対応KML生成（修正版）"""
    NM_O = "<name>"
    NM_C = "</name>"

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("  <Document>")
    lines.append("    " + NM_O + "spatial_data" + NM_C)
    lines += [
        '    <Style id="poly_style">',
        '      <LineStyle><color>ffff0000</color><width>2</width></LineStyle>',
        '      <PolyStyle><color>6400ffff</color><fill>1</fill><outline>1</outline></PolyStyle>',
        '    </Style>',
        '    <Style id="line_style">',
        '      <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>',
        '    </Style>',
        '    <Style id="point_style">',
        '      <IconStyle><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle>',
        '    </Style>',
    ]

    # ポイント出力
    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            pname = str(r["点名"]).replace("&", "&").replace("<", "<").replace(">", ">")
            lines += [
                "    <Placemark>",
                "      " + NM_O + pname + NM_C,
                "      <styleUrl>#point_style</styleUrl>",
                "      <Point>",
                "        <coordinates>" + f"{r['経度']},{r['緯度']},{r['楕円体高']}" + "</coordinates>",
                "      </Point>",
                "    </Placemark>",
            ]

    # ポリゴン・ライン出力（修正箇所：判定を「図形」に変更）
    if "図形" in kml_export_type and non_point_features:
        poly_count = 1
        line_count = 1
        for feat in non_point_features:
            geom = feat.get("geometry", {})

            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                if len(coords) < 3:
                    continue
                if coords[0] != coords[-1]:
                    coords = coords + [coords[0]]
                # Google Earth形式の座標文字列作成
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                lines += [
                    "    <Placemark>",
                    "      " + NM_O + f"Polygon_{poly_count}" + NM_C,
                    "      <styleUrl>#poly_style</styleUrl>",
                    "      <Polygon>",
                    "        <tessellate>1</tessellate>",
                    "        <altitudeMode>clampToGround</altitudeMode>",
                    "        <outerBoundaryIs>",
                    "          <LinearRing>",
                    "            <coordinates>" + coord_str + "</coordinates>",
                    "          </LinearRing>",
                    "        </outerBoundaryIs>",
                    "      </Polygon>",
                    "    </Placemark>",
                ]
                poly_count += 1

            elif geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) < 2:
                    continue
                coord_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
                lines += [
                    "    <Placemark>",
                    "      " + NM_O + f"Line_{line_count}" + NM_C,
                    "      <styleUrl>#line_style</styleUrl>",
                    "      <LineString>",
                    "        <tessellate>1</tessellate>",
                    "        <altitudeMode>clampToGround</altitudeMode>",
                    "        <coordinates>" + coord_str + "</coordinates>",
                    "      </LineString>",
                    "    </Placemark>",
                ]
                line_count += 1

    lines.append("  </Document>")
    lines.append("</kml>")
    return "\n".join(lines).encode("utf-8")

# =============================================================================
# session_state 初期化
# =============================================================================
if "saved_polygons" not in st.session_state:
    st.session_state.saved_polygons = []
if "registered_marker_coords" not in st.session_state:
    st.session_state.registered_marker_coords = []

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

st.sidebar.markdown("---")
st.sidebar.header("💾 成果品保存")

if "result" in st.session_state:
    res_data = st.session_state.result
    latlon_format = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"], index=0)

    disp_csv = res_data.copy()
    if latlon_format == "60進法 (DMS)":
        disp_csv["緯度"] = disp_csv["緯度"].map(decimal_to_dms)
        disp_csv["経度"] = disp_csv["経度"].map(decimal_to_dms)
    else:
        for c in ["緯度", "経度"]:
            disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]:
        disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")

    st.sidebar.download_button(
        label="📊 CSVを保存",
        data=disp_csv.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"result_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("計算を実行するとCSV保存ボタンが表示されます。")

# --- KML保存（サイドバー）---
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 KML保存")

_sb_polygons = st.session_state.saved_polygons
_sb_result = st.session_state.get("result", None)

kml_export_type = st.sidebar.selectbox(
    "KML出力対象",
    ["ポイントと図形の両方", "ポイントのみ", "図形のみ"],
    index=0,
    key="kml_type_select"
)

if _sb_polygons:
    st.sidebar.caption(f"✏️ 保存済み図形: {len(_sb_polygons)} 件")
else:
    st.sidebar.caption("図形未描画（マップでポリゴン等を描画してください）")

kml_bytes = build_kml(_sb_result, kml_export_type, _sb_polygons)
st.sidebar.download_button(
    label="🌍 KMLを保存",
    data=kml_bytes,
    file_name=f"spatial_data_{int(time.time())}.kml",
    mime="application/vnd.google-earth.kml+xml",
    use_container_width=True
)

# --- マーカー追加（サイドバー）---
st.sidebar.markdown("---")
st.sidebar.subheader("📍 マーカー追加")
st.sidebar.warning(f"⚠️ 現在の系番号: **第{zone}系**\n\nマーカー追加前に系番号を確認してください。")

_map_markers = st.session_state.get("map_markers", [])
_registered_sb = st.session_state.registered_marker_coords
_new_markers_sb = [
    (lat, lon) for lat, lon in _map_markers
    if (round(lat, 8), round(lon, 8)) not in _registered_sb
]

if _new_markers_sb:
    st.sidebar.info(f"📍 未登録マーカー: {len(_new_markers_sb)} 件")
    if st.sidebar.button("➕ マーカーを変換リストに追加", type="primary"):
        _transformer_inv_sb = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)
        _existing_result = st.session_state.get("result", pd.DataFrame(
            columns=["点名", "X", "Y", "標高H", "緯度", "経度", "ジオイド高", "楕円体高", "適用モデル"]
        ))
        _existing_click_nums = []
        for name in _existing_result["点名"].astype(str):
            if name.startswith("Click_"):
                try: _existing_click_nums.append(int(name.split("_")[1]))
                except ValueError: pass
        _next_num = max(_existing_click_nums, default=0) + 1
        _new_rows = []
        _newly_registered = []
        for m_lat, m_lon in _new_markers_sb:
            y_val, x_val = _transformer_inv_sb.transform(m_lon, m_lat)
            gh = get_geoid_height(m_lat, m_lon, use_geoid)
            _new_rows.append({
                "点名": f"Click_{_next_num}",
                "X": round(x_val, 4), "Y": round(y_val, 4), "標高H": 0.0,
                "緯度": m_lat, "経度": m_lon, "ジオイド高": gh,
                "楕円体高": round(0.0 + gh + offset_val, 4), "適用モデル": use_geoid
            })
            _next_num += 1
            _newly_registered.append((round(m_lat, 8), round(m_lon, 8)))
        st.session_state.result = pd.concat([_existing_result, pd.DataFrame(_new_rows)], ignore_index=True)
        st.session_state.registered_marker_coords.extend(_newly_registered)
        st.rerun()

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)

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
    st.rerun()

with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: p_name = st.text_input("点名", "Point_1")
    with col_b: p_x = st.number_input("X座標", value=0.0, format="%.4f")
    with col_c: p_y = st.number_input("Y座標", value=0.0, format="%.4f")
    with col_d: p_h = st.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行 (1点)", type="primary"):
        run_calculation_process(pd.DataFrame([{"点名": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

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
                        pts.append({"点名": p[1] if
