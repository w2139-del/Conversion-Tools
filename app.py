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
# 📍 デフォルト位置設定（福島県福島市）
# ==========================================
DEFAULT_LAT = 37.754483
DEFAULT_LON = 140.473454

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="緯度経度変換ツール", layout="wide")
st.title("緯度経度変換ツール")

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    """10進法を60進法(DMS)文字列に変換"""
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
    """ジオイド高を取得（タイポ修正済み）"""
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
        
        dr, dc = r - r0, c - c0  # ← ここを修正しました (dc を c0 に)
        return round((1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3], 4)
    except:
        return 0.0

def build_kml(res_data, kml_export_type, drawn_data):
    """KMLファイルを構築"""
    placemarks = []
    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            pname = str(r["点名"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            placemarks.append(f"""    <Placemark>
      <name>{pname}</name>
      <Point><coordinates>{r['経度']},{r['緯度']},{r['楕円体高']}</coordinates></Point>
    </Placemark>""")

    if ("図形" in kml_export_type or "ポリゴン" in kml_export_type) and drawn_data:
        features = drawn_data.get("all_drawings", [])
        for i, feat in enumerate(features):
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                placemarks.append(f"    <Placemark><name>Poly_{i}</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{coord_str}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
            elif geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                placemarks.append(f"    <Placemark><name>Line_{i}</name><LineString><coordinates>{coord_str}</coordinates></LineString></Placemark>")

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

# --- 4. メインコンテンツの計算ロジック準備 ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
transformer_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)

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

# --- 5. タブ表示 (入力セクション) ---
tab1, tab2, tab3 = st.tabs(["📝 1点手入力変換", "📂 ファイル一括変換", "📖 操作マニュアル"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    p_name = c1.text_input("点名", "Point_1")
    p_x = c2.number_input("X座標", value=0.0, format="%.4f")
    p_y = c3.number_input("Y座標", value=0.0, format="%.4f")
    p_h = c4.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行 (1点)", type="primary"):
        run_calculation_process(pd.DataFrame([{"点名": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up_file = st.file_uploader("CSV/SIMAアップロード", type=["csv", "sim"])
    if up_file and st.button("一括計算開始 🚀"):
        try:
            if up_file.name.lower().endswith(".sim"):
                pts = []
                for line in up_file.read().decode("shift-jis", errors="replace").splitlines():
                    p = line.split(",")
                    if len(p) >= 6 and p[0] in ["A01", "C00", "C01"]:
                        pts.append({"点名": p[1] if p[0].startswith("C") else p[2], "X": float(p[3]), "Y": float(p[4]), "H": float(p[5])})
                df_in = pd.DataFrame(pts)
            else:
                up_file.seek(0)
                df_in = pd.read_csv(up_file, encoding="shift-jis")
                df_in.columns = ["点名", "X", "Y", "H"]
            run_calculation_process(df_in)
        except Exception as e: st.error(f"エラー: {e}")

with tab3:
    st.markdown("""### 操作方法
1. 左側のサイドバーで**系番号**（平面直角座標系）を選択します。
2. CSVまたはSIMAファイルをアップロードするか、手動で座標を入力します。
3. 計算後、画面に表と地図が表示されます。
4. サイドバーの「緯度経度の表示形式」を切り替えると、表とCSV保存時の形式が変わります。""")

# --- 6. 結果表示セクション ---
st.divider()
if "result" in st.session_state:
    res = st.session_state.result
    # 表示用の整形
    disp = res.copy()
    if latlon_fmt == "60進法 (DMS)":
        disp["緯度"] = disp["緯度"].apply(decimal_to_dms)
        disp["経度"] = disp["経度"].apply(decimal_to_dms)
    else:
        disp["緯度"] = disp["緯度"].map(lambda x: f"{x:.8f}")
        disp["経度"] = disp["経度"].map(lambda x: f"{x:.8f}")
    
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]:
        disp[c] = disp[c].map(lambda x: f"{x:.4f}")

    st.subheader("✅ 変換結果")
    st.dataframe(disp, use_container_width=True)

    # サイドバーのダウンロードボタンを更新
    st.sidebar.markdown("---")
    st.sidebar.download_button("📊 CSVを保存", disp.to_csv(index=False).encode("utf-8-sig"), "result.csv", "text/csv", use_container_width=True)
    kml_b = build_kml(res, "ポイント", st.session_state.get("drawn_data", {}))
    st.sidebar.download_button("🌍 KMLを保存", kml_b, "data.kml", "application/vnd.google-earth.kml+xml", use_container_width=True)

# --- 7. マップセクション ---
st.subheader("🗺 マッププレビュー")
avg_lat, avg_lon = (res["緯度"].mean(), res["経度"].mean()) if ("result" in st.session_state and not res.empty) else (DEFAULT_LAT, DEFAULT_LON)
tiles = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg" if map_type == "航空写真" else "OpenStreetMap"
attr = "GSI" if map_type == "航空写真" else "OSM"

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=16, tiles=tiles, attr=attr)
if "result" in st.session_state:
    for _, row in res.iterrows():
        folium.Marker([row["緯度"], row["経度"]], popup=row["点名"], tooltip=row["点名"]).add_to(m)

Draw(export=False).add_to(m)
out = st_folium(m, width=None, height=500, use_container_width=True, key="main_map")

if out.get("all_drawings") and out.get("all_drawings") != st.session_state.get("drawn_data"):
    st.session_state.drawn_data = out["all_drawings"]
    st.rerun()
