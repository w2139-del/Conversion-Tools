import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import simplekml
import os
import time

# --- 1. ページ設定 ---
st.set_page_config(page_title="高精度座標変換ツール", layout="wide")
st.title("高精度 座標変換ツール")

# --- 2. ジオイド計算エンジン ---
@st.cache_resource
def load_geoid_data():
    data = {}
    if os.path.exists('geoid2024.npz'):
        data['2024'] = np.load('geoid2024.npz')['grid']
    if os.path.exists('geoid2011.npz'):
        loader = np.load('geoid2011.npz')
        data['2011'] = loader['grid']
        data['2011_h'] = loader['header']
    return data

geoid_db = load_geoid_data()

def get_geoid_height(lat, lon, model_name):
    if model_name == "使用しない" or not geoid_db:
        return 0.0
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get('2024')
            r, c = (50.0 - lat) * 60.0, (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "日本のジオイド2011":
            g, h = geoid_db.get('2011'), geoid_db.get('2011_h')
            r, c = (lat - h[0]) / h[2], (lon - h[1]) / h[3]
        else: return 0.0

        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        if r1 >= g.shape[0] or c1 >= g.shape[1] or r0 < 0 or c0 < 0: return 0.0
        v = [g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]]
        if any(x > 900 for x in v): return 0.0
        dr, dc = r - r0, c - c0
        return round((1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3], 4)
    except: return 0.0

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("使用するジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0

st.sidebar.markdown("---")
st.sidebar.header("🗺 地図表示")
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

# --- 【新機能】サイドバーへのダウンロードボタン配置 ---
if 'result' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("💾 成果品エクスポート")
    
    # ダウンロード用データの準備
    res = st.session_state.result
    disp_csv = res.copy()
    for c in ['緯度', '経度']: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高', 'X', 'Y', '標高H']: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")
    
    # CSVボタン
    st.sidebar.download_button(
        label="📊 CSVファイルを保存",
        data=disp_csv.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"conversion_result_{int(time.time())}.csv",
        mime='text/csv',
        use_container_width=True
    )
    
    # KMLボタン
    kml = simplekml.Kml()
    for _, r in res.iterrows():
        kml.newpoint(name=str(r['点名']), coords=[(r['経度'], r['緯度'], r['楕円体高'])])
    
    st.sidebar.download_button(
        label="🌍 KMLファイルを保存",
        data=kml.kml(),
        file_name=f"spatial_data_{int(time.time())}.kml",
        mime='application/vnd.google-earth.kml+xml',
        use_container_width=True
    )

# 座標変換準備
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)

# --- 4. メインコンテンツ ---
tab1, tab2 = st.tabs(["📝 1点手入力", "📂 ファイル一括変換"])

def run_calc(input_df):
    if input_df.empty
