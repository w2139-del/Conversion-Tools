import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import simplekml
import os
import hashlib

# --- 1. ページ設定 ---
st.set_page_config(page_title="高精度座標変換ツール", layout="wide")
st.title("高精度 座標変換ツール（表示バグ修正版）")

# --- 2. ジオイドデータ読み込み（高速バイナリ版） ---
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
    if not geoid_db: return 0.0
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get('2024')
            r = (50.0 - lat) * 60.0
            c = (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "日本のジオイド2011":
            g = geoid_db.get('2011')
            h = geoid_db.get('2011_h')
            r = (lat - h[0]) / h[2]
            c = (lon - h[1]) / h[3]
        else:
            return 0.0

        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        if r1 >= g.shape[0] or c1 >= g.shape[1] or r0 < 0 or c0 < 0: return 0.0
        v00, v01, v10, v11 = g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]
        if any(val > 900 for val in [v00, v01, v10, v11]): return 0.0
        dr, dc = r - r0, c - c0
        res = (1-dr)*(1-dc)*v00 + (1-dr)*dc*v01 + dr*(1-dc)*v10 + dr*dc*v11
        return round(res, 4)
    except:
        return 0.0

# --- 3. 解析関数 ---
def parse_sima(file):
    points = []
    content = file.read().decode('shift-jis', errors='replace')
    for line in content.splitlines():
        p = line.split(',')
        if len(p) >= 6:
            if p[0] == 'A01':
                points.append({'点名': p[2], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
            elif p[0] in ['C00', 'C01']:
                points.append({'点名': p[1], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
    return pd.DataFrame(points)

# --- 4. サイドバー設定 ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("使用するジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])
offset_val = 1.803 if st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True) else 0.0

st.sidebar.markdown("---")
st.sidebar.header("地図表示設定")
map_type = st.sidebar.radio("背景地図の選択", ["航空写真", "標準地図"])

# 座標変換エンジン
epsg = 6668 + zone
transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# --- 5. メイン処理 ---
uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])

if uploaded_file:
    if st.button("🚀 変換計算を開始"):
        try:
            if uploaded_file.name.lower().endswith('.sim'):
                df = parse_sima(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, encoding='shift-jis')
                df = df.rename(columns={df.columns[0]: '点名'})
            
            lons, lats = transformer.transform(df['Y'].values, df['X'].values)
            ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
            
            res = pd.DataFrame({
                "点名": df['点名'], "X座標": df['X'], "Y座標": df['Y'], "標高H": df['H'],
                "緯度": lats, "経度": lons, "適用モデル": use_geoid,
                "ジオイド高": ghs, "楕円体高": df['H'].values + ghs + offset_val
            })
            # 結果をセッションに保存
            st.session_state.result = res
            # 計算回数をカウント（地図の強制更新キー用）
            if 'calc_count' not in st.session_state:
                st.session_state.calc_count = 0
            st.session_state.calc_count += 1
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# --- 6. 結果表示・保存 ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.success(f"✅ 計算完了：【{res['適用モデル'].iloc[0]}】を使用して計算しました")
    
    disp = res.copy()
    for c in ['緯度', '経度']: disp[c] = disp[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高']: disp[c] = disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True)
    
    col1, col2, _ = st.columns([2, 2, 6])
    m_name = res['適用モデル'].iloc[0]
    
    with col1:
        st.download_button(f"📊 CSVとして保存", disp.to_csv(index=False).encode('utf-8-sig'), f"変換結果_{m_name}.csv", "text/csv")
    with col2:
        kml = simplekml.Kml()
        for _, r in res.iterrows():
            kml.newpoint(name=str(r['点名']), coords=[(r['経度'], r['緯度'], r['楕円体高'])])
        st.download_button(f"🌍 KMLとして保存", kml.kml(), f"変換結果_{m_name}.kml", "application/vnd.google-earth.kml+xml")

    # --- 7. 地図表示（バグ修正済み） ---
    st.subheader("🗺 マッププレビュー")
    avg_lat, avg_lon = res['緯度'].mean(), res['経度'].mean()
    
    if map_type == "航空写真":
        tiles = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg'
        attr = "地理院タイル (航空写真)"
    else:
        tiles = "OpenStreetMap"
        attr = "OSM"

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=19, tiles=tiles, attr=attr)
    
    for _, r in res.iterrows():
        folium.Marker([r['緯度'], r['経度']], tooltip=f"{r['点名']}").add_to(m)
        folium.map.Marker(
            [r['緯度'], r['経度']],
            icon=folium.DivIcon(
                icon_size=(150, 36), icon_anchor=(7, 20),
                html=f'''<div style="font-size: 12pt; color: red; font-weight: bold; 
                        text-shadow: 2px 2px 2px #fff, -2px -2px 2px #fff, 2px -2px 2px #fff, -2px -2px 2px #fff; 
                        white-space: nowrap;">{r["点名"]}</div>'''
            )
        ).add_to(m)
    
    # 【重要】keyに計算回数を含めることで、データ更新時に地図を強制リフレッシュさせる
    st_folium(m, width=1200, height=600, key=f"map_calc_{st.session_state.get('calc_count', 0)}")
