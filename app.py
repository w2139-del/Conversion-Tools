import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import simplekml
import os

# --- 1. ページ設定 ---
st.set_page_config(page_title="座標変換ツール", layout="wide")
st.title("高精度 座標変換ツール（ジオイドデータ内蔵版）")

# --- 2. ジオイドデータ読み込み関数 ---
@st.cache_resource
def load_geoid_data():
    """圧縮されたジオイドデータを読み込む"""
    data = {}
    if os.path.exists('geoid2024.npz'):
        data['2024'] = np.load('geoid2024.npz')['grid']
    if os.path.exists('geoid2011.npz'):
        loader = np.load('geoid2011.npz')
        data['2011'] = loader['grid']
        data['2011_h'] = loader['header'] # [lat_min, lon_min, d_lat, d_lon, rows, cols]
    return data

geoid_db = load_geoid_data()

def get_geoid_height(lat, lon, model_name):
    """内蔵データから線形補間でジオイド高を算出"""
    if not geoid_db:
        return 0.0
    
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get('2024')
            if g is None: return 0.0
            # 2024年版設定: 15-50N, 120-160E, 1分x1.5分間隔, 2101x1601 (N-to-S)
            r = (50.0 - lat) / (1/60)
            c = (lon - 120.0) / (1.5/60)
        else:
            g = geoid_db.get('2011')
            h = geoid_db.get('2011_h')
            if g is None or h is None: return 0.0
            # 2011年版設定: S-to-N
            r = (lat - h[0]) / h[2]
            c = (lon - h[1]) / h[3]

        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        
        # 4点補間計算
        v00, v01, v10, v11 = g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]
        if any(v > 900 for v in [v00, v01, v10, v11]): return 0.0
        
        dr, dc = r - r0, c - c0
        return (1-dr)*(1-dc)*v00 + (1-dr)*dc*v01 + dr*(1-dc)*v10 + dr*dc*v11
    except:
        return 0.0

# --- 3. SIMA解析関数 ---
def parse_sima(file):
    points = []
    content = file.read().decode('shift-jis', errors='replace')
    for line in content.splitlines():
        p = line.split(',')
        if len(p) >= 6:
            if p[0] == 'A01': # 座標データ
                points.append({'点名': p[2], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
            elif p[0] in ['C00', 'C01']: # 測設データ
                points.append({'点名': p[1], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
    return pd.DataFrame(points)

# --- 4. メイン画面（サイドバー） ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("使用するジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])
offset_val = 1.803 if st.sidebar.checkbox("アンテナ高(1.803m)を加算", value=True) else 0.0

# データ欠損警告
if use_geoid == "ジオイド2024" and '2024' not in geoid_db:
    st.sidebar.error("❌ geoid2024.npz が見つかりません")
if use_geoid == "日本のジオイド2011" and '2011' not in geoid_db:
    st.sidebar.error("❌ geoid2011.npz が見つかりません")

# 座標変換準備
epsg = 6668 + zone
transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# --- 5. メイン画面（アップロード・計算） ---
uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])

if uploaded_file:
    if st.button("🚀 変換計算を開始"):
        try:
            if uploaded_file.name.lower().endswith('.sim'):
                df = parse_sima(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, encoding='shift-jis')
                # 汎用的な列名対応
                df = df.rename(columns={df.columns[0]: '点名', 'X': 'X', 'Y': 'Y', 'H': 'H'})
            
            # 緯度経度変換
            lons, lats = transformer.transform(df['Y'].values, df['X'].values)
            
            # ジオイド高計算
            ghs = [get_geoid_height(la, lo, use_geoid) if use_geoid != "使用しない" else 0.0 for la, lo in zip(lats, lons)]
            
            # 結果まとめ
            res = pd.DataFrame({
                "点名": df['点名'],
                "緯度": lats,
                "経度": lons,
                "適用モデル": use_geoid,
                "ジオイド高": ghs,
                "楕円体高": df['H'].values + ghs + offset_val
            })
            st.session_state.result = res
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# --- 6. 結果表示とダウンロード ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.success(f"✅ 計算完了！ ({use_geoid})")
    
    # 表示用フォーマット
    disp = res.copy()
    disp['緯度'] = disp['緯度'].map(lambda x: f"{x:.8f}")
    disp['経度'] = disp['経度'].map(lambda x: f"{x:.8f}")
    disp['ジオイド高'] = disp['ジオイド高'].map(lambda x: f"{x:.4f}")
    disp['楕円体高'] = disp['楕円体高'].map(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True)
    
    col1, col2, _ = st.columns([2, 2, 6])
    with col1:
        st.download_button("📊 CSV保存", disp.to_csv(index=False).encode('utf-8-sig'), "result.csv", "text/csv")
    with col2:
        kml = simplekml.Kml()
        for _, r in res.iterrows():
            kml.newpoint(name=str(r['点名']), coords=[(r['経度'], r['緯度'], r['楕円体高'])])
        st.download_button("🌍 KML保存", kml.kml(), "result.kml", "application/vnd.google-earth.kml+xml")

    # 地図
    avg_lat, avg_lon = res['緯度'].mean(), res['経度'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18)
    for _, r in res.iterrows():
        folium.Marker([r['緯度'], r['経度']], tooltip=str(r['点名'])).add_to(m)
    st_folium(m, width=1000, height=500)
