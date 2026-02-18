import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import simplekml
import os
import requests
import time

# --- ページ設定 ---
st.set_page_config(page_title="高精度座標変換ツール", layout="wide")
st.title("高精度 座標変換ツール")

# --- ジオイド計算エンジン ---
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

def get_geoid_height_internal(lat, lon, model_name):
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
        else: return 0.0
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        v = g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]
        dr, dc = r - r0, c - c0
        res = (1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3]
        return round(res, 4)
    except: return 0.0

def get_geoid_height_api(lat, lon):
    """地理院APIを使用した計算（エラー詳細取得版）"""
    url = "https://vldb.gsi.go.jp/point/geoid/geoid_api.php"
    # 緯度経度を小数点第8位まで固定（APIの推奨形式）
    params = {
        "lat": f"{lat:.8f}",
        "lon": f"{lon:.8f}",
        "output": "json"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "geoid" in data:
                return float(data["geoid"])
            else:
                # APIは成功したがジオイド高がデータに含まれない場合（海域など）
                return 0.0
    except Exception as e:
        # 通信エラーが発生した場合はNoneを返し、後で警告を出す
        return None
    return None

# --- 解析・UI設定 ---
def parse_sima(file):
    points = []
    content = file.read().decode('shift-jis', errors='replace')
    for line in content.splitlines():
        p = line.split(',')
        if len(p) >= 6:
            if p[0] == 'A01': points.append({'点名': p[2], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
            elif p[0] in ['C00', 'C01']: points.append({'点名': p[1], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
    return pd.DataFrame(points)

# サイドバー
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("使用するジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])

calc_method = "内蔵データ"
if use_geoid == "日本のジオイド2011":
    calc_method = st.sidebar.radio("計算方法の選択", ["地理院API ", "内蔵データ (高速・オフライン)"])

offset_val = 1.803 if st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True) else 0.0
st.sidebar.markdown("---")
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

# 変換準備
epsg = 6668 + zone
transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# メイン処理
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
            ghs = []
            api_error_flag = False
            
            if calc_method == "地理院API ":
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, (la, lo) in enumerate(zip(lats, lons)):
                    status_text.text(f"API計算中... ({i+1}/{len(lats)} 点)")
                    val = get_geoid_height_api(la, lo)
                    if val is None:
                        api_error_flag = True
                        ghs.append(0.0)
                    else:
                        ghs.append(val)
                    progress_bar.progress((i + 1) / len(lats))
                    time.sleep(0.1) 
                status_text.empty()
                if api_error_flag:
                    st.warning("⚠️ 一部の点でAPI通信に失敗しました。ネット接続を確認してください。")
            else:
                ghs = [get_geoid_height_internal(la, lo, use_geoid) for la, lo in zip(lats, lons)]

            res = pd.DataFrame({
                "点名": df['点名'], "X座標": df['X'], "Y座標": df['Y'], "標高H": df['H'],
                "緯度": lats, "経度": lons, "適用モデル": f"{use_geoid}({calc_method})",
                "ジオイド高": ghs, "楕円体高": df['H'].values + np.array(ghs) + offset_val
            })
            st.session_state.result = res
            st.session_state.calc_count = st.session_state.get('calc_count', 0) + 1
            
        except Exception as e:
            st.error(f"エラー: {e}")

# 結果表示
if 'result' in st.session_state:
    res = st.session_state.result
    st.success(f"✅ 計算完了")
    disp = res.copy()
    for c in ['緯度', '経度']: disp[c] = disp[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高']: disp[c] = disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True)
    
    # 地図
    avg_lat, avg_lon = res['緯度'].mean(), res['経度'].mean()
    tiles = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg' if map_type == "航空写真" else "OpenStreetMap"
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=19, tiles=tiles, attr="GSI/OSM")
    for _, r in res.iterrows():
        folium.Marker([r['緯度'], r['経度']], tooltip=str(r['点名'])).add_to(m)
        folium.map.Marker(
            [r['緯度'], r['経度']],
            icon=folium.DivIcon(icon_size=(150, 36), icon_anchor=(7, 20),
                html=f'<div style="font-size: 12pt; color: red; font-weight: bold; text-shadow: 2px 2px 2px #fff; white-space: nowrap;">{r["点名"]}</div>')
        ).add_to(m)
    st_folium(m, width=1200, height=600, key=f"map_{st.session_state.calc_count}")
