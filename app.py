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
    """内蔵データの読み込み"""
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
    """バイリニア補間によるジオイド高計算"""
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
        
        # 配列範囲チェック
        if r1 >= g.shape[0] or c1 >= g.shape[1] or r0 < 0 or c0 < 0: return 0.0
        v = [g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]]
        if any(x > 900 for x in v): return 0.0 # 欠測値処理
        
        dr, dc = r - r0, c - c0
        res = (1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3]
        return round(res, 4)
    except: return 0.0

# --- 3. サイドバー設定 ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)

# 修正1: デフォルトを ジオイド2024 (index=1) に設定
use_geoid = st.sidebar.selectbox(
    "使用するジオイドモデル", 
    ["日本のジオイド2011", "ジオイド2024", "使用しない"], 
    index=1
)

# 修正3の補足: ジオイド使用時のみアンテナ高を選択可能にする
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0

st.sidebar.markdown("---")
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)

# --- 4. メインコンテンツ ---
tab1, tab2 = st.tabs(["📝 1点手入力", "📂 ファイル一括変換"])

def run_calc(input_df):
    """共通計算処理"""
    if input_df.empty: return
    lons, lats = transformer.transform(input_df['Y'].values, input_df['X'].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    
    res = pd.DataFrame({
        "点名": input_df['点名'], "X": input_df['X'], "Y": input_df['Y'], "標高H": input_df['H'],
        "緯度": lats, "経度": lons, "ジオイド高": ghs,
        "楕円体高": input_df['H'].values + np.array(ghs) + offset_val,
        "適用モデル": use_geoid
    })
    st.session_state.result = res
    st.session_state.calc_id = time.time()

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1: name = st.text_input("点名", "Point_1")
    with c2: x_val = st.number_input("X座標", value=0.0, format="%.4f")
    with c3: y_val = st.number_input("Y座標", value=0.0, format="%.4f")
    with c4: h_val = st.number_input("標高H", value=0.0, format="%.4f")
    if st.button("この地点を計算"):
        run_calc(pd.DataFrame([{"点名": name, "X": x_val, "Y": y_val, "H": h_val}]))

with tab2:
    uploaded_file = st.file_uploader("CSV/SIMAをアップロード", type=["csv", "sim"])
    if uploaded_file:
        if st.button("一括変換を開始 🚀"):
            try:
                if uploaded_file.name.lower().endswith('.sim'):
                    points = []
                    content = uploaded_file.read().decode('shift-jis', errors='replace')
                    for line in content.splitlines():
                        p = line.split(',')
                        if len(p) >= 6 and p[0] in ['A01', 'C00', 'C01']:
                            points.append({'点名': p[1] if p[0].startswith('C') else p[2], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
                    df_raw = pd.DataFrame(points)
                else:
                    df_raw = pd.read_csv(uploaded_file, encoding='shift-jis')
                    df_raw = df_raw.rename(columns={df_raw.columns[0]: '点名', df_raw.columns[1]: 'X', df_raw.columns[2]: 'Y', df_raw.columns[3]: 'H'})
                run_calc(df_raw)
            except Exception as e: st.error(f"解析エラー: {e}")

# --- 5. 結果表示 ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.divider()
    
    # 修正3の確認用メッセージ
    if use_geoid == "使用しない":
        st.info("ℹ️ 『使用しない』が選択されているため、ジオイド高およびアンテナ高を0として計算しています（標高＝楕円体高）。")

    # テーブル表示
    disp = res.copy()
    for c in ['緯度', '経度']: disp[c] = disp[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高', 'X', 'Y', '標高H']: disp[c] = disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(disp, use_container_width=True)
    
    col1, col2, _ = st.columns([2, 2, 6])
    with col1: st.download_button("📊 CSV保存", disp.to_csv(index=False).encode('utf-8-sig'), "result.csv")
    with col2:
        kml = simplekml.Kml()
        for _, r in res.iterrows(): kml.newpoint(name=str(r['点名']), coords=[(r['経度'], r['緯度'], r['楕円体高'])])
        st.download_button("🌍 KML保存", kml.kml(), "result.kml")

    # 修正2: 地図とピンの表示
    st.subheader("🗺 マッププレビュー")
    # 不正な座標（0,0など）を除外して平均位置を算出
    valid_res = res[(res['緯度'] > 20) & (res['経度'] > 120)]
    if not valid_res.empty:
        avg_lat, avg_lon = valid_res['緯度'].mean(), valid_res['経度'].mean()
        tiles = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg' if map_type == "航空写真" else "OpenStreetMap"
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18, tiles=tiles, attr="GSI/OSM")
        
        for _, r in res.iterrows():
            if 20 < r['緯度'] < 50: # 日本国内の座標のみピンを立てる
                # 通常のピン
                folium.Marker(
                    location=[r['緯度'], r['経度']],
                    popup=f"{r['点名']}<br>楕円体高: {r['楕円体高']:.3f}m",
                    tooltip=str(r['点名'])
                ).add_to(m)
                
                # 点名ラベル（赤い文字）
                folium.map.Marker(
                    [r['緯度'], r['経度']],
                    icon=folium.DivIcon(
                        icon_size=(150, 30),
                        icon_anchor=(7, 25),
                        html=f'<div style="font-size: 11pt; color: red; font-weight: bold; text-shadow: 2px 2px 2px #fff; white-space: nowrap;">{r["点名"]}</div>'
                    )
                ).add_to(m)
        
        st_folium(m, width=1200, height=550, key=f"map_{st.session_state.calc_id}")
    else:
        st.warning("⚠️ 座標が正しくないため地図を表示できません。系番号を確認してください。")

