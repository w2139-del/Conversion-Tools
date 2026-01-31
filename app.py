import streamlit as st
import pandas as pd
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import requests
import simplekml

# --- 設定 ---
st.set_page_config(page_title="座標変換ツール", layout="wide")
st.title("座標変換ツール（KML/KMZ出力対応）")

if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# --- 共通関数 ---
def get_geoid_height(lat, lon, ver="2024"):
    """国土地理院APIからジオイド高を取得（パラメータ名を修正）"""
    v = "2024" if "2024" in ver else "2011"
    try:
        # パラメータ名を latitude, longitude に修正
        url = f"https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl?outputType=json&latitude={lat}&longitude={lon}&gsigen={v}"
        response = requests.get(url, timeout=10)
        res = response.json()
        
        # APIのレスポンス構造に合わせて値を取得
        # 通常は OutputData 内にあるが、直接直下にあるケースも考慮
        out_data = res.get('OutputData', {})
        gh = out_data.get('geoidHeight') or res.get('geoidHeight')
        
        if gh is not None:
            return float(gh)
        else:
            return 0.0
    except Exception as e:
        st.error(f"ジオイド取得エラー: {e}")
        return 0.0

def parse_sima(uploaded_file):
    points = []
    content = uploaded_file.read().decode('shift-jis', errors='replace')
    for line in content.splitlines():
        parts = line.split(',')
        if len(parts) >= 6 and parts[0] in ['C00', 'C01']:
            try:
                points.append({
                    'Name': parts[1], 
                    'X': float(parts[3]), 
                    'Y': float(parts[4]), 
                    'H': float(parts[5]) if parts[5] else 0.0
                })
            except: continue
    return pd.DataFrame(points)

# --- サイドバー設定 ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8) # デフォルト9系
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])
add_offset = st.sidebar.checkbox("アンテナ高(1.803m)を加算", value=True)
offset_val = 1.803 if add_offset else 0.0

# 変換エンジンの準備
epsg = 6668 + zone
# always_xy=True: transform(Y, X) -> (Lon, Lat)
to_latlon = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# --- メイン UI ---
tab1, tab2 = st.tabs(["📍 1点直接入力", "📂 ファイル一括変換 (CSV/SIMA)"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: p_name = st.text_input("点名", value="Point01")
    with col2: p_x = st.number_input("X(北方向)", value=0.0, format="%.4f", key="p_x")
    with col3: p_y = st.number_input("Y(東方向)", value=0.0, format="%.4f", key="p_y")
    with col4: p_h = st.number_input("標高(H)", value=0.0, format="%.4f", key="p_h")
    
    if st.button("1点変換を実行"):
        lon, lat = to_latlon.transform(p_y, p_x)
        gh = get_geoid_height(lat, lon, use_geoid) if "ジオイド" in use_geoid else 0.0
        
        st.session_state.result_df = pd.DataFrame([{
            "点名": p_name,
            "変換前_X": p_x, "変換前_Y": p_y, "変換前_標高(H)": p_h,
            "変換後_緯度": lat, "変換後_経度": lon,
            "ジオイド高": gh, "変換後_楕円体高": p_h + gh + offset_val
        }])

with tab2:
    uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])
    if uploaded_file:
        if st.button("一括変換を開始"):
            df_in = parse_sima(uploaded_file) if uploaded_file.name.endswith('.sim') else pd.read_csv(uploaded_file)
            lons, lats = to_latlon.transform(df_in['Y'].values, df_in['X'].values)
            
            with st.spinner("ジオイド高を取得中..."):
                ghs = [get_geoid_height(la, lo, use_geoid) if "ジオイド" in use_geoid else 0.0 for la, lo in zip(lats, lons)]
            
            df_res = pd.DataFrame({
                "点名": df_in['Name'],
                "変換前_X": df_in['X'], "変換前_Y": df_in['Y'], "変換前_標高(H)": df_in['H'],
                "変換後_緯度": lats, "変換後_経度": lons,
                "ジオイド高": ghs,
                "変換後_楕円体高": df_in['H'] + ghs + offset_val
            })
            st.session_state.result_df = df_res

# --- 結果表示と保存 ---
if st.session_state.result_df is not None:
    # 表示用にフォーマットしたデータフレームを作成
    display_df = st.session_state.result_df.copy()
    
    # 桁数の指定適用
    display_df['変換後_緯度'] = display_df['変換後_緯度'].map(lambda x: f"{x:.8f}")
    display_df['変換後_経度'] = display_df['変換後_経度'].map(lambda x: f"{x:.8f}")
    display_df['ジオイド高'] = display_df['ジオイド高'].map(lambda x: f"{x:.4f}")
    display_df['変換後_楕円体高'] = display_df['変換後_楕円体高'].map(lambda x: f"{x:.4f}")
    
    st.write("---")
    st.subheader("✅ 変換結果（比較表）")
    st.dataframe(display_df)
    
    col_dl1, col_dl2, _ = st.columns([2, 2, 6])
    with col_dl1:
        csv_data = display_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 CSV保存", csv_data, "result.csv", "text/csv")
    
    with col_dl2:
        kml = simplekml.Kml()
        for _, r in display_df.iterrows():
            pnt = kml.newpoint(name=r['点名'], coords=[(float(r['変換後_経度']), float(r['変換後_緯度']), float(r['変換後_楕円体高']))])
            pnt.altitudemode = simplekml.AltitudeMode.absolute
        st.download_button("🌍 KML保存", kml.kml(), "result.kml", "application/vnd.google-earth.kml+xml")

    # マップ表示
    st.subheader("🗺 マッププレビュー")
    try:
        center_lat = float(display_df['変換後_緯度'].iloc[0])
        center_lon = float(display_df['変換後_経度'].iloc[0])
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
        for _, r in display_df.iterrows():
            folium.Marker([float(r['変換後_緯度']), float(r['変換後_経度'])], tooltip=r['点名']).add_to(m)
        st_folium(m, width=1000, height=500, key="survey_map_final")
    except:
        st.warning("マップを表示できる座標ではありません。")