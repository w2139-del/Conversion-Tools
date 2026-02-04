import streamlit as st
import pandas as pd
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import requests
import simplekml
import io

# --- 1. 基本設定 ---
st.set_page_config(page_title="座標変換ツール", layout="wide")
st.title("座標変換ツール（KML/KMZ出力対応）")

# データ保持用のメモリ
if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# --- 2. 共通関数（ジオイド取得・SIMA解析） ---
def get_geoid_height(lat, lon, ver="2024"):
    """国土地理院APIからジオイド高を取得"""
    v = "2024" if "2024" in ver else "2011"
    try:
        # パラメータは latitude, longitude
        url = f"https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl?outputType=json&latitude={lat}&longitude={lon}&gsigen={v}"
        response = requests.get(url, timeout=10)
        res = response.json()
        
        # APIのレスポンス構造（OutputData内）から取得
        out_data = res.get('OutputData', {})
        gh = out_data.get('geoidHeight') or res.get('geoidHeight')
        
        return float(gh) if gh is not None else 0.0
    except:
        return 0.0

def parse_sima(uploaded_file):
    """SIMAファイルを解析してDataFrameに変換"""
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

# --- 3. サイドバー設定 ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=9) # 山梨・静岡等の10系をデフォルトに設定
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])
add_offset = st.sidebar.checkbox("アンテナ高(1.803m)を加算", value=True)
offset_val = 1.803 if add_offset else 0.0

# 座標変換エンジンの準備 (JGD2011)
epsg = 6668 + zone
# always_xy=True: transform(Y, X) -> (Lon, Lat)
to_latlon = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# --- 4. メイン画面レイアウト ---
tab1, tab2 = st.tabs(["📍 1点直接入力", "📂 ファイル一括変換 (CSV/SIMA)"])

# 【1点入力】
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: p_name = st.text_input("点名", value="Point01")
    with col2: p_x = st.number_input("X(北方向)", value=0.0, format="%.4f")
    with col3: p_y = st.number_input("Y(東方向)", value=0.0, format="%.4f")
    with col4: p_h = st.number_input("標高(H)", value=0.0, format="%.4f")
    
    if st.button("1点変換を実行"):
        lon, lat = to_latlon.transform(p_y, p_x)
        gh = get_geoid_height(lat, lon, use_geoid) if "ジオイド" in use_geoid else 0.0
        
        st.session_state.result_df = pd.DataFrame([{
            "点名": p_name,
            "変換前_X": p_x, "変換前_Y": p_y, "変換前_標高(H)": p_h,
            "変換後_緯度": lat, "変換後_経度": lon,
            "ジオイド高": gh, "変換後_楕円体高": p_h + gh + offset_val
        }])

# 【ファイル一括変換】
with tab2:
    st.info("CSVの場合、ヘッダーなしで [点名, X, Y, H] の順のファイルを想定しています。")
    uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])
    
    if uploaded_file:
        if st.button("一括変換を開始"):
            # SIMAとCSVの読み分け
            if uploaded_file.name.endswith('.sim'):
                df_in = parse_sima(uploaded_file)
            else:
                # ヘッダーなしCSVに対応（1列目:点名, 2列目:X, 3列目:Y, 4列目:H）
                raw_df = pd.read_csv(uploaded_file, header=None)
                if raw_df.shape[1] >= 4:
                    df_in = raw_df.iloc[:, :4]
                    df_in.columns = ['Name', 'X', 'Y', 'H']
                else:
                    st.error("CSVの列数が足りません。点名, X, Y, H の順で4列必要です。")
                    st.stop()

            # 座標変換実行
            lons, lats = to_latlon.transform(df_in['Y'].values, df_in['X'].values)
            
            # ジオイド高取得
            with st.spinner("国土地理院からジオイド高を取得中..."):
                ghs = [get_geoid_height(la, lo, use_geoid) if "ジオイド" in use_geoid else 0.0 for la, lo in zip(lats, lons)]
            
            # 結果の統合
            st.session_state.result_df = pd.DataFrame({
                "点名": df_in['Name'],
                "変換前_X": df_in['X'], "変換前_Y": df_in['Y'], "変換前_標高(H)": df_in['H'],
                "変換後_緯度": lats, "変換後_経度": lons,
                "ジオイド高": ghs,
                "変換後_楕円体高": df_in['H'] + ghs + offset_val
            })

# --- 5. 結果の表示と書き出し ---
if st.session_state.result_df is not None:
    res = st.session_state.result_df.copy()
    
    # 表示用のフォーマット（緯度経度8桁、高さ4桁）
    disp = res.copy()
    disp['変換後_緯度'] = disp['変換後_緯度'].map(lambda x: f"{x:.8f}")
    disp['変換後_経度'] = disp['変換後_経度'].map(lambda x: f"{x:.8f}")
    disp['ジオイド高'] = disp['ジオイド高'].map(lambda x: f"{x:.4f}")
    disp['変換後_楕円体高'] = disp['変換後_楕円体高'].map(lambda x: f"{x:.4f}")
    
    st.write("---")
    st.subheader("✅ 変換結果")
    st.dataframe(disp)
    
    col_dl1, col_dl2, _ = st.columns([2, 2, 6])
    
    # CSV保存
    with col_dl1:
        csv_data = disp.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 CSVを保存", csv_data, "converted_result.csv", "text/csv")
    
    # KML保存
    with col_dl2:
        kml = simplekml.Kml()
        for _, r in res.iterrows():
            # KMLには数値データとして渡す
            pnt = kml.newpoint(name=r['点名'], coords=[(r['変換後_経度'], r['変換後_緯度'], r['変換後_楕円体高'])])
            pnt.altitudemode = simplekml.AltitudeMode.absolute
        st.download_button("🌍 KMLを保存", kml.kml(), "converted_result.kml", "application/vnd.google-earth.kml+xml")

    # 地図表示
    st.subheader("🗺 マッププレビュー")
    avg_lat = res['変換後_緯度'].mean()
    avg_lon = res['変換後_経度'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=16)
    for _, r in res.iterrows():
        folium.Marker(
            [r['変換後_緯度'], r['変換後_経度']], 
            tooltip=f"{r['点名']}: H={r['変換後_楕円体高']:.3f}"
        ).add_to(m)
    st_folium(m, width=1200, height=600, key="main_map")
