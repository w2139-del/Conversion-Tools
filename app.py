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

if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# --- 2. 共通関数 ---
def get_geoid_height(lat, lon, ver="2024"):
    """国土地理院APIからジオイド高を取得"""
    v = "2024" if "2024" in ver else "2011"
    try:
        url = f"https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl?outputType=json&latitude={lat}&longitude={lon}&gsigen={v}"
        response = requests.get(url, timeout=10)
        res = response.json()
        out_data = res.get('OutputData', {})
        gh = out_data.get('geoidHeight') or res.get('geoidHeight')
        return float(gh) if gh is not None else 0.0
    except:
        return 0.0

def parse_sima(uploaded_file):
    """SIMAファイルを解析"""
    points = []
    # 読み込み位置を先頭に戻す
    uploaded_file.seek(0)
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
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=9) # 10系デフォルト
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])
add_offset = st.sidebar.checkbox("アンテナ高(1.803m)を加算", value=True)
offset_val = 1.803 if add_offset else 0.0

# 座標変換エンジンの準備
epsg = 6668 + zone
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
            "点名": p_name, "変換前_X": p_x, "変換前_Y": p_y, "変換前_標高(H)": p_h,
            "変換後_緯度": lat, "変換後_経度": lon, "ジオイド高": gh, "変換後_楕円体高": p_h + gh + offset_val
        }])

# 【ファイル一括変換】
with tab2:
    st.info("CSVの場合、[点名, X, Y, H] の順で並んでいる必要があります（見出し行はあってもなくてもOK）。")
    uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])
    
    if uploaded_file:
        if st.button("一括変換を開始"):
            try:
                # 読み込み位置をリセット
                uploaded_file.seek(0)
                
                if uploaded_file.name.lower().endswith('.sim'):
                    df_in = parse_sima(uploaded_file)
                else:
                    # CSVの読み込み：1〜4列目を強制的に取得し、列名をセット
                    # header=Noneにすることで見出しがあっても1行のデータとして扱い、後で数値化できない行を除外する
                    df_in = pd.read_csv(uploaded_file, header=None, usecols=[0, 1, 2, 3], names=['Name', 'X', 'Y', 'H'])
                    # 数値であるべき列（X, Y, H）が数値に変換できない行（見出し行など）をエラーにせず除外
                    df_in['X'] = pd.to_numeric(df_in['X'], errors='coerce')
                    df_in['Y'] = pd.to_numeric(df_in['Y'], errors='coerce')
                    df_in['H'] = pd.to_numeric(df_in['H'], errors='coerce')
                    df_in = df_in.dropna(subset=['X', 'Y']) # X, Yが空の行を消す

                if df_in.empty:
                    st.error("有効なデータが見つかりませんでした。ファイルの形式を確認してください。")
                else:
                    # 座標変換
                    lons, lats = to_latlon.transform(df_in['Y'].values, df_in['X'].values)
                    
                    with st.spinner("ジオイド高を取得中..."):
                        ghs = [get_geoid_height(la, lo, use_geoid) if "ジオイド" in use_geoid else 0.0 for la, lo in zip(lats, lons)]
                    
                    # 結果保存
                    st.session_state.result_df = pd.DataFrame({
                        "点名": df_in['Name'],
                        "変換前_X": df_in['X'], "変換前_Y": df_in['Y'], "変換前_標高(H)": df_in['H'],
                        "変換後_緯度": lats, "変換後_経度": lons,
                        "ジオイド高": ghs,
                        "変換後_楕円体高": df_in['H'] + ghs + offset_val
                    })
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# --- 5. 結果表示 ---
if st.session_state.result_df is not None:
    res = st.session_state.result_df.copy()
    disp = res.copy()
    disp['変換後_緯度'] = disp['変換後_緯度'].map(lambda x: f"{x:.8f}")
    disp['変換後_経度'] = disp['変換後_経度'].map(lambda x: f"{x:.8f}")
    disp['ジオイド高'] = disp['ジオイド高'].map(lambda x: f"{x:.4f}")
    disp['変換後_楕円体高'] = disp['変換後_楕円体高'].map(lambda x: f"{x:.4f}")
    
    st.write("---")
    st.subheader("✅ 変換結果")
    st.dataframe(disp)
    
    col_dl1, col_dl2, _ = st.columns([2, 2, 6])
    with col_dl1:
        st.download_button("📊 CSVを保存", disp.to_csv(index=False).encode('utf-8-sig'), "result.csv", "text/csv")
    with col_dl2:
        kml = simplekml.Kml()
        for _, r in res.iterrows():
            pnt = kml.newpoint(name=r['点名'], coords=[(r['変換後_経度'], r['変換後_緯度'], r['変換後_楕円体高'])])
            pnt.altitudemode = simplekml.AltitudeMode.absolute
        st.download_button("🌍 KMLを保存", kml.kml(), "result.kml", "application/vnd.google-earth.kml+xml")

    # 地図
    avg_lat, avg_lon = res['変換後_緯度'].mean(), res['変換後_経度'].mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=16)
    for _, r in res.iterrows():
        folium.Marker([r['変換後_緯度'], r['変換後_経度']], tooltip=r['点名']).add_to(m)
    st_folium(m, width=1200, height=600, key="survey_map")
