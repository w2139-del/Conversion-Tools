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
st.title("座標変換ツール（9系・ジオイド切替完全版）")

# セッション状態の初期化
if 'result_df' not in st.session_state:
    st.session_state.result_df = None

# --- 2. 共通関数 ---
def get_geoid_height(lat, lon, model_label):
    """国土地理院APIからジオイド高を取得（モデル指定を厳密化）"""
    # API用パラメータの決定
    if "2024" in model_label:
        v_param = "gsig2024"
    elif "2011" in model_label:
        v_param = "gsig2011"
    else:
        return 0.0
    
    try:
        # 国土地理院 ジオイド高計算API
        url = "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl"
        params = {
            "outputType": "json",
            "latitude": lat,
            "longitude": lon,
            "gsigen": v_param
        }
        response = requests.get(url, params=params, timeout=10)
        res = response.json()
        # APIの戻り値からジオイド高を抽出
        gh = res.get('OutputData', {}).get('geoidHeight') or res.get('geoidHeight')
        return float(gh) if gh is not None else 0.0
    except Exception as e:
        return 0.0

def parse_sima(uploaded_file):
    points = []
    uploaded_file.seek(0)
    content = uploaded_file.read().decode('shift-jis', errors='replace')
    for line in content.splitlines():
        parts = line.split(',')
        if len(parts) >= 6:
            if parts[0] == 'A01': # 座標データ
                points.append({'Name': parts[2], 'X': float(parts[3]), 'Y': float(parts[4]), 'H': float(parts[5]) if parts[5] else 0.0})
            elif parts[0] in ['C00', 'C01']: # 測設データ
                points.append({'Name': parts[1], 'X': float(parts[3]), 'Y': float(parts[4]), 'H': float(parts[5]) if parts[5] else 0.0})
    return pd.DataFrame(points)

# --- 3. サイドバー設定 ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8) # デフォルト9系

# ジオイドモデルの選択
use_geoid = st.sidebar.selectbox(
    "使用するジオイドモデル", 
    ["ジオイド2024", "日本のジオイド2011", "使用しない"]
)

add_offset = st.sidebar.checkbox("アンテナ高(1.803m)を加算", value=True)
offset_val = 1.803 if add_offset else 0.0

st.sidebar.markdown("---")
st.sidebar.header("地図表示設定")
map_type = st.sidebar.radio("背景地図の選択", ["標準地図", "航空写真"])

# 座標変換エンジンの準備
epsg = 6668 + zone
to_latlon = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# --- 4. メイン画面 ---
tab1, tab2 = st.tabs(["📍 1点直接入力", "📂 ファイル一括変換 (CSV/SIMA)"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1: p_name = st.text_input("点名", value="Point01")
    with col2: p_x = st.number_input("X(北方向)", value=0.0, format="%.4f")
    with col3: p_y = st.number_input("Y(東方向)", value=0.0, format="%.4f")
    with col4: p_h = st.number_input("標高(H)", value=0.0, format="%.4f")
    
    if st.button("1点変換を実行"):
        lon, lat = to_latlon.transform(p_y, p_x)
        gh = get_geoid_height(lat, lon, use_geoid)
        st.session_state.result_df = pd.DataFrame([{
            "点名": p_name, "変換前_X": p_x, "変換前_Y": p_y, "変換前_標高(H)": p_h,
            "変換後_緯度": lat, "変換後_経度": lon, "使用モデル": use_geoid,
            "ジオイド高": gh, "変換後_楕円体高": p_h + gh + offset_val
        }])

with tab2:
    uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])
    if uploaded_file:
        if st.button("一括変換を開始（再計算）"):
            try:
                uploaded_file.seek(0)
                if uploaded_file.name.lower().endswith('.sim'):
                    df_in = parse_sima(uploaded_file)
                else:
                    # CSVの読み込み（ヘッダーがある場合を考慮）
                    try:
                        df_in = pd.read_csv(uploaded_file, encoding='shift-jis')
                        # 列名が期待通りでない場合はヘッダーなしとして再読み込み
                        if not all(c in df_in.columns for c in ['X', 'Y']):
                            uploaded_file.seek(0)
                            df_in = pd.read_csv(uploaded_file, header=None, names=['Name', 'X', 'Y', 'H'], encoding='shift-jis')
                    except:
                        uploaded_file.seek(0)
                        df_in = pd.read_csv(uploaded_file, header=None, names=['Name', 'X', 'Y', 'H'], encoding='utf-8')
                    
                    df_in['X'] = pd.to_numeric(df_in['X'], errors='coerce')
                    df_in['Y'] = pd.to_numeric(df_in['Y'], errors='coerce')
                    df_in['H'] = pd.to_numeric(df_in['H'], errors='coerce')
                    df_in = df_in.dropna(subset=['X', 'Y'])

                if not df_in.empty:
                    lons, lats = to_latlon.transform(df_in['Y'].values, df_in['X'].values)
                    with st.spinner(f"国土地理院APIから {use_geoid} を取得中..."):
                        ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
                    
                    st.session_state.result_df = pd.DataFrame({
                        "点名": df_in.iloc[:, 0], # 1列目を点名として使用
                        "変換前_X": df_in['X'], "変換前_Y": df_in['Y'], "変換前_標高(H)": df_in['H'],
                        "変換後_緯度": lats, "変換後_経度": lons,
                        "使用モデル": use_geoid,
                        "ジオイド高": ghs,
                        "変換後_楕円体高": df_in['H'].values + ghs + offset_val
                    })
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# --- 5. 結果表示 ---
if st.session_state.result_df is not None:
    res = st.session_state.result_df.copy()
    
    st.write("---")
    # 現在の計算に使用されたモデルを大きく表示
    current_model = res['使用モデル'].iloc[0] if '使用モデル' in res.columns else "不明"
    st.subheader(f"✅ 変換結果 （計算モデル: {current_model}）")
    
    disp = res.copy()
    for c in ['変換後_緯度', '変換後_経度']: disp[c] = disp[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '変換後_楕円体高']: disp[c] = disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(disp)
    
    # KML保存など
    col_dl1, col_dl2, _ = st.columns([2, 2, 6])
    with col_dl1:
        st.download_button("📊 CSV保存", disp.to_csv(index=False).encode('utf-8-sig'), "result.csv", "text/csv")
    with col_dl2:
        kml = simplekml.Kml()
        for _, r in res.iterrows():
            p = kml.newpoint(name=str(r['点名']), coords=[(r['変換後_経度'], r['変換後_緯度'], r['変換後_楕円体高'])])
            p.altitudemode = simplekml.AltitudeMode.absolute
        st.download_button("🌍 KML保存", kml.kml(), "result.kml", "application/vnd.google-earth.kml+xml")

    # 地図表示
    st.subheader("🗺 マッププレビュー")
    avg_lat, avg_lon = res['変換後_緯度'].mean(), res['変換後_経度'].mean()
    tiles = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg' if map_type == "航空写真" else 'OpenStreetMap'
    attr = '国土地理院' if map_type == "航空写真" else 'OpenStreetMap'
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18, tiles=tiles, attr=attr)

    for _, r in res.iterrows():
        folium.Marker([r['変換後_緯度'], r['変換後_経度']], tooltip=str(r['点名'])).add_to(m)
        folium.map.Marker(
            [r['変換後_緯度'], r['変換後_経度']],
            icon=folium.DivIcon(icon_size=(150,36), icon_anchor=(7,20),
                html=f'<div style="font-size: 11pt; color: red; font-weight: bold; text-shadow: 2px 2px 2px #fff; white-space: nowrap;">{r["点名"]}</div>')
        ).add_to(m)
    st_folium(m, width=1200, height=600, key="survey_map_v4")
