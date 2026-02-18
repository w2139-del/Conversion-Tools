import streamlit as st
import pandas as pd
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import requests
import simplekml
import time

# --- 1. 基本設定（キャッシュを完全にクリア） ---
st.set_page_config(page_title="座標変換ツール", layout="wide")

# アプリ起動時にキャッシュを空にする
if 'init' not in st.session_state:
    st.cache_data.clear()
    st.session_state.init = True

st.title("座標変換ツール（9系・ジオイド完全切替保証版）")

# --- 2. ジオイド取得関数（キャッシュを一切使わない設定） ---
def get_geoid_height_direct(lat, lon, model_label):
    # モデルの判定
    v_param = "gsig2024" if "2024" in model_label else "gsig2011"
    
    url = "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl"
    params = {
        "outputType": "json",
        "latitude": lat,
        "longitude": lon,
        "gsigen": v_param,
        "t": time.time() # 1秒ごとに変わる数値を送り、APIに「新しい計算」を強制する
    }
    
    try:
        # 毎回必ずリクエストを飛ばす
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        gh = data.get('OutputData', {}).get('geoidHeight') or data.get('geoidHeight')
        return float(gh) if gh is not None else 0.0
    except:
        return 0.0

# --- 3. サイドバー設定 ---
st.sidebar.header("共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("使用するジオイドモデル", ["ジオイド2024", "日本のジオイド2011", "使用しない"])

# モデルを変えたらセッションをリセット
if "prev_model" not in st.session_state:
    st.session_state.prev_model = use_geoid

if st.session_state.prev_model != use_geoid:
    st.cache_data.clear() # キャッシュを物理的に削除
    st.session_state.result_df = None
    st.session_state.prev_model = use_geoid

add_offset = st.sidebar.checkbox("アンテナ高(1.803m)を加算", value=True)
offset_val = 1.803 if add_offset else 0.0

# 座標変換エンジン
epsg = 6668 + zone
to_latlon = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

# --- 4. メイン処理 ---
uploaded_file = st.file_uploader("CSVまたはSIMAをアップロード", type=["csv", "sim"])

if uploaded_file:
    if st.button("🚀 変換を実行（最新データを取得）"):
        try:
            # データの読み込み
            uploaded_file.seek(0)
            if uploaded_file.name.lower().endswith('.sim'):
                points = []
                content = uploaded_file.read().decode('shift-jis', errors='replace')
                for line in content.splitlines():
                    parts = line.split(',')
                    if len(parts) >= 6:
                        if parts[0] == 'A01': points.append({'Name': parts[2], 'X': float(parts[3]), 'Y': float(parts[4]), 'H': float(parts[5])})
                        elif parts[0] in ['C00', 'C01']: points.append({'Name': parts[1], 'X': float(parts[3]), 'Y': float(parts[4]), 'H': float(parts[5])})
                df_in = pd.DataFrame(points)
            else:
                try:
                    df_in = pd.read_csv(uploaded_file, encoding='shift-jis')
                    if 'X' not in df_in.columns:
                        uploaded_file.seek(0)
                        df_in = pd.read_csv(uploaded_file, header=None, names=['Name', 'X', 'Y', 'H'], encoding='shift-jis')
                except:
                    uploaded_file.seek(0)
                    df_in = pd.read_csv(uploaded_file, header=None, names=['Name', 'X', 'Y', 'H'], encoding='utf-8')

            df_in = df_in.dropna(subset=['X', 'Y'])
            
            # 緯度経度変換
            lons, lats = to_latlon.transform(df_in['Y'].values, df_in['X'].values)
            
            # ジオイド高取得（プログレスバー表示）
            ghs = []
            progress_bar = st.progress(0)
            total = len(lats)
            
            for i, (la, lo) in enumerate(zip(lats, lons)):
                gh = get_geoid_height_direct(la, lo, use_geoid) if "ジオイド" in use_geoid else 0.0
                ghs.append(gh)
                progress_bar.progress((i + 1) / total)
            
            # 結果をセッションに保存
            st.session_state.result_df = pd.DataFrame({
                "点名": df_in.iloc[:, 0],
                "変換前_X": df_in['X'], "変換前_Y": df_in['Y'], "変換前_標高(H)": df_in['H'],
                "緯度": lats, "経度": lons,
                "適用モデル": use_geoid,
                "ジオイド高": ghs,
                "楕円体高": df_in['H'].values + ghs + offset_val
            })
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

# --- 5. 結果表示 ---
if st.session_state.result_df is not None:
    res = st.session_state.result_df
    st.success(f"✅ 【{res['適用モデル'].iloc[0]}】で計算しました。")
    
    # 小数点以下の表示調整
    disp = res.copy()
    disp['緯度'] = disp['緯度'].map(lambda x: f"{x:.8f}")
    disp['経度'] = disp['経度'].map(lambda x: f"{x:.8f}")
    disp['ジオイド高'] = disp['ジオイド高'].map(lambda x: f"{x:.4f}")
    disp['楕円体高'] = disp['楕円体高'].map(lambda x: f"{x:.4f}")
    
    st.dataframe(disp, use_container_width=True)
    
    # ダウンロードボタン
    csv_data = disp.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📊 CSVを保存", csv_data, f"result_{res['適用モデル'].iloc[0]}.csv", "text/csv")
