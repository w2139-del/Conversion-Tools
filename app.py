import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from streamlit_folium import st_folium
import simplekml
import os
import time

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="高精度座標変換ツール", layout="wide")
st.title("高精度 座標変換ツール")

# --- 2. ジオイド計算エンジン（内蔵データ版） ---
@st.cache_resource
def load_geoid_data():
    """内蔵バイナリデータの読み込み（2011 & 2024）"""
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
        if r1 >= g.shape[0] or c1 >= g.shape[1] or r0 < 0 or c0 < 0: return 0.0
        v = [g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]]
        if any(x > 900 for x in v): return 0.0 # 欠測値除外
        
        dr, dc = r - r0, c - c0
        return round((1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3], 4)
    except: return 0.0

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 共通設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)

# 修正：デフォルトを「ジオイド2024」に
use_geoid = st.sidebar.selectbox(
    "使用するジオイドモデル", 
    ["日本のジオイド2011", "ジオイド2024", "使用しない"], 
    index=1
)

is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
# 修正：「使用しない」時はアンテナ高も強制的に0mとする
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0

st.sidebar.markdown("---")
st.sidebar.header("🗺 地図表示設定")
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

# --- サイドバー：ダウンロードセクション ---
if 'result' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.header("💾 成果品保存")
    
    res_data = st.session_state.result
    
    # 表示用データ整形
    disp_csv = res_data.copy()
    for c in ['緯度', '経度']: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高', 'X', 'Y', '標高H']: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")
    
    st.sidebar.download_button(
        label="📊 CSVをダウンロード",
        data=disp_csv.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"coord_result_{int(time.time())}.csv",
        mime='text/csv',
        use_container_width=True
    )
    
    kml = simplekml.Kml()
    for _, r in res_data.iterrows():
        kml.newpoint(name=str(r['点名']), coords=[(r['経度'], r['緯度'], r['楕円体高'])])
    
    st.sidebar.download_button(
        label="🌍 KMLをダウンロード",
        data=kml.kml(),
        file_name=f"coord_result_{int(time.time())}.kml",
        mime='application/vnd.google-earth.kml+xml',
        use_container_width=True
    )

# 座標変換の準備
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)

# --- 4. メインコンテンツ（タブ構成） ---
tab1, tab2 = st.tabs(["📝 1点手入力変換", "📂 ファイル一括変換"])

def run_calculation_process(input_df):
    """計算処理の共通化"""
    if input_df.empty: return
    lons, lats = transformer.transform(input_df['Y'].values, input_df['X'].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    
    st.session_state.result = pd.DataFrame({
        "点名": input_df['点名'], "X": input_df['X'], "Y": input_df['Y'], "標高H": input_df['H'],
        "緯度": lats, "経度": lons, "ジオイド高": ghs,
        "楕円体高": input_df['H'].values + np.array(ghs) + offset_val,
        "適用モデル": use_geoid
    })
    # マップ強制更新用ID（時間+地図タイプ）
    st.session_state.calc_id = f"{time.time()}_{map_type}"

with tab1:
    st.markdown("##### 特定の1点を入力してください")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: p_name = st.text_input("点名", "Point_1")
    with col_b: p_x = st.number_input("X座標 (北方向)", value=0.0, format="%.4f")
    with col_c: p_y = st.number_input("Y座標 (東方向)", value=0.0, format="%.4f")
    with col_d: p_h = st.number_input("標高 H", value=0.0, format="%.4f")
    
    if st.button("この地点を計算する", type="primary"):
        run_calculation_process(pd.DataFrame([{"点名": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    st.markdown("##### CSVまたはSIMAファイルをアップロードしてください")
    up_file = st.file_uploader("ファイルを選択", type=["csv", "sim"])
    if up_file and st.button("一括変換を実行 🚀", type="primary"):
        try:
            if up_file.name.lower().endswith('.sim'):
                pts = []
                content = up_file.read().decode('shift-jis', errors='replace')
                for line in content.splitlines():
                    p = line.split(',')
                    if len(p) >= 6 and p[0] in ['A01', 'C00', 'C01']:
                        pts.append({'点名': p[1] if p[0].startswith('C') else p[2], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
                df_input = pd.DataFrame(pts)
            else:
                df_input = pd.read_csv(up_file, encoding='shift-jis')
                df_input = df_input.rename(columns={df_input.columns[0]: '点名', df_input.columns[1]: 'X', df_input.columns[2]: 'Y', df_input.columns[3]: 'H'})
            run_calculation_process(df_input)
        except Exception as e:
            st.error(f"解析エラー: {e}")

# --- 5. 計算結果の表示とマップ描画 ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.divider()
    
    if use_geoid == "使用しない":
        st.info("💡 ジオイド高とアンテナ高を『0』として計算しています。楕円体高と標高は一致します。")

    # データテーブル表示
    res_disp = res.copy()
    for c in ['緯度', '経度']: res_disp[c] = res_disp[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高', 'X', 'Y', '標高H']: res_disp[c] = res_disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(res_disp, use_container_width=True)
    
    # マップセクション
    st.subheader("🗺 変換位置の確認")
    # 座標が有効なデータのみ抽出
    valid_map_data = res[(res['緯度'] > 20) & (res['経度'] > 120)]
    
    if not valid_map_data.empty:
        avg_lat, avg_lon = valid_map_data['緯度'].mean(), valid_map_data['経度'].mean()
        
        # タイルと属性
        tiles_url = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg' if map_type == "航空写真" else "OpenStreetMap"
        map_attr = "地理院航空写真" if map_type == "航空写真" else "OpenStreetMap"
        
        # 地図の初期化
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18, tiles=tiles_url, attr=map_attr)
        
        # ピンとラベルのグループ化
        fg = folium.FeatureGroup(name="Markers")
        for _, row in valid_map_data.iterrows():
            # 通常ピン
            folium.Marker(
                location=[row['緯度'], row['経度']],
                popup=f"<b>{row['点名']}</b><br>楕円体高: {row['楕円体高']:.3f}m",
                tooltip=str(row['点名'])
            ).add_to(fg)
            
            # 赤文字ラベル
            folium.Marker(
                [row['緯度'], row['経度']],
                icon=folium.DivIcon(
                    icon_size=(150, 30), icon_anchor=(7, 25),
                    html=f'<div style="font-size: 11pt; color: red; font-weight: bold; text-shadow: 2px 2px 2px #fff; white-space: nowrap;">{row["点名"]}</div>'
                )
            ).add_to(fg)
        
        fg.add_to(m)
        
        # 地図の描画（keyを更新することで計算のたびに強制再描画）
        st_folium(
            m, 
            width=1200, 
            height=600, 
            key=st.session_state.calc_id,
            returned_objects=[] # 描画速度優先のため戻り値を制限
        )
    else:
        st.warning("⚠️ 日本国内の有効な座標が入力されていないため、地図を表示できません。")
