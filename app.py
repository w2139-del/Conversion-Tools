import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import simplekml
import os
import time

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="高精度座標変換ツール", layout="wide")
st.title("座標変換ツール")

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:07.4f}\""

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
        if any(x > 900 for x in v): return 0.0
        dr, dc = r - r0, c - c0
        return round((1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3], 4)
    except: return 0.0

def read_csv_auto(uploaded_file):
    uploaded_file.seek(0)
    first_line = uploaded_file.readline().decode('shift-jis', errors='replace').strip()
    uploaded_file.seek(0)
    cols = first_line.split(',')
    try:
        float(cols[1])
        float(cols[2])
        df = pd.read_csv(uploaded_file, encoding='shift-jis', header=None)
    except (ValueError, IndexError):
        df = pd.read_csv(uploaded_file, encoding='shift-jis', header=0)
    if df.shape[1] < 4: raise ValueError("CSVは最低4列必要です。")
    df = df.iloc[:, :4]
    df.columns = ['点名', 'X', 'Y', 'H']
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['H'] = pd.to_numeric(df['H'], errors='coerce')
    return df.dropna(subset=['X', 'Y', 'H'])

# --- 3. サイドバー設定 ---
st.sidebar.header("💾 成果品保存")
kml_export_type = st.sidebar.selectbox("KML出力対象", ["ポイントとポリゴンの両方", "ポイントのみ", "ポリゴンのみ"], index=0)

if 'result' in st.session_state:
    res_data = st.session_state.result
    latlon_format = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"], index=0)

    disp_csv = res_data.copy()
    if latlon_format == "60進法 (DMS)":
        disp_csv['緯度'] = disp_csv['緯度'].map(decimal_to_dms)
        disp_csv['経度'] = disp_csv['経度'].map(decimal_to_dms)
    else:
        for c in ['緯度', '経度']: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高', 'X', 'Y', '標高H']: disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")

    st.sidebar.download_button(label="📊 CSVを保存", data=disp_csv.to_csv(index=False).encode('utf-8-sig'), 
                               file_name=f"result_{int(time.time())}.csv", mime='text/csv', use_container_width=True)

    kml = simplekml.Kml()
    if "ポイント" in kml_export_type:
        pnt_folder = kml.newfolder(name="Points")
        for _, r in res_data.iterrows():
            pnt_folder.newpoint(name=str(r['点名']), coords=[(r['経度'], r['緯度'], r['楕円体高'])])
    
    if "ポリゴン" in kml_export_type and 'drawn_data' in st.session_state:
        poly_folder = kml.newfolder(name="Polygons")
        features = st.session_state.drawn_data.get('all_drawings', [])
        for i, feat in enumerate(features):
            geom = feat.get('geometry', {})
            if geom.get('type') == 'Polygon':
                coords = geom.get('coordinates', [[]])[0]
                poly = poly_folder.newpolygon(name=f"Polygon_{i+1}")
                poly.outerboundaryis = coords
                poly.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.cyan)
            elif geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                line = poly_folder.newlinestring(name=f"Line_{i+1}")
                line.coords = coords
                line.style.linestyle.color = simplekml.Color.red
                line.style.linestyle.width = 3

    st.sidebar.download_button(label="🌍 KMLを保存", data=kml.kml(), file_name=f"spatial_data_{int(time.time())}.kml", 
                               mime='application/vnd.google-earth.kml+xml', use_container_width=True)
else:
    st.sidebar.info("計算を実行すると保存ボタンが表示されます。")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
transformer_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)

tab1, tab2, tab3 = st.tabs(["📝 1点手入力変換", "📂 ファイル一括変換", "📖 操作マニュアル"])

def run_calculation_process(input_df):
    if input_df.empty: return
    lons, lats = transformer.transform(input_df['Y'].values, input_df['X'].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    st.session_state.result = pd.DataFrame({
        "点名": input_df['点名'], "X": input_df['X'], "Y": input_df['Y'], "標高H": input_df['H'],
        "緯度": lats, "経度": lons, "ジオイド高": ghs,
        "楕円体高": input_df['H'].values + np.array(ghs) + offset_val, "適用モデル": use_geoid
    })
    if 'registered_marker_coords' not in st.session_state:
        st.session_state.registered_marker_coords = []
    st.rerun()

with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: p_name = st.text_input("点名", "Point_1")
    with col_b: p_x = st.number_input("X座標", value=0.0, format="%.4f")
    with col_c: p_y = st.number_input("Y座標", value=0.0, format="%.4f")
    with col_d: p_h = st.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行 (1点)", type="primary"):
        run_calculation_process(pd.DataFrame([{"点名": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up_file = st.file_uploader("CSV/SIMAアップロード", type=["csv", "sim"])
    if up_file and st.button("一括計算開始 🚀", type="primary"):
        try:
            if up_file.name.lower().endswith('.sim'):
                pts = []
                content = up_file.read().decode('shift-jis', errors='replace')
                for line in content.splitlines():
                    p = line.split(',')
                    if len(p) >= 6 and p[0] in ['A01', 'C00', 'C01']:
                        pts.append({'点名': p[1] if p[0].startswith('C') else p[2], 'X': float(p[3]), 'Y': float(p[4]), 'H': float(p[5])})
                df_input = pd.DataFrame(pts)
            else: df_input = read_csv_auto(up_file)
            run_calculation_process(df_input)
        except Exception as e: st.error(f"エラー: {e}")

with tab3:
    st.markdown("""
    ### 📖 操作ガイド
    1. **座標変換**: 『1点入力』または『ファイル一括』で計算を実行します。
    2. **マップ表示**: 変換が完了すると計算地点にピンが立ちます。
    3. **マーカー追加**: マップ左側の **『ピンアイコン（Marker）』** で地図上に配置します。
    4. **位置修正**: 配置したマーカーを動かしたい場合は、左側の **『鉛筆アイコン（Edit）』** をクリックし、マーカーをドラッグ移動させてから **『Save』** で確定させてください。
    5. **リスト登録**: 位置が確定したら、マップ下の **『マーカーを変換リストに追加』** を押します。修正後の座標で登録されます。
    6. **安全設計**: 一度リストに追加された座標は、マップ上の操作（ドラッグなど）で上書きされることはありません。
    """)

# --- 5. 結果表示 & マップ描画 ---
if 'result' in st.session_state:
    res = st.session_state.result
    st.divider()

    # データ表示
    res_disp = res.copy()
    if latlon_format == "60進法 (DMS)":
        res_disp['緯度'] = res_disp['緯度'].map(decimal_to_dms); res_disp['経度'] = res_disp['経度'].map(decimal_to_dms)
    else:
        res_disp['緯度'] = res_disp['緯度'].map(lambda x: f"{x:.8f}"); res_disp['経度'] = res_disp['経度'].map(lambda x: f"{x:.8f}")
    for c in ['ジオイド高', '楕円体高', 'X', 'Y', '標高H']: res_disp[c] = res_disp[c].map(lambda x: f"{x:.4f}")
    st.dataframe(res_disp, use_container_width=True)

    st.subheader("🗺 マッププレビュー & 描画ツール")
    valid_map_data = res[(res['緯度'] > 20) & (res['経度'] > 120)]

    if not valid_map_data.empty:
        avg_lat, avg_lon = valid_map_data['緯度'].mean(), valid_map_data['経度'].mean()
        tiles_url = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg' if map_type == "航空写真" else "OpenStreetMap"
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18, tiles=tiles_url, attr="GSI" if map_type=="航空写真" else "OSM")

        # 確定済みポイントの描画（FeatureGroup）
        fg = folium.FeatureGroup(name="Registered Markers")
        for _, row in valid_map_data.iterrows():
            folium.Marker([row['緯度'], row['経度']], popup=str(row['点名']), tooltip=str(row['点名'])).add_to(fg)
            folium.Marker([row['緯度'], row['経度']], icon=folium.DivIcon(icon_size=(150, 30), icon_anchor=(7, 25),
                html=f'<div style="font-size:11pt;color:red;font-weight:bold;text-shadow:2px 2px 2px #fff;">{row["点名"]}</div>')).add_to(fg)
        fg.add_to(m)

        draw = Draw(export=False, edit_options={'edit': {}, 'remove': {}},
                    draw_options={'marker': True, 'polyline': True, 'rectangle': True, 'polygon': True, 'circle': False, 'circlemarker': False})
        draw.add_to(m)

        output = st_folium(m, width=1200, height=600, key="fixed_map_layer")

        # 描画データ保存
        if output.get('all_drawings') is not None:
            st.session_state.drawn_data = output
            all_drawings = output.get('all_drawings')
            placed_markers = [f for f in all_drawings if f.get('geometry', {}).get('type') == 'Point']

            # マーカー追加ロジック
            if placed_markers:
                st.divider()
                if 'registered_marker_coords' not in st.session_state: st.session_state.registered_marker_coords = []
                
                registered = st.session_state.registered_marker_coords
                new_markers = []
                for feat in placed_markers:
                    c = feat['geometry']['coordinates']
                    c_key = (round(c[1], 8), round(c[0], 8))
                    if c_key not in registered: new_markers.append((c[1], c[0]))

                if new_markers:
                    st.info(f"📍 未登録の新規マーカーが {len(new_markers)} 件あります。（位置を修正してから追加できます）")
                    if st.button("➕ マーカーを変換リストに追加", type="primary"):
                        existing_nums = [int(n.split('_')[1]) for n in st.session_state.result['点名'].astype(str) if n.startswith('Click_') and '_' in n]
                        next_num = max(existing_nums, default=0) + 1
                        
                        new_rows = []
                        new_reg_keys = []
                        for m_lat, m_lon in new_markers:
                            y_v, x_v = transformer_inv.transform(m_lon, m_lat)
                            gh = get_geoid_height(m_lat, m_lon, use_geoid)
                            new_rows.append({
                                "点名": f"Click_{next_num}", "X": round(x_v, 4), "Y": round(y_v, 4), "標高H": 0.0,
                                "緯度": m_lat, "経度": m_lon, "ジオイド高": gh,
                                "楕円体高": round(0.0 + gh + offset_val, 4), "適用モデル": use_geoid
                            })
                            new_reg_keys.append((round(m_lat, 8), round(m_lon, 8)))
                            next_num += 1
                        
                        st.session_state.result = pd.concat([st.session_state.result, pd.DataFrame(new_rows)], ignore_index=True)
                        st.session_state.registered_marker_coords.extend(new_reg_keys)
                        st.success(f"✅ {len(new_rows)} 件を追加しました。")
                        st.rerun()
                else:
                    st.success("✅ 地図上のすべてのマーカーは登録済みです。")
