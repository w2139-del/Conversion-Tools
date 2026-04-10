# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import os
import time

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="緯度経度変換ツール", layout="wide")
st.title("緯度経度変換ツール")

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}°{m:02d}'{s:07.4f}\""

@st.cache_resource
def load_geoid_data():
    data = {}
    if os.path.exists("geoid2024.npz"):
        data["2024"] = np.load("geoid2024.npz")["grid"]
    if os.path.exists("geoid2011.npz"):
        loader = np.load("geoid2011.npz")
        data["2011"] = loader["grid"]
        data["2011_h"] = loader["header"]
    return data

geoid_db = load_geoid_data()

def get_geoid_height(lat, lon, model_name):
    if model_name == "使用しない" or not geoid_db:
        return 0.0
    try:
        if model_name == "ジオイド2024":
            g = geoid_db.get("2024")
            r, c = (50.0 - lat) * 60.0, (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "日本のジオイド2011":
            g, h = geoid_db.get("2011"), geoid_db.get("2011_h")
            r, c = (lat - h[0]) / h[2], (lon - h[1]) / h[3]
        else:
            return 0.0
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        r1, c1 = r0 + 1, c0 + 1
        if r1 >= g.shape[0] or c1 >= g.shape[1] or r0 < 0 or c0 < 0:
            return 0.0
        v = [g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]]
        if any(x > 900 for x in v):
            return 0.0
        dr, dc = r - r0, c - c0
        return round((1-dr)*(1-dc)*v[0] + (1-dr)*dc*v[1] + dr*(1-dc)*v[2] + dr*dc*v[3], 4)
    except:
        return 0.0

def read_csv_auto(uploaded_file):
    uploaded_file.seek(0)
    first_line = uploaded_file.readline().decode("shift-jis", errors="replace").strip()
    uploaded_file.seek(0)
    cols = first_line.split(",")
    try:
        float(cols[1])
        float(cols[2])
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=None)
    except (ValueError, IndexError):
        df = pd.read_csv(uploaded_file, encoding="shift-jis", header=0)
    if df.shape[1] < 4:
        raise ValueError("CSVは最低4列（点名, X, Y, H）必要です。")
    df = df.iloc[:, :4]
    df.columns = ["点名", "X", "Y", "H"]
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    df = df.dropna(subset=["X", "Y", "H"])
    return df

def build_kml(res_data, kml_export_type, drawn_data):
    """Google Earth対応KML生成"""
    NM_O = "<" + "name" + ">"
    NM_C = "</" + "name" + ">"

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("  <Document>")
    lines.append("  " + NM_O + "spatial_data" + NM_C)
    # スタイルをDocument直下に定義（IDで参照するGoogle Earth推奨方式）
    lines += [
        "  <Style id=\"poly_style\">",
        "    <LineStyle><color>ffff0000</color><width>2</width></LineStyle>",
        "    <PolyStyle><color>6400ffff</color><fill>1</fill><outline>1</outline></PolyStyle>",
        "  </Style>",
        "  <Style id=\"line_style\">",
        "    <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>",
        "  </Style>",
        "  <Style id=\"point_style\">",
        "    <IconStyle><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle>",
        "  </Style>",
    ]

    # ポイント出力
    if "ポイント" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            pname = str(r["点名"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            lines += [
                "  <Placemark>",
                "    " + NM_O + pname + NM_C,
                "    <styleUrl>#point_style</styleUrl>",
                "    <Point>",
                "      <coordinates>" + str(r["経度"]) + "," + str(r["緯度"]) + "," + str(r["楕円体高"]) + "</coordinates>",
                "    </Point>",
                "  </Placemark>",
            ]

    # ポリゴン・ライン出力
    if "ポリゴン" in kml_export_type and drawn_data:
        features = drawn_data.get("all_drawings", [])
        poly_count = 1
        line_count = 1
        for feat in features:
            geom = feat.get("geometry", {})

            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                if len(coords) < 3:
                    continue
                if coords[0] != coords[-1]:
                    coords = coords + [coords[0]]
                coord_str = "\n".join(
                    "          " + str(c[0]) + "," + str(c[1]) + ",0"
                    for c in coords
                )
                lines += [
                    "  <Placemark>",
                    "    " + NM_O + "Polygon_" + str(poly_count) + NM_C,
                    "    <styleUrl>#poly_style</styleUrl>",
                    "    <Polygon>",
                    "      <tessellate>1</tessellate>",
                    "      <outerBoundaryIs>",
                    "        <LinearRing>",
                    "          <coordinates>",
                    coord_str,
                    "          </coordinates>",
                    "        </LinearRing>",
                    "      </outerBoundaryIs>",
                    "    </Polygon>",
                    "  </Placemark>",
                ]
                poly_count += 1

            elif geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) < 2:
                    continue
                coord_str = "\n".join(
                    "        " + str(c[0]) + "," + str(c[1]) + ",0"
                    for c in coords
                )
                lines += [
                    "  <Placemark>",
                    "    " + NM_O + "Line_" + str(line_count) + NM_C,
                    "    <styleUrl>#line_style</styleUrl>",
                    "    <LineString>",
                    "      <tessellate>1</tessellate>",
                    "      <coordinates>",
                    coord_str,
                    "      </coordinates>",
                    "    </LineString>",
                    "  </Placemark>",
                ]
                line_count += 1

    lines.append("  </Document>")
    lines.append("</kml>")
    return "\n".join(lines).encode("utf-8")

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 変換設定")
zone = st.sidebar.selectbox("系番号 (1-19系)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("ジオイドモデル", ["日本のジオイド2011", "ジオイド2024", "使用しない"], index=1)
is_antenna = st.sidebar.checkbox("アンテナ高(1.803m)加算", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "使用しない") else 0.0
map_type = st.sidebar.radio("背景地図", ["航空写真", "標準地図"])

st.sidebar.markdown("---")
st.sidebar.header("💾 成果品保存")

latlon_format = "10進法 (DD)"

if "result" in st.session_state:
    res_data = st.session_state.result
    latlon_format = st.sidebar.radio("緯度経度の形式", ["10進法 (DD)", "60進法 (DMS)"], index=0)

    disp_csv = res_data.copy()
    if latlon_format == "60進法 (DMS)":
        disp_csv["緯度"] = disp_csv["緯度"].map(decimal_to_dms)
        disp_csv["経度"] = disp_csv["経度"].map(decimal_to_dms)
    else:
        for c in ["緯度", "経度"]:
            disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]:
        disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")

    st.sidebar.download_button(
        label="📊 CSVを保存",
        data=disp_csv.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"result_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("計算を実行するとCSV保存ボタンが表示されます。")

# --- KML保存（サイドバー）---
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 KML保存")
_sb_drawn = st.session_state.get("drawn_data", {})
_sb_drawings = _sb_drawn.get("all_drawings") or []
_sb_non_point = [f for f in _sb_drawings if f.get("geometry", {}).get("type") != "Point"]
_sb_result = st.session_state.get("result", None)

kml_export_type = st.sidebar.selectbox(
    "KML出力対象",
    ["ポイントと図形の両方", "ポイントのみ", "図形のみ"],
    index=0,
    key="kml_type_select"
)
if _sb_non_point:
    st.sidebar.caption(f"✏️ 描画済み図形: {len(_sb_non_point)} 件")
else:
    st.sidebar.caption("図形未描画（マップでポリゴン等を描画してください）")
kml_bytes = build_kml(_sb_result, kml_export_type, _sb_drawn)
st.sidebar.download_button(
    label="🌍 KMLを保存",
    data=kml_bytes,
    file_name=f"spatial_data_{int(time.time())}.kml",
    mime="application/vnd.google-earth.kml+xml",
    use_container_width=True
)

# --- マーカー追加（サイドバー）---
st.sidebar.markdown("---")
st.sidebar.subheader("📍 マーカー追加")
st.sidebar.warning(f"⚠️ 現在の系番号: **第{zone}系**\n\nマーカー追加前に上記「変換設定」で正しい系番号を設定してください。")

_placed_markers_sb = [f for f in _sb_drawings if f.get("geometry", {}).get("type") == "Point"]

if _placed_markers_sb:
    if "registered_marker_coords" not in st.session_state:
        st.session_state.registered_marker_coords = []
    _registered_sb = st.session_state.registered_marker_coords
    _new_markers_sb = []
    for feat in _placed_markers_sb:
        coords = feat["geometry"]["coordinates"]
        coord_key = (round(coords[1], 8), round(coords[0], 8))
        if coord_key not in _registered_sb:
            _new_markers_sb.append((coords[1], coords[0]))

    if _new_markers_sb:
        st.sidebar.info(f"📍 未登録マーカー: {len(_new_markers_sb)} 件")
        st.sidebar.caption("💡 位置修正は鉛筆アイコン（Edit）でドラッグ → Save 後に追加してください。")
        if st.sidebar.button("➕ マーカーを変換リストに追加", type="primary", key="add_marker_sidebar"):
            _transformer_inv_sb = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)
            _existing_result = st.session_state.get("result", pd.DataFrame(
                columns=["点名", "X", "Y", "標高H", "緯度", "経度", "ジオイド高", "楕円体高", "適用モデル"]
            ))
            _existing_click_nums = []
            for name in _existing_result["点名"].astype(str):
                if name.startswith("Click_"):
                    try:
                        _existing_click_nums.append(int(name.split("_")[1]))
                    except ValueError:
                        pass
            _next_num = max(_existing_click_nums, default=0) + 1
            _new_rows = []
            _newly_registered = []
            for m_lat, m_lon in _new_markers_sb:
                y_val, x_val = _transformer_inv_sb.transform(m_lon, m_lat)
                gh = get_geoid_height(m_lat, m_lon, use_geoid)
                point_name = f"Click_{_next_num}"
                _next_num += 1
                _new_rows.append({
                    "点名": point_name,
                    "X": round(x_val, 4),
                    "Y": round(y_val, 4),
                    "標高H": 0.0,
                    "緯度": m_lat,
                    "経度": m_lon,
                    "ジオイド高": gh,
                    "楕円体高": round(0.0 + gh + offset_val, 4),
                    "適用モデル": use_geoid
                })
                _newly_registered.append((round(m_lat, 8), round(m_lon, 8)))
            st.session_state.result = pd.concat(
                [_existing_result, pd.DataFrame(_new_rows)],
                ignore_index=True
            )
            st.session_state.registered_marker_coords.extend(_newly_registered)
            # rerun前にdrawn_dataのポリゴンを明示的に保全する
            # （rerun後のマップ初期描画で all_drawings=[] が返っても消えないようにする）
            _cur_drawings = st.session_state.get("drawn_data", {}).get("all_drawings") or []
            _cur_non_points = [f for f in _cur_drawings if f.get("geometry", {}).get("type") != "Point"]
            # マーカーは登録済みになるのでPointは引き継がない（Drawプラグインが管理）
            st.session_state.drawn_data = {
                "all_drawings": _cur_non_points,
                "last_active_drawing": None,
            }
            st.rerun()
    else:
        st.sidebar.success("✅ 全マーカー登録済み")
else:
    st.sidebar.info("マップにマーカーを配置すると追加ボタンが表示されます。")

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
transformer_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)

tab1, tab2, tab3 = st.tabs(["📝 1点手入力変換", "📂 ファイル一括変換", "📖 操作マニュアル"])

def run_calculation_process(input_df):
    if input_df.empty:
        return
    lons, lats = transformer.transform(input_df["Y"].values, input_df["X"].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    st.session_state.result = pd.DataFrame({
        "点名": input_df["点名"],
        "X": input_df["X"],
        "Y": input_df["Y"],
        "標高H": input_df["H"],
        "緯度": lats,
        "経度": lons,
        "ジオイド高": ghs,
        "楕円体高": input_df["H"].values + np.array(ghs) + offset_val,
        "適用モデル": use_geoid
    })
    if "registered_marker_coords" not in st.session_state:
        st.session_state.registered_marker_coords = []
    # rerun前にdrawn_dataのポリゴンを保全
    _cur_drawings2 = st.session_state.get("drawn_data", {}).get("all_drawings") or []
    _cur_non_points2 = [f for f in _cur_drawings2 if f.get("geometry", {}).get("type") != "Point"]
    if _cur_non_points2:
        st.session_state.drawn_data = {
            "all_drawings": _cur_non_points2,
            "last_active_drawing": None,
        }
    st.rerun()

with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        p_name = st.text_input("点名", "Point_1")
    with col_b:
        p_x = st.number_input("X座標", value=0.0, format="%.4f")
    with col_c:
        p_y = st.number_input("Y座標", value=0.0, format="%.4f")
    with col_d:
        p_h = st.number_input("標高 H", value=0.0, format="%.4f")
    if st.button("計算実行 (1点)", type="primary"):
        run_calculation_process(pd.DataFrame([{"点名": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up_file = st.file_uploader("CSV/SIMAアップロード", type=["csv", "sim"])
    if up_file and st.button("一括計算開始 🚀", type="primary"):
        try:
            if up_file.name.lower().endswith(".sim"):
                pts = []
                content_sim = up_file.read().decode("shift-jis", errors="replace")
                for line in content_sim.splitlines():
                    p = line.split(",")
                    if len(p) >= 6 and p[0] in ["A01", "C00", "C01"]:
                        pts.append({
                            "点名": p[1] if p[0].startswith("C") else p[2],
                            "X": float(p[3]), "Y": float(p[4]), "H": float(p[5])
                        })
                df_input = pd.DataFrame(pts)
            else:
                df_input = read_csv_auto(up_file)
            run_calculation_process(df_input)
        except Exception as e:
            st.error(f"エラー: {e}")

with tab3:
    st.markdown("""
    ### 📖 操作ガイド
    1. **座標変換**: 「1点入力」または「ファイル一括」で計算を実行します。
    2. **CSV読み込み**: ヘッダー行があってもなくても自動判定して1点目から変換します。
    3. **マップ表示**: 変換が完了すると計算地点にピンが立ちます。
    4. **ポリゴン描画**: マップ左側の五角形アイコンで図形を描きます。ダブルクリックで確定。
    5. **マーカーで点追加**:
       - サイドバーの「変換設定」で**系番号を正しく設定**してください（変換精度に直結します）。
       - マップ左側の「ピンアイコン（Marker）」をクリックし地図上に配置します。
       - 位置を間違えた場合は「鉛筆アイコン（Edit）」でドラッグして移動 → 「Save」で確定してください。
       - 位置が確定したらサイドバーの「**➕ マーカーを変換リストに追加**」ボタンを押します。
       - 座標変換前（初期マップ状態）でもマーカーを追加できます。
    6. **KML保存**: ポリゴン描画後、サイドバーの「**🌍 KMLを保存**」ボタンから保存してください。
       ※ポリゴン描画後、**一度マップ外をクリックしてから**保存ボタンを押してください（描画確定のため）。
    """)

# --- 5. 結果表示 & マップ描画 ---
DEFAULT_LAT = 37.76162301842977
DEFAULT_LON = 140.47599584578793

if "result" in st.session_state:
    res = st.session_state.result
    st.divider()

    res_disp = res.copy()
    if latlon_format == "60進法 (DMS)":
        res_disp["緯度"] = res_disp["緯度"].map(decimal_to_dms)
        res_disp["経度"] = res_disp["経度"].map(decimal_to_dms)
    else:
        res_disp["緯度"] = res_disp["緯度"].map(lambda x: f"{x:.8f}")
        res_disp["経度"] = res_disp["経度"].map(lambda x: f"{x:.8f}")
    for c in ["ジオイド高", "楕円体高", "X", "Y", "標高H"]:
        res_disp[c] = res_disp[c].map(lambda x: f"{x:.4f}")

    st.dataframe(res_disp, use_container_width=True)

    valid_map_data = res[(res["緯度"] > 20) & (res["経度"] > 120)]
    if not valid_map_data.empty:
        center_lat = valid_map_data["緯度"].mean()
        center_lon = valid_map_data["経度"].mean()
        zoom_level = 18
    else:
        center_lat = DEFAULT_LAT
        center_lon = DEFAULT_LON
        zoom_level = 14
else:
    valid_map_data = pd.DataFrame()
    center_lat = DEFAULT_LAT
    center_lon = DEFAULT_LON
    zoom_level = 14

st.subheader("🗺 マッププレビュー & 描画ツール")

if map_type == "航空写真":
    tiles_url = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
    attr = "GSI"
else:
    tiles_url = "OpenStreetMap"
    attr = "OpenStreetMap"

m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level, tiles=tiles_url, attr=attr)

# --- 保存済みポリゴン・ラインをGeoJsonとして地図上に復元（rerun後も消えない）---
_restore_drawings = st.session_state.get("drawn_data", {}).get("all_drawings") or []
_non_point_restore = [f for f in _restore_drawings if f.get("geometry", {}).get("type") != "Point"]
if _non_point_restore:
    geojson_data = {"type": "FeatureCollection", "features": _non_point_restore}
    folium.GeoJson(
        geojson_data,
        name="saved_drawings",
        style_function=lambda feat: {
            "color": "#ff0000" if feat["geometry"]["type"] == "LineString" else "#0000ff",
            "weight": 2,
            "fillColor": "#00ffff",
            "fillOpacity": 0.25,
        }
    ).add_to(m)

if not valid_map_data.empty:
    fg = folium.FeatureGroup(name="Markers")
    for _, row in valid_map_data.iterrows():
        folium.Marker(
            [row["緯度"], row["経度"]],
            popup=str(row["点名"]),
            tooltip=str(row["点名"])
        ).add_to(fg)
        folium.Marker(
            [row["緯度"], row["経度"]],
            icon=folium.DivIcon(
                icon_size=(150, 30), icon_anchor=(7, 25),
                html='<div style="font-size:11pt;color:red;font-weight:bold;'
                     'text-shadow:2px 2px 2px #fff;">' + str(row["点名"]) + '</div>'
            )
        ).add_to(fg)
    fg.add_to(m)

draw = Draw(
    export=False,
    draw_options={
        "marker": True,
        "polyline": True,
        "rectangle": True,
        "polygon": True,
        "circle": False,
        "circlemarker": False,
    },
    edit_options={"edit": {}, "remove": {}}
)
draw.add_to(m)

map_key = st.session_state.get("map_key", "folium_map_fixed")
output = st_folium(m, width=1200, height=600, key=map_key)

# マップ描画直後にdrawn_dataを保存
# 【設計方針】
# - ポリゴン/ラインは session_state で永続管理し、rerun や マーカー追加で絶対に消さない
# - Drawプラグインが返す all_drawings はあくまでヒントとして使い、
#   非ポイント図形は「明示的に新しいものが描かれた時だけ」更新する
# - all_drawings が None の場合（マップ未操作）は一切上書きしない
raw_drawings = output.get("all_drawings")
if raw_drawings is not None:
    new_non_points = [f for f in raw_drawings if f.get("geometry", {}).get("type") != "Point"]
    new_points     = [f for f in raw_drawings if f.get("geometry", {}).get("type") == "Point"]

    # 非ポイント図形の更新ルール：
    #   Drawプラグインが1件以上の非ポイント図形を返した → ユーザーが描いた → 採用
    #   空リスト [] を返した               → rerun後の初期化や未操作 → 既存を維持
    if new_non_points:
        keep_non_points = new_non_points
    else:
        # 既存 drawn_data の非ポイント図形をそのまま維持
        keep_non_points = [
            f for f in (st.session_state.get("drawn_data", {}).get("all_drawings") or [])
            if f.get("geometry", {}).get("type") != "Point"
        ]

    # ポイント（マーカー）は Drawプラグインの値をそのまま使う
    st.session_state.drawn_data = {
        "all_drawings": keep_non_points + new_points,
        "last_active_drawing": output.get("last_active_drawing"),
    }
