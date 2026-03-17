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
st.set_page_config(page_title="高精度座標変換ツール", layout="wide")
st.title("高精度 座標変換ツール（ポリゴン描画＆KML高度エクスポート版）")

# --- 2. ユーティリティ関数 ---
def decimal_to_dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60) * 3600
    return f"{d}\u00b0{m:02d}'{s:07.4f}\""

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
    if model_name == "\u4f7f\u7528\u3057\u306a\u3044" or not geoid_db:
        return 0.0
    try:
        if model_name == "\u30b8\u30aa\u30a4\u30c92024":
            g = geoid_db.get("2024")
            r, c = (50.0 - lat) * 60.0, (lon - 120.0) * (60.0 / 1.5)
        elif model_name == "\u65e5\u672c\u306e\u30b8\u30aa\u30a4\u30c92011":
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
        raise ValueError("CSV\u306f\u6700\u4f534\u5217\uff08\u70b9\u540d, X, Y, H\uff09\u5fc5\u8981\u3067\u3059\u3002")
    df = df.iloc[:, :4]
    df.columns = ["\u70b9\u540d", "X", "Y", "H"]
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    df = df.dropna(subset=["X", "Y", "H"])
    return df

def build_kml(res_data, kml_export_type, drawn_data):
    """Google Maps\u5b8c\u5168\u5bfe\u5ffdKML\u3092\u76f4\u63a5\u6587\u5b57\u5217\u3067\u751f\u6210"""
    # nameタグを文字コードから直接生成（環境依存の表示問題を回避）
    NO = chr(60) + chr(110) + chr(97) + chr(109) + chr(101) + chr(62)   # <name>
    NC = chr(60) + chr(47) + chr(110) + chr(97) + chr(109) + chr(101) + chr(62)  # </name>

    placemarks = []

    # ポイント出力
    if "\u30dd\u30a4\u30f3\u30c8" in kml_export_type and res_data is not None:
        for _, r in res_data.iterrows():
            pname = str(r["\u70b9\u540d"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            placemarks.append(
                "    <Placemark>\n"
                f"      {NO}{pname}{NC}\n"
                "      <Style><IconStyle><Icon>\n"
                "        <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>\n"
                "      </Icon></IconStyle></Style>\n"
                "      <Point>\n"
                f"        <coordinates>{r[chr(32060)+chr(24230)]},{r[chr(32887)+chr(24230)]},{r[chr(26977)+chr(22280)+chr(20307)+chr(39640)]}</coordinates>\n"
                "      </Point>\n"
                "    </Placemark>"
            )

    # ポリゴン・ライン出力
    if "\u30dd\u30ea\u30b4\u30f3" in kml_export_type and drawn_data:
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
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                placemarks.append(
                    "    <Placemark>\n"
                    f"      {NO}Polygon_{poly_count}{NC}\n"
                    "      <Style>\n"
                    "        <LineStyle><color>ffff0000</color><width>2</width></LineStyle>\n"
                    "        <PolyStyle><color>6400ffff</color><fill>1</fill><outline>1</outline></PolyStyle>\n"
                    "      </Style>\n"
                    "      <Polygon>\n"
                    "        <outerBoundaryIs><LinearRing>\n"
                    f"          <coordinates>{coord_str}</coordinates>\n"
                    "        </LinearRing></outerBoundaryIs>\n"
                    "      </Polygon>\n"
                    "    </Placemark>"
                )
                poly_count += 1
            elif geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if len(coords) < 2:
                    continue
                coord_str = " ".join(f"{c[0]},{c[1]},0" for c in coords)
                placemarks.append(
                    "    <Placemark>\n"
                    f"      {NO}Line_{line_count}{NC}\n"
                    "      <Style><LineStyle><color>ff0000ff</color><width>3</width></LineStyle></Style>\n"
                    "      <LineString>\n"
                    f"        <coordinates>{coord_str}</coordinates>\n"
                    "      </LineString>\n"
                    "    </Placemark>"
                )
                line_count += 1

    body = "\n".join(placemarks)
    kml_str = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '  <Document>\n'
        f'    {NO}spatial_data{NC}\n'
        f'{body}\n'
        '  </Document>\n'
        '</kml>'
    )
    return kml_str.encode("utf-8")

# --- 3. サイドバー設定 ---
st.sidebar.header("\u2699\ufe0f \u5909\u63db\u8a2d\u5b9a")
zone = st.sidebar.selectbox("\u7cfb\u756a\u53f7 (1-19\u7cfb)", list(range(1, 20)), index=8)
use_geoid = st.sidebar.selectbox("\u30b8\u30aa\u30a4\u30c9\u30e2\u30c7\u30eb", ["\u65e5\u672c\u306e\u30b8\u30aa\u30a4\u30c92011", "\u30b8\u30aa\u30a4\u30c92024", "\u4f7f\u7528\u3057\u306a\u3044"], index=1)
is_antenna = st.sidebar.checkbox("\u30a2\u30f3\u30c6\u30ca\u9ad8(1.803m)\u52a0\u7b97", value=True)
offset_val = 1.803 if (is_antenna and use_geoid != "\u4f7f\u7528\u3057\u306a\u3044") else 0.0
map_type = st.sidebar.radio("\u80cc\u666f\u5730\u56f3", ["\u822a\u7a7a\u5199\u771f", "\u6a19\u6e96\u5730\u56f3"])

st.sidebar.markdown("---")
st.sidebar.header("\ud83d\udcbe \u6210\u679c\u54c1\u4fdd\u5b58")

latlon_format = "10\u9032\u6cd5 (DD)"

if "result" in st.session_state:
    res_data = st.session_state.result
    latlon_format = st.sidebar.radio("\u7def\u5ea6\u7d4c\u5ea6\u306e\u5f62\u5f0f", ["10\u9032\u6cd5 (DD)", "60\u9032\u6cd5 (DMS)"], index=0)

    disp_csv = res_data.copy()
    if latlon_format == "60\u9032\u6cd5 (DMS)":
        disp_csv["\u7def\u5ea6"] = disp_csv["\u7def\u5ea6"].map(decimal_to_dms)
        disp_csv["\u7d4c\u5ea6"] = disp_csv["\u7d4c\u5ea6"].map(decimal_to_dms)
    else:
        for c in ["\u7def\u5ea6", "\u7d4c\u5ea6"]:
            disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.8f}")
    for c in ["\u30b8\u30aa\u30a4\u30c9\u9ad8", "\u6977\u5186\u4f53\u9ad8", "X", "Y", "\u6a19\u9ad8H"]:
        disp_csv[c] = disp_csv[c].map(lambda x: f"{x:.4f}")

    st.sidebar.download_button(
        label="\ud83d\udcca CSV\u3092\u4fdd\u5b58",
        data=disp_csv.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"result_{int(time.time())}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.sidebar.info("\ud83c\udf0d KML\u306f\u30de\u30c3\u30d7\u306e\u4e0b\u306e\u300eKML\u3092\u4fdd\u5b58\u300f\u30dc\u30bf\u30f3\u304b\u3089\u4fdd\u5b58\u3057\u3066\u304f\u3060\u3055\u3044\u3002")
else:
    st.sidebar.info("\u8a08\u7b97\u3092\u5b9f\u884c\u3059\u308b\u3068\u4fdd\u5b58\u30dc\u30bf\u30f3\u304c\u8868\u793a\u3055\u308c\u307e\u3059\u3002")

# --- 4. メインコンテンツ ---
transformer = Transformer.from_crs(f"EPSG:{6668 + zone}", "EPSG:4326", always_xy=True)
transformer_inv = Transformer.from_crs("EPSG:4326", f"EPSG:{6668 + zone}", always_xy=True)

tab1, tab2, tab3 = st.tabs(["\ud83d\udcdd 1\u70b9\u624b\u5165\u529b\u5909\u63db", "\ud83d\udcc2 \u30d5\u30a1\u30a4\u30eb\u4e00\u62ec\u5909\u63db", "\ud83d\udcd6 \u64cd\u4f5c\u30de\u30cb\u30e5\u30a2\u30eb"])

def run_calculation_process(input_df):
    if input_df.empty:
        return
    lons, lats = transformer.transform(input_df["Y"].values, input_df["X"].values)
    ghs = [get_geoid_height(la, lo, use_geoid) for la, lo in zip(lats, lons)]
    st.session_state.result = pd.DataFrame({
        "\u70b9\u540d": input_df["\u70b9\u540d"],
        "X": input_df["X"],
        "Y": input_df["Y"],
        "\u6a19\u9ad8H": input_df["H"],
        "\u7def\u5ea6": lats,
        "\u7d4c\u5ea6": lons,
        "\u30b8\u30aa\u30a4\u30c9\u9ad8": ghs,
        "\u6977\u5186\u4f53\u9ad8": input_df["H"].values + np.array(ghs) + offset_val,
        "\u9069\u7528\u30e2\u30c7\u30eb": use_geoid
    })
    if "map_key" not in st.session_state:
        st.session_state.map_key = "folium_map_fixed"
    if "registered_marker_coords" not in st.session_state:
        st.session_state.registered_marker_coords = []
    st.rerun()

with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: p_name = st.text_input("\u70b9\u540d", "Point_1")
    with col_b: p_x = st.number_input("X\u5ea7\u6a19", value=0.0, format="%.4f")
    with col_c: p_y = st.number_input("Y\u5ea7\u6a19", value=0.0, format="%.4f")
    with col_d: p_h = st.number_input("\u6a19\u9ad8 H", value=0.0, format="%.4f")
    if st.button("\u8a08\u7b97\u5b9f\u884c (1\u70b9)", type="primary"):
        run_calculation_process(pd.DataFrame([{"\u70b9\u540d": p_name, "X": p_x, "Y": p_y, "H": p_h}]))

with tab2:
    up_file = st.file_uploader("CSV/SIMA\u30a2\u30c3\u30d7\u30ed\u30fc\u30c9", type=["csv", "sim"])
    if up_file and st.button("\u4e00\u62ec\u8a08\u7b97\u958b\u59cb \ud83d\ude80", type="primary"):
        try:
            if up_file.name.lower().endswith(".sim"):
                pts = []
                content_sim = up_file.read().decode("shift-jis", errors="replace")
                for line in content_sim.splitlines():
                    p = line.split(",")
                    if len(p) >= 6 and p[0] in ["A01", "C00", "C01"]:
                        pts.append({
                            "\u70b9\u540d": p[1] if p[0].startswith("C") else p[2],
                            "X": float(p[3]), "Y": float(p[4]), "H": float(p[5])
                        })
                df_input = pd.DataFrame(pts)
            else:
                df_input = read_csv_auto(up_file)
            run_calculation_process(df_input)
        except Exception as e:
            st.error(f"\u30a8\u30e9\u30fc: {e}")

with tab3:
    st.markdown("""
    ### \ud83d\udcd6 \u64cd\u4f5c\u30ac\u30a4\u30c9
    1. **\u5ea7\u6a19\u5909\u63db**: \u300c1\u70b9\u5165\u529b\u300d\u307e\u305f\u306f\u300c\u30d5\u30a1\u30a4\u30eb\u4e00\u62ec\u300d\u3067\u8a08\u7b97\u3092\u5b9f\u884c\u3057\u307e\u3059\u3002
    2. **CSV\u8aad\u307f\u8fbc\u307f**: \u30d8\u30c3\u30c0\u30fc\u884c\u304c\u3042\u3063\u3066\u3082\u306a\u304f\u3066\u3082\u81ea\u52d5\u5224\u5b9a\u3057\u3066\u5909\u63db\u3057\u307e\u3059\u3002
    3. **\u30de\u30c3\u30d7\u8868\u793a**: \u5909\u63db\u5b8c\u4e86\u5f8c\u3001\u8a08\u7b97\u5730\u70b9\u306b\u30d4\u30f3\u304c\u7acb\u3061\u307e\u3059\u3002
    4. **\u30dd\u30ea\u30b4\u30f3\u63cf\u753b**: \u30de\u30c3\u30d7\u5de6\u5074\u306e\u4e94\u89d2\u5f62\u30a2\u30a4\u30b3\u30f3\u3067\u56f3\u5f62\u3092\u63cf\u304d\u307e\u3059\u3002\u30c0\u30d6\u30eb\u30af\u30ea\u30c3\u30af\u3067\u78ba\u5b9a\u3002
    5. **\u30de\u30fc\u30ab\u30fc\u3067\u70b9\u8ffd\u52a0**: \u30de\u30c3\u30d7\u5de6\u5074\u306e\u300c\u30d4\u30f3\u30a2\u30a4\u30b3\u30f3\u300d\u3067\u914d\u7f6e\u3057\u3001\u4f4d\u7f6e\u78ba\u8a8d\u5f8c\u306b\u300c\u30de\u30fc\u30ab\u30fc\u3092\u5909\u63db\u30ea\u30b9\u30c8\u306b\u8ffd\u52a0\u300d\u30dc\u30bf\u30f3\u3092\u62bc\u3057\u3066\u304f\u3060\u3055\u3044\u3002
    6. **KML\u4fdd\u5b58**: \u30dd\u30ea\u30b4\u30f3\u3084\u30de\u30fc\u30ab\u30fc\u3092\u63cf\u753b\u5f8c\u3001**\u30de\u30c3\u30d7\u306e\u4e0b\u306b\u8868\u793a\u3055\u308c\u308bKML\u4fdd\u5b58\u30dc\u30bf\u30f3**\u304b\u3089\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u3057\u3066\u304f\u3060\u3055\u3044\u3002Google\u30de\u30c3\u30d7\u3067\u8aad\u307f\u8fbc\u3081\u307e\u3059\u3002
    """)

# --- 5. 結果表示 & マップ描画 ---
if "result" in st.session_state:
    res = st.session_state.result
    st.divider()

    res_disp = res.copy()
    if latlon_format == "60\u9032\u6cd5 (DMS)":
        res_disp["\u7def\u5ea6"] = res_disp["\u7def\u5ea6"].map(decimal_to_dms)
        res_disp["\u7d4c\u5ea6"] = res_disp["\u7d4c\u5ea6"].map(decimal_to_dms)
    else:
        res_disp["\u7def\u5ea6"] = res_disp["\u7def\u5ea6"].map(lambda x: f"{x:.8f}")
        res_disp["\u7d4c\u5ea6"] = res_disp["\u7d4c\u5ea6"].map(lambda x: f"{x:.8f}")
    for c in ["\u30b8\u30aa\u30a4\u30c9\u9ad8", "\u6977\u5186\u4f53\u9ad8", "X", "Y", "\u6a19\u9ad8H"]:
        res_disp[c] = res_disp[c].map(lambda x: f"{x:.4f}")

    st.dataframe(res_disp, use_container_width=True)

    st.subheader("\ud83d\uddfa \u30de\u30c3\u30d7\u30d7\u30ec\u30d3\u30e5\u30fc & \u63cf\u753b\u30c4\u30fc\u30eb")
    valid_map_data = res[(res["\u7def\u5ea6"] > 20) & (res["\u7d4c\u5ea6"] > 120)]

    if not valid_map_data.empty:
        avg_lat = valid_map_data["\u7def\u5ea6"].mean()
        avg_lon = valid_map_data["\u7d4c\u5ea6"].mean()

        if map_type == "\u822a\u7a7a\u5199\u771f":
            tiles_url = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
            attr = "GSI"
        else:
            tiles_url = "OpenStreetMap"
            attr = "OpenStreetMap"

        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18, tiles=tiles_url, attr=attr)

        fg = folium.FeatureGroup(name="Markers")
        for _, row in valid_map_data.iterrows():
            folium.Marker(
                [row["\u7def\u5ea6"], row["\u7d4c\u5ea6"]],
                popup=str(row["\u70b9\u540d"]),
                tooltip=str(row["\u70b9\u540d"])
            ).add_to(fg)
            folium.Marker(
                [row["\u7def\u5ea6"], row["\u7d4c\u5ea6"]],
                icon=folium.DivIcon(
                    icon_size=(150, 30), icon_anchor=(7, 25),
                    html=f'<div style="font-size:11pt;color:red;font-weight:bold;'
                         f'text-shadow:2px 2px 2px #fff;">{row[chr(28857)+chr(21517)]}</div>'
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

        # ★ マップ描画直後にdrawn_dataを保存（KML生成はこの後）
        if output.get("all_drawings") is not None:
            st.session_state.drawn_data = output

        all_drawings = output.get("all_drawings") or []

        non_point_count = sum(
            1 for f in all_drawings
            if f.get("geometry", {}).get("type") != "Point"
        )

        # ★ KMLダウンロードボタンをマップの直下に配置（drawn_data保存後なので確実に反映される）
        st.divider()
        st.subheader("\ud83d\udcbe \u6210\u679c\u54c1\u4fdd\u5b58")
        col_kml1, col_kml2 = st.columns(2)

        with col_kml1:
            kml_export_type = st.selectbox(
                "KML\u51fa\u529b\u5bfe\u8c61\u3092\u9078\u629e",
                ["\u30dd\u30a4\u30f3\u30c8\u3068\u30dd\u30ea\u30b4\u30f3\u306e\u4e21\u65b9", "\u30dd\u30a4\u30f3\u30c8\u306e\u307f", "\u30dd\u30ea\u30b4\u30f3\u306e\u307f"],
                index=0,
                key="kml_type_select"
            )

        with col_kml2:
            # ★ ここでdrawn_dataを使ってKMLを生成（必ずマップ描画後）
            current_drawn = st.session_state.get("drawn_data", {})
            kml_bytes = build_kml(res, kml_export_type, current_drawn)
            if non_point_count > 0:
                st.info(f"\u270f\ufe0f \u63cf\u753b\u6e08\u307f\u56f3\u5f62: {non_point_count} \u4ef6")
            st.download_button(
                label="\ud83c\udf0d KML\u3092\u4fdd\u5b58",
                data=kml_bytes,
                file_name=f"spatial_data_{int(time.time())}.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True
            )

        # マーカー追加UI
        placed_markers = [
            f for f in all_drawings
            if f.get("geometry", {}).get("type") == "Point"
        ]

        if placed_markers:
            st.divider()

            if "registered_marker_coords" not in st.session_state:
                st.session_state.registered_marker_coords = []

            registered = st.session_state.registered_marker_coords
            new_markers = []
            for feat in placed_markers:
                coords = feat["geometry"]["coordinates"]
                coord_key = (round(coords[1], 8), round(coords[0], 8))
                if coord_key not in registered:
                    new_markers.append((coords[1], coords[0]))

            if new_markers:
                st.info(f"\ud83d\udccd \u672a\u767b\u9332\u306e\u30de\u30fc\u30ab\u30fc\u304c {len(new_markers)} \u4ef6\u3042\u308a\u307e\u3059\u3002\u4f4d\u7f6e\u3092\u78ba\u8a8d\u3057\u3066\u304b\u3089\u8ffd\u52a0\u3057\u3066\u304f\u3060\u3055\u3044\u3002")
                st.caption("\ud83d\udca1 \u4f4d\u7f6e\u3092\u4fee\u6b63\u3059\u308b\u5834\u5408\u306f\u9c26\u7b46\u30a2\u30a4\u30b3\u30f3\uff08Edit\uff09\u3067\u30c9\u30e9\u30c3\u30b0 \u2192 Save \u3057\u3066\u304b\u3089\u8ffd\u52a0\u30dc\u30bf\u30f3\u3092\u62bc\u3057\u3066\u304f\u3060\u3055\u3044\u3002")

                if st.button("\u2795 \u30de\u30fc\u30ab\u30fc\u3092\u5909\u63db\u30ea\u30b9\u30c8\u306b\u8ffd\u52a0", type="primary"):
                    existing_click_nums = []
                    for name in st.session_state.result["\u70b9\u540d"].astype(str):
                        if name.startswith("Click_"):
                            try:
                                existing_click_nums.append(int(name.split("_")[1]))
                            except ValueError:
                                pass
                    next_num = max(existing_click_nums, default=0) + 1

                    new_rows = []
                    newly_registered = []
                    for m_lat, m_lon in new_markers:
                        y_val, x_val = transformer_inv.transform(m_lon, m_lat)
                        gh = get_geoid_height(m_lat, m_lon, use_geoid)
                        point_name = f"Click_{next_num}"
                        next_num += 1
                        new_rows.append({
                            "\u70b9\u540d": point_name,
                            "X": round(x_val, 4),
                            "Y": round(y_val, 4),
                            "\u6a19\u9ad8H": 0.0,
                            "\u7def\u5ea6": m_lat,
                            "\u7d4c\u5ea6": m_lon,
                            "\u30b8\u30aa\u30a4\u30c9\u9ad8": gh,
                            "\u6977\u5186\u4f53\u9ad8": round(0.0 + gh + offset_val, 4),
                            "\u9069\u7528\u30e2\u30c7\u30eb": use_geoid
                        })
                        newly_registered.append((round(m_lat, 8), round(m_lon, 8)))

                    st.session_state.result = pd.concat(
                        [st.session_state.result, pd.DataFrame(new_rows)],
                        ignore_index=True
                    )
                    st.session_state.registered_marker_coords.extend(newly_registered)
                    st.success(f"\u2705 {len(new_rows)} \u4ef6\u3092\u5909\u63db\u30ea\u30b9\u30c8\u306b\u8ffd\u52a0\u3057\u307e\u3057\u305f\u3002")
                    st.rerun()
            else:
                st.success("\u2705 \u914d\u7f6e\u6e08\u307f\u306e\u30de\u30fc\u30ab\u30fc\u306f\u3059\u3079\u3066\u30ea\u30b9\u30c8\u306b\u767b\u9332\u3055\u308c\u3066\u3044\u307e\u3059\u3002")
