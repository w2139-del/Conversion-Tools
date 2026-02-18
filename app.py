import streamlit as st
import pandas as pd
import numpy as np
from pyproj import Transformer
import os

# --- ジオイド読み込み（高速バイナリ版） ---
@st.cache_resource
def load_geoids():
    data = {}
    if os.path.exists('geoid2024.npz'):
        data['2024'] = np.load('geoid2024.npz')['grid']
    if os.path.exists('geoid2011.npz'):
        loader = np.load('geoid2011.npz')
        data['2011'] = loader['grid']
        data['2011_h'] = loader['header'] # lat_min, lon_min, d_lat, d_lon, rows, cols
    return data

geoid_db = load_geoids()

def get_geoid_height(lat, lon, model_name):
    if not geoid_db: return 0.0
    
    if model_name == "ジオイド2024":
        g = geoid_db.get('2024')
        if g is None: return 0.0
        # 2024設定: 15-50N, 120-160E, 1'x1.5'
        r = (50.0 - lat) / (1/60)
        c = (lon - 120.0) / (1.5/60)
    else: # 2011
        g = geoid_db.get('2011')
        h = geoid_db.get('2011_h')
        if g is None: return 0.0
        # 2011設定 (S-to-N)
        r = (lat - h[0]) / h[2]
        c = (lon - h[1]) / h[3]

    r0, c0 = int(np.floor(r)), int(np.floor(c))
    r1, c1 = r0 + 1, c0 + 1
    
    # 範囲外・無効値チェック
    try:
        v00, v01, v10, v11 = g[r0, c0], g[r0, c1], g[r1, c0], g[r1, c1]
        if any(val > 900 for val in [v00, v01, v10, v11]): return 0.0
        dr, dc = r - r0, c - c0
        return (1-dr)*(1-dc)*v00 + (1-dr)*dc*v01 + dr*(1-dc)*v10 + dr*dc*v11
    except:
        return 0.0

# --- 以降、座標変換・UIロジック ---
# (ここから先はこれまでのUIコードと同じです)
