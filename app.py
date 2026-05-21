import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
st.title("研磨率解析（完全UI版：4点ドラッグ）")

# =========================
# 画像読み込み
# =========================
uploaded = st.file_uploader("画像アップロード", type=["jpg","jpeg","png"])
if uploaded is None:
    st.stop()

img_pil = Image.open(uploaded)
img = np.array(img_pil)
img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

H, W = img.shape[:2]

# =========================
# 表示縮尺（重要）
# =========================
disp_w = 700
scale = W / disp_w
disp_h = int(H / scale)

# =========================
# ROI選択（ポリゴン）
# =========================
st.subheader("① 4点クリックでROI指定（ドラッグで修正可）")

canvas = st_canvas(
    background_image=img_pil,
    drawing_mode="polygon",         # ★ここがポイント
    display_toolbar=True
    stroke_width=0.5,                 # 細線
    stroke_color="#00FF00",
    fill_color="rgba(0,255,0,0.2)", # 半透明
    height=disp_h,
    width=disp_w,
    key="canvas",
    initial_drawing=None
    realtime_update=False
)

# =========================
# ROI取得
# =========================
if canvas.json_data and len(canvas.json_data["objects"]) > 0:

    obj = canvas.json_data["objects"][-1]
    path = obj["path"]

    pts = []
    for p in path:
        if len(p) >= 3:
            pts.append((p[1], p[2]))

    if len(pts) >= 4:

        pts = np.array(pts[:4], dtype="float32")

        # ★元画像座標へ変換（重要）
        pts[:,0] *= scale
        pts[:,1] *= scale

        # =========================
        # 台形補正
        # =========================
        def warp(img, pts):
            (tl, tr, br, bl) = pts

            w = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
            h = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

            dst = np.array([
                [0,0],[w,0],[w,h],[0,h]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(img, M, (w, h))

        roi = warp(img_cv, pts)

        # =========================
        # 画像補正（解析だけ）
        # =========================
        st.sidebar.header("画像補正")
        brightness = st.sidebar.slider("明るさ", -100, 100, 0)
        contrast = st.sidebar.slider("コントラスト", 0.5, 3.0, 1.0)

        roi_adj = cv2.convertScaleAbs(roi, alpha=contrast, beta=brightness)

        # =========================
        # 色範囲指定
        # =========================
        st.sidebar.header("色範囲（HSV）")

        h_min, h_max = st.sidebar.slider("H", 0, 179, (0, 30))
        s_min, s_max = st.sidebar.slider("S", 0, 255, (0, 120))
        v_min, v_max = st.sidebar.slider("V", 0, 255, (150, 255))

        hsv = cv2.cvtColor(roi_adj, cv2.COLOR_BGR2HSV)

        mask = (
            (hsv[:,:,0]>=h_min)&(hsv[:,:,0]<=h_max)&
            (hsv[:,:,1]>=s_min)&(hsv[:,:,1]<=s_max)&
            (hsv[:,:,2]>=v_min)&(hsv[:,:,2]<=v_max)
        )

        mask = mask.astype(np.uint8)*255
        ratio = np.mean(mask>0)*100

        # =========================
        # 可視化
        # =========================
        overlay = roi.copy()
        overlay[mask>0] = [255,255,255]

        # =========================
        # 表示
        # =========================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("ROI（元画像そのまま）")
            st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        with col2:
            st.subheader("マスク")
            st.image(mask)

        with col3:
            st.subheader(f"割合 {ratio:.2f}%")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

# =========================
# 元画像
# =========================
st.subheader("元画像")
st.image(img_pil)

import matplotlib.pyplot as plt

# =========================
# ヒストグラム表示（ImageJ風）
# =========================
def show_histogram(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0].flatten()
    s = hsv[:,:,1].flatten()
    v = hsv[:,:,2].flatten()

    fig, ax = plt.subplots(3,1, figsize=(4,6))

    ax[0].hist(h, bins=180)
    ax[0].set_title("Hue")

    ax[1].hist(s, bins=256)
    ax[1].set_title("Saturation")

    ax[2].hist(v, bins=256)
    ax[2].set_title("Value")

    return fig

# =========================
# 色プリセット
# =========================
st.sidebar.header("色プリセット")

preset = st.sidebar.selectbox(
    "プリセット",
    ["手動", "研磨（白）", "錆（赤）", "錆（茶）"]
)

if preset == "研磨（白）":
    h_min, h_max = 0, 179
    s_min, s_max = 0, 60
    v_min, v_max = 180, 255

elif preset == "錆（赤）":
    h_min, h_max = 0, 20
    s_min, s_max = 80, 255
    v_min, v_max = 50, 255

elif preset == "錆（茶）":
    h_min, h_max = 10, 30
    s_min, s_max = 50, 255
    v_min, v_max = 50, 255

# =========================
# 手動調整
# =========================
st.sidebar.header("色範囲（ImageJ風）")

h_min, h_max = st.sidebar.slider("Hue", 0, 179, (h_min, h_max))
s_min, s_max = st.sidebar.slider("Saturation", 0, 255, (s_min, s_max))
v_min, v_max = st.sidebar.slider("Value", 0, 255, (v_min, v_max))

# =========================
# ヒストグラム表示
# =========================
st.subheader("② 色分布（ROI）")

fig = show_histogram(roi)
st.pyplot(fig)

# =========================
# マスク生成
# =========================
hsv = cv2.cvtColor(roi_adj, cv2.COLOR_BGR2HSV)

mask = (
    (hsv[:,:,0]>=h_min)&(hsv[:,:,0]<=h_max)&
    (hsv[:,:,1]>=s_min)&(hsv[:,:,1]<=s_max)&
    (hsv[:,:,2]>=v_min)&(hsv[:,:,2]<=v_max)
)

mask = mask.astype(np.uint8)*255

ratio = np.mean(mask>0)*100

# =========================
# overlay（ImageJ風）
# =========================
overlay = roi.copy()
overlay[mask>0] = [255,255,255]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ROI")
    st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

with col2:
    st.subheader("マスク")
    st.image(mask)

with col3:
    st.subheader(f"割合 {ratio:.2f}%")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))