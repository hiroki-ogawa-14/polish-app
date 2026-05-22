import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("研磨率解析（安定版UI）")

uploaded = st.file_uploader("画像アップロード", type=["jpg","jpeg","png"])
if uploaded is None:
    st.stop()

img_pil = Image.open(uploaded)
img = np.array(img_pil)
img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

H,W = img.shape[:2]

# =========================
# 初期矩形
# =========================
st.sidebar.header("① ROI位置（ドラッグ代用）")

x = st.sidebar.slider("左X", 0, W, int(W*0.2))
y = st.sidebar.slider("上Y", 0, H, int(H*0.3))
w = st.sidebar.slider("幅", 10, W, int(W*0.3))
h = st.sidebar.slider("高さ", 10, H, int(H*0.3))

# =========================
# 頂点編集（重要）
# =========================
st.sidebar.header("② 頂点微調整（台形）")

tlx = st.sidebar.slider("左上X", 0, W, x)
tly = st.sidebar.slider("左上Y", 0, H, y)

trx = st.sidebar.slider("右上X", 0, W, x+w)
try_ = st.sidebar.slider("右上Y", 0, H, y)

brx = st.sidebar.slider("右下X", 0, W, x+w)
bry = st.sidebar.slider("右下Y", 0, H, y+h)

blx = st.sidebar.slider("左下X", 0, W, x)
bly = st.sidebar.slider("左下Y", 0, H, y+h)

pts = np.array([[tlx,tly],[trx,try_],[brx,bry],[blx,bly]], dtype="float32")

# =========================
# 台形補正
# =========================
def warp(img, pts):
    (tl, tr, br, bl) = pts

    w = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    h = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))

    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (w, h))

roi = warp(img_cv, pts)

# =========================
# 色判定
# =========================
st.sidebar.header("③ 色範囲（ImageJ風）")

h_min, h_max = st.sidebar.slider("Hue", 0, 179, (0,30))
s_min, s_max = st.sidebar.slider("Sat", 0, 255, (0,120))
v_min, v_max = st.sidebar.slider("Val", 0, 255, (150,255))

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = (
    (hsv[:,:,0]>=h_min)&(hsv[:,:,0]<=h_max)&
    (hsv[:,:,1]>=s_min)&(hsv[:,:,1]<=s_max)&
    (hsv[:,:,2]>=v_min)&(hsv[:,:,2]<=v_max)
)

mask = mask.astype(np.uint8)*255
ratio = np.mean(mask>0)*100

overlay = roi.copy()
overlay[mask>0] = [255,255,255]

# =========================
# 表示
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="ROI")

with col2:
    st.image(mask, caption="Mask")

with col3:
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"{ratio:.2f}%")

# ROI可視化
vis = img_cv.copy()
cv2.polylines(vis, [pts.astype(int)], True, (0,255,0), 1)

st.subheader("位置確認")
st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))