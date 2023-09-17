import streamlit as st
import numpy as np

from random import choice

from PIL import Image

from style_transfer import stylize, styles
from apply_style_to_nails import apply_style


def style(buffer):

    style = choice(styles)

    with st.spinner("Обработка изображения..."):
        stylized = stylize(buffer, style)

    return stylized


def fuse(buffer, stylized):

    buffer.seek(0)

    pil_img = Image.open(buffer)
    np_img = np.array(pil_img)[:,:,::-1]

    stylized_np = np.array(stylized)[:,:,::-1]

    return apply_style(np_img, stylized_np, 0.65)


st.set_page_config(page_title="Nogotochki", page_icon="💅", layout="wide")

st.title("Nogotochki 💅")

col1, col2 = st.columns(2)

with col1:

    on = st.toggle("Использовать камеру")

    if on:
        img_file_buffer = st.camera_input("Снять фото")
    else:
        img_file_buffer = st.file_uploader("Загрузить фото", type=["png", "jpg"])

    process_button = st.button("Обработать в случайном стиле")

with col2:

    img_placeholder = st.empty()

    if img_file_buffer and process_button:
        with st.spinner("Обработка изображения..."):
            stylized = style(img_file_buffer)
            fused = fuse(img_file_buffer, stylized)

        img_placeholder.image(fused, caption="Результат обработки")
