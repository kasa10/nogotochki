import streamlit as st

from random import choice

from style_transfer import stylize, styles


def style(buffer):

    style = choice(styles)

    with st.spinner("Обработка изображения..."):
        stylized = stylize(buffer, style)

    return stylized


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
        img_placeholder.image(stylized, caption="Результат обработки")
