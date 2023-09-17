import streamlit as st

from nails_segmantation_inference import segment_nails


on = st.toggle("Использовать камеру")

if on:
    img_file_buffer = st.camera_input("Снять фото")
else:
    img_file_buffer = st.file_uploader("Загрузить фото")

if img_file_buffer is not None:

    bytes_data = img_file_buffer.getvalue()

    st.image(bytes_data)
