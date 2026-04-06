"""Мінімальна сторінка — якщо вона не відкривається, проблема в Python/Streamlit, не в app.py."""
import streamlit as st

st.set_page_config(page_title="Тест", layout="wide")
st.title("OK — Streamlit працює")
st.write("Якщо ви це бачите, запускайте основний `app.py`.")
