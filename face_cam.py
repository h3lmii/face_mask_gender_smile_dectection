import streamlit as st
import cv2
def  main():
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])


if __name__ == '__main__':
    main()
