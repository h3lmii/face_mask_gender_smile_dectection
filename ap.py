import streamlit as st
import cv2
from PIL import Image
import numpy as np
import cvlib as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array



model_gender = load_model('gender_detection.model')
model_mask = load_model('my_model.h5')
model_smile = load_model('my_smile_model.h5')




classes_gender = ['man','woman']
classes_mask = ['without  mask','with mask']
classes_smile = ['No smile',' smile']



    
def main():
    st.title("Gender,MASK,SMILE Detection App :sunglasses: ")
    st.write("**Using the CNN**")

    image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file is not None:
        image = Image.open(image_file).convert('RGB')
        img = np.array(image)
        img_mask = np.array(image)
        img_smile = np.array(image)

        face, confidence = cv.detect_face(img)
        for idx, f in enumerate(face):

            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            face_crop = np.copy(img[startY:endY,startX:endX])
            
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop_mask=cv2.resize(face_crop, (116,116))

            face_crop_smile=cv2.resize(face_crop, (64,64))


            face_crop = face_crop.astype("float") / 255.0
            face_crop_mask = face_crop_mask.astype("float") / 255.0
            face_crop_smile = face_crop_smile.astype("float") / 255.0


            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            face_crop_mask = img_to_array(face_crop_mask)
            face_crop_mask = np.expand_dims(face_crop_mask, axis=0)

            face_crop_smile = img_to_array(face_crop_smile)
            face_crop_smile = np.expand_dims(face_crop_smile, axis=0)
            
            conf = model_gender.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
            conf_mask = model_mask.predict(face_crop_mask)[0]
            conf_smile = model_smile.predict(face_crop_smile)[0]



            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes_gender[idx]
            label = "{}: {:.2f}%".format(label,conf[idx] * 100)

            idx_mask = np.argmax(conf_mask)
            label_mask = classes_mask[idx]
            label_mask = "{}".format(label_mask)

            idx_smile = np.argmax(conf_smile)
            label_smile = classes_smile[idx_smile]
            label_smile = "{}".format(label_smile)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # draw rectangle over face
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
            cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.rectangle(img_mask, (startX,startY), (endX,endY), (255,0,0), 2)
            cv2.putText(img_mask, label_mask, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

            cv2.rectangle(img_smile, (startX,startY), (endX,endY), (255,0,0), 2)
            cv2.putText(img_smile, label_smile, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)



        if st.button("Process"):




            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Gender")
                st.image(img)

            with col2:
                st.header("Mask")
                st.image(img_mask)

            with col3:
                st.header("Smile")
                st.image(img_smile)





if __name__ == '__main__':
    main()
