import  cv2
import tensorflow as tf
from keras.models import load_model
import cvlib as cv
import  numpy   as np

def detect_smile():
    model=load_model('my_smile_model.h5')
    categories=['No Smile','Smile']
    #open the cam
    cam=cv2.VideoCapture(0)
    while (True):
        success,frame=cam.read()

        faces, confidences = cv.detect_face(frame)
        for idx, f in enumerate(faces):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            img = np.copy(frame[startY:endY,startX:endX])

            if (img.shape[0]) < 10 or (img.shape[1]) < 10:
                    continue
            t=[]
            img=cv2.resize(img,(64,64))
            t.append(img)

            t=np.array(t)/255
            y_pred = model.predict(t)
            y_classes = [np.argmax(element) for element in y_pred]
            
            if y_classes[0]==0:
                
                color=(0, 0, 255)
            else:
                color=(0, 255, 0)
                
            cv2.rectangle(frame,(startX, startY),(endX, endY),color,2)

            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame,categories[y_classes[0]],(startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
        cv2.imshow('cam',frame)



            #img=cv2.resize(img,(116,116))
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break


    cam.release()
    cv2.destroyAllWindows()
    #copy all the frame
    #preprocess  the frame
    #predict+print  the result
