import tkinter as tk
from tkinter import messagebox

import sys
from detect_gender_webcam import *
from cam import *
from smile_cam  import *

import  os

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

blockPrint()

window = tk.Tk()
window.title("Face system")
l = tk.Label(window, text = 'Ensemble Projects by helmi',font=("Algerian", 25))
l.place(x = 220,y = 10) 
b2=tk.Button(window,text="GENDER DETECTION",font=("Algerian",20),bg='green',fg='white',command=detect_gender)
b2.place(x=10, y=100)
b=tk.Button(window,text="FACE MASK DETECTION",font=("Algerian",20),bg='green',fg='white',command=detect_mask)
b.place(x=300, y=100)
b1=tk.Button(window,text="SMILE DETECTION",font=("Algerian",20),bg='green',fg='white',command=detect_smile)
b1.place(x=640, y=100)


b3=tk.Button(window,text="EXIT",font=("Algerian",20),bg='black',fg='white',command=window.destroy)
b3.place(x=800, y=230)

window.geometry("900x300")
window.mainloop()
