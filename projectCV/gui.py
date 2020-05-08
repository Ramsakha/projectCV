# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:40:13 2020

@author: Ram
"""

from tkinter import *
import os
from PIL import ImageTk,Image


root = Tk()

root.geometry('{}x{}'.format(900, 610))

root.title('my face recognition model')

root.iconbitmap('favicon.ico')

svalue = StringVar()


#e = Entry(root,width=35,borderwidth = 5)
#e.grid(row = 0,column=1)#,padx=100,pady=100)
#e.pack(side='left')


my_img= ImageTk.PhotoImage(Image.open('facial-recognition-use-triggers.jpg'))
my_label = Label(image=my_img)
#my_label.grid(row=0,column=0,rowspan=3)
my_label.pack(side="left")


w = Entry(root,textvariable=svalue) # adds a textarea widget
w.pack()
w.place(x=500,y=300)

def train_fn():
   name = w.get()
   os.system('python starting.py %s'%name)

def recog_fn():
   os.system('python end.py')

train_button = Button(root,text="train", command=train_fn)
#train_fisher_button.grid(row=1,column=1,padx=100,pady=100)
train_button.pack()
train_button.place(x=599,y=350)
recog_button = Button(root,text="recognize", command=recog_fn)
#recog_fisher_button.grid(row=2,column=1,padx=100,pady=100)
recog_button.pack()
recog_button.place(x=565,y=400)



root.mainloop()