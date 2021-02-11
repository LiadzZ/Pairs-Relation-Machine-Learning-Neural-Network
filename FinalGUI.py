from keras.models import load_model
from keras import backend
import matplotlib.pyplot as plt
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import pandas as pd
import numpy as np
from FinalLoadBatch import LoadBatch
from keras.preprocessing.sequence import pad_sequences


class FinalGUI():
    def __init__(self):
        self.title = "FinalProject GUI"
        self.root = tk.Tk()
        self.root.title(self.title)
        w = 390  # width for the Tk root
        h = 520  # height for the Tk root

        # get screen width and height
        ws = self.root.winfo_screenwidth()  # width of the screen
        hs = self.root.winfo_screenheight()  # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)

        # set the dimensions of the screen
        # and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.frame = Frame(self.root)
        self.frame.grid()
        self.Output = None
        self.data = None
        self.tokenizer = None
        self.max_len_description = None
        self.model = None

        self.max_len_review = None
        self.max_words = None
        self.df = None




    def rmse(self,y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def Make_prediction(self,input2):
        test = []
        test1 = ["Hello this is my first time"]
        test1.append(input2)

        test3 = self.tokenizer.texts_to_sequences(test1)
        temp = pad_sequences(test3, maxlen=self.max_len_description, padding='post')

        test.append(temp[1])
        test2 = self.tokenizer.sequences_to_texts([test[0]])

        test.append((self.df['right'].iloc[100]).tolist())

        left = []
        right = []
        for x in range(500):
            left.append(test[0])
            temp = (self.df['right'].iloc[x]).tolist()
            right.append(temp)
        query = []
        query.append(left)
        query.append(right)

        print("prepare for predict\n")
        y = self.model.predict(query)
        print(y)
        predictions = np.array([a[0] for a in y])
        recommended = (-predictions).argsort()[:10]
        print("Top 10:")


        for num in recommended:
            sent = self.tokenizer.sequences_to_texts([right[num]])

            self.Output.insert(END,test2)
            self.Output.insert(END, sent)
            self.Output.insert(END, "\n")
            print("Sent1:", test2)

            print("Sent2:", sent)
        return 0

    def Take_input(self,inputtxt):
        INPUT = inputtxt.get("1.0", "end-1c")
        print(INPUT)
        self.Make_prediction(INPUT)



    def start(self):
        self.loadData()

        text = tk.Label(self.root, text="Enter Question:")
        text.grid(row=0, column=1)
        inputtxt = Text(self.root, height=10,
                        width=25,
                        bg="light yellow")
        inputtxt.grid(row=1,column = 1)

        self.Output = Text(self.root, height=50,
                      width=50,
                      bg="light cyan")
        self.Output.grid(row=3,column =1)


        Display = Button(self.root, height=2,
                         width=20,
                         text="Show",
                         # command=lambda: self.Take_input(inputtxt))
                         command=lambda: self.Take_input(inputtxt))
        Display.grid(row=2,column = 1)

        self.root.mainloop()







    def loadData(self):
        # self.data = LoadBatch('batch-1-07-05-2020')
        self.data = LoadBatch('QQAbatch_400k_lstm_pre-1-19-06-2020')
        self.tokenizer = self.data.tokenizer
        self.max_len_description = self.data.left_vector_max_size
       

        self.max_len_review = self.data.right_vector_max_size
        self.max_words = self.data.vocab_size
        self.df = self.data.convert_to_dataframe()
        # self.model = load_model('embedding_model2_07-05-2020.h5')
        self.model = load_model('400k_model_embedding_TEST-LIAD_3_QQAbatch-1-20-06-2020.h5')




def main():
    program = FinalGUI()
    program.start()



main()