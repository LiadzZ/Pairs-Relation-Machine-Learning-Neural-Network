import os
import pandas as pd
import pickle
from keras import backend
from datetime import date
import numpy as np
from ast import literal_eval
import random
#from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import re
import spacy
import pandas as pd
stopwordfile='stopwords.txt'
nlp=spacy.load('en_core_web_sm',disable=['tagger','parser','ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))
def get_stopwords():
    "Return a set of stopwords read in from a file."
    with open(stopwordfile) as f:
        stopwords = []
        for line in f:
            stopwords.append(line.strip("\n"))
    # Convert to set for performance
    stopwords_set = set(stopwords)
    return stopwords_set

stopwords = get_stopwords()

def lemmatize_pipe(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords]
    return lemma_list

def preprocess_pipe(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe

class dataGenerator:
    def __init__(self):
        self.data = None
        self.all_words = None
        self.max_len_question1 = None
        self.max_len_question2 = None
        self.tokenizer = None
        self.vocab_size = None
        self.num_of_rows = None
        self.batch_id = 0
        self.batch = None
        self.batch_length = None
        self.num_of_positive_labels = None
        self.num_of_negative_labels = None
        self.question1 = None
        self.question2 = None
        self.padded_encoded_question1 = None
        self.padded_encoded_question2 = None
        self.label = None
        # self.wrong_answer = None



    def load_files(self,num_of_rows=None):
        print("load activated")
        if num_of_rows:
            self.data = pd.read_csv("p_c_questions_data_3.csv",converters={"p_c_question1": literal_eval ,
                                                                         "p_c_question2": literal_eval,}
                               ,nrows=num_of_rows)
        else:
            self.data = pd.read_csv("p_c_questions_data_3.csv",converters={"p_c_question1": literal_eval ,
                                                                         "p_c_question2": literal_eval,
                                                                        })
        self.num_of_rows = self.data.shape[0]
    def data_preperation(self):
        print("data_preperation Activated")

        question1 = self.data['p_c_question1']
        question2 = self.data['p_c_question2']
        label = self.data['is_duplicate']

        question1 = question1.tolist()
        question2 = question2.tolist()
        label = label.tolist()

        self.label=label
        self.question1 = question1
        self.question2 = question2

        all_words = []
        max_len_question1 = 0
        idx = 0

        for quest in question1:
            for word in quest:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_question1):
                max_len_question1 = idx
            idx = 0

        max_len_question2 = 0
        idx = 0
        for ans in question2:
            for word in ans:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_question2):
                max_len_question2 = idx
            idx = 0

        self.all_words = all_words
        self.max_len_question1 = max_len_question1
        self.max_len_question2 = max_len_question2

        # Encoding 2
        self.vocab_size = len(all_words)

        max_words = self.vocab_size + 5
        t = Tokenizer(num_words=max_words)
        # fit_text = ["The earth is an awesome place live"]
        # words --> integers
        t.fit_on_texts(question1 + question2)
        encoded_question1 = list(t.texts_to_sequences(question1))
        encoded_question2 = list(t.texts_to_sequences(question2))

        self.tokenizer = t
        # Pad-Sequence - Zero Padding
        # self.padded_encoded_description = pad_sequences(encoded_des, maxlen=self.max_len_description, padding='post')
        # self.padded_encoded_review = pad_sequences(encoded_rev, maxlen=self.max_len_review, padding='post')
        self.padded_encoded_question1 = pad_sequences(encoded_question1, maxlen=self.max_len_question1, padding='pre')
        self.padded_encoded_question2 = pad_sequences(encoded_question2, maxlen=self.max_len_question2, padding='pre')
        # self.padded_encoded_wrong_answer = pad_sequences(encoded_wrong_answer, maxlen=self.max_len_answer, padding='post')
        print(self.padded_encoded_question1[0])





    def generate_batch(self):
        print("generate_batch Activated")
        batch=[]
        self.batch_length=self.num_of_rows
        while(True):
            for idx in range(self.num_of_rows):
                batch.append((self.padded_encoded_question1[idx],
                             self.padded_encoded_question2[idx],
                             self.label[idx]))
            np.random.shuffle(batch)
            self.batch_id += 1
            self.batch = batch
            yield batch

    def arrange_batch(self,batch,batch_size):
        print("arrange_batch Activated")
        left = []
        right = []
        label = []
        count = 0
        batch_size = self.num_of_rows
        for k in range(batch_size):
            left.append(batch[k][0])
            right.append(batch[k][1])
            label.append(batch[k][2])
            count += 1
        left = pd.Series(left)
        right = pd.Series(right)
        label = pd.Series(label)
        return {'left':left,'right':right,'label':label}


    def save_batch(self,batch,n_positive,negative_ratio):
        print("save_batch Activated")
        current_directory = os.getcwd()
        file_name = "batches"
        mid_directory = os.path.join(current_directory, file_name)
        if not os.path.exists(mid_directory):
            os.makedirs(mid_directory)



        today = date.today()
        d1 = today.strftime("%d-%m-%Y")

        final_directory_name = file_name + "/" + "QQAbatch_400k_lstm_pre-" + str(self.batch_id) + "-" + str(d1) + "/"
        final_directory = os.path.join(current_directory, final_directory_name)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        new_batch_name = final_directory_name + "batch-" + str(self.batch_id) + "-" + str(d1) + ".pkl"
        print("new_batch_name:",new_batch_name)
        output = open(new_batch_name, "wb")
        pickle.dump(batch, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        tokenizer_file_name = final_directory_name + "tokenizer.pkl"
        output = open(tokenizer_file_name, "wb")
        pickle.dump(self.tokenizer, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        question_file_name = final_directory_name + "question1_encoded.pkl"
        output = open(question_file_name, "wb")
        pickle.dump(self.padded_encoded_question1, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        question_file_name = final_directory_name + "questions1.pkl"
        output = open(question_file_name, "wb")
        pickle.dump(self.question1, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        answer_file_name = final_directory_name + "question2_encoded.pkl"
        output = open(answer_file_name, "wb")
        pickle.dump(self.padded_encoded_question2, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        answer_file_name = final_directory_name + "question2.pkl"
        output = open(answer_file_name, "wb")
        pickle.dump(self.question2, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close




        headers_list = ['num of rows:','num of question1:', 'num of question2:', 'left vector size:',
                   'right vector size:', 'pos:', 'neg:', 'vocab size:']

        values_list = []
        values_list.append(self.num_of_rows)
        values_list.append(int(self.batch_length/2))
        values_list.append(self.batch_length)


        values_list.append(self.max_len_question1)
        values_list.append(self.max_len_question2)

        values_list.append(7)
        values_list.append(7)
        values_list.append(self.vocab_size)
        newList = []
        for index in range(len(values_list)):
            temp = str(headers_list[index]) + str(values_list[index])
            newList.append(temp)

        meta_data_final_directory = final_directory_name + "meta_data.txt"
        output = open(meta_data_final_directory, "w")
        for line in newList:
            output.write(str(line) + "\n")
        output.close()

    def pre_process(self):
        data = pd.read_csv("questions.csv")
        data = pd.DataFrame(data[['question1', 'question2', 'is_duplicate']])
        nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        pattern = re.compile(r"[A-Za-z0-9\-]{0,50}")
        data['clean_question1'] = data['question1'].str.findall(pattern).str.join(' ')
        data['clean_question2'] = data['question2'].str.findall(pattern).str.join(' ')

        data['p_c_question1'] = preprocess_pipe(data['clean_question1'])
        data['p_c_question2'] = preprocess_pipe(data['clean_question2'])

        data.to_csv('p_c_questions_data_2.csv')

    def main(self):
        self.pre_process()
        self.load_files(200000)
        self.data_preperation()
        batch2 = next(self.generate_batch())
        batch2 = self.arrange_batch(batch2,self.batch_length)
        self.save_batch(batch2,self.num_of_rows,1)


a = dataGenerator()
a.main()