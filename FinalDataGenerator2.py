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

from joblib import Parallel, delayed
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
        self.max_len_question = None
        self.max_len_answer = None
        self.tokenizer = None
        self.vocab_size = None
        self.num_of_rows = None
        self.batch_id = 0
        self.batch = None
        self.batch_length = None
        self.num_of_positive_labels = None
        self.num_of_negative_labels = None
        self.question = None
        self.answer = None
        self.wrong_answer = None



    def load_files(self,num_of_rows=None):
        print("load activated")
        if num_of_rows:
            self.data = pd.read_csv("precovid_data.csv",converters={"p_c_question": literal_eval , "p_c_answer": literal_eval,"p_c_wrong_answer":literal_eval}
                               ,nrows=num_of_rows)
        else:
            #data = pd.read_csv("precovid_data.csv",converters={"p_c_question": lambda x: x.strip("[]").split(", ")})
            self.data = pd.read_csv("precovid_data.csv",converters={"p_c_question": literal_eval , "p_c_answer": literal_eval,"p_c_wrong_answer":literal_eval})
        self.num_of_rows = self.data.shape[0]
    def data_preperation(self):
        print("data_preperation Activated")

        question = self.data['p_c_question']
        answer = self.data['p_c_answer']
        wrong_answer = self.data['p_c_wrong_answer']
        question = question.tolist()
        answer = answer.tolist()
        wrong_answer = wrong_answer.tolist()

        self.question = question
        self.answer = answer
        self.wrong_answer = wrong_answer



        all_words = []
        max_len_question = 0
        idx = 0

        for quest in question:
            for word in quest:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_question):
                max_len_question = idx
            idx = 0

        max_len_answer = 0
        idx = 0
        for ans in answer:
            for word in ans:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_answer):
                max_len_answer = idx
            idx = 0

        idx = 0
        for ans in wrong_answer:
            for word in ans:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_answer):
                max_len_answer = idx
            idx = 0

        self.all_words = all_words
        self.max_len_question = max_len_question
        self.max_len_answer = max_len_answer

        # Encoding 2
        self.vocab_size = len(all_words)

        max_words = self.vocab_size + 5
        t = Tokenizer(num_words=max_words)
        # words --> integers
        t.fit_on_texts(question + answer + wrong_answer)
        encoded_question = list(t.texts_to_sequences(question))
        encoded_answer = list(t.texts_to_sequences(answer))
        encoded_wrong_answer = list(t.texts_to_sequences(wrong_answer))

        self.tokenizer = t
        # Pad-Sequence - Zero Padding
        # self.padded_encoded_description = pad_sequences(encoded_des, maxlen=self.max_len_description, padding='post')
        # self.padded_encoded_review = pad_sequences(encoded_rev, maxlen=self.max_len_review, padding='post')
        self.padded_encoded_question = pad_sequences(encoded_question, maxlen=self.max_len_question, padding='post')
        self.padded_encoded_answer = pad_sequences(encoded_answer, maxlen=self.max_len_answer, padding='post')
        self.padded_encoded_wrong_answer = pad_sequences(encoded_wrong_answer, maxlen=self.max_len_answer, padding='post')
        print(self.padded_encoded_question[0])



    def generate_batch(self, n_positive=None, negative_ratio=1):
        print("generate_batch Activated")
        if not n_positive:
            n_positive = self.num_of_rows

        batch_size = n_positive * (1 + negative_ratio)
        self.batch_length = batch_size

        self.num_of_positive_labels = n_positive
        self.num_of_negative_labels = batch_size - n_positive
        neg_label = 0
        pos_label = 1

        # This creates a generator
        while True:
            positive_samples = []
            negative_samples = []
            # randomly choose positive examples
            print("n_positive: ",n_positive)
            print("num_of_positive_labels: ",self.num_of_positive_labels)
            print("num_of_negative_labels: ",self.num_of_negative_labels)
            print("self.num of rows: ",self.num_of_rows)

            for idx in range(n_positive):
                positive_samples.append(
                    (self.padded_encoded_question[idx], self.padded_encoded_answer[idx], pos_label))
                negative_samples.append(
                    (self.padded_encoded_question[idx], self.padded_encoded_wrong_answer[idx], neg_label))
            batch = positive_samples + negative_samples
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

        final_directory_name = file_name + "/" + "QAbatch-" + str(self.batch_id) + "-" + str(d1) + "/"
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

        question_file_name = final_directory_name + "question_encoded.pkl"
        output = open(question_file_name, "wb")
        pickle.dump(self.padded_encoded_question, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        question_file_name = final_directory_name + "questions.pkl"
        output = open(question_file_name, "wb")
        pickle.dump(self.question, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        answer_file_name = final_directory_name + "answer_encoded.pkl"
        output = open(answer_file_name, "wb")
        pickle.dump(self.padded_encoded_answer, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        answer_file_name = final_directory_name + "answer.pkl"
        output = open(answer_file_name, "wb")
        pickle.dump(self.answer, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        wrong_answer_file_name = final_directory_name + "wrong_answer_encoded.pkl"
        output = open(wrong_answer_file_name, "wb")
        pickle.dump(self.padded_encoded_answer, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        wrong_answer_file_name = final_directory_name + "wrong_answer.pkl"
        output = open(wrong_answer_file_name, "wb")
        pickle.dump(self.wrong_answer, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close


        headers_list = ['num of rows:','num of question:', 'num of answer:', 'left vector size:',
                   'right vector size:', 'pos:', 'neg:', 'vocab size:']

        values_list = []
        values_list.append(self.num_of_rows)
        values_list.append(int(self.batch_length/2))
        values_list.append(self.batch_length)


        values_list.append(self.max_len_question)
        values_list.append(self.max_len_answer)
        values_list.append(self.num_of_positive_labels)
        values_list.append(self.num_of_negative_labels)
        values_list.append(self.vocab_size)
        print("--------------------------------------------------------------------------")
        print(values_list)
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
        data = pd.read_csv("general.csv")
        data = pd.DataFrame(data[['question', 'answer', 'wrong_answer']])
        # nlp=spacy.load('en_core_web_sm',disable=['tagger','parser','ner'])
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))
        pattern = re.compile(r"[A-Za-z0-9\-]{2,50}")
        data['clean_question'] = data['question'].str.findall(pattern).str.join(' ')
        data['clean_answer'] = data['answer'].str.findall(pattern).str.join(' ')
        data['clean_wrong_answer'] = data['wrong_answer'].str.findall(pattern).str.join(' ')

        data['p_c_question']=preprocess_pipe(data['clean_question'])
        data['p_c_answer']=preprocess_pipe(data['clean_answer'])
        data['p_c_wrong_answer']=preprocess_pipe(data['clean_wrong_answer'])

        #
        # data['p_c_question']=preprocess_parallel(data['clean_question'])
        # data['p_c_answer']=preprocess_parallel(data['clean_answer'])
        # data['p_c_wrong_answer']=preprocess_parallel(data['clean_wrong_answer'])

        data.to_csv('precovid_data.csv')
    def main(self):
        #self.load_files(500)
        self.pre_process()
        self.load_files(130000)
        self.data_preperation()

        batch2 = next(self.generate_batch(130000,1))
        batch2 = self.arrange_batch(batch2,self.batch_length)
        self.save_batch(batch2,self.num_of_rows,1)


a = dataGenerator()
a.main()