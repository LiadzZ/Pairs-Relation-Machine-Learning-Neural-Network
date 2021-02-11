import os
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer

class LoadBatch:
    def __init__(self,dir_name):
        self.batch_name = dir_name
        self.batch = None
        self.tokenizer = None
        self.padded_encoded_review = None
        self.review_list = None
        self.num_of_recipes = None
        self.num_of_scenario = None
        self.num_of_requirement = None
        self.left_vector_max_size = None
        self.right_vector_max_size = None
        self.num_of_pos = None
        self.num_of_neg = None
        self.vocab_size = None
        self.load()

    def __repr__(self):
        return {'name': self.batch_name}

    def __str__(self):
        return 'LoadBatch(name=' + self.batch_name + ')'


    def load(self):
        print("load_batch Activated")
        dir_name = "batches/" + self.batch_name
        arr = os.listdir(dir_name)
        print(arr)
        for f in arr:
            if 'batch' in f:
                f_name = dir_name + "/" + f
                with open(f_name, 'rb') as file:
                    self.batch = pickle.load(file)
            elif 'tokenizer' in f:
                f_name = dir_name + "/" + f
                with open(f_name, 'rb') as file:
                    self.tokenizer = pickle.load(file)
            elif 'reviews_encoded' in f:
                f_name = dir_name + "/" + f
                with open(f_name, 'rb') as file:
                    self.padded_encoded_review = pickle.load(file)
            elif 'reviews' in f:
                f_name = dir_name + "/" + f
                with open(f_name, 'rb') as file:
                    self.review_list = pickle.load(file)
            elif 'meta_data' in f:
                f_name = dir_name + "/" + f
                data=[]
                with open(f_name, 'r') as file:
                    for line in file:
                        temp = line
                        temp = temp.split(':')[1]
                        temp = temp.split('\n')[0]
                        data.append(temp)
                self.num_of_recipes = data[0]
                self.num_of_scenario = data[1]
                self.num_of_requirement = data[2]
                self.left_vector_max_size = int(data[3])
                self.right_vector_max_size = int(data[4])
                self.num_of_pos = data[5]
                self.num_of_neg = data[6]
                self.vocab_size = int(data[7])
                #self.get_vocab_size()
            else:
                print("Unknown file - ", f)
                continue

    def print_batch(self):
        print("print_batch details Activated:")
        print("batch class's:",self.batch.keys())

    def get_vocab_size(self):
        all_words = []
        max_len_description = 0
        idx = 0
        left_list = list(self.batch['left'])
        right_list = list(self.batch['right'])
        for val in left_list:
            for word in val:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_description):
                max_len_description = idx
            idx = 0

        max_len_review = 0
        idx = 0
        for val in right_list:
            for word in val:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_review):
                max_len_review = idx
            idx = 0


        self.left_vector_max_size = max_len_description
        self.right_vector_max_size = max_len_review
        self.vocab_size = len(all_words)-1


    def convert_to_dataframe(self,left_name='left',right_name='right'):
        print("convert_to_dataframe  Activated:")
        return pd.DataFrame({left_name:self.batch['left'] , right_name:self.batch['right'], 'label':self.batch['label']})


# a = LoadBatch("batch-1-07-05-2020")
# a.print_batch()