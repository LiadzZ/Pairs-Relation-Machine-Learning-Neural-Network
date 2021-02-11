import os
import pandas as pd
import pickle
from keras import backend
from datetime import date
import numpy as np
import random
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer




class dataGenerator:
    def __init__(self):
        self.data = None
        self.num_of_recipes = None
        self.description_list = None
        self.review_list = None
        self.tokenized_review_list = None
        self.tokenized_description_list = None
        self.tokenizer = None
        self.all_words = None
        self.max_len_description = None
        self.max_len_review = None
        self.vocab_size = None
        self.pairs = None
        self.padded_encoded_description = None
        self.padded_encoded_review = None
        self.batch = None
        self.batch_id = 0
        self.batch_length = None
        self.num_of_rows = None
        self.num_of_positive_labels = None
        self.num_of_negative_labels = None
        self.left_ids = []
        self.right_ids = []
        self.unwanted_chars = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%',
                          '=', '#', '*', '+', '\\', '•', '~', '@', '£',
                          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″',
                          '′', 'Â', '█', '½', 'à', '…',
                          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦',
                          '║', '―', '¥', '▓', '—', '‹', '─',
                          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
                          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
                          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤',
                          'ï', 'Ø', '¹', '≤', '‡', '√', ]

        self.unwanted_words = ['\r\n']

    def load_files(self,num_of_recipes=None):
        print("load_files Activated")
        if num_of_recipes:
            data_description = pd.read_csv("RAW_recipes.csv", converters={"ingredients": lambda x: x.strip("[]").split(", "),
                                                              "tags": lambda y: y.strip("[]").split(", ")},
                               nrows=num_of_recipes)
        else: # read all recipes
            data_description = pd.read_csv("RAW_recipes.csv",
                                           converters={"ingredients": lambda x: x.strip("[]").split(", "),
                                                       "tags": lambda y: y.strip("[]").split(", ")})
        self.num_of_recipes = data_description.shape[0]
        data_review = pd.read_csv('RAW_interactions.csv')
        data_description = data_description.rename({'id': 'recipe_id'}, axis='columns')


        df = pd.merge(data_description, data_review, on='recipe_id')
        df.head()

        self.data = pd.DataFrame(df[['description', 'review']])
        self.num_of_rows = self.data.shape[0]


    def clean_data(self):
        '''Pre-Processing and creates 2 lists'''
        print("clean_data Activated")
        self.data.apply(self.clean_text)
        self.data.head()

        self.data.dropna(inplace=True)
        self.data.head()

        self.description_list = list(self.data['description'])
        self.review_list = list(self.data['review'])
        self.num_of_rows=self.data.shape[0]

    def clean_text(self,x):
        x = str(x)
        for char in self.unwanted_chars:
            if char in x:
                x = x.replace(char, f' {char} ')
                #x = x.replace(char, 'TEST')
        for word in self.unwanted_words:
            if word in x:
                x = x.replace(word, '')
        return x

    def data_preperation(self):
        print("data_preperation Activated")
        delimiter = RegexpTokenizer('\s+', gaps=True)  # delimiters matching
        self.tokenized_review_list = [delimiter.tokenize(i) for i in self.review_list]

        self.tokenized_description_list = [delimiter.tokenize(i) for i in self.description_list]



        all_words = []
        max_len_description = 0
        idx = 0

        for recipe in self.tokenized_description_list:
            for word in recipe:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_description):
                max_len_description = idx
            idx = 0

        max_len_review = 0
        idx = 0
        for recipe in self.tokenized_review_list:
            for word in recipe:
                idx += 1
                if word not in all_words:
                    all_words.append(word)
            if (idx > max_len_review):
                max_len_review = idx
            idx = 0

        self.all_words = all_words
        self.max_len_description = max_len_description
        self.max_len_review = max_len_review

        # zipped = zip(description_list,review_list)

        # Encoding 1
        # encoded_description = [one_hot(d, vocab_size) for d in description_list]
        # encoded_review = [one_hot(d, vocab_size) for d in review_list]
        # print(len(encoded_description[0]))

        # Encoding 2
        self.vocab_size = len(all_words)

        max_words = self.vocab_size + 5
        t = Tokenizer(num_words=max_words)

        # words --> integers
        t.fit_on_texts(self.description_list + self.review_list)
        encoded_des = list(t.texts_to_sequences(self.description_list))
        encoded_rev = list(t.texts_to_sequences(self.review_list))
        self.tokenizer = t

        # Pad-Sequence - Zero Padding
        # self.padded_encoded_description = pad_sequences(encoded_des, maxlen=self.max_len_description, padding='post')
        # self.padded_encoded_review = pad_sequences(encoded_rev, maxlen=self.max_len_review, padding='post')
        self.padded_encoded_description = pad_sequences(encoded_des, maxlen=self.max_len_description, padding='pre')
        self.padded_encoded_review = pad_sequences(encoded_rev, maxlen=self.max_len_review, padding='pre')
        print(self.padded_encoded_description[0])



    def pairs_creator(self):
        print("pairs_creator Activated")
        review_index = {review: x for x, review in enumerate(self.review_list)}  # for prediction
        index_review = {x: review for review, x in review_index.items()}

        print("number of reviews:",len(index_review))
        #print(len(review_index))

        description_index = {description: y for y, description in enumerate(set(self.description_list))}
        index_description = {y: description for description, y in description_index.items()}

        print("number of description:",len(index_description))
        #print(len(description_index))

        review_byIndex_list = []
        for x in (self.review_list):
            review_byIndex_list.append(review_index[x])

        description_byIndex_list = []
        for x in (self.description_list):
            description_byIndex_list.append(description_index[x])

        pairs = []
        for x in range(len(review_byIndex_list)):
            pairs.append((description_byIndex_list[x], review_byIndex_list[x]))

        print("Description Example: ",index_description[pairs[0][0]])
        print("Review Example: ", index_review[pairs[0][1]])
        # pairs by index --> (5,17) (description,review)
        self.pairs = pairs

    def rmse(self,y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

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
            for idx, (description_id, review_id) in enumerate(random.sample(self.pairs, n_positive)):
                self.left_ids.append(description_id)
                self.right_ids.append(review_id)
                positive_samples.append(
                    (self.padded_encoded_description[description_id], self.padded_encoded_review[review_id], pos_label))

            idx += 1

            while idx < batch_size:
                # random selection
                random_description = random.randrange(len(set(self.description_list)))
                random_review = random.randrange(len(self.review_list))

                # Check to make sure this is not a positive example
                if (random_description, random_review) not in self.pairs:
                    self.left_ids.append(random_description)
                    self.right_ids.append(random_review)
                    negative_samples.append((self.padded_encoded_description[random_description],
                                             self.padded_encoded_review[random_review], neg_label))
                    idx += 1

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

        final_directory_name = file_name + "/" + "batch-" + str(self.batch_id) + "-" + str(d1) + "/"
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

        reviews_file_name = final_directory_name + "reviews_encoded.pkl"
        output = open(reviews_file_name, "wb")
        pickle.dump(self.padded_encoded_review, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        reviews_file_name = final_directory_name + "reviews.pkl"
        output = open(reviews_file_name, "wb")
        pickle.dump(self.review_list, output, protocol=pickle.HIGHEST_PROTOCOL)
        output.close

        headers_list = ['num of recipe:', 'num of description:', 'num of review:', 'left vector size:',
                   'right vector size:', 'pos:', 'neg:', 'vocab size:']

        values_list = []
        values_list.append(self.num_of_recipes)
        values_list.append(len(set(self.left_ids)))
        values_list.append(len(set(self.right_ids)))

        values_list.append(self.max_len_description)
        values_list.append(self.max_len_review)
        values_list.append(self.num_of_positive_labels)
        values_list.append(self.num_of_negative_labels)
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

    def load_batch(self,batch_dir):
        print("load_batch Activated")
        dir_name = "batches/" + batch_dir
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




    def main(self):
        #self.load_files(500)
        self.load_files(25000)
        self.clean_data()
        self.data_preperation()
        self.pairs_creator()
        print("LIAD TEST: ",self.num_of_rows)
        batch2 = next(self.generate_batch(self.num_of_rows,1))
        batch2 = self.arrange_batch(batch2,self.batch_length)
        self.save_batch(batch2,self.num_of_rows,1)
        # batch1 = next(self.generate_batch(1000,1))  # 2000
        # batch1 = self.arrange_batch(batch1,self.batch_length)
        # self.save_batch(batch1,1000,1)

        # self.load_batch('batch-1-21-04-2020')
        # print(self.batch['left'])
        # print(self.tokenizer)


        # batch1 = self.arrange_batch(batch1,self.batch_length)
        # self.save_batch(batch1)



        # batch2 = self.generate_batch(n_positive=1000,negative_ratio=1.5)  # 2500
        # batch3 = self.generate_batch(n_positive=3000,negative_ratio=1)  # 6000
        # batch4 = self.generate_batch(n_positive=3000,negative_ratio=1.5)  # 7500



a = dataGenerator()
a.main()