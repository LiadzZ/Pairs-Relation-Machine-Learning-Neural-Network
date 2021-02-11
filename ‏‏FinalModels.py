from keras import backend
from keras.layers import  Dropout,Conv1D,Input, Embedding, Dot, Reshape, Dense, Flatten, Concatenate, LSTM, Subtract, Multiply, Add, Lambda , GlobalMaxPool1D
from keras.optimizers import Adam , SGD , Adadelta
from keras.models import Model
from FinalLoadBatch import LoadBatch
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from keras.callbacks import CSVLogger

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from keras import backend as K

plt.style.use('ggplot')
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class MyCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
        print("Custom callback filename: ",filename)

    def on_test_begin(self, logs=None):
        # open csv file
        self.csvFile=open(self.filename,'w')
        fields = ['batch', 'size', 'loss', 'accuracy']
        # fields = ['batch', 'size', 'loss', 'accuracy', 'f1_m', 'precision_m', 'recall_m']

        self.writer = csv.DictWriter(self.csvFile, fieldnames=fields)
        self.writer.writeheader()
        print('test begin')

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        self.writer.writerows([logs])
        #
        # print(logs)
    def on_test_end(self, logs=None):
        # close csv file
        self.csvFile.close()
        print('test end')







class Models:
    def __init__(self,model_name,batch_name):
        self.max_len_description = None
        self.max_len_review = None
        self.max_words = None
        self.batch_name = batch_name
        self.batch = None

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.df = None

        self.set_model_data()
        self.set_model()


    def __repr__(self):
        return {'name': self.model_name}

    def __str__(self):
        return 'Models(name=' + self.model_name + 'batch name=' + self.batch_name + ')'

    def manhattan_dist(self, x, hidden_size=50):
        return K.exp(-K.sum(K.abs(x[:,:hidden_size] - x[:,hidden_size:]), axis=1))
    def embedding_model(self,embedding_size=50, classification=False):
        print("embedding_model Activated")
        self.model_name = "embedding_model"
        # Both inputs are 1-dimensional
        description = Input(name='description', shape=[1])
        review = Input(name='review', shape=[1])

        # Embedding the description , We will enter new descriptions, So they might be longer
        description_embedding = Embedding(name='description_embedding',
                                          input_dim=(self.max_len_description + 50),
                                          output_dim=embedding_size)(description)

        # Embedding the review , review cant be bigger
        review_embedding = Embedding(name='review_embedding',
                                     input_dim=self.max_len_review,
                                     output_dim=embedding_size)(review)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name='dot_product', normalize=True, axes=2)([description_embedding, review_embedding])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape=[1])(merged)

        # If classifcation, add extra layer and loss function is binary cross entropy
        if classification:
            merged = Dense(1, activation='sigmoid')(merged)
            model = Model(inputs=[description, review], outputs=merged)
            model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs=[description, review], outputs=merged)
            model.compile(optimizer='Adam', loss='mse')

        return model

    def embedding_model2(self,embedding_size=50):
        print("embedding_model2 Activated")
        self.model_name = "embedding_model2"
        self.max_words = self.max_words + 1000
        recipe_input = Input(shape=[self.max_len_description], name="description-Input")
        recipe_embedding = Embedding(self.max_words, embedding_size, name="description-Embedding")(recipe_input)
        recipe_vec = Flatten(name="Flatten-description")(recipe_embedding)

        user_input = Input(shape=[self.max_len_review], name="review-Input")
        user_embedding = Embedding(self.max_words, embedding_size, name="review-Embedding")(user_input)
        user_vec = Flatten(name="Flatten-review")(user_embedding)

        conc = Concatenate()([recipe_vec, user_vec])


        fc1 = Dense(128, activation='relu')(conc)
        fc2 = Dense(32, activation='relu')(fc1)
        out = Dense(1,activation='sigmoid')(fc2)
        # model = Model([user_input, book_input], prod)
        model = Model([recipe_input, user_input], out)
        # model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        # opt = SGD(lr=0.01,momentum=0.9)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def embedding_model2_test(self,embedding_size=50):
        print("embedding_model 3 Activated")
        self.model_name = "embedding_model2_test"

        self.max_words = self.max_words + 1000


        recipe_input = Input(shape=[self.max_len_description], name="description-Input")
        recipe_embedding = Embedding(self.max_words, embedding_size, name="description-Embedding")(recipe_input)
        des_conv = Conv1D(256,8,activation='relu',border_mode='valid')(recipe_embedding)
        des_drop = Dropout(0.2)(des_conv)
        #recipe_vec = Flatten(name="Flatten-description")(recipe_embedding)
        des_vec = GlobalMaxPool1D(name='MaxPool-description')(des_drop)

        user_input = Input(shape=[self.max_len_review], name="review-Input")
        user_embedding = Embedding(self.max_words, embedding_size, name="review-Embedding")(user_input)
        rev_conv = Conv1D(256, 8, activation='relu',border_mode='valid')(user_embedding)
        rev_drop = Dropout(0.2)(rev_conv)
        #user_vec = Flatten(name="Flatten-review")(user_embedding)
        rev_vec = GlobalMaxPool1D(name='MaxPool-review')(rev_drop)

        conc = Concatenate(axis=1)([des_vec, rev_vec])


        fc1 = Dense(256, activation='relu')(conc)
        fc2 = Dense(32, activation='relu')(fc1)
        out = Dense(1,activation='sigmoid')(fc2)
        # model = Model([user_input, book_input], prod)
        model = Model([recipe_input, user_input], out)
        # model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
        #opt = SGD(lr=0.0001,momentum=0.9)

        # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model
    def embedding_lstm_model_manhattan_dist(self,embedding_size=5):  # manhattan_dist for Siamese Network
        print("embedding_lstm_model Activated")
        self.model_name = "embedding_lstm_model"
        self.max_words = self.max_words + 1000
        recipe_input = Input(shape=[self.max_len_description], name="description-Input")
        recipe_embedding = Embedding(self.max_words, embedding_size, name="description-Embedding")(recipe_input)
        lstm1 = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(recipe_embedding)


        user_input = Input(shape=[self.max_len_review], name="review-Input")
        user_embedding = Embedding(self.max_words, embedding_size, name="review-Embedding")(user_input)
        lstm2 = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(user_embedding)




        conc = Concatenate(axis=-1)([lstm1, lstm2])



        dist = Lambda(self.manhattan_dist,output_shape=(1,))(conc)

        fc1 = Dense(128, activation='relu', name='dist_conc_layer')(dist)
        fc2 = Dense(64, activation='relu', name='hidden_layer')(fc1)
        fc3 = Dense(32, activation='relu', name='hidden_layer_2')(fc2)
        out = Dense(1, activation='sigmoid', name='out_layer')(fc3)
        model = Model([recipe_input, user_input], out)

        opt = Adadelta(lr=1,rho=0.95)
        model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
        return model

    def embedding_lstm_model(self,embedding_size=50):  # Since this is a siamese network, both sides share the same LSTM
        print("embedding_lstm_model Activated")
        self.model_name = "embedding_lstm_model"
        self.max_words = self.max_words + 1000
        def cosine_distance(vests):
            x, y = vests
            x = backend.l2_normalize(x, axis=-1)
            y = backend.l2_normalize(y, axis=-1)
            return -backend.mean(x * y, axis=-1, keepdims=True)

        def cos_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        shared_lstm = LSTM(64, dropout=0.2, recurrent_dropout=0.2)
        shared_emb = Embedding(self.max_words, embedding_size)

        recipe_input = Input(shape=[self.max_len_description], name="description-Input")
        recipe_embedding = shared_emb(recipe_input)
        lstm1 = shared_lstm(recipe_embedding)
        # recipe_vec = Flatten(name="Flatten-description")(lstm1)

        user_input = Input(shape=[self.max_len_review], name="review-Input")
        user_embedding = shared_emb(user_input)
        lstm2 = shared_lstm(user_embedding)
        # user_vec = Flatten(name="Flatten-review")(lstm2)

        x3 = Subtract()([lstm1, lstm2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([lstm1, lstm1])
        x2_ = Multiply()([lstm2, lstm2])
        x4 = Subtract()([x1_, x2_])

        x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([lstm1, lstm2])

        conc = Concatenate(axis=-1)([x5, x4, x3])


        fc1 = Dense(128, activation='relu', name='conc_layer')(conc)
        fc2 = Dense(64, activation='relu', name='hidden_layer')(fc1)
        fc3 = Dense(32, activation='relu', name='hidden_layer_2')(fc2)
        out = Dense(1, activation='sigmoid', name='out_layer')(fc3)

        model = Model([recipe_input, user_input], out)

        # model.compile(loss="binary_crossentropy", metrics=['accuracy',f1_m,precision_m, recall_m], optimizer=Adam(0.00001))
        model.compile(loss="binary_crossentropy", metrics=['accuracy'],
                      optimizer=Adam(0.00001))
        return model

    def set_model_data(self):
        data = LoadBatch(self.batch_name)
        self.batch = data.batch
        data.print_batch()
        self.tokenizer = data.tokenizer
        self.max_len_description = data.left_vector_max_size


        self.max_len_review = data.right_vector_max_size

        self.max_words = data.vocab_size
        self.df = data.convert_to_dataframe()


    def set_model(self):
        if self.model_name == "embedding_model":
            self.model = self.embedding_model()
        elif self.model_name == "embedding_model2":
            self.model = self.embedding_model2()
        elif self.model_name == "embedding_model2_test":
            self.model = self.embedding_model2_test()
        elif self.model_name == "embedding_lstm_model":
            self.model = self.embedding_lstm_model()
        elif self.model_name == "embedding_lstm_model_manhattan_dist":
            self.model = self.embedding_lstm_model_manhattan_dist()
        else:
            print("Unknown model name")

    def start(self):
        print("start activated")
        kfoldCounter=1
        des = self.batch['left']
        rev = self.batch['right']
        label = self.batch['label']


        kf = KFold(n_splits=10, shuffle=False)

        # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
        accuracy_model = []

        # Iterate over each train-test split
        for train_index, test_index in kf.split(des):
            # Split train-test
            X1_train, X1_test = des.iloc[train_index], des.iloc[test_index]
            X2_train, X2_test = rev.iloc[train_index], rev.iloc[test_index]
            y_train, y_test = label.iloc[train_index], label.iloc[test_index]
            # Train the model


            X1_train = X1_train.tolist()
            X2_train = X2_train.tolist()
            y_train = y_train.tolist()

            encoder = LabelEncoder()
            encoder.fit(y_train)
            encoded_y_train = encoder.transform(y_train)

            X1_test = X1_test.tolist()
            X2_test = X2_test.tolist()
            y_test = y_test.tolist()
            encoded_y_test = encoder.transform(y_test)

            # model = clf.fit(X_train, y_train)
            csv_logger = CSVLogger("QQA_embeddingIGNORE_model_test_2_history_log_fold_400k_10_Epochs fold-" + str(kfoldCounter) + ".csv", append=True)
            self.set_model()
            history = self.model.fit([X1_train, X2_train], encoded_y_train,
                                     batch_size=128,
                                     epochs=20,
                                     verbose=1,
                                     callbacks=[csv_logger]
                                     )




            # pyplot.plot(history.history['rmse'])
            # pyplot.show()



            csv_logger2 = MyCSVLogger("QQA_embedding_model_test_history_log_fold_400k_10_Epochs fold-" + str(kfoldCounter) + "-EVALUATION.csv")

            accuracy_model.append(self.model.evaluate([X1_test, X2_test], encoded_y_test,
                                                      batch_size=16,
                                                      callbacks=[csv_logger2]))
            # self.plot_history(history)

            if kfoldCounter==1:
                # Print the accuracy
                print(self.model.metrics_names)
                print(accuracy_model)

                model.save("400k_model_embedding_TEST-LIAD_3_QQAbatch-1-18-06-2020.h5")
                # model.save("25k_model_lstm_test_CALLBACK_02-08-2020.h5")
                return model

            kfoldCounter = kfoldCounter + 1

        # Print the accuracy
        print(self.model.metrics_names)
        print(accuracy_model)

        model.save("400k_model_embedding_TEST-LIAD_QQAbatch-3-18-06-2020.h5")
        # model.save("25k_model_lstm_test_CALLBACK_02-08-2020.h5")
        return model



    def plot_history(sel,history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

a = Models("embedding_lstm_model","QQAbatch_400k_lstm_pre-1-15-06-2020")
# a = Models("embedding_model2_test","QQAbatch_400k_3-1-15-06-2020")
# a = Models("embedding_lstm_model_pre","QQAbatch_400k_lstm_pre-1-15-06-2020")
# a = Models("embedding_model2","QQAbatch_400k_lstm_pre-1-19-06-2020")
# a = Models("embedding_model2","batch-1-07-05-2020")
# a = Models("embedding_model2","batch-1-02-06-2020")

model = a.model
print(model.summary())
model = a.start()






