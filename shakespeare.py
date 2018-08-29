from PIL import Image
import pytesseract
import sympy
import pylatexenc
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, Embedding
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical, plot_model
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from faker import Faker
import random
import csv
import sys  
from nltk import sent_tokenize, word_tokenize, pos_tag
from gensim.models import Word2Vec, KeyedVectors
import os
import h5py
import pydot 
import graphviz
import random
from nltk.corpus import wordnet as wn
import string

input_seq = []
output_seq = []
raw_text = ""
original = open("/Users/parthpatel/Desktop/original.txt","r",encoding='utf-8')
lines = original.readlines()
original.close()
for line in lines:
	t = line.lower()
	for c in string.punctuation:
		t = t.replace(c,"")
	input_seq.append(t)
	raw_text+=' '+t
modern = open("/Users/parthpatel/Desktop/modern.txt","r",encoding='utf-8')
lines2 = modern.readlines()
modern.close()
for line in lines2:
	t2 =line.lower()
	for c in string.punctuation:
		t2 = t2.replace(c,"")
	output_seq.append('S '+t2+' E')
	raw_text+=' '+t2
raw_text+=' '+'S'+' '+'E'
vocab_list = word_tokenize(raw_text)
np.save("/Users/parthpatel/Desktop/vocab_list.npy",vocab_list)
vocab = sorted(set(vocab_list))
vocab_size = len(vocab)
#word_to_id = dict.fromkeys(vocab,1)
#id_to_word = dict.fromkeys(np.arange(vocab_size),"")
word_to_id = {}
id_to_word = {}
x = 1;
for i in vocab:
	word_to_id[i] = x
	id_to_word[x] = i
	x=x+1
id_to_word[vocab_size+1] = '<unk>'
word_to_id['<unk>'] = vocab_size+1
x = np.zeros((len(input_seq),35))
y = np.zeros((len(input_seq),35))
y_target = np.zeros((len(input_seq),35,vocab_size+1))
for i in range(len(input_seq)):
	sequence = np.zeros((1,35))
	z=0
	for e in word_tokenize(input_seq[i]):
		if(z < 35):
			if(e in vocab):
				index = word_to_id[e]
				sequence[0][z] = index
			else:
				sequence[0][z] = word_to_id['<unk>']
			z=z+1
	x[i] = sequence[0]
for i in range(len(output_seq)):
	sequence = np.zeros((1,35))
	z=0
	for e in word_tokenize(output_seq[i]):
		if(z < 35):
			if(e in vocab):
				index = word_to_id[e]
				sequence[0][z] = index
			else:
				sequence[0][z] = word_to_id['<unk>']
			z=z+1
	y[i] = sequence

for a in range(len(output_seq)):
	r = np.zeros((35,vocab_size+1))
	for b in range(35):
		seq = y[a]
		fill = np.zeros((1,vocab_size+1))
		if(b < 34):
			fill[0][int(y[a][b+1])] = 1
		r[b] = fill
	y_target[a] = r
np.save("/Users/parthpatel/Desktop/en6.npy",x)
np.save("/Users/parthpatel/Desktop/de6.npy",y)
np.save("/Users/parthpatel/Desktop/tar6.npy",y_target)


encoder_inputs = Input(shape = (50,))
embed = Embedding(input_dim=vocab_size+1,output_dim=64,mask_zero=True,input_length=50)(encoder_inputs)
encoder = Bidirectional(LSTM(64,return_state = True))
encoder_outputs, state_h_for, state_c_for,state_h_back,state_c_back = encoder(embed) 
state_h = Concatenate()([state_h_for,state_h_back])
state_c = Concatenate()([state_c_for,state_c_back])
hidden_layer = Dense(64,activation = None)
encoder_dense = Dense(3, activation='softmax')
encoder_outputs = hidden_layer(encoder_outputs)
encoder_outputs = encoder_dense(encoder_outputs)
model = Model(inputs=encoder_inputs,outputs= encoder_outputs)
opt = RMSprop(lr=0.001)
#model.load_weights("Classification17.h5")
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(x,y,epochs=6,batch_size=100, shuffle=True, validation_split = 0.1)
model.save_weights("Classification17.h5")


#fill = np.zeros((1,50))
#fill[0] = x[67]
#print(model.predict(fill))
#print(y[67])
#plot_model(model, to_file='model.png')
texto = "wow this was amazing"


sequence = np.zeros((1,50))
z=0
for e in word_tokenize(texto.lower()):
	if(z < 50):
		if(e in word_to_id):
			index = word_to_id[e]
		else:
			index = word_to_id['<unk>']
		sequence[0][z] = index
		z=z+1
print(sequence)
#print(x)

print(model.predict(sequence))
