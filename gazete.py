import tweepy
import scipy
import tensorflow as tf
import os
import time
import pandas as pd     
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.feature_extraction import stop_words
from collections import Counter 
from IPython.display import display
from sklearn.datasets import fetch_20newsgroups
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn import preprocessing
import csv

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

my_csv = pd.read_csv("articles1.csv")
column = my_csv.content
print(len(column))


sorted(list(stop_words.ENGLISH_STOP_WORDS))[:20]
 
tweets = column[20:35]

split_it =  list()
raw_sentences = list()
firs_word = list()
intquot = ["http", "@", "&amp", "…", "…"]
for tweet in tweets[:50]:
    tweet = tweet
    tweet = tweet.replace("RT", "")
    tweet = tweet.replace("the U.S.", "the US")
    split_it = split_it+ tweet.split()  
    for q in intquot:
        if(not q in tweet): 
            raw_sentences = raw_sentences + tweet.split('.')  

begin = list()
end = list()

for tweet in tweets: 
    this = tweet.split() 
    for i in range(len(this)):
        if ("http" in this[i]) or  ("@" in this[i]) or ("&amp" in this[i]) or ("RT" in this[i]):
            this[i] = ''
        if(this[i] in stop_words.ENGLISH_STOP_WORDS):
            # print("STOP WORD FOUND")
            this[i] = ''  
    while("" in this) : 
        this.remove("")   
    begin.append(this[0])
    end.append(this[len(this)-1])

words = []
quot = ["...", "\"", ".", "”", "…", "…", "RT", ","]
for i in range(len(split_it)):
    for q in quot:
        if(q in split_it[i]): 
            split_it[i] = split_it[i].replace(q,"") 
    for q in intquot:
        if( q in  split_it[i]): 
            split_it[i] = ''  
    if("15M" in split_it[i]): 
        split_it[i] = split_it[i].replace("15M","5M")

         
    if split_it[i] != '.': # because we don't want to treat . as a word
        words.append(split_it[i]) 


  
dm = ["suit,", ]   
quot = ["...", "\"", ".", "”", "…", "…", "RT", ",", "\'"]

for i in range(len(raw_sentences)): 
    for q in intquot:
        if(q in raw_sentences[i]): 
            raw_sentences[i] = '' 
    for q in quot:
        if(q in raw_sentences[i]): 
            raw_sentences[i] = '' 
    # if "suit," in raw_sentences[i]: 
    #     raw_sentences[i] = ""    
            
words = set(words) # so that all duplicate words are removed
word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words
print("policy" in words)


for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())


data = []
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                data.append([word, nb_word]) 

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp
x_train = [] # input word
y_train = [] # output word 
for data_word in data: 
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train.shape, y_train.shape)

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

print("Session starting")
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10000

#*************************************
# train for n_iter iterations
# for _ in range(n_iters):
#     sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
#     print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
#*************************************

def euclidean_dist(vec1, vec2):
    # return scipy.spatial.distance.cdist(vec1, vec2, 'euclidean') 
    return np.sqrt(np.sum((vec1-vec2)**2))
def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    order = []
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
            order.append(min_index)
    return order

vectors = sess.run(W1 + b1)
# print(vectors)
# print(vectors[ word2int['president'] ])  

# word = 'president'
# for k in range (10):
#     res = find_closest(word2int[word], vectors) 
#     print(int2word[res])
#     # for i in range(len(res)):
#     #     print(str(i) + "= " + int2word[res[i]])
#     print("break \n")
        
sentence = word
# first_words[random.randint(0,len(first_words-1))]

for b in range(len(begin)):
    if begin[b] not in words or "#" in begin[b]:
        begin[b] = "" 
    
while '' in begin:
    begin.remove('')

for e in range(len(end)):
    if end[e] not in words:
        end[e] = "" 
while '' in end:
    end.remove('')


# for i in range(20):
#     s_len = random.randint(5,30)
#     # sentence = first_words[random.randint(0,len(first_words)-1)]
#     # word = split_it[random.randint(0,len(split_it)-1)]
#     # word = firs_word[random.randint(0,len(firs_word)-1)]
#     word = begin[random.randint(0,len(begin)-1)]
#     sentence = word
#     for i in range(s_len):
#         res = find_closest(word2int[word], vectors)
#         # word = split_it[random.randint(0,len(split_it)-1)]
#         # word = int2word[find_closest(word2int[word], vectors)] 
#         pre = int2word[res[len(res)-1]]
#         i = 2
#         while pre in sentence and len(res)-i >= 0: 
#             pre = int2word[res[len(res)-i]]
#             i = i+1
#         word = pre
#         sentence = sentence + " " + word
#     print(sentence + "\n")



# prepare_for_w2v('data/War and Peace by Leo Tolstoy (ru).txt', 'train_war_and_peace_ru.txt', 'russian')
# model_wp = train_word2vec('train_war_and_peace_ru.txt')
model_wp = vectors
words_wp = []
embeddings_wp = []
for word in split_it:
    embeddings_wp.append(word2int[word])
    words_wp.append(word)
    

tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)




def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.show()

# res = []
# for i in vectors[ word2int['president'] ]:
#     res.append(int2word[int(i)])
# print(res)
#---------------------------------------------------
 
# model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# vectors = model.fit_transform(vectors)
# normalizer = preprocessing.Normalizer()
# vectors =  normalizer.fit_transform(vectors, 'l2')
# fig, ax = plt.subplots()
# for word in words:
#     # print(word, vectors[word2int[word]][1])
#     ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
# plt.show()


# print(int2word[find_closest(word2int['president'], vectors)])



# def write_tweets(contweets):
#     #write tweets
#     f= open("trtweets.txt","w+")
#     for line in contweets: 
#             line = line.replace("  ", " ")
#             # api.update_status(line)
#             # time.sleep(20)
#             f.write(line + "\n\n")
#     f.close() 

# # Pass the split_it list to instance of Counter class. 
# Counter = Counter(split_it) 
  
# # most_common() produces k frequently encountered 
# # input values and their respective counts. 
# most_occur = Counter.most_common(50) 
  
# print(most_occur) 
 
 