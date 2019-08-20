# DL and Glove
# Combining features and then applying this thing
# Implement UNDER sampling and both sampling -->>>
import numpy as np
import ast
import pandas as pd
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
# Making UrlsData.csv--->>>
'''
with open('Urls_data.txt') as json_file:
    data = json_file.readlines()
    
user_id=[]
title=[]
link=[]
description=[]
long_desc=[]
brand=[]
entities=[]
tags = []

count=0
for i in data:
    count+=1
    x = ast.literal_eval(i)
    print(count)
    user_id.append(x['id'])
    try:
        title.append(x['title'])
    except:
        title.append('none')
    try:
        link.append(x['link'])
    except:
        link.append('none')
    try:   
        description.append(x['description'])
    except:
        description.append('none')
    try:
        long_desc.append(x['long_description'])
    except:
        long_desc.append('none')
    try:
        brand.append(x['brand'])
    except:
        brand.append('none')
    try:
        tags.append(x['tags'])
    except:
        tags.append('none')
    try:
        entities.append(x['entities'])
    except:
        entities.append('none')
    

urls_data = pd.DataFrame({
              'id': user_id,
              'title': title,
              'link': link,
              'desc': description,
              'long_desc': long_desc,
              'brand': brand,
              'tags': tags,
              'entities': entities
              })  

urls_data.to_csv('urls_data.csv')

ast.literal_eval(data[109443]).keys()
'''
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
# Combining UserIdtoUrl folder (will be used for train and test data features!) 
# inner merging it with train data making intermediate training data -->>>
'''
user_id=[]
url=[]
count=0

for i in range(0,10):
    print(i)
    j = '0000'+str(i)    
    with open('UserIdToUrl/part-'+j) as file:
        data = file.readlines()
    for i in data[1:]:
        count+=1
        print(count)
        print(j)
        av = i.split(',')
        user_id.append(str(av[0]))
        url.append(str(av[1]))

for i in range(10,12):
    print(i)
    j = '000'+str(i)    
    with open('UserIdToUrl/part-'+j) as file:
        data = file.readlines()
    for i in data[1:]:
        count+=1
        print(count)
        print(j)
        av = i.split(',')
        user_id.append(str(av[0]))
        url.append(str(av[1]))
print('dataframe--')     
user_id_url_data = pd.DataFrame({
              'userid': user_id,
              'url': url
              })  
print('dataframe to csv--')     
user_id_url_data.to_csv('user_id_url_data.csv')

UserIdToGender_Train = pd.read_csv('UserIdToGender_Train.csv')
print('string me convert--')     
UserIdToGender_Train = UserIdToGender_Train.astype(str)
print('merge--')     

df_merged = pd.merge(user_id_url_data, UserIdToGender_Train, on='userid')
print('mergefile to csv--')     
df_merged.to_csv('df_merged.csv')


test_df = pd.read_csv('UserId_Test.csv')

df_merged_nonduplicates_train = df_merged.drop_duplicates()

df_merged_nonduplicates_train.to_csv('df_merged_nonduplicates_train.csv')
'''
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

user_id_url_data = pd.read_csv('user_id_url_data.csv')
user_id_url_data = user_id_url_data.loc[:, ~user_id_url_data.columns.str.contains('^Unnamed')]
user_id_url_data = user_id_url_data.astype(str)
UserIdToGender_Train = pd.read_csv('UserIdToGender_Train.csv')
UserIdToGender_Train = UserIdToGender_Train.loc[:, ~UserIdToGender_Train.columns.str.contains('^Unnamed')]
UserIdToGender_Train = UserIdToGender_Train.astype(str)
urls_data = pd.read_csv('urls_data.csv')
urls_data = urls_data.loc[:, ~urls_data.columns.str.contains('^Unnamed')]
urls_data = urls_data.astype(str)

#len(urls_data.url.unique())
def preprocess_strip(x):
    x= str(x)
    x= x.strip()
    return x

user_id_url_data['url'] = user_id_url_data['url'].map(preprocess_strip)
# df_merged = Contain data for both train and test--
df_merged = pd.merge(user_id_url_data, UserIdToGender_Train, on='userid')
df_merged['url'] = df_merged['url'].map(preprocess_strip)
df_merged = df_merged.drop_duplicates()
# This will be only used only for training-->>>
df_merged_train = pd.merge(df_merged, urls_data, on='url')


# To CSVs-->>>
'''
user_id_url_data.to_csv('user_id_url_data.csv')
UserIdToGender_Train.to_csv('UserIdToGender_Train.csv')
urls_data.to_csv('urls_data.csv')
df_merged.to_csv('df_merged.csv')
df_merged_train.to_csv('df_merged_train.csv')
'''
#==

# Making Final Training datsets-->>>
df_merged_train_final = df_merged_train.loc[:,['desc','gender']]
df_merged_train_final = df_merged_train_final.drop_duplicates()
df_merged_train_final = df_merged_train_final.astype(str)
#df_merged_train_final['desc'] = df_merged_train['desc']+df_merged_train['desc']
#df_merged_train_final.to_csv('df_merged_train_final.csv')

#len(df_merged_train_final[df_merged_train_final['desc']=='none'])

'''# Under Sampling-->>
# Class count
count_class_M, count_class_F = df_merged_train_final.gender.value_counts()
# Divide by class
df_class_M = df_merged_train_final[df_merged_train_final['gender'] == 'M']
df_class_F = df_merged_train_final[df_merged_train_final['gender'] == 'F']
df_class_M_under = df_class_M.sample(count_class_F+20000,random_state=42)
df_merged_train_final = pd.concat([df_class_F, df_class_M_under], axis=0)
print('Random over-sampling:')
print(df_merged_train_final.gender.value_counts())'''

# Class count Under
count_class_M, count_class_F = df_merged_train_final.gender.value_counts()
# Divide by class
df_class_M = df_merged_train_final[df_merged_train_final['gender'] == 'M']
df_class_F = df_merged_train_final[df_merged_train_final['gender'] == 'F']
df_class_M_under = df_class_M.sample(count_class_F+25000,random_state=42)
df_merged_train_final = pd.concat([df_class_F, df_class_M_under], axis=0)
print('Random over-sampling:')
print(df_merged_train_final.gender.value_counts())

# Class count Over
count_class_M, count_class_F = df_merged_train_final.gender.value_counts()
# Divide by class
df_class_M = df_merged_train_final[df_merged_train_final['gender'] == 'M']
df_class_F = df_merged_train_final[df_merged_train_final['gender'] == 'F']
df_class_F_over = df_class_F.sample(count_class_M-25000, replace=True, random_state=42)
df_merged_train_final = pd.concat([df_class_M, df_class_F_over], axis=0)
df_merged_train_final = df_merged_train_final.astype(str)
print('Random over-sampling:')
print(df_merged_train_final.gender.value_counts())

'''
#### Preprocessing-->>>
# Normalizing and encoding
import unicodedata
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer

# making short name-
lemma=WordNetLemmatizer()
token=ToktokTokenizer()
from nltk.corpus import stopwords
stopWordList=stopwords.words('english')
stopWordList.remove('no')
stopWordList.remove('not')

#import spacy
#nlp = spacy.load('en_core', parse=True, tag=True, entity=True)

# NFKD - Compatibility Decomposition 
def removeAscendingChar(data):
    data=unicodedata.normalize('NFKD', data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return data

# Removing diff characters-
def removeCharDigit(text):
    str='`1234567890-=~@#$%^&*()_+[!{;":\'><.,/?"}]'
    for w in text:
        if w in str:
            text=text.replace(w,'')
    return text

# choosing root word-
def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w,'v')
        #print(x)
        listLemma.append(x)
    return text

# Removing stop words-
def stopWordsRemove(text):
    wordList=[x.lower().strip() for x in token.tokenize(text)]
    removedList=[x for x in wordList if not x in stopWordList]
    text=' '.join(removedList)
    #print(text)
    return text

# Running above functions-
def PreProcessing(text):
    text=removeCharDigit(text)
    #print(text)
    text=removeAscendingChar(text)
    #print(text)
    text=lemitizeWords(text)
    #print(text)
    text=stopWordsRemove(text)
    #print(text)
    return(text)'''

'''
totalText=''
count=0
for x in df_merged_train_final['desc']:
    ps=PreProcessing(x)
    totalText=totalText+" "+ps # Single variable with all the body
    print (count)
    count+=1

f= open("/Users/anilvyas/Desktop/TILCompleteDataSet/totalText.txt","w+")
f.write(totalText)'''
#0-F
#1-M


import os
import numpy as np
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets.base import get_data_home
from keras.metrics import categorical_accuracy
texts = df_merged_train_final.desc # Extract text
target = df_merged_train_final.gender # Extract target
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
target[:] = labelencoder_X.fit_transform(target[:])
vocab_size = 25000

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>') # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Generate sequences

word_index = tokenizer.word_index
print('Found {:,} unique words.'.format(len(word_index)))

# Create inverse index mapping numbers to words
inv_index = {v: k for k, v in tokenizer.word_index.items()}

# Print out text again
for w in sequences[0]:
    x = inv_index.get(w)
    print(x,end = ' ')

# Get the average length of a text
avg = sum(map(len, sequences)) / len(sequences)

# Get the standard deviation of the sequence length
std = np.sqrt(sum(map(lambda x: (len(x) - avg)**2, sequences)) / len(sequences))

avg,std

max_length = 30
data = pad_sequences(sequences, maxlen=max_length)

from keras.utils import to_categorical
labels = to_categorical(np.asarray(target))
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)

glove_dir = 'glove' # This is the folder with the dataset

embeddings_index = {} # We create a dictionary of word -> embedding

with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0] # The first value is the word, the rest are the values of the embedding
        embedding = np.asarray(values[1:], dtype='float32') # Load embedding
        embeddings_index[word] = embedding # Add embedding to our embedding dictionary

print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))

embedding_dim = 100 # We use 100 dimensional glove vectors

word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) # How many words are there actually

embedding_matrix = np.zeros((nb_words, embedding_dim))

# The vectors need to be in the same position as their index. 
# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= vocab_size: 
        continue
    # Get the embedding vector for the word
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=max_length, 
                    weights = [embedding_matrix], 
                    trainable = False))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.utils import shuffle
data, labels = shuffle(data, labels, random_state=42)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(target),
                                                 target)

model.fit(data, labels, validation_split=0.2, batch_size=32,epochs=10, class_weight = {0: 1.50, 1: 0.69})


test_df = pd.read_csv('UserId_Test.csv')
test_df = test_df.drop_duplicates()
test_df = test_df.astype('str')
test_df_merged = pd.merge(user_id_url_data, test_df, on='userid')
test_df_merged['url'] = test_df_merged['url'].map(preprocess_strip)
test_df_merged = test_df_merged.drop_duplicates()
test_df_merged_final = pd.merge(urls_data, test_df_merged, on='url')
#test_df_merged_final['desc']= test_df_merged_final['desc']+ test_df_merged_final['desc']

## Prediction
#av = pd.Series(['av cool'])
#text_clf.predict(av)

'''count = 0
def prediction(tx):
    global count
    print(count)
    count+=1
    cool = np.array([tx])
    sequences = tokenizer.texts_to_sequences(cool) # Generate sequences
    #print(len(sequences))
    da = pad_sequences(sequences, maxlen=max_length)
    av = model.predict(da.reshape(1,10))
    return av'''

#predicted = pd.Series()
#predicted = test_df_merged_final['desc'][:].map(prediction)
#predicted = model.predict(test_df_merged_final['desc'][:200].values.reshape(1,10)) #747
#predicted_copy = predicted.copy()
prediction_values = test_df_merged_final.desc.values
seqs = tokenizer.texts_to_sequences(prediction_values) # Generate sequences
#print(len(sequences))
data_to_pred = pad_sequences(seqs, maxlen=max_length)
#av = np.array(predicted_copy)
predicted = model.predict(data_to_pred)


from collections import OrderedDict
test_cases = OrderedDict()
test_cases2 = OrderedDict()

for i in test_df.iloc[:,0]:
    test_cases[str(i)] = []
    test_cases2[str(i)] = []
    
#predicted=predicted.values
#predicted=predicted.reshape(1834125,1)
size = test_df_merged_final.shape[0]
for i in range(size):
    print(i)
    #test_df_merged_final.userid.get()
    #predicted = text_clf.predict(pd.Series([test_df_merged_final.loc[i,'desc']]))
    av = str(labelencoder_X.inverse_transform([np.argmax(predicted[i])])[0])
    usr_id = str(test_df_merged_final.userid[i])
    test_cases[usr_id].append(av)
    #test_cases[str(test_df_merged_final.loc[i,'userid'])].append(predicted[0])

count=0
from collections import Counter
for i in test_cases.keys():
    print(count)
    count+=1
    av = list(Counter(test_cases[i]).keys())
    cool = list(Counter(test_cases[i]).values())
    #print(cool)
    if 'F' in test_cases[i]:
        av = list(Counter(test_cases[i]).keys())
        cool = list(Counter(test_cases[i]).values())
        ind = av.index('F')
        if cool[ind]/sum(cool)>0.9:
            test_cases2[str(i)] = ['F']
        else:
            test_cases2[str(i)] = ['M']
    else:
        test_cases2[str(i)] = ['M']

count=0
from collections import Counter
for i in test_cases.keys():
    print(count)
    count+=1
    #av = list(Counter(test_cases[i]).keys())
    #cool = list(Counter(test_cases[i]).values())
    #print(cool)
    if 'F' in test_cases[i]:
        av = list(Counter(test_cases[i]).keys())
        ind = av.index('F')
        cool = list(Counter(test_cases[i]).values())
        if cool[ind]>5:
            test_cases2[str(i)] = ['F']
        else:
            test_cases2[str(i)] = ['M']
    else:
        test_cases2[str(i)] = ['M']



submission = pd.DataFrame.from_dict(test_cases2, orient='index')
submission['userid'] = submission.index
submission['gender'] = submission[0]
submission = submission.iloc[:,1:]
submission = submission.astype('str')
submission.to_csv('submission.csv',index=False)
print(submission['gender'].value_counts())


#shuffling, over sampling
# Add rows od data of Male into Female category























