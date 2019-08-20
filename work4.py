# Implement OVER sampling -->>>
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
df_merged_train_final = df_merged_train.loc[:,['long_desc','gender']]
df_merged_train_final = df_merged_train_final.drop_duplicates()
df_merged_train_final = df_merged_train_final.astype(str)
#df_merged_train_final.to_csv('df_merged_train_final.csv')

# Over Sampling-->>
# Class count
count_class_M, count_class_F = df_merged_train_final.gender.value_counts()
# Divide by class
df_class_M = df_merged_train_final[df_merged_train_final['gender'] == 'M']
df_class_F = df_merged_train_final[df_merged_train_final['gender'] == 'F']
df_class_F_over = df_class_F.sample(count_class_M-22000, replace=True)
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
for x in df_merged_train_final['title']:
    ps=PreProcessing(x)
    totalText=totalText+" "+ps # Single variable with all the body
    print (count)
    count+=1

f= open("/Users/anilvyas/Desktop/TILCompleteDataSet/totalText.txt","w+")
f.write(totalText)'''

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_merged_train_final.long_desc)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
'''clf = MultinomialNB().fit(X_train_tfidf, df_merged_train_final.gender)'''



from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()) ])
text_clf = text_clf.fit(df_merged_train_final.long_desc, df_merged_train_final.gender)

len(df_merged_train_final[(df_merged_train_final.gender=='F')])
len(df_merged_train_final[(df_merged_train_final.long_desc=='none')])
### Check that the desc is none--ABOVE
#For Test-->>>
test_df = pd.read_csv('UserId_Test.csv')
test_df = test_df.drop_duplicates()
test_df = test_df.astype('str')
test_df_merged = pd.merge(user_id_url_data, test_df, on='userid')
test_df_merged['url'] = test_df_merged['url'].map(preprocess_strip)
test_df_merged = test_df_merged.drop_duplicates()
test_df_merged_final = pd.merge(urls_data, test_df_merged, on='url')


## Prediction
#av = pd.Series(['av cool'])
#text_clf.predict(av)
predicted = text_clf.predict(test_df_merged_final.long_desc) #747
#np.mean(predicted == twenty_test.target)

'''
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(df_merged_train_final.title, df_merged_train_final.gender)
'''

from collections import OrderedDict
test_cases = OrderedDict()
test_cases2 = OrderedDict()

for i in test_df.iloc[:,0]:
    test_cases[str(i)] = []
    test_cases2[str(i)] = []
    
predicted=predicted.reshape(1834125,1)
size = test_df_merged_final.shape[0]
for i in range(size):
    print(i)
    #predicted = text_clf.predict(pd.Series([test_df_merged_final.loc[i,'title']]))
    av=str(predicted[i,0])
    usr_id = str(test_df_merged_final.loc[i,'userid'])
    if usr_id not in test_cases.keys():
        test_cases[usr_id]=[]
        test_cases[usr_id].append(av)
    else:
        test_cases[usr_id].append(av)
        #test_cases[str(test_df_merged_final.loc[i,'userid'])].append(predicted[0])
'''count=0
from collections import Counter
for i in test_cases.keys():
    print(count)
    count+=1
    av = list(Counter(test_cases[i]).keys())
    cool = list(Counter(test_cases[i]).values())
    print(cool)
    try:
        if cool[0]>cool[1]:
            test_cases2[i] = av[0]
        elif cool[0]<cool[1]:
            test_cases2[i] = av[1]
        else:
            test_cases2[i] = 'M'
    except:
        test_cases2[i] = []'''

#gk = test_df_merged_final.groupby('userid').get_group('1000021')
#gk.first() 

'''submission = test_df.copy()
submission['gender'] = ['M']*88384
submission.to_csv('submission.csv',index=False)
'''

'''test_on_predicted=list(predicted.reshape(1834125,))
cc = pd.Series(test_on_predicted[:])
cc.value_counts()'''
#small 
#classify acc to subcategory like business

'''# Single presence-->>>
count=0
from collections import Counter
for i in test_cases.keys():
    print(count)
    count+=1
    #av = list(Counter(test_cases[i]).keys())
    #cool = list(Counter(test_cases[i]).values())
    #print(cool)
    if 'F' in test_cases[i]:
        test_cases2[str(i)] = ['F']
    else:
        test_cases2[str(i)] = ['M']'''

#test_cases2 = OrderedDict()

# Another logic-->>
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
        if cool[ind]/sum(cool)>0.80:
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
submission['gender'].value_counts()



# Resampling over wali and also to dec the no pf male exampkes
# Bert classifier
# Using long dess,desc and combi to detect the same as above

















