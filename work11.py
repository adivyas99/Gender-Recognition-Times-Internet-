# Adding 
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
df_merged_train = df_merged_train.drop_duplicates()

# To CSVs-->>>
'''
user_id_url_data.to_csv('user_id_url_data.csv')
UserIdToGender_Train.to_csv('UserIdToGender_Train.csv')
urls_data.to_csv('urls_data.csv')
df_merged.to_csv('df_merged.csv')
df_merged_train.to_csv('df_merged_train.csv')
'''
#==
#df_merged_train_exp = df_merged_train.drop_duplicates(subset = 'tags')
## Female Keywords-
Female_df = df_merged_train[df_merged_train['gender'] == 'F']
#Female_df = Female_df.drop_duplicates(subset = 'tags')
Female_df = Female_df[Female_df['tags'] != 'none']
keywords_F = []
for i in Female_df.tags[:]:
    print(i)
    #i= exec('%s'%(i))
    exec('k=list(%s)'%(i))
    #k=list(k)
    #print(k)
    for j in k:
        keywords_F.append(j)

keywords_F = list(set(keywords_F))
keywords_F.remove('')

## Male Keywords-
Male_df = df_merged_train[df_merged_train['gender'] == 'M']
#Male_df = Male_df.drop_duplicates(subset = 'tags')
Male_df = Male_df[Male_df['tags'] != 'none']
keywords_M = []
for i in Male_df.tags[:]:
    print(i)
    #i= exec('%s'%(i))
    exec('k=list(%s)'%(i))
    #k=list(k)
    #print(k)
    for j in k:
        keywords_M.append(j)

keywords_M = list(set(keywords_M))
keywords_M.remove('')
#--------------------
## Converting for Male-
Male_df = urls_data.copy()
Male_df = Male_df[Male_df['tags'] != 'none']
#dele = Male_df[Male_df.duplicated(['desc'])]
#dele = Male_df.duplicated(subset='desc', keep='first')
Male_df = Male_df.drop_duplicates(subset ='desc')
#Male_df = Male_df.drop_duplicates()
Male_df_copy = Male_df.copy()
Male_df_copy['gender']=[''for i in range(Male_df_copy.shape[0])]

sz = Male_df.shape[0]
indices = list(Male_df.index)
c=0
j=0
for i in indices:
    #print(type(i))
    av = Male_df.tags[i]
    print(c)
    c+=1
    exec('k=list(%s)'%(av))
    a_set = set(keywords_F)
    b_set = set(k)
    if len(a_set.intersection(b_set))>4:
        j+=1
        print('--')
        Male_df_copy.gender[i]='F'

Male_df_copy = Male_df_copy[Male_df_copy['gender']=='F']
## Converting for Female-
Female_df = urls_data.copy()
Female_df = Female_df[Female_df['tags'] != 'none']
#dele = Female_df[Female_df.duplicated(['desc'])]
#dele = Female_df.duplicated(subset='desc', keep='first')
Female_df = Female_df.drop_duplicates(subset ='desc')
#Female_df = Female_df.drop_duplicates()
Female_df_copy = Female_df.copy()
Female_df_copy['gender']=[''for i in range(Female_df_copy.shape[0])]

sz = Female_df.shape[0]
indices = list(Female_df.index)
c=0
j=0
for i in indices:
    #print(type(i))
    av = Female_df.tags[i]
    print(c)
    c+=1
    exec('k=list(%s)'%(av))
    a_set = set(keywords_M)
    b_set = set(k)
    if len(a_set.intersection(b_set))>4:
        j+=1
        print('--')
        Female_df_copy.gender[i]='M'

Female_df_copy = Female_df_copy[Female_df_copy['gender']=='M']

#--------------------
df_merged_train_c = df_merged_train.copy()
df_merged_train_c = df_merged_train_c.drop('userid', axis=1)
columns_for_d_fmerged_train = ['id', 'title', 'url', 'desc', 'long_desc', 'brand', 'tags', 'entities', 'gender']
df_merged_train_c = df_merged_train_c[columns_for_d_fmerged_train]
df_merged_train_c_F = df_merged_train_c.copy()[df_merged_train_c['gender']=='F']
df_merged_train_c_F['gender'] = ['M' for i in range(df_merged_train_c_F.shape[0])]
df_merged_train_c_M = df_merged_train_c.copy()[df_merged_train_c['gender']=='M']
df_merged_train_c_M['gender'] = ['F' for i in range(df_merged_train_c_M.shape[0])]

df_all_rows = pd.concat([Female_df_copy, Male_df_copy, df_merged_train_c, df_merged_train_c_F,df_merged_train_c_M])
# Making Final Training datsets-->>>
#df_merged_train_final = df_merged_train.loc[:,['desc','gender']]
df_merged_train_final = df_all_rows.loc[:,['desc','gender']]
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


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_merged_train_final.desc)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
df_merged_train_final.gender = labelencoder_X.fit_transform(df_merged_train_final.gender)
labelencoder_X.inverse_transform([0])
from sklearn.pipeline import Pipeline
'''
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier( random_state=42))])

text_clf_svm = text_clf_svm.fit(df_merged_train_final.desc, df_merged_train_final.gender)
'''

'''
from sklearn.tree import DecisionTreeClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', DecisionTreeClassifier()) ])
text_clf = text_clf.fit(df_merged_train_final.desc, df_merged_train_final.gender)

'''

'''
from sklearn import ensemble,feature_extraction
clf=Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1))
    ])
clf.fit(df_merged_train_final.desc,df_merged_train_final.gender)
'''

'''
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=True))])

text_mnb_stemmed = text_mnb_stemmed.fit(df_merged_train_final.desc, df_merged_train_final.gender)

predicted_mnb_stemmed = text_mnb_stemmed.predict(test_df_merged_final.desc)

predicted = predicted_mnb_stemmed
'''
#from sklearn.naive_bayes import GaussianNB

'''clf = MultinomialNB().fit(X_train_tfidf, df_merged_train_final.gender)'''


'''
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()) ])
text_clf = text_clf.fit(df_merged_train_final.desc, df_merged_train_final.gender)
'''

len(df_merged_train_final.loc[df_merged_train_final['gender']==1])
len(df_merged_train_final[(df_merged_train_final.desc=='none')])
### Check that the desc is none--ABOVE
#For Test-->>>
test_df = pd.read_csv('UserId_Test.csv')
test_df = test_df.drop_duplicates()
test_df = test_df.astype('str')
test_df_merged = pd.merge(user_id_url_data, test_df, on='userid')
test_df_merged['url'] = test_df_merged['url'].map(preprocess_strip)
test_df_merged = test_df_merged.drop_duplicates()
test_df_merged_final = pd.merge(urls_data, test_df_merged, on='url')
#test_df_merged_final['desc']= test_df_merged_final['title']+ test_df_merged_final['desc']

## Prediction
#av = pd.Series(['av cool'])
#text_clf.predict(av)
predicted = text_clf.predict(test_df_merged_final.desc) #747
#np.mean(predicted == twenty_test.target)

'''
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(df_merged_train_final.desc, df_merged_train_final.gender)
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
    #predicted = text_clf.predict(pd.Series([test_df_merged_final.loc[i,'desc']]))
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
        test_cases2[str(i)] = ['M']
'''

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
    if 'F' in test_cases[i] and 'M' not in test_cases[i]:
       test_cases2[str(i)] = ['F']
    elif 'F' in test_cases[i]:
        av = list(Counter(test_cases[i]).keys())
        cool = list(Counter(test_cases[i]).values())
        ind = av.index('F')
        if cool[ind]/sum(cool)> 1-cool[ind]/sum(cool):
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



#print('\a')
# 81.6%
# Resampling over wali and also to dec the no pf male exampkes
# Bert classifier
# Using long dess,desc and combi to detect the same as above



# Classifier
# both over and under resampling
# COlumn that is most efficient in predicting that shit- Desc_col is good one 













