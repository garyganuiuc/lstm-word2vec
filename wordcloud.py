#code for generating the wordcloud

import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.cross_validation import train_test_split
from wordcloud import WordCloud,STOPWORDS
import re
import nltk
from nltk.corpus import stopwords
#df = pd.read_csv('./reddit/Combined_News_DJIA.csv')
df = pd.read_csv('./reuters/simple.csv')
print(df.shape)
import matplotlib
matplotlib.rcParams["figure.figsize"] = "8, 8"

# uncomment this for reddit dataset
#df['Combined']=df.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
simple = df.drop(['Unnamed: 0'], axis=1)

# uncomment this for reddit dataset
#train,test = train_test_split(df,test_size=0.2,random_state=42)

# uncomment this for reddit dataset
#non_decrease = train[train['Label']==1]
#decrease = train[train['Label']==0]
#print(len(non_decrease)/len(df))

# comment this for reddit dataset
non_decrease = simple[simple['Diff']>=0]
decrease = simple[simple['Diff']<0]
print(len(non_decrease)/len(simple))
print(len(non_decrease))


def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 
	
# uncomment this for reddit dataset
'''
non_decrease_word=[]
decrease_word=[]
for each in non_decrease['Combined']:
    non_decrease_word.append(to_words(each))

for each in decrease['Combined']:
    decrease_word.append(to_words(each))
'''
# comment this for reddit dataset
non_decrease_word=[]
decrease_word=[]
for each in non_decrease['Title']:
    non_decrease_word.append(to_words(each))

for each in decrease['Title']:
    decrease_word.append(to_words(each))

	
# generate negative wordcloud
wordcloud1 = WordCloud(background_color='black',
                      width=3000,
                      height=2500
                     ).generate(decrease_word[0])



plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()

# generate positive wordcloud
wordcloud2 = WordCloud(background_color='white',
                      width=3000,
                      height=2500
                     ).generate(non_decrease_word[0])



plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud2)
plt.axis('off')
plt.show()






