#!/usr/bin/env python
# coding: utf-8

# # The comments are the same as previous ipynb file except the new ones which are added only in this notebook

# In[50]:





import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from pattern.en import sentiment, Sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from numpy import loadtxt
#import gensim


# In[51]:


# df_tt=pd.read_json('Toys_and_Games/test2/product_test.json')
# df_tt.shape
#pickled_model = pickle.load(open('final_late_fusion_model.pkl', 'rb'))


# In[52]:


# df_tt=pd.read_json('predictions.json')
# df_tt


# In[53]:


df=pd.read_json('Toys_and_Games/train/review_training.json')
df.head(5)
df.shape


# In[ ]:





# In[54]:


df_test=pd.read_json('Toys_and_Games/test3/review_test3.json')
df_test.head(5)
df_test.shape


# In[ ]:





# In[55]:


#preprocess image
df['image'] = df['image'].replace([None], 0)
df['image'] = df['image'].apply(lambda x: 1 if x!=0 else 0)
df['image'].value_counts()


# In[56]:


#preprocess image
df_test['image'] = df_test['image'].replace([None], 0)
df_test['image'] = df_test['image'].apply(lambda x: 1 if x!=0 else 0)
df_test['image'].value_counts()


# In[57]:


#preprocess votes
df['vote'] =df['vote'].replace([None], 0)
df['vote'].value_counts()


# In[58]:


#preprocess votes
df_test['vote'] =df_test['vote'].replace([None], 0)
df_test['vote'].value_counts()


# In[59]:


#preprocess verified
df['verified'] = df['verified'].apply(lambda x: 1 if x==True else 0)
df['verified'].value_counts()


# In[60]:


#preprocess verified
df_test['verified'] = df_test['verified'].apply(lambda x: 1 if x==True else 0)
df_test['verified'].value_counts()


# In[61]:


df1=pd.read_json('Toys_and_Games/train/product_training.json')
df=pd.merge(df,df1,on=['asin'])
df.head(3)


# In[62]:


df=df.drop(columns=['reviewerID','reviewTime','reviewerName','style'])


# In[63]:


df_test=df_test.drop(columns=['reviewerID','reviewTime','reviewerName','style'])


# In[64]:


df['reviewText'] = df['reviewText'].astype(str) +" "
df['summary'] = df['summary'].astype(str) +" "


# In[65]:


df_test['reviewText'] = df_test['reviewText'].astype(str) +" "
df_test['summary'] = df_test['summary'].astype(str) +" "


# In[66]:


df['summary']=df['summary'].str.replace('Five Stars ', 'great ')
df['summary']=df['summary'].str.replace('Four Stars ', 'neutral ')
df['summary']=df['summary'].str.replace('Three Stars ', 'not good ')
df['summary']=df['summary'].str.replace('Two Stars ', 'bad ')
df['summary']=df['summary'].str.replace('One Star ', 'bad bad ')


# In[67]:


df_test['summary']=df_test['summary'].str.replace('Five Stars ', 'great ')
df_test['summary']=df_test['summary'].str.replace('Four Stars ', 'neutral ')
df_test['summary']=df_test['summary'].str.replace('Three Stars ', 'not good ')
df_test['summary']=df_test['summary'].str.replace('Two Stars ', 'bad ')
df_test['summary']=df_test['summary'].str.replace('One Star ', 'bad bad ')


# In[68]:


sent = SentimentIntensityAnalyzer()
polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in df['reviewText']]
df['sentiment_score_review'] = polarity
polarity_s = [round(sent.polarity_scores(i)['compound'], 2) for i in df['summary']]
df['sentiment_score_summary'] = polarity_s


# In[69]:


sent = SentimentIntensityAnalyzer()
# df = pd.read_csv('', usecols = ['body'])
polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in df_test['reviewText']]
df_test['sentiment_score_review'] = polarity
polarity_s = [round(sent.polarity_scores(i)['compound'], 2) for i in df_test['summary']]
df_test['sentiment_score_summary'] = polarity_s


# In[70]:


#a new sentiment analyzer
# # import pattern
# from pattern.en import sentiment, Sentence
# # import flair


# # sentences = ["great"," you are bad"]
# polarity_review_pattern = [round(sentiment(sentence)[0],2) for sentence in df['reviewText']]  # polarity
# df['pattern_sentiment_review']=polarity_review_pattern

# polarity_summary_pattern = [round(sentiment(sentence)[0],2) for sentence in df['summary']]  # polarity
# df['pattern_sentiment_summary']=polarity_summary_pattern
# df.head()


# In[71]:


# a new feature #of positive and negative words
# file = open('negative-words.txt', 'r')
# neg_words = file.read().split()
# file = open('positive-words.txt', 'r')
# pos_words = file.read().split()
# num_pos = df_agg['reviewText'].map(lambda x: len([i for i in x if i in pos_words]))
# df_agg['pos_count_review'] = num_pos
# num_neg = df_agg['reviewText'].map(lambda x: len([i for i in x if i in neg_words]))
# df_agg['neg_count_review'] = num_neg

# num_pos = df_agg['summary'].map(lambda x: len([i for i in x if i in pos_words]))
# df_agg['pos_count_summary'] = num_pos
# num_neg = df_agg['summary'].map(lambda x: len([i for i in x if i in neg_words]))
# df_agg['neg_count_summary'] = num_neg
# df_agg


# In[72]:


aggregation_functions = {'unixReviewTime': 'mean', 'verified': 'mean', 'image': 'mean', 'reviewText': 'sum', 'summary': 'sum','sentiment_score_review': 'mean','sentiment_score_summary': 'mean'}
df_agg = df.groupby(df['asin']).aggregate(aggregation_functions)


# In[73]:


aggregation_functions = {'unixReviewTime': 'mean', 'verified': 'mean', 'image': 'mean', 'reviewText': 'sum', 'summary': 'sum','sentiment_score_review': 'mean','sentiment_score_summary': 'mean'}
df_agg_test = df_test.groupby(df_test['asin']).aggregate(aggregation_functions)


# In[74]:


df_agg=pd.merge(df_agg,df1,on=['asin'])
df_agg.shape


# In[75]:


df_agg_test=df_agg_test.reset_index()


# In[76]:


df_agg['reviewText'] = df_agg['reviewText'].str.replace('[^\w\s]', '').str.lower()
df_agg['summary'] = df_agg['summary'].str.replace('[^\w\s]', '').str.lower()


# In[77]:


df_agg_test['reviewText'] = df_agg_test['reviewText'].str.replace('[^\w\s]', '').str.lower()
df_agg_test['summary'] = df_agg_test['summary'].str.replace('[^\w\s]', '').str.lower()


# In[78]:


stemmer = PorterStemmer()

# Define a function to perform stemming on each document in the corpus
def stem_words(text):
    # Split the text into individual words
    words = text.split()
    # Apply stemming to each word in the text
    stemmed_words = [stemmer.stem(word) for word in words]
    # Join the stemmed words back into a single string
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

# Preprocess the corpus by applying stemming
stemmed_summary = [stem_words(text) for text in df_agg['summary']]
stemmed_review = [stem_words(text) for text in df_agg['reviewText']]
df_agg['summary']=stemmed_summary
df_agg['reviewText']=stemmed_review

stemmed_summary = [stem_words(text) for text in df_agg_test['summary']]
stemmed_review = [stem_words(text) for text in df_agg_test['reviewText']]
df_agg_test['summary']=stemmed_summary
df_agg_test['reviewText']=stemmed_review


# In[79]:


independent_train=df_agg[['unixReviewTime','verified','image','sentiment_score_review','sentiment_score_summary','reviewText','summary']]
dependent_train=df_agg['awesomeness']


# In[80]:


independent_test=df_agg_test[['asin','unixReviewTime','verified','image','sentiment_score_review','sentiment_score_summary','reviewText','summary']]


# In[81]:


X_train=independent_train
X_test=independent_test
y_train=dependent_train


# In[82]:


model_log = LogisticRegression(max_iter=20,solver='saga',C=10)
model_ran = RandomForestClassifier(max_depth=40,criterion="entropy",min_samples_split=200)
model_grad = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)

model = VotingClassifier(estimators=[('LogisticRegression',model_log),("randomforest",model_ran),("GradientBoost",model_grad)],voting='soft')

vector_review_train = TfidfVectorizer(analyzer='word',stop_words= 'english',max_features=10000,ngram_range=(1,2))
vector_review_train.fit(X_train['reviewText'])

vector_summary_train = TfidfVectorizer(analyzer='word',stop_words= 'english',max_features=5000,ngram_range=(1,2))
vector_summary_train.fit(X_train['summary'])

review_train=vector_review_train.transform(X_train['reviewText'])
summary_train=vector_summary_train.transform(X_train['summary'])
#test
review_test=vector_review_train.transform(X_test['reviewText'])
summary_test=vector_summary_train.transform(X_test['summary'])


#adds tfidf of review+summary
sum_rev_train=hstack([review_train,summary_train])
sum_rev_test=hstack([review_test,summary_test])

#X_train[['unixReviewTime','verified','image','sentiment_score_review','sentiment_score_summary']].to_numpy()
df_training=X_train[['unixReviewTime','verified','image','sentiment_score_review','sentiment_score_summary']].to_numpy()
df_test=X_test[['unixReviewTime','verified','image','sentiment_score_review','sentiment_score_summary']].to_numpy()

scaler = StandardScaler()
train_features = scaler.fit_transform(df_training)
test_features= scaler.transform(df_test)
#     train_features=df_training
#     test_features=df_test





X_train_final=hstack([train_features,sum_rev_train])
X_test_final=hstack([test_features,sum_rev_test])
#     X_train_final=train_features
#     X_test_final=test_features

y_train_final=y_train.to_numpy()
# y_test_final=y_test.to_numpy()



#toarray()for NB
# model_log.fit(X_train_final,y_train_final)
# model_ran.fit(X_train_final,y_train_final)
# model_grad.fit(X_train_final,y_train_final)
model.fit(X_train_final,y_train_final)


# pred_values_log = model_log.predict(X_test_final)
# pred_values_ran = model_ran.predict(X_test_final)
# pred_values_grad = model_grad.predict(X_test_final)
pred_values=model.predict(X_test_final)


# In[83]:


pickle.dump(model, open('individual_model_py.pkl', 'wb'))

# In[84]:


##############################################################################
final_df=independent_test[['asin']]
final_df['awesomeness']=pred_values
final_df


# In[85]:


#####################################################################################
# final_df.to_json('predictions.json')
# read_d=pd.read_json('predictions.json')
# read_d

# test2_y=pd.read_json('Toys_and_Games/test2/test2_y.json')
# final_df2=test2_y.merge(final_df,on=['asin'])
# final_df2.head()
# print(f1_score(final_df2['awesomeness'],final_df2['awesomeness1']))
# print(accuracy_score(final_df2['awesomeness'],final_df2['awesomeness1']))
final_df.to_json('individual_predictions_py.json')


# In[97]:


#############################################################################################
# grad_pre=pd.read_json('grad_predictions.json')
# ran_pre=pd.read_json('ranforest_predictions.json')
# log_pre=pd.read_json('logistic_predictions.json')
# pre_merge=pd.merge(grad_pre,pd.merge(ran_pre,log_pre,on=['asin']),on=['asin'])
# pre_merge
# mode_series = pre_merge.mode(axis=1)

# # create a new column containing the mode of row values
# pre_merge['final_pred'] = mode_series
# pre_merge
# test3_yy=pd.read_json('Toys_and_Games/test3/product_test.json')
# test3_yy.shape
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# # Confusion matrix data
# confusion_matrix = np.array([[17597, 12227],
#  [ 8522, 27249]]
# )

# # Plotting the confusion matrix
# sns.set(font_scale=1.4)
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.show()


# In[ ]:





# In[87]:


# final_df.to_json('predictions.json')
# test1_y=pd.read_json('Toys_and_Games_test1_labels.json')
# test1_y


# In[88]:


# pred_values_log

# final_df=independent_test[['asin']]
# final_df['awesomeness_grad']=pred_values_grad
# final_df['awesomeness_ran']=pred_values_ran
# final_df['awesomeness_log']=pred_values_log
# mode_series = final_df.mode(axis=1)

# # create a new column containing the mode of row values
# final_df['final_pred'] = mode_series
# final_df


# In[89]:


# df_pred=pd.read_json('predictions.json')
# df_pred.head()



# df_merged=pd.merge(test1_y,final_df,on=['asin'])
# # df_merged=pd.merge(test1_y,final_df[['asin','final_pred']],on=['asin'])
# df_merged


# In[90]:


# print(f1_score(df_merged['awesomeness'],df_merged['awesomeness_grad']))
# print(accuracy_score(df_merged['awesomeness'],df_merged['awesomeness_grad']))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




