#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd


# In[10]:


df=pd.read_csv("Bengaluru_House_Data.csv")


# In[11]:


df.shape


# In[12]:


df.isna().sum()


# In[13]:


df1=df.drop(["society"],axis="columns")


# In[14]:


df2=df1.dropna()


# In[15]:


df2.isna().any()


# In[16]:


df2.shape


# In[17]:


df2.head(2)


# In[18]:


df2["bhk"]=df2["size"].apply(lambda x: int(x.split(' ')[0]))


# In[19]:


df2.head(3)


# In[20]:


df2.total_sqft.unique()


# In[21]:


def set_total_sqft(x):
    x=str(x)
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[22]:


df2["total_sqft"]=df2["total_sqft"].apply(set_total_sqft)


# In[23]:


df2.total_sqft.unique()


# In[24]:


def set_float(x):
    try:
         float(x)
    except:
        return False
    return True


# In[27]:






len(df2[~(df2["total_sqft"].apply(set_float))])


# In[28]:


df2.head(3)


# In[29]:


df3=df2.drop(["availability"],axis="columns")


# In[30]:


df3.isna().any()


# In[31]:


df3.shape


# In[32]:


df3.head(3)


# In[33]:


##df3[df3["total_sqft"]/df3["bhk"]<300]


# In[34]:


location_stat=df3.groupby("location")["location"].agg("count").sort_values(ascending=False)


# In[35]:


location_less_than_ten=location_stat[location_stat<=10]


# In[36]:


df3["location"]=df3["location"].apply(lambda x:"other" if x in location_less_than_ten else x)


# In[37]:


df3.groupby("location")["location"].agg("count").sort_values(ascending=False)


# In[38]:


df3.head(3)


# In[39]:


df3["price_per_sqft"]=(df3["price"]*100000)/df3["total_sqft"]


# In[40]:


df3.head(3)


# In[41]:


df3["bhk"].unique()


# In[42]:


df4=df3[df3["total_sqft"]/df3["bhk"]>300]


# In[43]:


df4.shape


# In[44]:


df4.head(2)


# In[45]:


df5=df4[df4["bath"]<df4["bhk"]+2]


# In[46]:




df5.shape


# In[47]:


df5.head()


# In[48]:


df5.price_per_sqft.describe()


# In[49]:


df6=df5.drop(["area_type"],axis="columns")


# In[50]:


df6.head()
import numpy as np


# In[51]:


def removeol(df):
    out_df=pd.DataFrame()
    for key,subdf in df.groupby("location"):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>=(m-st)) & (subdf.price_per_sqft<=(m+st))]
        out_df=pd.concat([out_df,reduced_df],ignore_index=True)
    return out_df


# In[52]:



df7=removeol(df6)


# In[53]:


df7.head()


# In[ ]:





# In[54]:


df7.isna().any()


# In[55]:


final=df7


# In[56]:


final.isna().any()


# In[57]:


from sklearn.preprocessing import LabelEncoder


# In[58]:


le=LabelEncoder()


# In[59]:


final.head()


# In[60]:


new_final=df7


# In[61]:


new_final.head()


# In[62]:


new_final.groupby("location")["location"].agg("count")
import matplotlib.pyplot as plt


# In[ ]:





# In[63]:


df7.head()


# In[64]:


final=df7


# In[65]:


final.head()


# In[66]:


def plots(df,location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    
    plt.scatter(bhk2.price_per_sqft,bhk2.total_sqft,marker="*",color="blue",label="bhk2",s=50)
    plt.scatter(bhk3.price_per_sqft,bhk3.total_sqft,marker="+",color="red",label="bhk3",s=50)
    plt.xlabel("price per sq ft")
    plt.ylabel("total price")
   
    plt.legend()


# In[67]:


plots(final,'Rajaji Nagar')
new_final=final.drop(["size"],axis=1)


# In[68]:


def remove_ol_bhk(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std' :np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[(bhk_df.price_per_sqft<(stats["mean"]))].index.values)
    return df.drop(exclude_indices,axis='index')


# In[69]:


cfinal=remove_ol_bhk(new_final)


# In[70]:


new_final.head(2)


# In[71]:


plots(cfinal,'Rajaji Nagar')


# In[72]:


plt.hist(cfinal.price_per_sqft,rwidth=0.8)
plt.xlabel("price per sqft")
plt.ylabel('count')
plt.show()


# In[73]:


cfinal.shape


# In[74]:


cfinal.head(3)


# In[75]:


n_final=cfinal.drop(["price_per_sqft","balcony"],axis="columns")


# In[76]:


n_final.head(2)


# In[77]:


dummies=pd.get_dummies(n_final.location)


# In[78]:


dummies


# In[79]:


last_df=pd.concat([n_final,dummies],axis="columns")


# In[80]:


last_df.head(2)


# In[81]:


last_df=last_df.drop(["location"],axis=1)


# In[82]:


last_df.head(2)


# In[83]:


last_df.shape


# In[84]:


x=last_df.drop(["price"],axis="columns")


# In[85]:


y=last_df.price


# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


# In[88]:


x_train.shape


# In[89]:


x_test.shape


# In[90]:


from sklearn.linear_model import LinearRegression


# In[91]:


lr=LinearRegression()


# In[92]:


lr.fit(x_train,y_train)


# In[93]:


lr.score(x_test,y_test)


# In[94]:


pred=lr.predict(x_test)


# In[95]:


from sklearn.model_selection import ShuffleSplit


# In[96]:


from sklearn.model_selection import cross_val_score


# In[97]:


cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)


# In[98]:


cross_val_score(LinearRegression(),x,y,cv=cv)


# In[99]:


from sklearn.linear_model import Lasso


# In[108]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


# In[111]:


def find_best_model_using_gridsearchcv(x,y):
    algos={
        'LinearRegression':{
        'model':LinearRegression(),
        'params':{
            'normalize':[True,False]
        }
    },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'DecisionTreeRegressor':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


# In[112]:



find_best_model_using_gridsearchcv(x,y)


# In[ ]:





# In[ ]:





# In[138]:


np.where(x.columns=='6th Phase JP Nagar')[0][0]


# In[139]:


len(x.columns)


# In[142]:


x=last_df.drop(["price"],axis="columns")
y=last_df.price


# In[143]:


x.shape


# In[144]:


def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    xa=np.zeros(len(x.columns))
    xa[0]=sqft
    xa[1]=bath
    xa[2]=bhk
    if loc_index>=0:
        xa[loc_index]=1
    return lr.predict([xa])[0]


# In[145]:


predict_price('2nd Phase Judicial Layout',1200.0,2,2)


# In[ ]:




