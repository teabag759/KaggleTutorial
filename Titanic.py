#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install missingno


# In[2]:


# 필요한 라이브러리 설치 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#plt.style.use('seaborn')
    # 버전 변경으로 인해 더이상 사용하지 않음 
sns.set(font_scale=2.5)

import missingno as msno

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Dataset 확인

# In[3]:


df_train = pd.read_csv('./KaggleData/Titanic/train.csv')
df_test = pd.read_csv('./KaggleData/Titanic/test.csv')


# In[4]:


df_train.head()


# # Dataset 확인

# - 해당 문제에서 다루는 feature : Pclass, Age, SibSp, Parch, Fare
# - target label : Survived
# 
# ## variable
# 
# - Pclass : 티켓 클래스 / 1, 2, 3 class로 나뉘며 categorial feature / integer
# - Age : 나이 / continuous / interger 
# - SibSp : 동승한 형제와 배우자 수 / quantitative / integer
# - Parch : 동승한 부모, 아이의 수 / quantitative / integer
# - Fare : 탑승료 / continuous / float
# - Survived : 생존여부 / 1, 0으로 표현 / integer
# - Embared : 탑승 항구 / C=Cherbourg, Q=Queentown, S=Southampton / string

# In[5]:


df_train.describe()


# In[6]:


df_test.describe()


# ## Null data check

# In[7]:


for col in df_train.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
    print(msg)


# In[8]:


for col in df_test.columns:
    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))
    print(msg)


# - Train, Test set 에서 Age, Cabin, Embarked(in Train) 에서 null data 존재

# In[9]:


msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# In[10]:


msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# In[11]:


msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))


# - MANO 라이브러리를 사용하면 null data 존재를 쉽게 살펴볼 수 있음

# ## Target label 확인 
# 
# - target label의 distribution 확인 
# - binary classification 문제의 경우, 1과 0의 분포가 어떠냐에 따라 모델의 평가 방법이 달라짐 

# In[12]:


# Target Lable 확인하기 

f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct = '%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot(x='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()


# - 38.4% 의 사람들이 살아남음 
# - target label의 분포가 제법 균일(balanced)한 편
#     - 불균일한 경우, 100 중 1이 99, 0이 1개인 경우, 0을 찾는 문제라면 해당 모델은 원하는 결과를 줄 수 없음 

# # Exploratory data analysis 
# 
# - 시각화 라이브러리 : matplotlib, seaborn, plotly 
# 
# ## Pclass
# - Pclass는 ordinal, 서수형 데이터 
#     - 카테고리이면서, 순서가 있는 데이터 타입 
#     
# ### Pclass에 따른 생존률 차이 
# - groupby, pivot method 
# 1. 'Pclass', 'Survived' 를 가져온 후, 'Pclass'로 묶기 
# 2. 각 'Pclass' 마다 count된 0, 1을 평균 내기 -> 각 Pclass 별 생존률 확인

# In[13]:


# 각 class의 탑승자 수 확인 (count)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()


# In[14]:


# 각 class 탑승자 중 생존자의 수 확인 (sum)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()


# In[15]:


# crosstab : compute a simple cross-tabulation of two(or more) factors 
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# In[16]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()


# - Pclass가 좋을 수록(1st) 생존률이 높음 

# In[17]:


# sns.countplot을 활용하여 특정 label의 개수 확인하기 

y_position = 1.02
    # 제목의 위치를 조절하기 위한 변수 
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')

sns.countplot(x='Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()


# - 클래스가 높을 수록 생존 확률이 높음 
#     - 1st : 63%, 2nd : 48%, 3rd : 25%
#     - 생존에 'Pclass'가 큰 영향을 미침 

# ## Sex
# - 성별에 따른 생존률 확인

# In[18]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot(x='Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')

plt.show()


# - 여성일수록 생존할 확률이 높음 

# In[19]:


df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[20]:


pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')


# 

# ## Both Sex and Pclass
# - Sex, Pclass에 따라 생존률이 어떻게 달라지는 지 확인하기 

# In[21]:


# sns.factorplot은 더이상 지원하지 않음 
#sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)

# catplot을 활용해서 생존률서로 확인해보기 
sns.catplot(x='Pclass', y='Survived', hue='Sex', kind='bar', data=df_train, 
               height=6, aspect=1.5)


# - 모든 클래스에서 female의 생존률이 높음 
# - 성별과 관계없이 class가 높을 수록 생존율이 높음 

# In[37]:


#sns.factorplot(x='Sex', y='Survived', col='Pclass', data=df_train, satureation=.5, size=9, aspect=1)

# 성별의 순서 고정하기
df_train['Sex'] = pd.Categorical(df_train['Sex'], categories=['female', 'male'], ordered=True)

f, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

sns.lineplot(x='Sex', y='Survived', hue='Pclass', data=df_train[df_train['Pclass'] == 1], palette='viridis', ax=ax[0])
ax[0].set_title('Pclass 1')

sns.lineplot(x='Sex', y='Survived', hue='Pclass', data=df_train[df_train['Pclass'] == 2], palette='viridis', ax=ax[1])
ax[1].set_title('Pclass 2')

sns.lineplot(x='Sex', y='Survived', hue='Pclass', data=df_train[df_train['Pclass'] == 3], palette='viridis', ax=ax[2])
ax[2].set_title('Pclass 3')

plt.tight_layout()
plt.show()


# ## Age

# In[23]:


print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))
print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))
print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))


# In[24]:


fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])
    # legend : 범례
plt.show()


# - 생존자 중 나이가 어린 경우가 많음 

# In[25]:


# Age distribution within classes
plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])


# - class가 높을 수록 나이 많은 사람의 비중이 커짐 

# In[26]:


cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum(0)
                                   / len(df_train[df_train['Age'] < i]['Survived']))

plt.figure(figsize=(7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()


# - 나이가 어릴 수록 생존률이 확실히 높음 

# ## Pclass, Sex, Age
# 
# - seaborn의 violinplot을 활용하여 Pclass, Sex, Age, Survived 모두 확인하기 
# - x축은 case(Pclass, Sex)
# - y축은 distribution(Age)

# In[27]:


f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=df_train, 
               scale='count', split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

sns.violinplot(x='Sex', y='Age', hue='Survived', data=df_train, 
              scale='count', split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))

plt.show()


# - 왼쪽 그림 : Pclass 별로 Age의 distribution이 어떻게 다른지, 생존여부에 따라 구분
# - 오른쪽 그림 : Sex, Survived에 따른 distribution이 어떻게 다른지 보여줌 
# - Survived만 봤을 때, 모든 클래스에서 나이가 어릴 수록 생존률이 높음 
# - 오른쪽 그림을 보면, 명확히 여자가 많이 생존함
# - 여성과 아이를 먼저 챙김 

# ## Embarked
# - Embarked : 탑승한 항구
# - 탑승한 곳에 따른 생존률 확인하기 

# In[28]:


f, ax = plt.subplots(1, 1, figsize=(7, 7))
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)


# - C의 생존률이 가장 높으나, 거의 비슷
# 
#     - 모델을 만들고 나면 사용한 feature가 얼마나 중요한 역할을 했는지 확인해볼 수 있음 

# In[29]:


f, ax = plt.subplots(2, 2, figsize=(20, 15))

sns.countplot(x='Embarked', data=df_train, ax=ax[0, 0])
ax[0, 0].set_title('(1) No. Of Passengers Boarded')

sns.countplot(x='Embarked', data=df_train, hue='Sex', ax=ax[0, 1])
ax[0, 1].set_title('(2) Male-Female Split for Embarked')

sns.countplot(x='Embarked', data=df_train, hue='Survived', ax=ax[1, 0])
ax[1, 0].set_title('(3) Embarked vs Survived')

sns.countplot(x='Embarked', data=df_train, hue='Pclass', ax=ax[1, 1])
ax[1, 1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()


# - Figure(1) : S에서 가장 많은 사람이 탑승함 
# - Figure(2) : C, Q는 성별의 비율이 비슷하고, S의 경우 남성이 비율이 더 높음 
# - Figure(3) : S의 경우 생존률이 가장 낮고, C의 생존률이 높음 
# - Figure(4) : S는 3rd class의 탑승자가, C는 1st class 탑승자가 많은 편 -> S는 3rd class가 많아서 생존률이 낮게 나옴 

# ## Famly - SibSp(형제 자매) + Parch(부모, 자녀)

# In[30]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
    # 자신을 포함해야 하므로 1을 더함 
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
    # 자신을 포함해야 하므로 1을 더함 


# In[31]:


print('Maximum size of Family: ', df_train['FamilySize'].max())
print('Minimum size of Family: ', df_train['FamilySize'].min())


# In[32]:


f, ax = plt.subplots(1, 3, figsize=(40, 10))

sns.countplot(x='FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot(x='FamilySize', data=df_train, hue='Survived', ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# - Figure(1) : 가족의 크기를 확인
#     - 1명 ~ 11명, 대부분 1명 
# - Figure(2), (3) : 가족 크기에 따른 생존 비교
#     - 가족이 4명이 경우 생존률이 가장 높음  
#     - 가족 수가 많아질수록, 생존률이 낮아짐 
#     - 3~4명 선에서 생존률이 높다는 것을 알 수 있음 

# ## Fare 
# - contious feature 

# In[33]:


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')


# - distribution이 매우 비대칭 (high skewness)
#     - 모델이 잘못 학습할 수도 있으므로 outlier의 영향을 줄이기 위해 log 취하기 
#     - Fare columns의 데이터 모두를 log 값 취할 때, lambda 함수를 이용해 간단한 로그를 적용하는 함수를 map에 인수로 넣어주면 Fare columns 데이터에 그대로 적용됨

# In[39]:


# testset에 있는 nan value를 평균값으로 치환
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)


# In[40]:


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')


# - log를 취하니 비대칭성이 많이 사라짐 
# - feature engineering 
#     - 모델을 학습시키기 위해, 모델의 성능을 높이기 위해 feature에 여러 조작을 가하거나, 새로운 feature을 추가하는 것 

# ## Cabin 
# - 해당 feature은 NaN이 80% 이므로, 모델에 포함시키지 않기로 함

# In[34]:


df_train.head()


# ## Ticket
# - string data이므로 작업 이후 실제 모델에 사용됨
# - 어떻게 사용할 지 고민해보기!

# In[35]:


df_train['Ticket'].value_counts()


# In[ ]:




