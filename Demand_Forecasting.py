#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 3-month item-level sales forecast for different store.
# There are 10 different stores and 50 different items in a 5-year dataset.
# Accordingly, we need to give forecasts for 3 months after the store-item breakdown.

#---------------------------------------------------------------

# Farklı store için 3 aylık item-level sales tahmini.
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pd.set_option("display.max_columns",None)
pd.set_option("display.float_format",lambda x : "%.4f" %x)
pd.set_option("display.width",200)
#pd.set_option("display.max_rows",200)


# In[4]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[5]:


train_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/demand_forecasting/train.csv"
test_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/demand_forecasting/test.csv"
sample_path = "/Users/gokhanersoz/Desktop/VBO_Dataset/demand_forecasting/sample_submission.csv"


# In[6]:


########################
# Loading the data
########################


# In[7]:


train = pd.read_csv(train_path,parse_dates = ["date"])
test = pd.read_csv(test_path,parse_dates=["date"])
sample_sub = pd.read_csv(sample_path)

print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))
print("Sample Shape : {}".format(sample_sub.shape))


# In[8]:


df = pd.concat([train,test],axis = 0 ,sort = False)

print("DataFrame Shape : {}".format(df.shape))


# In[9]:


#####################################################
# EDA
#####################################################


# In[10]:


def check_df(dataframe, num = 5):
    
    print(" shape ".upper().center(50,"#"),end = "\n\n" )
    print(dataframe.shape,end = "\n\n")
    
    print(" types ".upper().center(50,"#"),end = "\n\n" )
    print(dataframe.dtypes,end = "\n\n")
    
    print(" head ".upper().center(50,"#"),end = "\n\n" )
    print(dataframe.head(num),end = "\n\n")
    
    print(" tail ".upper().center(50,"#"),end = "\n\n" )
    print(dataframe.tail(num),end = "\n\n")
    
    print(" quantiles ".upper().center(50,"#"),end = "\n\n" )
    print(dataframe.quantile([0,.01 ,.05, .50, .95 ,.99, 1]).T,end = "\n\n")       
                                   


# In[11]:


print("For Train ;")
for col in train:
    if col  not in "date":
        print(f"{col.upper()} Nunique : {train[col].nunique()}")

print("\n")
        
print("For Test ;")
for col in test:
    if col not in "date":
        print(f"{col.upper()} Nunique : {test[col].nunique()}")

print("\n")
        
print("For Sample ;")
for col in sample_sub:
        print(f"{col.upper()} Nunique : {sample_sub[col].nunique()}")


# In[12]:


print("Train DataFrame:\nTrain Min Date : {}\nTrain Max Date : {}\n ".format(train.date.min(),train.date.max()))
print("Test DataFrame:\nTest Min Date : {}\nTest Max Date : {} ".format(test.date.min(),test.date.max()))


# In[13]:


check_df(train)


# In[14]:


check_df(test)


# In[15]:


check_df(sample_sub)


# In[16]:


check_df(df)


# In[17]:


# Sales Distribution

df[["sales"]].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T


# In[18]:


df.groupby(["store"])["item"].nunique()


# In[19]:


df.groupby(["store","item"]).agg({"sales" : "sum"}).T


# In[20]:


df.groupby(["store","item"]).agg({"sales" : ["sum","mean","median","std"]}).T


# In[21]:


def store_sales(dataframe):
    
    plt.figure(figsize = (20,12))
    for i in range(1,11):
    
        plt.subplot(2,5,i)
        dataframe[dataframe["store"] == i]["sales"].hist()
        plt.title(f"For Store {i}",fontsize = 15)
        plt.tight_layout()        


# In[22]:


store_sales(train)


# In[23]:


# I want to do

plt.figure(figsize = (15,6))
example = train[train["store"] == 1].set_index("date")
example[example["item"] == 1]["sales"].plot(label = "For Items 1")
plt.legend(loc = "upper left")
plt.title("Store 1 Distributions")
plt.show()


# In[24]:


def store_item_sales(dataframe,store = 1):
    
    plt.figure(figsize = (50,50))
    store_groupby = train[train["store"] == store].set_index("date")
    
    for item in range(1,51):
            
            plt.subplot(10,5,item)
            store_groupby[store_groupby["item"] == item]["sales"].plot(label = f"Item {item} Sales")
            plt.legend(loc = "upper left")
    
    plt.tight_layout(pad = 4.5)
    plt.suptitle(f"For Store {store} Item Sales Distribution",y = 1.01 ,x=0.5,fontsize = 16)
    plt.show()


# In[25]:


store_item_sales(train,store = 1)


# In[26]:


#####################################################
# FEATURE ENGINEERING
#####################################################


# In[27]:


from scipy.stats import shapiro,levene,mannwhitneyu,ttest_ind

def ab_test(GroupA,GroupB):
    
    #print("""

    #Normality Test

    # H0: Normal dağılım varsayımı sağlanmaktadır.
    # H1:..sağlanmamaktadır.
    
    #Homogeneity
    
    # H0: Varyanslar Homojendir
    # H1: Varyanslar Homojen Değildir
        

    # 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
    # 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

    # Eğer normallik sağlanmazsa her türlü nonparametrik test yapacağız.
    # Eger normallik sağlanır varyans homojenliği sağlanmazsa ne olacak?
    # T test fonksiyonuna arguman gireceğiz.

    # H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
    # H1: M1 != M2 (...vardır)
    
    #-----------------------------------------------------------
    
    #Normality Test

     # H0: Assumption of normal distribution is provided.
     # H1:..not provided.
    
     #homogeneity
    
     # H0: Variances are Homogeneous
     # H1: Variances Are Not Homogeneous
        

     #1. Independent two-sample t-test (parametric test) if assumptions are met
     # 2. Mannwhitneyu test if assumptions are not provided (non-parametric test)

     # If normality is not achieved, we will do all kinds of nonparametric tests.
     # What will happen if normality is achieved and variance homogeneity is not achieved?
     # We will enter arguments to the t test function.

     # H0: M1 = M2 (... there is no significant difference between the mean of the two groups.)
     # H1: M1 != M2 (...is)
    
    #""")
    
    
    testA = shapiro(GroupA)[1] < 0.05
    testB = shapiro(GroupB)[1] < 0.05
    
    if (testA == False) & (testB == False):
        
        leveneTest = levene(GroupA, GroupB)[1] < 0.05
        
        if leveneTest == False:
            
            ttest = ttest_ind(GroupA, GroupB, equal_var=True)[1]
            
        else:
            
            ttest = ttest_ind(GroupA, GroupB, equal_var=False)[1]
            
    else:
        
        ttest = mannwhitneyu(GroupA, GroupB)[1]
        
    results = pd.DataFrame({"Compare_Two_Groups" : [ttest < 0.05],
                            "p-value" : [round(ttest,5)],  
                            "GroupA_Mean"   : [GroupA.mean()]   ,"GroupB_Mean"   : [GroupB.mean()],
                            "GroupA_Median" : [GroupA.median()] ,"GroupB_Median" : [GroupB.median()],
                            "GroupA_Count"  : [GroupA.count()]  ,"GroupB_Count"  : [GroupB.count()]
                           })
    
    # H0 < 0.05 ise zaten True olur ve gruplar arasında istatiksel olarak fark vardır !!!! Red ediyor...
    # Değil ise zaten bunlar eşittir...
    # Kafanı boşa yordun zaten biliyorsun bunu neden yordun ki....
    
    # If H0 < 0.05 it is already True and there is a statistical difference between groups !!!! He refuses...
    # If not, they are already equal...
    # You're wasting your mind, you already know, why did you get tired of this....
    
    results["Compare_Two_Groups"] = np.where(results["Compare_Two_Groups"] ==True,"Different_Groups",
                                                                                  "Similar_Groups")
    
    results["TestType"] = np.where( (testA == False) & (testB == False) , "Parametric", "Non-Parametric")
    
    return results


# In[28]:


# Her mağazada 50 tane item var 1 store da 50 tane item var ve onların toplamını alıyoruz....
# There are 50 items in each store. There are 50 items in 1 store and we take their sum....

import itertools

store_sales = df.groupby(["date","store"])["sales"].sum().reset_index().set_index("date")

combination_stores = list(itertools.combinations( df["store"].unique(), 2))

AB_Stores = pd.DataFrame()

for i,j in combination_stores:
    
    GroupA = store_sales[store_sales["store"] == i]["sales"]
    GroupB = store_sales[store_sales["store"] == j]["sales"]
    
    data = ab_test(GroupA,GroupB)
    data.index = [(i,j)]
    
    data = data[["TestType","Compare_Two_Groups","p-value","GroupA_Mean","GroupB_Mean",
                 "GroupA_Median","GroupB_Median","GroupA_Count","GroupB_Count",]]
    
    AB_Stores = pd.concat([AB_Stores,data])


# In[29]:


test_1 = df[df["date"] == "2013-01-01"][["store","sales","item","date"]].set_index("date")
test_1[test_1["store"] == 1].sum()


# In[30]:


store_sales.T


# In[31]:


AB_Stores


# In[32]:


AB_Stores[AB_Stores["Compare_Two_Groups"] == "Similar_Groups"]


# In[33]:


similar_stores = AB_Stores[AB_Stores["Compare_Two_Groups"] == "Similar_Groups"].index.tolist()
similar_stores


# ---

# In[34]:


item_sales = df.groupby(["date","item"])["sales"].sum().reset_index().set_index("date")

combination_items = list(itertools.combinations(item_sales["item"].unique(), 2))

AB_items = pd.DataFrame()

for i , j in combination_items:
    
    GroupA = item_sales[item_sales["item"] == i]["sales"]
    GroupB = item_sales[item_sales["item"] == j]["sales"]
    
    data = ab_test(GroupA,GroupB)
    data.index = [(i,j)]
    
    data = data[["TestType","Compare_Two_Groups","p-value","GroupA_Mean","GroupB_Mean",
                 "GroupA_Median","GroupB_Median","GroupA_Count","GroupB_Count"]]
    
    AB_items = pd.concat([AB_items,data],axis = 0)


# In[35]:


test_2 = df[df["date"] == "2013-01-01"][["item","sales","date"]].set_index("date")
test_2[test_2["item"] == 1][["sales"]].sum()


# In[36]:


item_sales.head()


# In[37]:


AB_items


# In[38]:


AB_items[AB_items["Compare_Two_Groups"] == "Similar_Groups"]


# In[39]:


similar_items = AB_items[AB_items["Compare_Two_Groups"] == "Similar_Groups"].index.tolist()
similar_items[:10]


# In[40]:


df.head()


# In[41]:


for i in range(7):
    print(f"{i} : {i//4}")


# In[42]:


def create_date_features(dataframe):
    
    dataframe["day"]   = dataframe.date.dt.day
    dataframe["month"] = dataframe.date.dt.month
    dataframe["year"]  = dataframe.date.dt.year
    
    dataframe["day_of_week"]  = dataframe.date.dt.day_of_week
    dataframe["day_of_year"]  = dataframe.date.dt.day_of_year
    dataframe["week_of_year"] = dataframe.date.dt.weekofyear
    
    dataframe["day_name"]   = dataframe.date.dt.day_name()
    dataframe["month_name"] = dataframe.date.dt.month_name()
    
    # Friday Contains !!!
    dataframe["is_wknd"] = dataframe.date.dt.weekday // 4
    
        
    # We need to convert int because results return bools
    
    dataframe["is_month_start"] = dataframe.date.dt.is_month_start.astype(int)
    dataframe["is_month_end"]   = dataframe.date.dt.is_month_end.astype(int)
    
    dataframe["is_year_start"]  = dataframe.date.dt.is_year_start.astype(int)
    dataframe["is_year_end"]    = dataframe.date.dt.is_year_end.astype(int)
    
    dataframe["is_quarter_start"] = dataframe.date.dt.is_quarter_start.astype(int)
    dataframe["is_quarter_end"]   = dataframe.date.dt.is_quarter_end.astype(int)
    
    
    #  How much is this month 
    dataframe["days_in_month"] = dataframe.date.dt.days_in_month
    
    # Winter : 0
    # Spring : 1
    # Summer : 2
    # Fail : 3
    
    # just list !!!!
    
    dataframe["season"] = np.where(dataframe.month.isin([12,1,2]),  0, np.nan)
    dataframe["season"] = np.where(dataframe.month.isin([3,4,5]),   1, dataframe["season"])
    dataframe["season"] = np.where(dataframe.month.isin([6,7,8]),   2, dataframe["season"])
    dataframe["season"] = np.where(dataframe.month.isin([9,10,11]), 3, dataframe["season"])
    
    #*******
    
    # Stores Sales Similar
    
    dataframe["StoreSalesSimilar"] = np.where(dataframe.store.isin([3,10])  ,1, 0)
    dataframe["StoreSalesSimilar"] = np.where(dataframe.store.isin([4,9] )  ,1, 0)
    dataframe["StoreSalesSimilar"] = np.where(dataframe.store.isin([5,6] )  ,1, 0)
    
    # Items Sales Similar
    
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([1,4,27,41,47])  , 1, 0)
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([2,6,7,14,31,46]), 2, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([3,42])          , 3, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([8,36])          , 4, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([9,43,48])       , 5, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([11,12,29,33])   , 6, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([13,18])         , 7, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([15,28])         , 8, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([16,34])         , 9, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([19,21,30])      , 10, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([20,26])         , 11, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([22,25,38,45])   , 12, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([23,37,40,44,49]), 13, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([24,35,50])      , 14, dataframe["ItemSalesSimilar"])
    dataframe["ItemSalesSimilar"] = np.where(dataframe.item.isin([32,39])         , 15, dataframe["ItemSalesSimilar"])


    import holidays
    
    holidays_date = holidays.USA(years = [2013,2014,2015,2016,2017,2018]).keys()
    
    dataframe["holidays"] = dataframe.date.isin(holidays_date).astype(int)
        
    return dataframe


# In[43]:


df = create_date_features(df)


# In[44]:


df.isnull().sum()


# In[45]:


df.head()


# In[46]:


df.groupby(["store","item","month"]).agg({"sales":["sum","mean","median","std"]})


# In[47]:


########################
# Random Noise
########################


# In[48]:


def random_noise(dataframe):
    return np.random.normal(scale = 1.6, size = (len(dataframe),))


# In[49]:


########################
# Lag/Shifted Features
########################


# In[50]:


df.head()


# In[51]:


df = df.sort_values(by = ["store","item","date"] ,axis = 0)
df.head()


# In[52]:


pd.DataFrame({"sales" : df["sales"].values[:10],
              "lag1"  : df["sales"].shift(1).values[:10],
              "lag2"  : df["sales"].shift(2).values[:10],
              "lag3"  : df["sales"].shift(3).values[:10],
              "lag4"  : df["sales"].shift(4).values[:10]} )


# In[53]:


df.groupby(["store","item"])["sales"].head()


# In[54]:


df.groupby(["store","item"])["sales"].transform(lambda x : x.shift(1))


# In[55]:


exp = df.loc[912991: ,].groupby(["store","item"])[["sales"]].head(15)
exp["shift_sales"] = df.loc[912991:,].groupby(["store","item"])[["sales"]].shift(5).head(15)
exp


# In[56]:


def lag_features(dataframe, lags):
    
    for lag in lags:
        
        dataframe["sales_lag_"+str(lag)] = dataframe.groupby(["store","item"])["sales"].transform(                                            lambda x : x.shift(lag)) + random_noise(dataframe)
        
    return dataframe


# In[57]:


# Test verisinde +90 gün tahmin edilmesi isteniyor bu yüzden
# Lag featureları en az 91 olmalı!

# It is requested to estimate +90 days in the test data so
# Lag features must be at least 91!

df = lag_features(df , [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


# In[58]:


df.loc[:,df.columns.str.contains("sales_lag_")]


# In[59]:


########################
# Rolling Mean Features
########################


# In[60]:


df.sales.values[:10]


# In[61]:


# Shift Olunca Kendisi Dahil değil !!!
# Ama Normalde Kendisi Dahil !!!

# It Is Not Included With Shift !!!
# But Normally Himself Included!!!

df.sales.rolling(window = 2).mean().values[:10]


# In[62]:


# Biz shift koyarak bizden öncekiler ile tahmin yapmak !!!!

# We make guesses with our predecessors by putting the shift !!!!

pd.DataFrame({"sales"       : df["sales"].values[:10],
              "shif1"       : df["sales"].shift(1).values[:10],
              "roll2"       : df["sales"].rolling(window = 2).mean().values[:10],
              "roll2_shift1": df["sales"].shift(1).rolling(window = 2).mean().values[:10],
              "roll2_shift1_min": df["sales"].shift(1).rolling(window = 2,min_periods=2).mean().values[:10],
              "roll3"       : df["sales"].rolling(window = 3).mean().values[:10],
              "roll3_shift1": df["sales"].shift(1).rolling(window = 3).mean().values[:10],
              "roll3_shift1_min": df["sales"].shift(1).rolling(window = 3,min_periods=3).mean().values[:10],
              "roll4"       : df["sales"].rolling(window = 4).mean().values[:10],
              "roll4_shift1_min": df["sales"].shift(1).rolling(window = 4,min_periods=3).mean().values[:10]})


# In[63]:


roll = pd.DataFrame({'B': [0,1,2, np.nan, 4,5,6,7,np.nan,8,9,10]})

roll["roll3"] = roll["B"].rolling(window = 3, min_periods=2).mean()
roll["roll3_shift"] = roll["B"].shift(1).rolling(window = 3, min_periods=2).mean()
roll


# In[64]:


def roll_mean_features(dataframe,windows):
    
    for window in windows:
        dataframe["sales_window_"+str(window)] =         dataframe.groupby(["store","item"])["sales"].transform(        lambda x : x.shift(1).rolling(window = window, min_periods = 10, win_type = "triang").mean()) +        random_noise(dataframe)
        
    return dataframe


# In[65]:


windows = [91,98,105,112,119,126,182,364,546,728]

df = roll_mean_features(df,windows)


# In[66]:


df.loc[:, df.columns.str.contains("sales_window_")]


# In[67]:


########################
# Exponentially Weighted Mean Features
########################


# In[68]:


pd.DataFrame({"sales" : df["sales"].values[:10],
              "sales_shift" : df["sales"].shift(1).values[:10],
              "roll2": df["sales"].rolling(window = 2).mean().values[:10],
              "roll2_shift": df["sales"].shift(1).rolling(window = 2).mean().values[:10],
              "ewm.99" : df["sales"].shift(1).ewm(alpha = 0.99).mean().values[:10],
              "ewm.90" : df["sales"].shift(1).ewm(alpha = 0.90).mean().values[:10]})


# In[69]:


13*0.99 + 14*0.01


# In[70]:


def ewm_features(dataframe, alphas, lags):
    
    for alpha in alphas:
        for lag in lags:
            dataframe["sales_ewm_alpha_"+str(alpha)+"_lag_"+str(lag)] =              dataframe.groupby(["store","item"])["sales"].transform(lambda x :                                                                     x.shift(lag).ewm(alpha=alpha).mean())
            
    return dataframe


# In[71]:


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)


# In[72]:


########################
# One-Hot Encoding
########################


# In[73]:


df.head()


# In[74]:


df = df.drop(["day_name","month_name"], axis = 1)


# In[75]:


pd.DataFrame({"Name" : df.columns,
              "Nunique" : [df[col].nunique() for col in df.columns]}).set_index("Name").sort_values("Nunique").T


# In[76]:


df = pd.get_dummies(df , columns = ["store","item","month","year","day_of_week"])
df.shape


# In[77]:


pd.DataFrame({"Name"   : df.columns,
              "Dtypes" : [df[col].dtype for col in df.columns]}).set_index("Name").T


# In[78]:


########################
# Converting sales to log(1+sales)
########################


# In[79]:


df["sales"] = np.log1p(df["sales"].values)
df.sales


# In[80]:


import pickle

pd.to_pickle(df, open("DemandForecast/Demand_Forecast.pkl","wb"))


# ---

# In[81]:


#####################################################
# Model
#####################################################


# In[82]:


########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


# In[83]:


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# In[84]:


########################
# Time-Based Validation Sets
########################


# In[85]:


print("For Train : ")
print("Train Min Date : {}".format(train.date.min()))
print("Train Max Date : {}".format(train.date.max()))
print("Train Shape : {}".format(train.shape))


# In[86]:


# Train DataFrame Set 

train_df = df.loc[(df.date < "2017-01-01")]
print("Train DataFrame Shape : {}".format(train_df.shape))


# In[87]:


# Test DataFrame Set

valid_df = df.loc[ (df.date >= "2017-01-01") & (df.date < "2017-04-01"), :]
print("Validation DataFrame : {}".format(valid_df.shape))


# In[88]:


df.head()


# In[89]:


# Not Necessary

cols = [col for col in train_df.columns if col not in ["date","id","sales","year"]]


# In[90]:


Y_train = train_df["sales"]
X_train = train_df[cols]

print("Y_Train Shape : {}".format(Y_train.shape))
print("X_Train Shape : {}".format(X_train.shape))


# In[91]:


Y_valid = valid_df["sales"]
X_valid = valid_df[cols]

print("Y_Train Shape : {}".format(Y_valid.shape))
print("X_Train Shape : {}".format(X_valid.shape))


# In[92]:


########################
# LightGBM Model
########################


# In[93]:


# LightGBM parameters

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate : shrinkage_rate, eta
# feature_fraction : rf'in rundom subspace özelliği. her iterasyonda rastgele bulundurulacak iterasyon sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations, en az 10000-15000 civarı yapmak lazım

# early_stopping_rounds validasyon setindeki metric belirli bir early_stopping_rounds da ilerlemiyorsa yani hata
# düşmüyorsa modellemeyi durdurur
# hem train süresini kısaltır hem de overfit'e engel olur
# nthread: num_thread, nthreads, n_jobs


# In[94]:


from lightgbm import LGBMRegressor
import lightgbm as lgb


# In[95]:


lgb_params = {"metric" : {"mae"},
              "num_leaves" : 10,
              "learning_rate" :0.02,
              "feature_fraction" : 0.8,
              "max_depth" : 5,
              "verbose" : 0,
              "num_boost_round" : 1000,
              "early_stopping_rounds" : 200,
              "nthread" : -1}


# In[96]:


# Bu veri yapılarıyla işlemlerin daha hızlı sürdüğünü düşünüyorlar.
# Bu yüzden lgb nesnesinin içindeki dataseti kullanıyorum.

# They think transactions go faster with these data structures.
# That's why I'm using the dataset inside the lgb object.

lgbtrain = lgb.Dataset(data = X_train, label = Y_train , feature_name=cols)
lgbvalid = lgb.Dataset(data = X_valid, label = Y_valid, reference=lgbtrain , feature_name=cols)


# In[97]:


# Burdaki lgbvalid değeri ile lgbtrain arasında karşılaştırma yapmak için aldık
# Ve referance olarak lgbtrain alsın !!!!

# We took it to compare the lgbvalid value here and lgbtrain
# And take lgbtrain as reference !!!!

model = lgb.train(params= lgb_params,
                  train_set=lgbtrain,
                  num_boost_round=lgb_params["num_boost_round"],
                  early_stopping_rounds=lgb_params["early_stopping_rounds"],
                  valid_sets=[lgbtrain,lgbvalid],
                  feval = lgbm_smape,
                  verbose_eval=100)


# In[98]:


Y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)


# In[99]:


smape(np.expm1(Y_pred_valid) ,np.expm1(Y_valid))


# In[100]:


from sklearn.metrics import mean_absolute_error

mea_values = mean_absolute_error(Y_valid,Y_pred_valid)
mea_values


# In[101]:


########################
# Feature Importance
########################


# In[102]:


def plot_lgb_importance(model , plot = False , nums = 10):
    
    gain  = model.feature_importance(importance_type = "gain")
    split = model.feature_importance(importance_type = "split")
    feature = model.feature_name()
    
    feat_imp = pd.DataFrame({"Feature" : feature,
                             "Gain" : 100* gain / gain.sum(),
                             "split" : split}).sort_values(by = "Gain", ascending = False)
    
    if plot :
        
        plt.figure(figsize = (15,8))
        sns.barplot(y = "Feature", x = "Gain", data = feat_imp[:nums])
        plt.title("FEATURES GAIN", fontsize = 15)
        plt.xlabel("Gain",fontsize = 15)
        plt.ylabel("Features" , fontsize = 15)
        plt.tight_layout()
        plt.show()
        
    return feat_imp


# In[103]:


feature_importance = plot_lgb_importance(model,plot = True, nums = 20)


# In[104]:


zero_gain = feature_importance[feature_importance["Gain"] == 0]["Feature"].tolist()
zero_gain[:10]


# In[105]:


########################
# Final Model
########################


# In[106]:


new_cols = [col for col in cols if col not in zero_gain]
len(new_cols),len(zero_gain),len(cols)


# In[107]:


final_train = df.loc[~df.sales.isna()]

print("Final Train DataFrame Shape : {}".format(final_train.shape))


# In[108]:


final_Y_train = final_train["sales"]
final_X_train = final_train[cols]
final_X_train_new = final_train[new_cols]

print("Final Y Train DataFrame Shape : {}".format(final_Y_train.shape))
print("Final X Train DataFrame Shape : {}".format(final_X_train.shape))
print("Final X Train New DataFrame Shape : {}".format(final_X_train_new.shape))


# In[109]:


final_test = df.loc[df.sales.isna()]
final_X_test = final_test[cols]
final_X_test_new = final_test[new_cols]

print("Final X Test DataFrame Shape : {}".format(final_X_test.shape))
print("Final X Test New DataFrame Shape : {}".format(final_X_test_new.shape))


# In[110]:


final_lgbtrain = lgb.Dataset(data = final_X_train, label = final_Y_train, feature_name=cols)
final_lgbtrain_new = lgb.Dataset(data = final_X_train_new, label = final_Y_train, feature_name=new_cols)


# In[111]:


lgb_params = {"metric" : {"mae"},
              "num_leaves" : 10,
              "learning_rate" :0.02,
              "feature_fraction" : 0.8,
              "max_depth" : 5,
              "verbose" : 0,
              "num_boost_round" : model.best_iteration,
              "nthread" : -1}


# In[112]:


best_model = lgb.train(params = lgb_params,
                       train_set=final_lgbtrain,
                       num_boost_round=lgb_params["num_boost_round"])


# In[113]:


best_model_new = lgb.train(params = lgb_params,
                           train_set=final_lgbtrain_new,
                           num_boost_round=lgb_params["num_boost_round"])


# In[114]:


final_predictions = best_model.predict(data = final_X_test, num_iteration=best_model.best_iteration)
final_predictions[:10]


# In[115]:


final_predictions_new = best_model_new.predict(data = final_X_test_new, num_iteration=best_model_new.best_iteration)
final_predictions_new[:10]


# In[116]:


#####################################################
# Create submission
#####################################################


# In[117]:


test.head()


# In[118]:


submission_df = test.loc[:,["id"]]
submission_df["sales"] = np.expm1(final_predictions)
submission_df["id"] = submission_df.id.astype(int)
submission_df.to_csv("DemandForecast/submission_demand.csv",index = False)
submission_df.head()


# In[119]:


#####################################################
#Graphically Comparison Actual And Estimated Values
#####################################################


# In[120]:


review_test = pd.concat([test[["date","store","item"]], submission_df["sales"]], axis = 1)
review_test.head()


# In[121]:


def final_store_item_sales(train,test,store = 1):
    
    train_groupby = train[train["store"] == store].set_index("date")
    test_groupby = test[test["store"] == store].set_index("date")
    
    plt.figure(figsize = (50,50))
    
    for i in range(1,51):
        
        plt.subplot(10,5,i)
        train_groupby[train_groupby["item"] == i]["sales"].plot(legend = "Actuals")
        test_groupby[test_groupby["item"] == i]["sales"].plot(legend= "Predictions")
        plt.title(f"For Item {i} Sales Distribution")
        plt.legend(loc = "upper left")
    
    plt.tight_layout()
    plt.suptitle(f"For Store {store} Sales Distribution",y = 1.01, x= 0.5, fontsize = 15)
    plt.show()   


# In[122]:


final_store_item_sales(train, review_test, store = 1)


# In[123]:


pd.to_pickle(best_model, open("DemandForecast/Final_LGBM.pkl", "wb"))


# In[ ]:




