import pandas as pd
import numpy as np

df1 = pd.read_csv(r"C:\Users\Asus\Downloads\Bengaluru_House_Data.csv")
# print(df1.head())
# print(df1.shape)

df2 = df1.drop(['area_type','availability','society','balcony'],axis='columns')
# print(df2.head())
# print(df2.shape)

# print(df2.isnull().sum())
df3 =df2.dropna()
# print(df3.shape)

# print(df3.head(10))

# print(df3['size'].unique())
df3 = df3.copy()
df3['bhk'] =df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3['bhk'].unique())
# print(df3.loc[30])

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
float_val = df3[~df3['total_sqft'].apply(is_float)]
# print(float_val.head(10))


def convert_sqft_to_num(x):
    tokens  =x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4 = df3.copy()
df4.total_sqft =df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notna()]
# print(df4.head())
# print(df4.loc[30])

df5 =df4.copy()
df5['price_per_sqft'] =df5['price']*100000/df5['total_sqft']
# print(df5.head())

# print(df5['price_per_sqft'].describe())

# df5.to_csv('bhp.csv',index=False)

# print(len(df5.location.unique()))
df5.location =df5.location.apply(lambda x: x.strip())
location_stats =df5['location'].value_counts(ascending=False)
# print(len(location_stats))
# print(location_stats.values.sum())
# print(len(location_stats[location_stats>10]))
# print(len(location_stats))
# print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
# print(location_stats_less_than_10)

# print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(len(df5.location.unique()))


df5[df5.total_sqft/df5.bhk<300].head()
# print(df5.shape)

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
# print(df6.shape)

# print(df6.price_per_sqft.describe())

# Outlier Removal Using Standard Deviation and Mean
def remove_pps_outlier(df):
    df_out =pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st =np.std(subdf.price_per_sqft)
        reduced_out = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out =pd.concat([df_out,reduced_out],ignore_index=True)
    return df_out

df7 = remove_pps_outlier(df6)
# print(df7.shape)

# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like
import matplotlib.pyplot as plt
import matplotlib
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location)&(df.bhk==2)]
    bhk3 = df[(df.location == location)&(df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] =(15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color= 'blue', label ='2 bhk', s =50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color= 'green', label ='3 bhk', s =50, marker='+')
    plt.xlabel('Total sqft area')
    plt.ylabel('Price(Lakh INR)')
    plt.title(location)
    plt.legend()
    plt.show()


# plot_scatter_chart(df7,"Hebbal")

# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment
def remove_bhk_outliers(df):
    exclude_indices= np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats ={}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
# print(df8.shape)

# Outlier Removal Using Bathrooms Feature
# print(df8[df8.bath>10])
# It is unusual to have 2 more bathrooms than number of bedrooms in a home

# Again the business manager has a conversation with you (i.e. a data scientist)
#  that if you have 4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, 
#  you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

df9= df8[df8.bath<df8.bhk+2]
# print(df9.shape)

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
# print(df10.head(3))


# Use One Hot Encoding For Location
dummies = pd.get_dummies(df10.location)
df11= pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df12 = df11.drop('location',axis='columns')

# Build a Model Now...
X =df12.drop(['price'], axis ='columns')
# print(x)
y= df12.price
# print(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
# print(lr_clf.score(X_test,y_test))
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

# cv = ShuffleSplit(n_splits=5, test_size=0.2,random_state=10)
# print(cross_val_score(LinearRegression(),x,y,cv=cv))
# print(cross_val_score(LinearRegression(),x,y,cv=cv).mean())

# Test the model for few properties
def predict_price(location,sqft,bath,bhk):
    # loc_index =np.where(X.columns==location)[0][0] not needed
    x=np.zeros(len(X.columns))
    x[0]= sqft
    x[1] = bath
    x[2] =bhk  
    # if loc_index >=0: not needed
    #     x[loc_index]=1 not needed
    # return lr_clf.predict([x])[0] not needed

    if location in X.columns:
        loc_index = X.columns.get_loc(location)
        x[loc_index] = 1  # One-hot encoding for location
    
    # Convert to DataFrame to match training feature names
    x_df = pd.DataFrame([x], columns=X.columns)
    
    return lr_clf.predict(x_df)[0]

predicted =predict_price('1st Phase JP Nagar',1000, 2, 2)
print(predicted)

# Export the tested model to a pickle file
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# Export location and column information to a file
# that will be useful later on in our prediction application

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))