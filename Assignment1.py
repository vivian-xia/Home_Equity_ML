#Vivian Xia
#Assignment 1: Data Preparation – Graphs and Statistical Output


import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set



pd.set_option('display.max_rows', None) #no limit on number of rows
pd.set_option('display.max_columns', None) #no limit on number of columns
pd.set_option('display.width', None) 
pd.set_option('display.max_colwidth', -1) #sets width of columns


INFILE = 'HMEQ_Loss.csv'
df = pd.read_csv('HMEQ_Loss.csv')
print(df.head().T)


"""
Explore both the input and target variables using statistical techniques.

Note that there are missing values in almost every
column as seen from customer 3 and DEBTINC.

REASON and JOB are categorical variables. 

"""

#create a variable for target columns to make it easier to refer to in the future
TARGET_FLAG = 'TARGET_BAD_FLAG'
TARGET_LOSS = 'TARGET_LOSS_AMT'

#check data types of variables
dt = df.dtypes
print(dt)

"""

REASON and JOB are objects (categorical variables)
and the other variables are numerical.

"""

#descriptive statistics of data
x = df.describe().T 
print(x)



#classiify variables into list based on their datatype
objList = []
intList = []
floatList = []

for i in dt.index:
  if i in ([TARGET_FLAG, TARGET_LOSS]) : continue 
  if dt[i] in (["object"]): objList.append(i)
  if dt[i] in (["float64"]): floatList.append(i)
  if dt[i] in (["int64"]): intList.append(i)


print("-- OBJECTS --")
for i in objList:
  print(i)
print("\n")

print("-- INTEGER --")
for i in intList: 
  print(i)
print("\n")

print("-- FLOAT --")
for i in floatList: 
  print(i)
print("\n")



#explore the probability of a loan being defaulted and the loss amount from the categorical variables
for i in objList:
  print("Class = ", i)
  g = df.groupby(i)
  print(g[i].count())
  x = g[TARGET_FLAG].mean()
  print("Bad Loan Prob", x)
  print(".................")
  x=g[TARGET_LOSS].mean()
  print("Loss Amount", x)
  print("=============\n\n\n ")



"""

Explore variables with visualization and graphs.

"""


#pie chart of categorical variables
for i in objList:
  x = df[i].value_counts(dropna=False) #count values but do not drop NaN
  theLabels = x.axes[0].tolist()
  theSlices = list(x) #tell the slices/no. of each label
  plt.pie(theSlices, labels=theLabels, startangle=90, shadow=True, autopct="%1.1f%%")
  plt.title("Pie Chart: " + i)
  plt.show()


#histogram of numerical variables
for i in intList:
  plt.hist(df[i])
  plt.xlabel(i)
  plt.show()

for i in floatList:
  plt.hist(df[i])
  plt.xlabel(i)
  plt.show()

plt.hist(df[TARGET_LOSS])
plt.xlabel("Loss Amount")
plt.show()



#observe unique, most commonm, and number of missing category values in the categorical variables
for i in objList :
    print( i )
    print( df[i].unique() )
    g = df.groupby( i )
    print( g[i].count() )
    print( "MOST COMMON = ", df[i].mode()[0] )   
    print( "MISSING = ", df[i].isna().sum() )
    print( "\n\n")


#create new variable with filled in "MISSING" value for NaN values 
for i in objList :
    if df[i].isna().sum() == 0 : continue
    print( i ) 
    print("HAS MISSING")
    NAME = "IMP_"+i
    print( NAME ) 
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna("MISSING")
    print( "variable",i," has this many missing", df[i].isna().sum() )
    print( "variable",NAME," has this many missing", df[NAME].isna().sum() )
    g = df.groupby( NAME )
    print( g[NAME].count() )
    print( "\n\n")
    df = df.drop( i, axis=1 )



#observe new variables IMP_REASON and IMP_JOB and the absence of REASON and JOB
print(df.head().T)


#create a new list of categorical and numerical variables
dt = df.dtypes
objList = []
numList = []
for i in dt.index :
    if i in ( [ TARGET_FLAG, TARGET_LOSS ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )
    if dt[i] in (["float64","int64"]) : numList.append( i )


print(" OBJECTS ")
print(" ------- ")
for i in objList :
    print( i )
print(" ------- \n\n")


print(" NUMBER ")
print(" ------- ")
for i in numList :
    print( i )
print(" ------- ")



#encode the objects IMP_REASON and IMP_JOB then drop the columns
for i in objList :
    thePrefix = "z_" + i
    y = pd.get_dummies( df[i], prefix=thePrefix, drop_first=True ) 
    df = pd.concat( [df, y], axis=1 )
    df = df.drop( i, axis=1 )



#replace missing values in numerical variables to median value
missing_numList = []
for i in numList :
    if df[i].isna().sum() == 0 : continue
    missing_numList.append(i)

print(missing_numList)


for i in numList :
    if df[i].isna().sum() == 0 : continue #if no values are missing then continue
    FLAG = "M_" + i #notes that there was a missing value here -- highly predictive to know it was missing 
    IMP = "IMP_" + i
    print("\n-------")
    print(i)
    #print( df[i].isna().sum() ) #print how many missing values there are
    print( FLAG )
    print( IMP )
    print(" -------\n")
    df[ FLAG ] = df[i].isna() + 0 #add zero and Python will know that if TRUE it's a 1 and FALSE it's a 0
    df[ IMP ] = df[ i ] #create an exact copy of the column
    df.loc[ df[IMP].isna(), IMP ] = df[i].median() #fill missing with the median of the variable
    print(df.head().T)
    df = df.drop( i, axis=1 )




dt = df.dtypes
objList = []
numList = []
for i in dt.index :
    if i in ( [ TARGET_FLAG, TARGET_LOSS ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )
    if dt[i] in (["float64","int64"]) : numList.append( i )


print(" OBJECTS ")
print(" ------- ")
for i in objList :
    print( i )
print(" ------- \n\n")


print(" NUMBER ")
print(" ------- ")
for i in numList :
    print( i )
print(" ------- ")




print(df.head().T)




#fill missing values in TARGET_LOSS with median value
i = TARGET_LOSS
if df[i].isna().sum() != 0: #if no values are missing then continue
  FLAG = "M_" + i #notes that there was a missing value here -- highly predictive to know it was missing 
  IMP = "IMP_" + i
  print("\n-------")
  print(i)
  #print( df[i].isna().sum() ) #print how many missing values there are
  print( FLAG )
  print( IMP )
  print(" -------\n")
  df[ FLAG ] = df[i].isna() + 0 #add zero and Python will know that if TRUE it's a 1 and FALSE it's a 0
  df[ IMP ] = df[ i ] #create an exact copy of the column
  df.loc[ df[IMP].isna(), IMP ] = df[i].median() #fill missing with the median of the variable
  print(df.head().T)
  df = df.drop( i, axis=1 )




print(df.head().T)
