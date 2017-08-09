import pandas as pd
df=pd.read_csv("data.csv")
'''
df=df.groupby('name').agg({'symptom':'count'}).reset_index().rename(columns={'symptom':'countsym'})
print(df)
print(df.loc[:,'countsym'])
'''
