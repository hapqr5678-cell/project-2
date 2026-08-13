import pandas as pd 

df = pd.read_csv("tokyo_places.csv")
print(df.category.unique())

