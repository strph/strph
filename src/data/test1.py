# 假设这是加载用户和商品特征的代码
import pandas as pd

users = pd.read_csv('data/raw/users.csv')
items = pd.read_csv('data/raw/items.csv')

print("Users shape:", users.shape)
print("Items shape:", items.shape)

