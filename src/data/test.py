import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 定义用户数量和商品数量
num_users = 1000
num_items = 5000

# 生成用户数据
user_ids = np.arange(num_users)
# 假设用户特征：年龄、性别、职业
ages = np.random.randint(18, 60, num_users)
genders = np.random.choice(['Male', 'Female'], num_users)
occupations = np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist'], num_users)

# 对非数值类型的特征进行编码
gender_encoder = LabelEncoder()
occupation_encoder = LabelEncoder()

genders_encoded = gender_encoder.fit_transform(genders)
occupations_encoded = occupation_encoder.fit_transform(occupations)

# 对编码后的分类特征进行独热编码
onehot_encoder = OneHotEncoder(sparse_output=False)
genders_onehot = onehot_encoder.fit_transform(genders_encoded.reshape(-1, 1))
occupations_onehot = onehot_encoder.fit_transform(occupations_encoded.reshape(-1, 1))

# 合并所有特征
users_features = np.hstack([ages.reshape(-1, 1), genders_onehot, occupations_onehot])

users = pd.DataFrame({
    'user_id': user_ids,
    **{f'feature_{i}': users_features[:, i] for i in range(users_features.shape[1])}
})

# 生成商品数据
item_ids = np.arange(num_items)
# 假设商品特征：类别、价格
categories = np.random.choice(['Electronics', 'Clothing', 'Books', 'Home Decor'], num_items)
prices = np.random.uniform(10, 200, num_items)

# 对商品类别进行编码
category_encoder = LabelEncoder()
categories_encoded = category_encoder.fit_transform(categories)

# 对商品类别进行独热编码
categories_onehot = onehot_encoder.fit_transform(categories_encoded.reshape(-1, 1))

# 合并商品特征
items_features = np.hstack([categories_onehot, prices.reshape(-1, 1)])

items = pd.DataFrame({
    'item_id': item_ids,
    **{f'feature_{i}': items_features[:, i] for i in range(items_features.shape[1])}
})

# 生成交互数据
# 假设每个用户平均与 10 个商品有交互
interaction_count_per_user = 10
total_interactions = num_users * interaction_count_per_user

user_indices = np.repeat(user_ids, interaction_count_per_user)
item_indices = np.random.choice(item_ids, total_interactions)
# 假设评分范围是 1 到 5
ratings = np.random.randint(1, 6, total_interactions)

interactions = pd.DataFrame({
    'user_id': user_indices,
    'item_id': item_indices,
    'rating': ratings,
    'interaction_type': ratings  # 假设交互类型和评分相同
})

# 保存数据集
users.to_csv('data/raw/users.csv', index=False)
items.to_csv('data/raw/items.csv', index=False)
interactions.to_csv('data/raw/interactions.csv', index=False)

print("数据集生成完成，已保存为 users.csv, items.csv, interactions.csv")