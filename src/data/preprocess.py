import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(user_data_path, item_data_path, interaction_data_path):
    """
    加载并预处理用户、商品和交互数据
    """
    # 加载数据
    users = pd.read_csv(user_data_path)
    items = pd.read_csv(item_data_path)
    interactions = pd.read_csv(interaction_data_path)

    # 对用户和商品ID进行编码
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    users['user_id'] = user_encoder.fit_transform(users['user_id'])
    items['item_id'] = item_encoder.fit_transform(items['item_id'])
    interactions['user_id'] = user_encoder.transform(interactions['user_id'])
    interactions['item_id'] = item_encoder.transform(interactions['item_id'])

    # 创建用户和商品特征
    user_features = pd.get_dummies(users.drop('user_id', axis=1))
    item_features = pd.get_dummies(items.drop('item_id', axis=1))

    return users, items, interactions, user_features, item_features

def create_user_item_matrix(interactions, n_users, n_items):
    """
    创建用户-商品交互矩阵
    """
    user_item_matrix = np.zeros((n_users, n_items))
    for _, row in interactions.iterrows():
        user_item_matrix[row['user_id'], row['item_id']] = row['interaction_type']
    return user_item_matrix

if __name__ == "__main__":
    # 示例用法
    users, items, interactions, user_features, item_features = load_and_preprocess_data(
        'data/raw/users.csv',
        'data/raw/items.csv',
        'data/raw/interactions.csv'
    )
    
    n_users = users['user_id'].nunique()
    n_items = items['item_id'].nunique()
    
    user_item_matrix = create_user_item_matrix(interactions, n_users, n_items)
    print("用户-商品交互矩阵形状:", user_item_matrix.shape)
    print("用户特征维度:", user_features.shape)
    print("商品特征维度:", item_features.shape)