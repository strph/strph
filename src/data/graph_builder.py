import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, HeteroData

class GraphBuilder:
    """
    构建用户-商品交互图
    """
    def __init__(self):
        pass
    
    def build_homogeneous_graph(self, interactions, user_features, item_features, n_users):
        """
        构建同构图（将用户和商品视为同一类型的节点）
        
        Args:
            interactions: 用户-商品交互数据
            user_features: 用户特征矩阵
            item_features: 商品特征矩阵
            n_users: 用户数量
            
        Returns:
            torch_geometric.data.Data: 构建的图数据对象
        """
        # 构建边索引
        edge_index = torch.tensor([
            # 源节点索引
            np.concatenate([interactions['user_id'].values, interactions['item_id'].values + n_users]),
            # 目标节点索引
            np.concatenate([interactions['item_id'].values + n_users, interactions['user_id'].values])
        ], dtype=torch.long)
        
        # 边权重/类型
        edge_attr = torch.tensor(
            np.concatenate([interactions['interaction_type'].values, interactions['interaction_type'].values]),
            dtype=torch.float
        ).unsqueeze(1)
        
        # 节点特征
        x = torch.cat([
            torch.tensor(user_features.values, dtype=torch.float),
            torch.tensor(item_features.values, dtype=torch.float)
        ], dim=0)
        
        # 节点类型（0表示用户，1表示商品）
        node_type = torch.cat([
            torch.zeros(n_users, dtype=torch.long),
            torch.ones(len(item_features), dtype=torch.long)
        ])
        
        # 创建图数据对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_type,
            num_nodes=len(x)
        )
        
        return data
    
    def build_heterogeneous_graph(self, interactions, user_features, item_features):
        """
        构建异构图（用户和商品作为不同类型的节点）
        
        Args:
            interactions: 用户-商品交互数据
            user_features: 用户特征矩阵
            item_features: 商品特征矩阵
            
        Returns:
            torch_geometric.data.HeteroData: 构建的异构图数据对象
        """
        # 创建异构图数据对象
        data = HeteroData()
        
        # 添加节点特征
        data['user'].x = torch.tensor(user_features.values, dtype=torch.float)
        data['item'].x = torch.tensor(item_features.values, dtype=torch.float)
        
        # 添加边（用户->商品）
        data['user', 'interacts', 'item'].edge_index = torch.tensor([
            interactions['user_id'].values,
            interactions['item_id'].values
        ], dtype=torch.long)
        
        # 添加边属性
        data['user', 'interacts', 'item'].edge_attr = torch.tensor(
            interactions['interaction_type'].values, dtype=torch.float
        ).unsqueeze(1)
        
        # 添加反向边（商品->用户）
        data['item', 'interacted_by', 'user'].edge_index = torch.tensor([
            interactions['item_id'].values,
            interactions['user_id'].values
        ], dtype=torch.long)
        
        data['item', 'interacted_by', 'user'].edge_attr = torch.tensor(
            interactions['interaction_type'].values, dtype=torch.float
        ).unsqueeze(1)
        
        return data

if __name__ == "__main__":
    # 示例用法
    from preprocess import load_and_preprocess_data
    
    users, items, interactions, user_features, item_features = load_and_preprocess_data(
        '../../data/raw/users.csv',
        '../../data/raw/items.csv',
        '../../data/raw/interactions.csv'
    )
    
    n_users = users['user_id'].nunique()
    
    graph_builder = GraphBuilder()
    
    # 构建同构图
    homo_graph = graph_builder.build_homogeneous_graph(
        interactions, user_features, item_features, n_users
    )
    print("同构图信息:", homo_graph)
    
    # 构建异构图
    hetero_graph = graph_builder.build_heterogeneous_graph(
        interactions, user_features, item_features
    )
    print("异构图信息:", hetero_graph)