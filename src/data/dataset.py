import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, download_url, Data
from src.data.graph_builder import GraphBuilder
from src.data.preprocess import load_and_preprocess_data

class ECommerceDataset(Dataset):
    """
    电商平台图数据集
    """
    def __init__(self, root, raw_user_data, raw_item_data, raw_interaction_data, 
                 transform=None, pre_transform=None, pre_filter=None,
                 use_heterogeneous=True):
        """
        Args:
            root: 数据保存的根目录
            raw_user_data: 原始用户数据路径
            raw_item_data: 原始商品数据路径
            raw_interaction_data: 原始交互数据路径
            transform: 数据转换函数
            pre_transform: 数据预转换函数
            pre_filter: 数据预过滤函数
            use_heterogeneous: 是否使用异构图
        """
        self.raw_user_data = raw_user_data
        self.raw_item_data = raw_item_data
        self.raw_interaction_data = raw_interaction_data
        self.use_heterogeneous = use_heterogeneous
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return [self.raw_user_data, self.raw_item_data, self.raw_interaction_data]
    
    @property
    def processed_file_names(self):
        if self.use_heterogeneous:
            return ['heterogeneous_graph.pt']
        else:
            return ['homogeneous_graph.pt']
    
    def download(self):
        # 如果数据需要从网络下载，可以在这里实现
        # 本例中假设数据已经存在
        pass
    
    def process(self):
        # 加载并预处理数据
        users, items, interactions, user_features, item_features = load_and_preprocess_data(
            os.path.join(self.raw_dir, os.path.basename(self.raw_user_data)),
            os.path.join(self.raw_dir, os.path.basename(self.raw_item_data)),
            os.path.join(self.raw_dir, os.path.basename(self.raw_interaction_data))
        )
        
        n_users = users['user_id'].nunique()
        
        # 构建图
        graph_builder = GraphBuilder()
        
        if self.use_heterogeneous:
            graph = graph_builder.build_heterogeneous_graph(
                interactions, user_features, item_features
            )
            torch.save(graph, os.path.join(self.processed_dir, 'heterogeneous_graph.pt'))
        else:
            graph = graph_builder.build_homogeneous_graph(
                interactions, user_features, item_features, n_users
            )
            torch.save(graph, os.path.join(self.processed_dir, 'homogeneous_graph.pt'))
    
    def len(self):
        return 1
    
    def get(self, idx):
        if idx > 0:
            raise IndexError('Dataset contains only one graph')
        
        if self.use_heterogeneous:
            data = torch.load(os.path.join(self.processed_dir, 'heterogeneous_graph.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 'homogeneous_graph.pt'))
        
        return data

class RecommendationDataset(Dataset):
    """
    推荐系统训练数据集，用于生成训练样本
    """
    def __init__(self, root, graph_data, split='train', transform=None, 
                 pre_transform=None, pre_filter=None, negative_samples=4):
        """
        Args:
            root: 数据保存的根目录
            graph_data: 图数据
            split: 数据集划分（'train', 'val', 'test'）
            transform: 数据转换函数
            pre_transform: 数据预转换函数
            pre_filter: 数据预过滤函数
            negative_samples: 每个正样本对应的负样本数量
        """
        self.graph_data = graph_data
        self.split = split
        self.negative_samples = negative_samples
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'{self.split}_samples.pt']
    
    def download(self):
        pass
    
    def process(self):
        # 从图数据中提取用户-商品交互
        if hasattr(self.graph_data, 'edge_index'):  # 同构图
            user_item_edges = self.graph_data.edge_index[:, self.graph_data.edge_index[0] < self.graph_data.node_type.sum()]
            n_users = (self.graph_data.node_type == 0).sum().item()
            n_items = (self.graph_data.node_type == 1).sum().item()
        else:  # 异构图
            user_item_edges = self.graph_data['user', 'interacts', 'item'].edge_index
            n_users = self.graph_data['user'].x.size(0)
            n_items = self.graph_data['item'].x.size(0)
        
        # 创建正样本
        positive_samples = user_item_edges.t()
        
        # 划分数据集
        n_samples = positive_samples.size(0)
        indices = torch.randperm(n_samples)
        
        if self.split == 'train':
            samples = positive_samples[indices[:int(0.8 * n_samples)]]
        elif self.split == 'val':
            samples = positive_samples[indices[int(0.8 * n_samples):int(0.9 * n_samples)]]
        else:  # test
            samples = positive_samples[indices[int(0.9 * n_samples):]]
        
        # 生成负样本
        users = samples[:, 0]
        neg_samples = []
        
        for user_idx in users:
            user_neg_items = []
            while len(user_neg_items) < self.negative_samples:
                neg_item = torch.randint(0, n_items, (1,)).item()
                # 检查是否为正样本
                if not ((user_item_edges[0] == user_idx) & (user_item_edges[1] == neg_item)).any():
                    user_neg_items.append(neg_item)
            neg_samples.append(user_neg_items)
        
        neg_samples = torch.tensor(neg_samples)
        
        # 保存数据
        data = {
            'users': users,
            'pos_items': samples[:, 1],
            'neg_items': neg_samples
        }
        
        torch.save(data, os.path.join(self.processed_dir, f'{self.split}_samples.pt'))
    
    def len(self):
        data = torch.load(os.path.join(self.processed_dir, f'{self.split}_samples.pt'))
        return len(data['users'])

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.split}_samples.pt'))
        # 创建 Data 对象
        graph_data = Data(
            user=data['users'][idx],
            pos_item=data['pos_items'][idx],
            neg_items=data['neg_items'][idx],
            # 这里需要添加 edge_index，你可以根据实际情况修改
            edge_index=self.graph_data['user', 'interacts', 'item'].edge_index
        )
        return graph_data