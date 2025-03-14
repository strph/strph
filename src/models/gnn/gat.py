import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        
        # 第一层GAT，多头注意力
        self.conv1 = GATConv(
            num_node_features, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        
        # 第二层GAT，将多头注意力合并为单一表示
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=1, 
            concat=False,
            dropout=dropout
        )
        
        # 输出层
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. 获取节点特征
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 2. 图级别的池化
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类层
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x

class GATRecommender(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, heads=8, dropout=0.6):
        super(GATRecommender, self).__init__()
        
        self.user_embedding = torch.nn.Embedding(num_users, hidden_channels)
        self.item_embedding = torch.nn.Embedding(num_items, hidden_channels)
        
        self.gat = GAT(hidden_channels, hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        
        self.num_users = num_users
        self.num_items = num_items

    def forward(self, edge_index, batch=None):
        # 获取用户和商品的嵌入
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # 如果没有提供batch信息，创建默认的batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 通过GAT处理
        x = self.gat(x, edge_index, batch)
        
        return x
    
    def get_embeddings(self, edge_index, batch=None):
        # 获取所有节点的嵌入
        x = self.forward(edge_index, batch)
        
        # 分离用户和商品的嵌入
        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]
        
        return user_emb, item_emb
    
    def predict(self, user_indices, item_indices, edge_index, batch=None):
        # 获取所有节点的嵌入
        x = self.forward(edge_index, batch)
        
        # 获取指定用户和商品的嵌入
        user_emb = x[user_indices]
        item_emb = x[self.num_users + item_indices]
        
        # 计算用户-商品对的得分
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores
    
    def recommend(self, user_indices, edge_index, top_k=10, batch=None):
        # 获取所有节点的嵌入
        x = self.forward(edge_index, batch)
        
        # 获取指定用户的嵌入
        user_emb = x[user_indices]
        
        # 获取所有商品的嵌入
        item_emb = x[self.num_users:]
        
        # 计算用户与所有商品的得分
        scores = torch.matmul(user_emb, item_emb.t())
        
        # 获取top-k推荐
        top_scores, top_indices = scores.topk(top_k, dim=1)
        return top_scores, top_indices

# 示例用法
if __name__ == "__main__":
    num_users, num_items = 1000, 5000
    hidden_channels = 64
    model = GATRecommender(num_users, num_items, hidden_channels)
    
    # 模拟输入数据
    edge_index = torch.randint(0, num_users + num_items, (2, 10000))
    batch = torch.zeros(num_users + num_items, dtype=torch.long)
    
    # 获取嵌入
    user_emb, item_emb = model.get_embeddings(edge_index, batch)
    print("User embeddings shape:", user_emb.shape)
    print("Item embeddings shape:", item_emb.shape)
    
    # 预测评分
    user_indices = torch.randint(0, num_users, (100,))
    item_indices = torch.randint(0, num_items, (100,))
    scores = model.predict(user_indices, item_indices, edge_index, batch)
    print("Prediction scores shape:", scores.shape)
    
    # 获取推荐
    top_scores, top_indices = model.recommend(user_indices[:10], edge_index, top_k=5, batch=batch)
    print("Top-5 recommendations shape:", top_indices.shape)
    print("Top-5 scores shape:", top_scores.shape)