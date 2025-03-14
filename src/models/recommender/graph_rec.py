import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import to_undirected

class GraphRecommender(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, num_layers=2, dropout=0.5, model_type='sage'):
        super(GraphRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type
        
        # 用户和商品的嵌入层
        self.user_embedding = torch.nn.Embedding(num_users + 1, hidden_channels)  # +1 for cold start users
        self.item_embedding = torch.nn.Embedding(num_items + 1, hidden_channels)  # +1 for cold start items
        
        # 冷启动用户和商品的平均嵌入
        self.cold_user_embedding = torch.nn.Parameter(torch.zeros(hidden_channels))
        self.cold_item_embedding = torch.nn.Parameter(torch.zeros(hidden_channels))
        
        # 图神经网络层
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if model_type == 'sage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            elif model_type == 'gat':
                self.convs.append(GATConv(hidden_channels, hidden_channels))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # 输出层
        self.lin = torch.nn.Linear(hidden_channels, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.1)
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, edge_index):
        # 获取用户和商品的初始嵌入
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # 确保边是无向的
        edge_index = to_undirected(edge_index)
        
        # 应用图卷积层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict(self, user_indices, item_indices, x):
        # 获取用户和商品的最终嵌入
        user_emb = x[user_indices]
        item_emb = x[self.num_users + item_indices]
        
        # 计算用户-商品对的得分
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores
    
    def recommend(self, user_indices, x, top_k=10):
        # 获取用户的最终嵌入
        user_emb = x[user_indices]
        
        # 计算用户与所有商品的得分
        item_emb = x[self.num_users:]
        scores = torch.matmul(user_emb, item_emb.t())
        
        # 获取top-k推荐
        top_scores, top_indices = scores.topk(top_k, dim=1)
        return top_scores, top_indices

    def cold_start_recommend(self, user_features=None, item_features=None, top_k=10):
        if user_features is not None:
            # 为新用户生成推荐
            user_emb = self.get_cold_user_embedding(user_features)
            item_emb = self.item_embedding.weight[:-1]  # Exclude cold start item
        elif item_features is not None:
            # 为新商品生成相似商品
            item_emb = self.get_cold_item_embedding(item_features)
            user_emb = self.user_embedding.weight[:-1]  # Exclude cold start user
        else:
            raise ValueError("Either user_features or item_features must be provided")

        scores = torch.matmul(user_emb, item_emb.t())
        top_scores, top_indices = scores.topk(top_k, dim=1)
        return top_scores, top_indices

    def get_cold_user_embedding(self, user_features):
        # 使用用户特征生成冷启动用户的嵌入
        return self.cold_user_embedding + torch.matmul(user_features, self.user_embedding.weight[:-1].t())

    def get_cold_item_embedding(self, item_features):
        # 使用商品特征生成冷启动商品的嵌入
        return self.cold_item_embedding + torch.matmul(item_features, self.item_embedding.weight[:-1].t())

# 示例用法
if __name__ == "__main__":
    num_users, num_items = 1000, 5000
    hidden_channels = 64
    model = GraphRecommender(num_users, num_items, hidden_channels)
    
    # 模拟输入数据
    edge_index = torch.randint(0, num_users + num_items, (2, 10000))
    
    # 前向传播
    x = model(edge_index)
    print("Node embeddings shape:", x.shape)
    
    # 预测评分
    user_indices = torch.randint(0, num_users, (100,))
    item_indices = torch.randint(0, num_items, (100,))
    scores = model.predict(user_indices, item_indices, x)
    print("Prediction scores shape:", scores.shape)
    
    # 获取推荐
    top_scores, top_indices = model.recommend(user_indices[:10], x, top_k=5)
    print("Top-5 recommendations shape:", top_indices.shape)
    print("Top-5 scores shape:", top_scores.shape)