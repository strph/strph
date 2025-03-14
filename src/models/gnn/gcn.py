import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. 获取节点特征
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)

        # 2. 图级别的池化
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类层
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GCNRecommender(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels):
        super(GCNRecommender, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, hidden_channels)
        self.item_embedding = torch.nn.Embedding(num_items, hidden_channels)
        self.gcn = GCN(hidden_channels, hidden_channels, hidden_channels)

    def forward(self, user_ids, item_ids, edge_index, batch):
        # 获取用户和商品的嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 合并用户和商品节点
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # 通过GCN处理
        x = self.gcn(x, edge_index, batch)
        
        # 分离用户和商品的嵌入
        user_emb, item_emb = x[:len(user_ids)], x[len(user_ids):]
        
        return user_emb, item_emb

    def predict(self, user_emb, item_emb):
        return (user_emb * item_emb).sum(dim=-1)

# 示例用法
if __name__ == "__main__":
    num_users, num_items = 100, 50
    hidden_channels = 64
    model = GCNRecommender(num_users, num_items, hidden_channels)
    
    # 模拟输入数据
    user_ids = torch.randint(0, num_users, (10,))
    item_ids = torch.randint(0, num_items, (20,))
    edge_index = torch.randint(0, num_users + num_items, (2, 100))
    batch = torch.zeros(num_users + num_items, dtype=torch.long)
    
    user_emb, item_emb = model(user_ids, item_ids, edge_index, batch)
    print("User embeddings shape:", user_emb.shape)
    print("Item embeddings shape:", item_emb.shape)
    
    # 预测评分
    scores = model.predict(user_emb, item_emb[:10])
    print("Prediction scores shape:", scores.shape)