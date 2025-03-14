import torch
from torch_geometric.data import Data
from ..models.recommender.graph_rec import GraphRecommender

class OnlineLearner:
    def __init__(self, model: GraphRecommender, optimizer, device, batch_size=64):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.model.to(device)

    def update(self, new_interactions):
        self.model.train()
        total_loss = 0

        for i in range(0, len(new_interactions), self.batch_size):
            batch = new_interactions[i:i+self.batch_size]
            users = torch.tensor([interaction[0] for interaction in batch], dtype=torch.long).to(self.device)
            items = torch.tensor([interaction[1] for interaction in batch], dtype=torch.long).to(self.device)
            ratings = torch.tensor([interaction[2] for interaction in batch], dtype=torch.float).to(self.device)

            self.optimizer.zero_grad()

            # 获取用户和商品的嵌入
            user_emb = self.model.user_embedding(users)
            item_emb = self.model.item_embedding(items)

            # 计算预测评分
            pred_ratings = (user_emb * item_emb).sum(dim=1)

            # 计算损失
            loss = torch.nn.functional.mse_loss(pred_ratings, ratings)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(new_interactions)

    def update_graph(self, new_edges):
        # 更新图结构
        current_edges = self.model.edge_index
        updated_edges = torch.cat([current_edges, new_edges], dim=1)
        self.model.edge_index = updated_edges

    def add_new_user(self, user_id, user_features):
        with torch.no_grad():
            new_user_emb = self.model.get_cold_user_embedding(user_features)
            self.model.user_embedding.weight[user_id] = new_user_emb

    def add_new_item(self, item_id, item_features):
        with torch.no_grad():
            new_item_emb = self.model.get_cold_item_embedding(item_features)
            self.model.item_embedding.weight[item_id] = new_item_emb

# 使用示例
if __name__ == "__main__":
    # 假设我们已经有了训练好的模型和优化器
    model = GraphRecommender(num_users=1000, num_items=5000, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    online_learner = OnlineLearner(model, optimizer, device)

    # 模拟新的交互数据
    new_interactions = [(0, 100, 5), (1, 200, 4), (2, 300, 3)]  # (user_id, item_id, rating)
    loss = online_learner.update(new_interactions)
    print(f"Online learning loss: {loss}")

    # 模拟新的边（用户-商品交互）
    new_edges = torch.tensor([[0, 1, 2], [100, 200, 300]], dtype=torch.long)
    online_learner.update_graph(new_edges)

    # 添加新用户
    new_user_id = 1000  # 假设这是新用户的ID
    new_user_features = torch.randn(64)  # 假设我们有64维的用户特征
    online_learner.add_new_user(new_user_id, new_user_features)

    # 添加新商品
    new_item_id = 5000  # 假设这是新商品的ID
    new_item_features = torch.randn(64)  # 假设我们有64维的商品特征
    online_learner.add_new_item(new_item_id, new_item_features)