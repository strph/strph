import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import numpy as np

class GraphRecommenderTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train_epoch(self, train_data, batch_size):
        self.model.train()
        total_loss = 0
        loader = NeighborLoader(train_data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=True)
        
        for batch in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            
            # 获取节点嵌入
            node_emb = self.model(batch.edge_index)
            
            # 获取用户和正样本商品的嵌入
            user_emb = node_emb[batch.user_index]
            pos_item_emb = node_emb[self.model.num_users + batch.pos_item_index]
            neg_item_emb = node_emb[self.model.num_users + batch.neg_item_index]
            
            # 计算正样本和负样本的得分
            pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
            neg_scores = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=-1)
            
            # 计算 BPR 损失
            loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)

    def evaluate(self, eval_data, batch_size, k=10):
        self.model.eval()
        loader = NeighborLoader(eval_data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=False)
        
        hr_list = []
        ndcg_list = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                batch = batch.to(self.device)
                node_emb = self.model(batch.edge_index)
                
                user_emb = node_emb[batch.user_index]
                item_emb = node_emb[self.model.num_users:]
                
                # 计算用户与所有商品的得分
                scores = torch.matmul(user_emb, item_emb.t())
                
                # 获取 top-k 推荐
                _, top_indices = scores.topk(k, dim=1)
                
                # 计算 HR@k 和 NDCG@k
                for i, user_items in enumerate(top_indices):
                    pos_item = batch.pos_item_index[i].item()
                    if pos_item in user_items:
                        rank = torch.where(user_items == pos_item)[0].item()
                        hr_list.append(1)
                        ndcg_list.append(1 / np.log2(rank + 2))
                    else:
                        hr_list.append(0)
                        ndcg_list.append(0)
        
        hr = np.mean(hr_list)
        ndcg = np.mean(ndcg_list)
        
        return hr, ndcg

    def train(self, train_data, val_data, num_epochs, batch_size, early_stopping=5):
        best_hr = 0
        no_improve = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_data, batch_size)
            hr, ndcg = self.evaluate(val_data, batch_size)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")
            
            if hr > best_hr:
                best_hr = hr
                no_improve = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improve += 1
            
            if no_improve == early_stopping:
                print("Early stopping triggered")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

# 示例用法
if __name__ == "__main__":
    from recommender.src.models.recommender.graph_rec import GraphRecommender
    from recommender.src.data.dataset import ECommerceDataset, RecommendationDataset
    
    # 加载数据
    graph_data = ECommerceDataset(root='data', raw_user_data='users.csv', 
                                  raw_item_data='items.csv', raw_interaction_data='interactions.csv')
    train_data = RecommendationDataset(root='data', graph_data=graph_data[0], split='train')
    val_data = RecommendationDataset(root='data', graph_data=graph_data[0], split='val')
    
    # 创建模型
    num_users = graph_data[0]['user'].num_nodes
    num_items = graph_data[0]['item'].num_nodes
    hidden_channels = 64
    model = GraphRecommender(num_users, num_items, hidden_channels)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = GraphRecommenderTrainer(model, optimizer, device)
    
    # 训练模型
    trainer.train(train_data, val_data, num_epochs=50, batch_size=1024, early_stopping=5)