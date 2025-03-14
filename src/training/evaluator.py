import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader

class GraphRecommenderEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, test_data, batch_size, k_list=[1, 5, 10, 20]):
        loader = NeighborLoader(test_data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=False)
        
        metrics = {k: {'HR': [], 'NDCG': [], 'MRR': []} for k in k_list}
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                batch = batch.to(self.device)
                node_emb = self.model(batch.edge_index)
                
                user_emb = node_emb[batch.user_index]
                item_emb = node_emb[self.model.num_users:]
                
                # 计算用户与所有商品的得分
                scores = torch.matmul(user_emb, item_emb.t())
                
                # 获取 top-k 推荐
                _, top_indices = scores.topk(max(k_list), dim=1)
                
                # 计算各项指标
                for i, user_items in enumerate(top_indices):
                    pos_item = batch.pos_item_index[i].item()
                    if pos_item in user_items:
                        rank = torch.where(user_items == pos_item)[0].item()
                        for k in k_list:
                            if rank < k:
                                metrics[k]['HR'].append(1)
                                metrics[k]['NDCG'].append(1 / np.log2(rank + 2))
                                metrics[k]['MRR'].append(1 / (rank + 1))
                            else:
                                metrics[k]['HR'].append(0)
                                metrics[k]['NDCG'].append(0)
                                metrics[k]['MRR'].append(0)
                    else:
                        for k in k_list:
                            metrics[k]['HR'].append(0)
                            metrics[k]['NDCG'].append(0)
                            metrics[k]['MRR'].append(0)
        
        # 计算平均值
        for k in k_list:
            for metric in metrics[k]:
                metrics[k][metric] = np.mean(metrics[k][metric])
        
        return metrics

    def evaluate_cold_start(self, cold_start_data, batch_size, k_list=[1, 5, 10, 20]):
        # 实现冷启动用户的评估逻辑
        # 这里可以使用与 evaluate 方法类似的逻辑，但只针对冷启动用户
        pass

    def evaluate_diversity(self, test_data, batch_size, k=10):
        loader = NeighborLoader(test_data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=False)
        
        all_recommendations = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating Diversity"):
                batch = batch.to(self.device)
                node_emb = self.model(batch.edge_index)
                
                user_emb = node_emb[batch.user_index]
                item_emb = node_emb[self.model.num_users:]
                
                # 计算用户与所有商品的得分
                scores = torch.matmul(user_emb, item_emb.t())
                
                # 获取 top-k 推荐
                _, top_indices = scores.topk(k, dim=1)
                all_recommendations.extend(top_indices.cpu().numpy())
        
        all_recommendations = np.array(all_recommendations)
        
        # 计算推荐列表的多样性
        unique_items = np.unique(all_recommendations)
        diversity = len(unique_items) / (len(all_recommendations) * k)
        
        return diversity

    def evaluate_novelty(self, test_data, item_popularity, batch_size, k=10):
        loader = NeighborLoader(test_data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=False)
        
        novelty_scores = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating Novelty"):
                batch = batch.to(self.device)
                node_emb = self.model(batch.edge_index)
                
                user_emb = node_emb[batch.user_index]
                item_emb = node_emb[self.model.num_users:]
                
                # 计算用户与所有商品的得分
                scores = torch.matmul(user_emb, item_emb.t())
                
                # 获取 top-k 推荐
                _, top_indices = scores.topk(k, dim=1)
                
                # 计算新颖性得分
                for user_items in top_indices:
                    user_novelty = -np.mean([np.log2(item_popularity[item.item()]) for item in user_items])
                    novelty_scores.append(user_novelty)
        
        average_novelty = np.mean(novelty_scores)
        return average_novelty

# 示例用法
if __name__ == "__main__":
    from recommender.src.models.recommender.graph_rec import GraphRecommender
    from recommender.src.data.dataset import ECommerceDataset, RecommendationDataset
    
    # 加载数据
    graph_data = ECommerceDataset(root='data', raw_user_data='users.csv', 
                                  raw_item_data='items.csv', raw_interaction_data='interactions.csv')
    test_data = RecommendationDataset(root='data', graph_data=graph_data[0], split='test')
    
    # 创建模型
    num_users = graph_data[0]['user'].num_nodes
    num_items = graph_data[0]['item'].num_nodes
    hidden_channels = 64
    model = GraphRecommender(num_users, num_items, hidden_channels)
    
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 创建评估器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = GraphRecommenderEvaluator(model, device)
    
    # 评估模型
    metrics = evaluator.evaluate(test_data, batch_size=1024)
    for k, values in metrics.items():
        print(f"Top-{k} recommendations:")
        for metric, score in values.items():
            print(f"  {metric}: {score:.4f}")
    
    # 评估多样性
    diversity = evaluator.evaluate_diversity(test_data, batch_size=1024)
    print(f"Recommendation Diversity: {diversity:.4f}")
    
    # 评估新颖性 (需要先计算商品流行度)
    item_popularity = np.random.rand(num_items)  # 这里使用随机值代替实际的商品流行度
    novelty = evaluator.evaluate_novelty(test_data, item_popularity, batch_size=1024)
    print(f"Recommendation Novelty: {novelty:.4f}")