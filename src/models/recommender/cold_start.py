import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional

class ColdStartHandler:
    """
    处理冷启动问题的高级策略类
    """
    def __init__(self, model, item_features, user_features, popularity_scores=None, similarity_threshold=0.5):
        """
        初始化冷启动处理器
        
        Args:
            model: 推荐模型
            item_features: 商品特征矩阵 (n_items, n_features)
            user_features: 用户特征矩阵 (n_users, n_features)
            popularity_scores: 商品流行度分数 (可选)
            similarity_threshold: 相似度阈值，用于基于内容的推荐
        """
        self.model = model
        self.item_features = item_features
        self.user_features = user_features
        self.popularity_scores = popularity_scores
        self.similarity_threshold = similarity_threshold
        
        # 计算商品之间的相似度矩阵
        self.item_similarity = cosine_similarity(item_features)
        
        # 如果没有提供流行度分数，则使用均匀分布
        if popularity_scores is None:
            self.popularity_scores = np.ones(len(item_features)) / len(item_features)
        else:
            # 归一化流行度分数
            self.popularity_scores = popularity_scores / np.sum(popularity_scores)
    
    def recommend_for_new_user(self, user_features: np.ndarray, 
                              interaction_history: Optional[List[int]] = None, 
                              top_k: int = 10, 
                              strategy: str = 'hybrid') -> List[int]:
        """
        为新用户推荐商品
        
        Args:
            user_features: 新用户的特征向量
            interaction_history: 新用户的交互历史（如果有）
            top_k: 推荐的商品数量
            strategy: 推荐策略，可选 'content', 'popularity', 'hybrid'
            
        Returns:
            List[int]: 推荐的商品ID列表
        """
        if strategy == 'content':
            return self._content_based_recommendation(user_features, interaction_history, top_k)
        elif strategy == 'popularity':
            return self._popularity_based_recommendation(top_k)
        elif strategy == 'hybrid':
            return self._hybrid_recommendation(user_features, interaction_history, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def recommend_for_new_item(self, item_features: np.ndarray, top_k: int = 10) -> List[int]:
        """
        为新商品找到相似的商品
        
        Args:
            item_features: 新商品的特征向量
            top_k: 返回的相似商品数量
            
        Returns:
            List[int]: 相似商品ID列表
        """
        # 计算新商品与所有现有商品的相似度
        similarities = cosine_similarity([item_features], self.item_features)[0]
        
        # 获取相似度最高的商品
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices.tolist()
    
    def _content_based_recommendation(self, user_features: np.ndarray, 
                                    interaction_history: Optional[List[int]], 
                                    top_k: int) -> List[int]:
        """
        基于内容的推荐策略
        """
        # 如果用户有交互历史，基于历史计算偏好
        if interaction_history and len(interaction_history) > 0:
            # 获取用户交互过的商品的特征
            interacted_items_features = self.item_features[interaction_history]
            
            # 计算用户偏好向量（交互商品特征的平均值）
            user_preference = np.mean(interacted_items_features, axis=0)
            
            # 计算所有商品与用户偏好的相似度
            similarities = cosine_similarity([user_preference], self.item_features)[0]
            
            # 排除已交互的商品
            similarities[interaction_history] = -np.inf
        else:
            # 如果没有交互历史，直接使用用户特征
            similarities = cosine_similarity([user_features], self.item_features)[0]
        
        # 获取相似度最高的商品
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices.tolist()
    
    def _popularity_based_recommendation(self, top_k: int) -> List[int]:
        """
        基于流行度的推荐策略
        """
        # 获取流行度最高的商品
        top_indices = np.argsort(self.popularity_scores)[::-1][:top_k]
        
        return top_indices.tolist()
    
    def _hybrid_recommendation(self, user_features: np.ndarray, 
                             interaction_history: Optional[List[int]], 
                             top_k: int, 
                             alpha: float = 0.7) -> List[int]:
        """
        混合推荐策略，结合内容和流行度
        
        Args:
            alpha: 内容推荐的权重 (0-1)，流行度推荐的权重为 1-alpha
        """
        # 获取基于内容的推荐
        content_scores = np.zeros(len(self.item_features))
        content_recommendations = self._content_based_recommendation(user_features, interaction_history, len(self.item_features))
        for i, item_id in enumerate(content_recommendations):
            content_scores[item_id] = 1.0 / (i + 1)  # 根据排名赋予分数
        
        # 结合流行度分数
        hybrid_scores = alpha * content_scores + (1 - alpha) * self.popularity_scores
        
        # 如果有交互历史，排除已交互的商品
        if interaction_history and len(interaction_history) > 0:
            hybrid_scores[interaction_history] = -np.inf
        
        # 获取得分最高的商品
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        return top_indices.tolist()
    
    def update_user_profile(self, user_id: int, new_interactions: List[Tuple[int, float]]):
        """
        更新用户的偏好模型
        
        Args:
            user_id: 用户ID
            new_interactions: 新的交互数据，格式为 [(item_id, rating), ...]
        """
        # 获取交互的商品ID和评分
        item_ids = [item_id for item_id, _ in new_interactions]
        ratings = np.array([rating for _, rating in new_interactions])
        
        # 获取这些商品的特征
        items_features = self.item_features[item_ids]
        
        # 根据评分加权计算用户偏好
        weighted_features = items_features * ratings.reshape(-1, 1)
        user_preference = np.sum(weighted_features, axis=0) / np.sum(ratings)
        
        # 更新用户特征
        self.user_features[user_id] = user_preference
    
    def get_item_neighbors(self, item_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        获取商品的最相似邻居
        
        Args:
            item_id: 商品ID
            top_k: 返回的相似商品数量
            
        Returns:
            List[Tuple[int, float]]: (商品ID, 相似度分数) 的列表
        """
        # 获取商品的相似度向量
        similarities = self.item_similarity[item_id]
        
        # 排除自身
        similarities[item_id] = -np.inf
        
        # 获取相似度最高的商品
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        return list(zip(top_indices.tolist(), top_scores.tolist()))
    
    def get_diverse_recommendations(self, user_features: np.ndarray, top_k: int = 10, diversity_weight: float = 0.5) -> List[int]:
        """
        生成多样化的推荐
        
        Args:
            user_features: 用户特征
            top_k: 推荐的商品数量
            diversity_weight: 多样性的权重 (0-1)
            
        Returns:
            List[int]: 推荐的商品ID列表
        """
        # 先获取候选集（比所需的大一些）
        candidate_size = min(top_k * 3, len(self.item_features))
        candidates = self._content_based_recommendation(user_features, None, candidate_size)
        
        # 初始化推荐列表
        recommendations = [candidates[0]]
        candidates = candidates[1:]
        
        # 贪心地选择多样化的商品
        while len(recommendations) < top_k and candidates:
            # 计算候选商品与已推荐商品的最大相似度
            max_similarities = []
            for candidate in candidates:
                candidate_similarities = [self.item_similarity[candidate][rec] for rec in recommendations]
                max_similarities.append(max(candidate_similarities))
            
            # 计算候选商品与用户的相似度
            user_similarities = cosine_similarity([user_features], self.item_features[candidates])[0]
            
            # 结合相关性和多样性
            scores = (1 - diversity_weight) * user_similarities - diversity_weight * np.array(max_similarities)
            
            # 选择得分最高的商品
            best_idx = np.argmax(scores)
            recommendations.append(candidates[best_idx])
            candidates.pop(best_idx)
        
        return recommendations

# 示例用法
if __name__ == "__main__":
    # 创建模拟数据
    n_items = 1000
    n_users = 500
    n_features = 50
    
    # 随机生成特征矩阵
    item_features = np.random.rand(n_items, n_features)
    user_features = np.random.rand(n_users, n_features)
    
    # 生成模拟的流行度分数（使用幂律分布）
    popularity = np.random.power(0.5, size=n_items)
    
    # 创建冷启动处理器
    cold_start = ColdStartHandler(None, item_features, user_features, popularity)
    
    # 为新用户生成推荐
    new_user_features = np.random.rand(n_features)
    recommendations = cold_start.recommend_for_new_user(new_user_features, top_k=10)
    print("Content-based recommendations for new user:", recommendations)
    
    # 使用流行度策略
    pop_recommendations = cold_start.recommend_for_new_user(new_user_features, strategy='popularity', top_k=10)
    print("Popularity-based recommendations:", pop_recommendations)
    
    # 使用混合策略
    hybrid_recommendations = cold_start.recommend_for_new_user(new_user_features, strategy='hybrid', top_k=10)
    print("Hybrid recommendations:", hybrid_recommendations)
    
    # 为新商品找到相似商品
    new_item_features = np.random.rand(n_features)
    similar_items = cold_start.recommend_for_new_item(new_item_features, top_k=5)
    print("Similar items for new item:", similar_items)
    
    # 获取多样化推荐
    diverse_recommendations = cold_start.get_diverse_recommendations(new_user_features, top_k=10, diversity_weight=0.7)
    print("Diverse recommendations:", diverse_recommendations)