import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Tuple
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class RecommenderVisualizer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        
    def visualize_embeddings(self, n_components: int = 2, perplexity: int = 30, 
                           sample_size: Optional[int] = None) -> plt.Figure:
        """
        使用t-SNE可视化用户和商品的嵌入向量
        
        Args:
            n_components: t-SNE降维后的维度
            perplexity: t-SNE的perplexity参数
            sample_size: 可选的采样大小，用于大数据集
            
        Returns:
            matplotlib.figure.Figure: 可视化图形
        """
        # 获取嵌入
        with torch.no_grad():
            user_emb = self.model.user_embedding.weight.cpu().numpy()
            item_emb = self.model.item_embedding.weight.cpu().numpy()
        
        # 采样（如果需要）
        if sample_size is not None:
            user_indices = np.random.choice(len(user_emb), min(sample_size, len(user_emb)), replace=False)
            item_indices = np.random.choice(len(item_emb), min(sample_size, len(item_emb)), replace=False)
            user_emb = user_emb[user_indices]
            item_emb = item_emb[item_indices]
        
        # 合并用户和商品嵌入
        embeddings = np.vstack([user_emb, item_emb])
        labels = ['User'] * len(user_emb) + ['Item'] * len(item_emb)
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 创建可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=[0 if label == 'User' else 1 for label in labels],
                            cmap='viridis', alpha=0.6)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Users', 'Items'])
        plt.title('t-SNE Visualization of User and Item Embeddings')
        
        return plt.gcf()
    
    def visualize_interaction_graph(self, max_nodes: int = 100) -> go.Figure:
        """
        使用Plotly可视化用户-商品交互图
        
        Args:
            max_nodes: 最大显示的节点数量
            
        Returns:
            plotly.graph_objects.Figure: 交互式图形
        """
        # 获取边信息
        edge_index = self.model.edge_index.cpu().numpy()
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点
        n_users = self.model.num_users
        n_items = self.model.num_items
        
        # 限制节点数量
        users = list(range(min(n_users, max_nodes // 2)))
        items = list(range(n_users, n_users + min(n_items, max_nodes // 2)))
        
        # 添加节点
        for user in users:
            G.add_node(user, type='user')
        for item in items:
            G.add_node(item, type='item')
            
        # 添加边
        for i in range(edge_index.shape[1]):
            source, target = edge_index[:, i]
            if source in users and target-n_users+n_users in items:
                G.add_edge(source, target)
        
        # 使用spring_layout布局
        pos = nx.spring_layout(G)
        
        # 创建边的轨迹
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
        # 创建节点的轨迹
        node_trace = go.Scatter(
            x=[], y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Type',
                    xanchor='left',
                    titleside='right'
                )
            ))

        # 添加节点位置
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            node_trace['text'] += ('User ' + str(node),) if node < n_users else ('Item ' + str(node-n_users),)
            
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='User-Item Interaction Graph',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
        
        return fig
    
    def plot_recommendation_metrics(self, metrics_history: Dict[str, List[float]]) -> plt.Figure:
        """
        绘制推荐指标随时间的变化
        
        Args:
            metrics_history: 包含不同指标历史记录的字典
            
        Returns:
            matplotlib.figure.Figure: 指标变化图
        """
        plt.figure(figsize=(12, 6))
        for metric_name, values in metrics_history.items():
            plt.plot(values, label=metric_name)
        
        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('Recommendation Metrics Over Time')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_user_item_distribution(self, user_interactions: Dict[int, int], 
                                  item_interactions: Dict[int, int]) -> Tuple[plt.Figure, plt.Figure]:
        """
        绘制用户和商品的交互分布
        
        Args:
            user_interactions: 用户ID到交互次数的映射
            item_interactions: 商品ID到交互次数的映射
            
        Returns:
            Tuple[plt.Figure, plt.Figure]: 用户和商品交互分布图
        """
        # 用户交互分布
        plt.figure(figsize=(10, 6))
        plt.hist(list(user_interactions.values()), bins=50)
        plt.xlabel('Number of Interactions')
        plt.ylabel('Number of Users')
        plt.title('Distribution of User Interactions')
        user_fig = plt.gcf()
        
        # 商品交互分布
        plt.figure(figsize=(10, 6))
        plt.hist(list(item_interactions.values()), bins=50)
        plt.xlabel('Number of Interactions')
        plt.ylabel('Number of Items')
        plt.title('Distribution of Item Interactions')
        item_fig = plt.gcf()
        
        return user_fig, item_fig
    
    def plot_cold_start_analysis(self, cold_start_performance: Dict[str, List[float]], 
                               baseline_performance: Dict[str, List[float]]) -> plt.Figure:
        """
        绘制冷启动用户/商品的性能分析
        
        Args:
            cold_start_performance: 冷启动情况下的性能指标
            baseline_performance: 基准性能指标
            
        Returns:
            matplotlib.figure.Figure: 冷启动分析图
        """
        metrics = list(cold_start_performance.keys())
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, [cold_start_performance[m][-1] for m in metrics], width, label='Cold Start')
        ax.bar(x + width/2, [baseline_performance[m][-1] for m in metrics], width, label='Baseline')
        
        ax.set_ylabel('Performance')
        ax.set_title('Cold Start vs Baseline Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        return fig

# 使用示例
if __name__ == "__main__":
    from recommender.src.models.recommender.graph_rec import GraphRecommender
    from recommender.src.data.dataset import ECommerceDataset
    
    # 创建模型和数据集（这里仅作示例）
    model = GraphRecommender(num_users=1000, num_items=5000, hidden_channels=64)
    dataset = None  # 实际使用时需要提供真实的数据集
    
    visualizer = RecommenderVisualizer(model, dataset)
    
    # 可视化嵌入
    emb_fig = visualizer.visualize_embeddings()
    plt.show()
    
    # 可视化交互图
    graph_fig = visualizer.visualize_interaction_graph()
    graph_fig.show()
    
    # 可视化指标历史
    metrics_history = {
        'HR@10': [0.5, 0.6, 0.7, 0.75],
        'NDCG@10': [0.3, 0.4, 0.45, 0.5],
        'Recall@10': [0.4, 0.5, 0.55, 0.6]
    }
    metrics_fig = visualizer.plot_recommendation_metrics(metrics_history)
    plt.show()
    
    # 可视化用户和商品交互分布
    user_interactions = {i: np.random.randint(1, 100) for i in range(1000)}
    item_interactions = {i: np.random.randint(1, 200) for i in range(5000)}
    user_fig, item_fig = visualizer.plot_user_item_distribution(user_interactions, item_interactions)
    plt.show()
    
    # 可视化冷启动分析
    cold_start_performance = {
        'HR@10': [0.3, 0.35, 0.4],
        'NDCG@10': [0.2, 0.25, 0.3],
        'Recall@10': [0.25, 0.3, 0.35]
    }
    baseline_performance = {
        'HR@10': [0.6, 0.65, 0.7],
        'NDCG@10': [0.4, 0.45, 0.5],
        'Recall@10': [0.5, 0.55, 0.6]
    }
    cold_start_fig = visualizer.plot_cold_start_analysis(cold_start_performance, baseline_performance)
    plt.show()