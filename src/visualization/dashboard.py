import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.recommender.graph_rec import GraphRecommender
from src.models.gnn.gat import GATRecommender
from src.utils.visualization import RecommenderVisualizer
from src.utils.ab_testing import ABTest
from src.models.recommender.cold_start import ColdStartHandler

class RecommenderDashboard:
    def __init__(self, gcn_model: GraphRecommender, gat_model: GATRecommender, 
                 dataset, cold_start_handler: ColdStartHandler):
        self.gcn_model = gcn_model
        self.gat_model = gat_model
        self.dataset = dataset
        self.cold_start_handler = cold_start_handler
        self.visualizer = RecommenderVisualizer(gcn_model, dataset)
    
    def run(self):
        st.set_page_config(page_title="Recommender System Dashboard", layout="wide")
        st.title("Graph-based Recommender System Dashboard")
        
        menu = ["Overview", "Model Comparison", "User Analysis", "Item Analysis", "Cold Start Analysis"]
        choice = st.sidebar.selectbox("Menu", menu)
        
        if choice == "Overview":
            self.show_overview()
        elif choice == "Model Comparison":
            self.show_model_comparison()
        elif choice == "User Analysis":
            self.show_user_analysis()
        elif choice == "Item Analysis":
            self.show_item_analysis()
        elif choice == "Cold Start Analysis":
            self.show_cold_start_analysis()
    
    def show_overview(self):
        st.header("System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.write(f"Number of Users: {self.dataset.num_users}")
            st.write(f"Number of Items: {self.dataset.num_items}")
            st.write(f"Number of Interactions: {len(self.dataset.interactions)}")
        
        with col2:
            st.subheader("Model Information")
            st.write(f"GCN Model Hidden Channels: {self.gcn_model.hidden_channels}")
            st.write(f"GAT Model Hidden Channels: {self.gat_model.hidden_channels}")
        
        st.subheader("Interaction Graph")
        fig = self.visualizer.visualize_interaction_graph()
        st.plotly_chart(fig)
    
    def show_model_comparison(self):
        st.header("Model Comparison")
        
        # 假设我们有一些模型性能指标
        metrics = {
            'HR@10': {'GCN': 0.75, 'GAT': 0.78},
            'NDCG@10': {'GCN': 0.65, 'GAT': 0.68},
            'Recall@10': {'GCN': 0.70, 'GAT': 0.72}
        }
        
        ab_test = ABTest(name="GCN vs GAT Recommender", control_group="GCN", test_group="GAT")
        for metric, values in metrics.items():
            ab_test.add_metric(metric, higher_is_better=True)
            ab_test.add_observation("GCN", metric, values['GCN'])
            ab_test.add_observation("GAT", metric, values['GAT'])
        
        results = ab_test.run_test()
        summary = ab_test.summary()
        
        st.subheader("A/B Test Results")
        st.dataframe(summary)
        
        st.subheader("Performance Visualization")
        fig = go.Figure()
        for metric in metrics.keys():
            fig.add_trace(go.Bar(x=['GCN', 'GAT'], y=[metrics[metric]['GCN'], metrics[metric]['GAT']], name=metric))
        fig.update_layout(barmode='group', xaxis_title="Model", yaxis_title="Score")
        st.plotly_chart(fig)
    
    def show_user_analysis(self):
        st.header("User Analysis")
        
        # 假设我们有一些用户交互数据
        user_interactions = np.random.randint(1, 100, size=self.dataset.num_users)
        
        st.subheader("User Interaction Distribution")
        fig = px.histogram(user_interactions, nbins=50, labels={'value': 'Number of Interactions', 'count': 'Number of Users'})
        fig.update_layout(title="Distribution of User Interactions")
        st.plotly_chart(fig)
        
        st.subheader("User Embeddings Visualization")
        emb_fig = self.visualizer.visualize_embeddings()
        st.pyplot(emb_fig)
    
    def show_item_analysis(self):
        st.header("Item Analysis")
        
        # 假设我们有一些商品交互数据
        item_interactions = np.random.randint(1, 200, size=self.dataset.num_items)
        
        st.subheader("Item Popularity Distribution")
        fig = px.histogram(item_interactions, nbins=50, labels={'value': 'Number of Interactions', 'count': 'Number of Items'})
        fig.update_layout(title="Distribution of Item Popularity")
        st.plotly_chart(fig)
        
        st.subheader("Item Similarity Analysis")
        item_id = st.number_input("Enter Item ID", min_value=0, max_value=self.dataset.num_items-1, value=0)
        neighbors = self.cold_start_handler.get_item_neighbors(item_id, top_k=10)
        
        neighbor_df = pd.DataFrame(neighbors, columns=['Item ID', 'Similarity Score'])
        st.dataframe(neighbor_df)
    
    def show_cold_start_analysis(self):
        st.header("Cold Start Analysis")
        
        st.subheader("New User Recommendation")
        new_user_features = np.random.rand(self.dataset.user_feature_dim)
        
        strategy = st.selectbox("Recommendation Strategy", ['content', 'popularity', 'hybrid'])
        recommendations = self.cold_start_handler.recommend_for_new_user(new_user_features, strategy=strategy)
        
        st.write(f"Top 10 recommendations for new user using {strategy} strategy:")
        st.write(recommendations)
        
        st.subheader("New Item Analysis")
        new_item_features = np.random.rand(self.dataset.item_feature_dim)
        similar_items = self.cold_start_handler.recommend_for_new_item(new_item_features)
        
        st.write("Top 10 similar items for new item:")
        st.write(similar_items)
        
        st.subheader("Diverse Recommendations")
        diverse_recommendations = self.cold_start_handler.get_diverse_recommendations(new_user_features)
        st.write("Top 10 diverse recommendations:")
        st.write(diverse_recommendations)

if __name__ == "__main__":
    # 这里需要加载实际的模型、数据集和冷启动处理器
    # 为了演示，我们使用一些模拟数据
    class DummyDataset:
        def __init__(self):
            self.num_users = 1000
            self.num_items = 5000
            self.interactions = [(i, j, 1) for i in range(100) for j in range(50)]
            self.user_feature_dim = 64
            self.item_feature_dim = 64
    
    dataset = DummyDataset()
    gcn_model = GraphRecommender(dataset.num_users, dataset.num_items, hidden_channels=64)
    gat_model = GATRecommender(dataset.num_users, dataset.num_items, hidden_channels=64)
    
    item_features = np.random.rand(dataset.num_items, dataset.item_feature_dim)
    user_features = np.random.rand(dataset.num_users, dataset.user_feature_dim)
    popularity = np.random.power(0.5, size=dataset.num_items)
    
    cold_start_handler = ColdStartHandler(gcn_model, item_features, user_features, popularity)
    
    dashboard = RecommenderDashboard(gcn_model, gat_model, dataset, cold_start_handler)
    dashboard.run()