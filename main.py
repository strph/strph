import torch
import argparse
import matplotlib.pyplot as plt
from src.data.dataset import ECommerceDataset, RecommendationDataset
from src.models.recommender.graph_rec import GraphRecommender
from src.models.gnn.gat import GATRecommender
from src.training.trainer import GraphRecommenderTrainer
from src.training.evaluator import GraphRecommenderEvaluator
from src.training.online_learner import OnlineLearner
from src.utils.ab_testing import ABTest
from src.utils.visualization import RecommenderVisualizer

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    graph_data = ECommerceDataset(root=args.data_dir, raw_user_data='users.csv', 
                                  raw_item_data='items.csv', raw_interaction_data='interactions.csv')
    train_data = RecommendationDataset(root=args.data_dir, graph_data=graph_data[0], split='train')
    val_data = RecommendationDataset(root=args.data_dir, graph_data=graph_data[0], split='val')
    test_data = RecommendationDataset(root=args.data_dir, graph_data=graph_data[0], split='test')

    # 创建模型
    print("Creating models...")
    num_users = graph_data[0]['user'].num_nodes
    num_items = graph_data[0]['item'].num_nodes
    gcn_model = GraphRecommender(num_users, num_items, args.hidden_channels)
    gat_model = GATRecommender(num_users, num_items, args.hidden_channels)

    # 设置优化器
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=args.lr)
    gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=args.lr)

    # 创建训练器
    gcn_trainer = GraphRecommenderTrainer(gcn_model, gcn_optimizer, device)
    gat_trainer = GraphRecommenderTrainer(gat_model, gat_optimizer, device)

    # 训练模型
    print("Training models...")
    gcn_trainer.train(train_data, val_data, num_epochs=args.epochs, batch_size=args.batch_size, early_stopping=args.early_stopping)
    gat_trainer.train(train_data, val_data, num_epochs=args.epochs, batch_size=args.batch_size, early_stopping=args.early_stopping)

    # 创建评估器
    gcn_evaluator = GraphRecommenderEvaluator(gcn_model, device)
    gat_evaluator = GraphRecommenderEvaluator(gat_model, device)

    # 在测试集上评估模型
    print("Evaluating models...")
    gcn_metrics = gcn_evaluator.evaluate(test_data, batch_size=args.batch_size)
    gat_metrics = gat_evaluator.evaluate(test_data, batch_size=args.batch_size)

    # 创建A/B测试
    ab_test = ABTest(name="GCN vs GAT Recommender", control_group="GCN", test_group="GAT")
    
    # 添加指标
    for k in gcn_metrics.keys():
        for metric in gcn_metrics[k].keys():
            ab_test.add_metric(f"{metric}@{k}", higher_is_better=True)
    
    # 添加A/B测试数据
    for k, values in gcn_metrics.items():
        for metric, score in values.items():
            ab_test.add_observation("GCN", f"{metric}@{k}", score)
    
    for k, values in gat_metrics.items():
        for metric, score in values.items():
            ab_test.add_observation("GAT", f"{metric}@{k}", score)
    
    # 运行A/B测试
    ab_results = ab_test.run_test()
    print("\nA/B Test Results:")
    print(ab_test.summary())
    
    # 可视化A/B测试结果
    ab_fig = ab_test.plot_results()
    ab_fig.savefig('ab_test_results.png')
    print("A/B test results visualization saved as 'ab_test_results.png'")

    # 创建在线学习器
    online_learner = OnlineLearner(gcn_model, gcn_optimizer, device)

    # 模拟在线学习
    print("\nSimulating online learning...")
    for i in range(10):
        # 模拟新的交互数据
        new_interactions = [(i, i+100, 5) for i in range(10)]  # (user_id, item_id, rating)
        loss = online_learner.update(new_interactions)
        print(f"Online learning iteration {i+1}, loss: {loss:.4f}")

    # 创建可视化器
    visualizer = RecommenderVisualizer(gcn_model, graph_data)

    # 可视化嵌入
    print("\nVisualizing embeddings...")
    emb_fig = visualizer.visualize_embeddings()
    emb_fig.savefig('embeddings_visualization.png')
    print("Embeddings visualization saved as 'embeddings_visualization.png'")

    # 可视化交互图
    print("\nVisualizing interaction graph...")
    graph_fig = visualizer.visualize_interaction_graph()
    graph_fig.write_html('interaction_graph.html')
    print("Interaction graph visualization saved as 'interaction_graph.html'")

    # 可视化推荐指标历史
    print("\nVisualizing recommendation metrics history...")
    metrics_history = {
        'HR@10': [0.5, 0.6, 0.7, 0.75],
        'NDCG@10': [0.3, 0.4, 0.45, 0.5],
        'Recall@10': [0.4, 0.5, 0.55, 0.6]
    }
    metrics_fig = visualizer.plot_recommendation_metrics(metrics_history)
    metrics_fig.savefig('metrics_history.png')
    print("Metrics history visualization saved as 'metrics_history.png'")

    print("\nEvaluation complete. Visualizations have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph-based Recommender System')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for storing input data')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
    parser.add_argument('--batch_size', type=int, default=1024, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=5, help='Number of epochs for early stopping')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
    args = parser.parse_args()

    main(args)