import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any

class ABTest:
    def __init__(self, name: str, control_group: str = "A", test_group: str = "B"):
        """
        初始化A/B测试
        
        Args:
            name: 测试名称
            control_group: 对照组名称
            test_group: 测试组名称
        """
        self.name = name
        self.control_group = control_group
        self.test_group = test_group
        self.metrics = {}
        self.results = {}
        
    def add_metric(self, metric_name: str, higher_is_better: bool = True):
        """
        添加要测试的指标
        
        Args:
            metric_name: 指标名称
            higher_is_better: 该指标是否越高越好
        """
        self.metrics[metric_name] = {
            "higher_is_better": higher_is_better,
            "control_data": [],
            "test_data": []
        }
        
    def add_observation(self, group: str, metric_name: str, value: float):
        """
        添加观测值
        
        Args:
            group: 组名 (A或B)
            metric_name: 指标名称
            value: 观测值
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric {metric_name} not defined. Add it first with add_metric().")
        
        if group == self.control_group:
            self.metrics[metric_name]["control_data"].append(value)
        elif group == self.test_group:
            self.metrics[metric_name]["test_data"].append(value)
        else:
            raise ValueError(f"Group {group} not recognized. Use {self.control_group} or {self.test_group}.")
            
    def add_batch_observations(self, group: str, observations: Dict[str, List[float]]):
        """
        批量添加观测值
        
        Args:
            group: 组名 (A或B)
            observations: 指标名称到观测值列表的映射
        """
        for metric_name, values in observations.items():
            for value in values:
                self.add_observation(group, metric_name, value)
    
    def run_test(self, alpha: float = 0.05):
        """
        运行A/B测试
        
        Args:
            alpha: 显著性水平
            
        Returns:
            Dict: 测试结果
        """
        results = {}
        
        for metric_name, metric_data in self.metrics.items():
            control_data = np.array(metric_data["control_data"])
            test_data = np.array(metric_data["test_data"])
            
            # 基本统计
            control_mean = np.mean(control_data)
            test_mean = np.mean(test_data)
            control_std = np.std(control_data)
            test_std = np.std(test_data)
            
            # 计算相对变化
            relative_change = (test_mean - control_mean) / control_mean * 100
            
            # 执行t检验
            t_stat, p_value = stats.ttest_ind(control_data, test_data, equal_var=False)
            
            # 确定是否有统计显著性差异
            is_significant = p_value < alpha
            
            # 确定哪个组更好
            if metric_data["higher_is_better"]:
                better_group = self.test_group if test_mean > control_mean else self.control_group
            else:
                better_group = self.test_group if test_mean < control_mean else self.control_group
                
            # 如果差异不显著，则没有明确的赢家
            winner = better_group if is_significant else "No clear winner"
            
            results[metric_name] = {
                "control_mean": control_mean,
                "test_mean": test_mean,
                "control_std": control_std,
                "test_std": test_std,
                "relative_change": relative_change,
                "p_value": p_value,
                "is_significant": is_significant,
                "better_group": better_group,
                "winner": winner
            }
            
        self.results = results
        return results
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8)):
        """
        可视化A/B测试结果
        
        Args:
            figsize: 图形大小
        """
        if not self.results:
            raise ValueError("No test results available. Run run_test() first.")
        
        n_metrics = len(self.metrics)
        fig, axes = plt.subplots(n_metrics, 2, figsize=figsize)
        
        # 如果只有一个指标，确保axes是二维的
        if n_metrics == 1:
            axes = np.array([axes])
            
        for i, (metric_name, result) in enumerate(self.results.items()):
            # 绘制均值比较条形图
            ax1 = axes[i, 0]
            means = [result["control_mean"], result["test_mean"]]
            stds = [result["control_std"], result["test_std"]]
            bars = ax1.bar([self.control_group, self.test_group], means, yerr=stds, capsize=10)
            
            # 设置颜色
            if result["is_significant"]:
                if result["better_group"] == self.test_group:
                    bars[1].set_color('green')
                else:
                    bars[0].set_color('green')
            
            ax1.set_title(f"{metric_name} - Mean Comparison")
            ax1.set_ylabel("Value")
            
            # 添加p值和相对变化的标注
            p_value_text = f"p-value: {result['p_value']:.4f}"
            change_text = f"Change: {result['relative_change']:.2f}%"
            significance_text = "Significant" if result["is_significant"] else "Not Significant"
            ax1.annotate(f"{p_value_text}\n{change_text}\n{significance_text}", 
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 绘制分布图
            ax2 = axes[i, 1]
            sns.kdeplot(self.metrics[metric_name]["control_data"], ax=ax2, label=self.control_group)
            sns.kdeplot(self.metrics[metric_name]["test_data"], ax=ax2, label=self.test_group)
            ax2.set_title(f"{metric_name} - Distribution")
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Density")
            ax2.legend()
            
        plt.tight_layout()
        return fig
    
    def summary(self):
        """
        生成A/B测试结果摘要
        
        Returns:
            pd.DataFrame: 结果摘要表格
        """
        if not self.results:
            raise ValueError("No test results available. Run run_test() first.")
        
        summary_data = []
        
        for metric_name, result in self.results.items():
            summary_data.append({
                "Metric": metric_name,
                f"{self.control_group} Mean": result["control_mean"],
                f"{self.test_group} Mean": result["test_mean"],
                "Relative Change (%)": result["relative_change"],
                "p-value": result["p_value"],
                "Significant": "Yes" if result["is_significant"] else "No",
                "Winner": result["winner"]
            })
            
        return pd.DataFrame(summary_data)

# 使用示例
if __name__ == "__main__":
    # 创建A/B测试
    ab_test = ABTest(name="Recommender Comparison", control_group="GCN", test_group="GAT")
    
    # 添加要测试的指标
    ab_test.add_metric("CTR", higher_is_better=True)  # 点击率
    ab_test.add_metric("Conversion Rate", higher_is_better=True)  # 转化率
    ab_test.add_metric("Average Session Duration", higher_is_better=True)  # 平均会话时长
    
    # 添加模拟数据
    np.random.seed(42)
    
    # 对照组数据
    control_ctr = np.random.beta(10, 90, size=1000) * 100  # 点击率约10%
    control_conversion = np.random.beta(3, 97, size=1000) * 100  # 转化率约3%
    control_duration = np.random.gamma(5, 1, size=1000)  # 平均会话时长
    
    # 测试组数据 (假设有一些提升)
    test_ctr = np.random.beta(11, 89, size=1000) * 100  # 点击率约11%
    test_conversion = np.random.beta(3.5, 96.5, size=1000) * 100  # 转化率约3.5%
    test_duration = np.random.gamma(5.5, 1, size=1000)  # 平均会话时长增加
    
    # 批量添加观测值
    ab_test.add_batch_observations("GCN", {
        "CTR": control_ctr,
        "Conversion Rate": control_conversion,
        "Average Session Duration": control_duration
    })
    
    ab_test.add_batch_observations("GAT", {
        "CTR": test_ctr,
        "Conversion Rate": test_conversion,
        "Average Session Duration": test_duration
    })
    
    # 运行测试
    results = ab_test.run_test(alpha=0.05)
    
    # 打印结果摘要
    print(ab_test.summary())
    
    # 可视化结果
    fig = ab_test.plot_results()
    plt.show()