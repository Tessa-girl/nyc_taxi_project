"""
实验系统单元测试
验证5个核心实验的功能和结果记录
"""
import pytest
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from main import NYCTaxiProject, ExperimentRecorder


class TestExperimentRecorder:
    """测试实验记录器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.recorder = ExperimentRecorder()
    
    def test_record_experiment(self):
        """测试记录单个实验"""
        results = {'accuracy': 0.95, 'rmse': 123.45}
        self.recorder.record_experiment(
            exp_name="Test Experiment",
            exp_type="Unit Test",
            results=results,
            metadata={'test_key': 'test_value'}
        )
        
        assert "Test Experiment" in self.recorder.experiments
        assert self.recorder.experiments["Test Experiment"]['results'] == results
        assert self.recorder.experiments["Test Experiment"]['type'] == "Unit Test"
    
    def test_save_all_experiments(self, tmp_path):
        """测试保存所有实验结果到JSON"""
        # 记录两个实验
        self.recorder.record_experiment("Exp1", "Type1", {'value': 1})
        self.recorder.record_experiment("Exp2", "Type2", {'value': 2})
        
        # 保存到临时文件
        output_file = tmp_path / "test_results.json"
        self.recorder.save_all_experiments(str(output_file))
        
        # 验证文件存在且内容正确
        assert output_file.exists()
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'experiments' in data
        assert len(data['experiments']) == 2
        assert 'Exp1' in data['experiments']
        assert 'Exp2' in data['experiments']
    
    def test_generate_summary_report(self):
        """测试生成总结报告"""
        self.recorder.record_experiment("Test Exp", "Test Type", {
            'metric1': 123.456,
            'metric2': 0.789
        })
        
        report = self.recorder.generate_summary_report()
        
        assert isinstance(report, str)
        assert "Test Exp" in report
        assert "Test Type" in report
        assert "123.4560" in report  # 检查浮点数格式化


class TestExperimentResults:
    """测试实验结果的完整性"""
    
    @pytest.fixture
    def experiment_results(self):
        """加载实验结果文件"""
        results_file = Path("results/experiment_results.json")
        if not results_file.exists():
            pytest.skip("实验结果文件不存在，请先运行实验")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_experiment_results_structure(self, experiment_results):
        """测试实验结果的基本结构"""
        assert 'project' in experiment_results
        assert 'experiments' in experiment_results
        assert 'start_time' in experiment_results
        assert 'end_time' in experiment_results
    
    def test_all_five_experiments_exist(self, experiment_results):
        """测试5个实验都存在"""
        experiments = experiment_results['experiments']
        
        expected_experiments = [
            "Experiment 1: Data Loading Comparison",
            "Experiment 2: Memory Optimization",
            "Experiment 3: Vectorization Speedup",
            "Experiment 4: Model Comparison",
            "Experiment 5: Feature Ablation Study"
        ]
        
        for exp_name in expected_experiments:
            assert exp_name in experiments, f"缺少实验: {exp_name}"
    
    def test_experiment_1_results(self, experiment_results):
        """测试实验1的结果"""
        exp1 = experiment_results['experiments']["Experiment 1: Data Loading Comparison"]
        results = exp1['results']
        
        assert 'pandas' in results
        assert 'method' in results['pandas']
        assert 'read_time_seconds' in results['pandas']
    
    def test_experiment_2_results(self, experiment_results):
        """测试实验2的结果"""
        exp2 = experiment_results['experiments']["Experiment 2: Memory Optimization"]
        results = exp2['results']
        
        assert 'before_optimization' in results
        assert 'after_optimization' in results
        assert 'optimization_effect' in results
        assert 'savings_percentage' in results['optimization_effect']
    
    def test_experiment_3_results(self, experiment_results):
        """测试实验3的结果"""
        exp3 = experiment_results['experiments']["Experiment 3: Vectorization Speedup"]
        results = exp3['results']
        
        assert 'loop_method' in results
        assert 'vectorized_method' in results
        assert 'speedup' in results
        assert 'acceleration_factor' in results['speedup']
    
    def test_experiment_3_no_simulated_data(self, experiment_results):
        """测试实验3未使用模拟数据（关键测试）"""
        exp3 = experiment_results['experiments']["Experiment 3: Vectorization Speedup"]
        metadata = exp3['metadata']
        results = exp3['results']
        
        # 验证元数据中标记为不使用模拟数据
        assert metadata.get('uses_simulated_data') == False, \
            "实验3不应使用模拟数据"
        
        # 验证使用了真实的LocationID字段
        assert metadata.get('has_real_location_ids') == True, \
            "实验3应使用真实的LocationID字段"
        
        # 验证计算类型是LocationID差值
        assert 'LocationID' in results['loop_method']['computation_type'], \
            "循环方法应使用LocationID计算"
        assert 'LocationID' in results['vectorized_method']['computation_type'], \
            "向量化方法应使用LocationID计算"
        
        # 验证没有经纬度相关的字段
        assert 'longitude' not in str(results).lower(), \
            "实验结果中不应包含经度字段"
        assert 'latitude' not in str(results).lower(), \
            "实验结果中不应包含纬度字段"

    def test_experiment_4_results(self, experiment_results):
        """测试实验4的结果"""
        exp4 = experiment_results['experiments']["Experiment 4: Model Comparison"]
        results = exp4['results']
        
        assert 'model_metrics' in results
        assert 'best_model' in results
        assert 'lightgbm' in results['model_metrics']
        
        lgbm_metrics = results['model_metrics']['lightgbm']
        assert 'RMSE' in lgbm_metrics
        assert 'R2' in lgbm_metrics
        assert lgbm_metrics['R2'] > 0.9  # LightGBM应该达到较高的R²
    
    def test_experiment_5_results(self, experiment_results):
        """测试实验5的结果"""
        exp5 = experiment_results['experiments']["Experiment 5: Feature Ablation Study"]
        results = exp5['results']
        
        assert 'baseline_no_features' in results
        assert 'all_features' in results
        assert 'feature_engineering_impact' in results
    
    def test_experiment_5_baseline_has_fewer_features(self, experiment_results):
        """测试实验5的baseline配置使用更少的特征(关键修复验证)"""
        exp5 = experiment_results['experiments']["Experiment 5: Feature Ablation Study"]
        results = exp5['results']
        
        baseline = results['baseline_no_features']
        all_feat = results['all_features']
        
        # Baseline应该有更少的特征
        assert baseline['num_features'] < all_feat['num_features'], \
            f"Baseline应该有更少特征: {baseline['num_features']} vs {all_feat['num_features']}"
        
        # Baseline的特征数应该在5-10之间(基础特征)
        assert 3 <= baseline['num_features'] <= 10, \
            f"Baseline特征数不合理: {baseline['num_features']}"
        
        # All features应该有更多特征(20+)
        assert all_feat['num_features'] >= 20, \
            f"All features特征数过少: {all_feat['num_features']}"
    
    def test_experiment_5_feature_engineering_improves_performance(self, experiment_results):
        """测试特征工程确实提升了模型性能"""
        exp5 = experiment_results['experiments']["Experiment 5: Feature Ablation Study"]
        results = exp5['results']
        
        baseline_rmse = results['baseline_no_features']['RMSE']
        all_rmse = results['all_features']['RMSE']
        
        # 全量特征应该比baseline有更好的RMSE
        assert all_rmse < baseline_rmse, \
            f"特征工程应提升性能: baseline RMSE={baseline_rmse}, all features RMSE={all_rmse}"
        
        # 性能提升至少1%
        improvement_pct = (baseline_rmse - all_rmse) / baseline_rmse * 100
        assert improvement_pct > 1.0, \
            f"特征工程性能提升过小: {improvement_pct:.2f}%"

class TestGeneratedFiles:
    """测试生成的文件"""
    
    def test_experiment_results_json_exists(self):
        """测试JSON结果文件存在"""
        assert Path("results/experiment_results.json").exists()
    
    def test_comprehensive_report_exists(self):
        """测试综合报告文件存在"""
        assert Path("results/comprehensive_experiment_report.txt").exists()
    
    def test_visualization_files_exist(self):
        """测试可视化图表文件存在"""
        expected_files = [
            "results/figures/exp2_memory_comparison.png",
            "results/figures/exp3_vectorization_comparison.png",
            "results/figures/exp5_feature_ablation.png",
            "results/figures/model_comparison.png",
            "results/figures/training_time_comparison.png"
        ]
        
        for file_path in expected_files:
            assert Path(file_path).exists(), f"文件不存在: {file_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
