"""
Visualizer Module for Enhanced PHPA Benchmark
"""

import os
import json
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """Generate benchmark reports and visualizations."""
    
    def __init__(self, config):
        self.config = config
    
    def generate_comprehensive_report(self, results, output_dir, timestamp):
        """Generate HTML report with charts."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate basic chart
            self._generate_basic_chart(results, output_dir, timestamp)
            
            # Generate HTML report
            self._generate_html_report(results, output_dir, timestamp)
            
            logger.info(f"Report generated in {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def _generate_basic_chart(self, results, output_dir, timestamp):
        """Generate basic accuracy chart."""
        try:
            plt.figure(figsize=(10, 6))
            
            llm_names = list(results.get('overall_results', {}).keys())
            accuracies = []
            
            for llm_name in llm_names:
                llm_data = results['overall_results'][llm_name]
                overall_metrics = llm_data.get('overall_accuracy', {})
                accuracy = overall_metrics.get('pattern_accuracy', 0) * 100
                accuracies.append(accuracy)
            
            plt.bar(llm_names, accuracies, alpha=0.8)
            plt.title('LLM Pattern Recognition Accuracy')
            plt.ylabel('Accuracy (%)')
            plt.xlabel('LLM')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_path = os.path.join(output_dir, f'accuracy_chart_{timestamp}.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")
    
    def _generate_html_report(self, results, output_dir, timestamp):
        """Generate basic HTML report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PHPA Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Enhanced PHPA Benchmark Results</h1>
    
    <div class="metric">
        <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
    
    <h2>LLM Performance Summary</h2>
    <table>
        <tr>
            <th>LLM</th>
            <th>Pattern Accuracy</th>
            <th>Model Accuracy</th>
        </tr>
"""
            
            for llm_name, llm_data in results.get('overall_results', {}).items():
                overall_metrics = llm_data.get('overall_accuracy', {})
                pattern_acc = overall_metrics.get('pattern_accuracy', 0) * 100
                model_acc = overall_metrics.get('model_accuracy', 0) * 100
                
                html_content += f"""
        <tr>
            <td>{llm_name}</td>
            <td>{pattern_acc:.1f}%</td>
            <td>{model_acc:.1f}%</td>
        </tr>
"""
            
            html_content += f"""
    </table>
    
    <h2>Accuracy Chart</h2>
    <img src="accuracy_chart_{timestamp}.png" alt="Accuracy Chart" style="max-width: 100%;">
    
</body>
</html>
"""
            
            html_path = os.path.join(output_dir, f'benchmark_report_{timestamp}.html')
            with open(html_path, 'w') as f:
                f.write(html_content)
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}") 