#!/usr/bin/env python3
import subprocess
import os
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import concurrent.futures
import multiprocessing
import time
import random
import sys
import seaborn as sns
import glob
from scipy.cluster import hierarchy
import matplotlib.dates as mdates
from tqdm import tqdm
import warnings
import argparse
import importlib
import matplotlib
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

# Parse arguments
args = None  # Initialize args at the global level

# Path to generated datasets
scenarios_path = "generated_datasets"

# Check if GPU-related packages are available
def check_gpu_requirements():
    """Check if required packages for GPU models are installed and available."""
    has_gpu = False
    try:
        # Try to import torch
        import torch
        has_torch = True
        has_cuda = torch.cuda.is_available()
        
        if has_cuda:
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}"
            logging.info(f"PyTorch with CUDA is available: {gpu_info}")
            has_gpu = True
        else:
            logging.warning("PyTorch is available but CUDA is not. Checking for Apple Silicon MPS...")
        
        # Check for Apple Silicon MPS support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logging.info("Apple Silicon MPS is available and will be used for GPU acceleration")
            has_gpu = True
    except ImportError:
        has_torch = False
        has_cuda = False
        logging.warning("PyTorch is not installed. GPU models may not work properly.")
    
    required_packages = ['torch', 'tensorflow', 'keras']
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        logging.warning(f"Missing required packages for GPU models: {', '.join(missing_packages)}")
        logging.warning("Use 'pip install torch tensorflow keras' to install them.")
    
    return has_gpu

# Local imports
from models import TimeSeriesModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Utility functions for consistent handling of scenario names
def normalize_scenario_name(scenario):
    """
    Normalize scenario name by removing 'cleaned_data_' prefix if present.
    This function might not be needed anymore if scenario names don't have this prefix.
    Keeping it for now for potential backward compatibility if needed.
    """
    if isinstance(scenario, str) and scenario.startswith("cleaned_data_"):
        return scenario.replace("cleaned_data_", "")
    return scenario

def get_original_scenario_name(scenario):
    """
    Get the original scenario name as it appears in the directory structure.
    Since files are now directly in scenarios_path, this just returns the scenario name.
    """
    # if isinstance(scenario, str) and not scenario.startswith("cleaned_data_"):
    #     return f"cleaned_data_{scenario}" # Original logic assumed subdirectories
    return scenario # New logic: scenario name is the file prefix

# Parse arguments
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate time series forecasting models.")
    
    # Add all existing arguments
    parser.add_argument('--test', action='store_true', help='Run in test mode - runs all models on a single dataset and reports execution times')
    parser.add_argument('--dataset', type=str, help='Specific dataset to use in test mode (if not specified, first available will be used)')
    parser.add_argument('--gpu', action='store_true', help='Test only GPU models')
    parser.add_argument('--cpu', action='store_true', help='Test only CPU models')
    parser.add_argument('--visualize-only', action='store_true', help='Skip training and only create visualizations from existing results')
    parser.add_argument('--results-file', type=str, help='Path to specific results file to use with --visualize-only')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save output visualizations')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output plots (default: 300)')
    parser.add_argument('--plot-size', type=str, default='medium', choices=['small', 'medium', 'large'], help='Size of output plots')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to train')
    parser.add_argument('--scenarios', type=str, help='Comma-separated list of scenarios to use')
    parser.add_argument('--scenarios-list', type=str, help='Path to a file containing scenarios, one per line')
    parser.add_argument('--limit', type=int, help='Limit the number of scenarios to process')
    parser.add_argument('--load-existing', action='store_true', help='Load existing results if available')
    
    # Add visualization-related arguments
    visualization_group = parser.add_argument_group('Visualization options')
    visualization_group.add_argument('--group-scenarios', action='store_true',
                                     help="Group similar scenarios together for visualization")
    visualization_group.add_argument('--html-report', action='store_true',
                                     help="Generate detailed HTML report with all comparisons")
    
    # Parse the arguments
    args = parser.parse_args()
    
    return args

# Function to test all the important functions before continuing
def test_functions():
    """Test critical functions to ensure they work correctly before running the full pipeline."""
    logging.info("Testing critical functions...")
    
    # Test directory creation
    try:
        test_dir = "results/test_dir"
        os.makedirs(test_dir, exist_ok=True)
        logging.info("✓ Directory creation works")
        # Clean up
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
    except Exception as e:
        logging.error(f"✗ Directory creation failed: {e}")
        return False
    
    # Test pandas DataFrame operations
    try:
        # Create a simple test DataFrame
        test_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10),
            'scenario': ['test'] * 10,
            'model': ['test_model'] * 10,
            'rmse': np.random.rand(10),
            'mae': np.random.rand(10),
            'mape': np.random.rand(10)
        })
        test_df.to_csv('results/test_df.csv', index=False)
        test_df_read = pd.read_csv('results/test_df.csv')
        logging.info("✓ Pandas DataFrame operations work")
        # Clean up
        if os.path.exists('results/test_df.csv'):
            os.remove('results/test_df.csv')
    except Exception as e:
        logging.error(f"✗ Pandas DataFrame operations failed: {e}")
        return False
    
    # Test matplotlib plotting
    try:
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.savefig('results/test_plot.png')
        plt.close()
        logging.info("✓ Matplotlib plotting works")
        # Clean up
        if os.path.exists('results/test_plot.png'):
            os.remove('results/test_plot.png')
    except Exception as e:
        logging.error(f"✗ Matplotlib plotting failed: {e}")
        return False
    
    # Test data directories
    if not os.path.exists(scenarios_path):
        logging.error(f"✗ Scenarios path '{scenarios_path}' does not exist")
        return False
    else:
        scenario_count = len([d for d in os.listdir(scenarios_path) if os.path.isdir(os.path.join(scenarios_path, d))])
        if scenario_count == 0:
            logging.warning(f"⚠ No scenario directories found in '{scenarios_path}'")
        else:
            logging.info(f"✓ Found {scenario_count} scenario directories in '{scenarios_path}'")
    
    # Test model script paths
    model_scripts_exist = True
    for model_name, model_info in MODELS.items():
        script_path = model_info["script"]
        if not os.path.exists(script_path):
            logging.error(f"✗ Model script for '{model_name}' not found at '{script_path}'")
            model_scripts_exist = False
    
    if model_scripts_exist:
        logging.info("✓ All model scripts exist")
    else:
        logging.error("✗ Some model scripts are missing")
        return False
    
    logging.info("All critical functions tested successfully!")
    return True

# Define constants
# Path to scenarios (updated path - using the new time_horizon_datasets directory)
scenarios_path = "generated_datasets"
models_path = "cpu-models"
# models_path = "gpu-models"

# Global visualization settings - will be updated by setup_visualization_params
PLOT_DPI = 300
FIGURE_SIZES = {
    'default': (12, 8),
    'wide': (16, 8),
    'square': (12, 12),
    'tall': (10, 14),
    'radar': (14, 14),
    'bar': (14, 8),
    'boxplot': (16, 8),
    'heatmap': (18, 12),
    'grid': (16, 10)
}
FONT_SIZES = {
    'title': 14,
    'axis_label': 12,
    'tick_label': 10,
    'legend': 10,
    'annotation': 9
}

# Define the scenarios (get all directories from time_horizon_datasets)
# This will automatically pick up all the pattern datasets with different time horizons
# SCENARIOS = [
#     d for d in os.listdir(scenarios_path) 
#     if os.path.isdir(os.path.join(scenarios_path, d))
# ]

# Updated: Find scenarios by looking for _train.csv files
SCENARIOS = []
if os.path.exists(scenarios_path):
    for f in os.listdir(scenarios_path):
        if f.endswith("_train.csv"):
            # Extract scenario name (remove _train.csv suffix)
            scenario_name = f.replace("_train.csv", "")
            SCENARIOS.append(scenario_name)
else:
    logging.error(f"Scenarios path '{scenarios_path}' does not exist.")

# Define the models to run with their respective scripts
MODELS = {
    "prophet": {
        "script": f"{models_path}/prophet_model.py",
        "gpu": False
    },
    "xgboost": {
        "script": f"{models_path}/train_xgboost.py",
        "gpu": False
    },
    "lightgbm": {
        "script": f"{models_path}/train_lightgbm.py",
        "gpu": False
    },
    "catboost": {
        "script": f"{models_path}/train_catboost.py",
        "gpu": False
    },
    "gbdt": {
        "script": f"{models_path}/train_gbdt.py",
        "gpu": False
    },
    "var": {
        "script": f"{models_path}/train_var.py",
        "gpu": False
    },
    "baseline": {
        "script": f"{models_path}/train_baseline.py",
        "gpu": False
    }
}

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Function to run a model and capture its output
def run_model(model_name, scenario, all_results=None, measure_time=False):
    model_info = MODELS[model_name]
    script_path = model_info["script"]
    
    # Use the original scenario name with cleaned_data_ prefix for file paths -> NO LONGER NEEDED
    # scenario_dir_name = get_original_scenario_name(scenario) # Old logic
    # train_data = f"{scenarios_path}/{scenario_dir_name}/data_train.csv" # Old logic
    # test_data = f"{scenarios_path}/{scenario_dir_name}/data_test.csv" # Old logic
    
    # New logic: Files are directly in scenarios_path, named after the scenario
    train_data = f"{scenarios_path}/{scenario}_train.csv"
    test_data = f"{scenarios_path}/{scenario}_test.csv"
    
    # Normalize scenario name for run_id and results (this should still work)
    normalized_scenario = normalize_scenario_name(scenario)
    
    # Check if data files exist
    if not os.path.exists(train_data) or not os.path.exists(test_data):
        # # Try the alternative path format -> No longer needed with direct file check
        # train_data = f"{scenarios_path}/{scenario}/data_train.csv"
        # test_data = f"{scenarios_path}/{scenario}/data_test.csv"
        
        # if not os.path.exists(train_data) or not os.path.exists(test_data):
            logging.warning(f"Data files {train_data} or {test_data} not found for {model_name} in {scenario}. Skipping.")
            return model_name, scenario, None, 0
    
    # Handle model dependencies if any
    if "depends_on" in model_info and all_results is not None:
        for dependency in model_info["depends_on"]:
            if scenario not in all_results or dependency not in all_results[scenario]:
                # If the dependency hasn't been run, run it first
                logging.info(f"{model_name} depends on {dependency}, running it first...")
                dep_model_name, dep_scenario, dep_metrics, dep_time = run_model(dependency, scenario, all_results, measure_time)
                
                # Update results with dependency results
                if dep_metrics:
                    if dep_scenario not in all_results:
                        all_results[dep_scenario] = {}
                    all_results[dep_scenario][dep_model_name] = dep_metrics
    
    logging.info(f"Running {model_name} on {scenario}...")
    
    # Create a run ID using model name and normalized scenario name
    run_id = f"{model_name}_{normalized_scenario}"
    
    # Ensure model run directory exists (using normalized scenario in run_id)
    model_run_dir = os.path.join("train/models", f"{model_name}_model", "runs", run_id)
    os.makedirs(model_run_dir, exist_ok=True)
    
    # Ensure scenario results directory exists (using normalized scenario name)
    scenario_results_dir = f"results/{normalized_scenario}" # Use normalized name for results dir
    os.makedirs(scenario_results_dir, exist_ok=True)
    
    try:
        # Add a small random delay to prevent resource conflicts
        if model_info.get("gpu", False):
            time.sleep(random.uniform(0.5, 2.0))
        
        # Run the model script using the current Python interpreter
        cmd = [
            sys.executable,
            script_path, 
            "--train-file", train_data, 
            "--test-file", test_data,
            "--run-id", run_id
        ]
        
        # Start timing if requested
        start_time = time.time() if measure_time else 0
        
        # Set timeout based on model type (GPU models get longer timeout)
        timeout = 7200 if model_info.get("gpu", False) else 3600  # 2 hours for GPU, 1 hour for CPU
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout
        )
        
        # End timing if requested
        execution_time = time.time() - start_time if measure_time else 0
        
        # Try to parse metrics from stdout
        output = process.stdout
        metrics = {}
        
        try:
            # Try to parse JSON output
            metrics = json.loads(output)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract metrics using regex
            mae_match = re.search(r'MAE:\s*([\d.]+)', output)
            rmse_match = re.search(r'RMSE:\s*([\d.]+)', output)
            mape_match = re.search(r'MAPE:\s*([\d.]+)', output)
            
            if mae_match:
                metrics['mae'] = float(mae_match.group(1))
            if rmse_match:
                metrics['rmse'] = float(rmse_match.group(1))
            if mape_match:
                metrics['mape'] = float(mape_match.group(1))
        
        # If we have metrics, save them
        if metrics:
            # Add execution time to metrics if measuring time
            if measure_time:
                metrics['execution_time'] = execution_time
            
            # Create a unique output file for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/{model_name}_{normalized_scenario}_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Copy predictions.csv to results directory
            predictions_file = os.path.join(model_run_dir, "predictions.csv")
            if os.path.exists(predictions_file):
                # Copy and rename the predictions file to results/normalized_scenario/model_predictions.csv
                results_predictions_file = os.path.join(scenario_results_dir, f"{model_name}_predictions.csv")
                try:
                    # Load and reformat predictions if needed
                    predictions_df = pd.read_csv(predictions_file)
                    
                    # Ensure the predictions DataFrame has the expected columns
                    if "actual" in predictions_df.columns and "predicted" in predictions_df.columns:
                        # Rename columns if needed for consistency
                        predictions_df = predictions_df.rename(columns={
                            "actual": "actual",
                            "predicted": "prediction"
                        })
                    
                    # Save to the results directory
                    predictions_df.to_csv(results_predictions_file, index=False)
                    logging.info(f"Copied predictions from {predictions_file} to {results_predictions_file}")
                except Exception as e:
                    logging.error(f"Error copying predictions file: {e}")
            
            return model_name, scenario, metrics, execution_time
        else:
            # Try to find metrics.json in the model's output directory
            metrics_path = None
            
            # Check common output directories based on model type
            # Use normalized_scenario here as well
            possible_paths = [
                f"train/models/{model_name}_model/runs/{run_id}/metrics.json",
                f"train/models/{model_name}/runs/{run_id}/metrics.json",
                f"./train/models/{model_name}_model/runs/{run_id}/metrics.json",
                f"./train/models/{model_name}/runs/{run_id}/metrics.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    metrics_path = path
                    break
            
            if metrics_path:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                if measure_time:
                    metrics['execution_time'] = execution_time
                return model_name, scenario, metrics, execution_time
            else:
                logging.error(f"Could not extract metrics from {model_name} output")
                return model_name, scenario, None, execution_time
    
    except subprocess.TimeoutExpired:
        timeout_duration = 7200 if model_info.get("gpu", False) else 3600
        logging.error(f"Timeout running {model_name} on {scenario} (exceeded {timeout_duration//3600} hour(s))")
        return model_name, scenario, None, timeout_duration  # Return max time
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {model_name} on {scenario}: {e}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        return model_name, scenario, None, time.time() - start_time if measure_time else 0
    except Exception as e:
        logging.error(f"Unexpected error running {model_name} on {scenario}: {e}")
        return model_name, scenario, None, time.time() - start_time if measure_time else 0

def run_model_parallel(task):
    """Wrapper function for running a model in parallel using ProcessPoolExecutor"""
    model_name, scenario, all_results, measure_time = task
    return run_model(model_name, scenario, all_results, measure_time)

# Function to compare model performance
def compare_models(all_results):
    if not all_results:
        logging.warning("No results to compare")
        return
    
    # Create a DataFrame to store all results
    all_results_df = []
    
    for scenario, scenario_results in all_results.items():
        # Always use normalized scenario name
        normalized_scenario = normalize_scenario_name(scenario)
        
        for model_name, metrics in scenario_results.items():
            if metrics:
                all_results_df.append({
                    'scenario': normalized_scenario,
                    'model': model_name,
                    'mae': metrics.get('mae', float('inf')),
                    'rmse': metrics.get('rmse', float('inf')),
                    'mape': metrics.get('mape', float('inf'))
                })
    
    if not all_results_df:
        logging.warning("No valid results found")
        return
    
    results_df = pd.DataFrame(all_results_df)
    
    # Save the complete results and skip redundant visualizations
    # We'll handle visualizations here instead
    results_df.to_csv('results/all_model_comparisons.csv', index=False)
    
    # Save all metrics in a standardized format but skip visualizations
    # to avoid duplication with what we'll create here
    save_summary_metrics(all_results, skip_visualization=True)
    
    # Find the best model for each scenario based on different metrics
    best_models = {}
    
    for scenario in results_df['scenario'].unique():
        scenario_df = results_df[results_df['scenario'] == scenario]
        
        best_mae = scenario_df.loc[scenario_df['mae'].idxmin()]
        best_rmse = scenario_df.loc[scenario_df['rmse'].idxmin()]
        best_mape = scenario_df.loc[scenario_df['mape'].idxmin()]
        
        best_models[scenario] = {
            'best_mae': best_mae['model'],
            'mae_value': best_mae['mae'],
            'best_rmse': best_rmse['model'],
            'rmse_value': best_rmse['rmse'],
            'best_mape': best_mape['model'],
            'mape_value': best_mape['mape']
        }
    
    # Create a summary DataFrame
    summary_rows = []
    for scenario, bests in best_models.items():
        summary_rows.append({
            'scenario': scenario,
            'best_mae_model': bests['best_mae'],
            'best_mae_value': bests['mae_value'],
            'best_rmse_model': bests['best_rmse'],
            'best_rmse_value': bests['rmse_value'],
            'best_mape_model': bests['best_mape'],
            'best_mape_value': bests['mape_value']
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('results/best_models_summary.csv', index=False)
    
    # Count how many times each model was the best
    model_wins = {model: {'mae': 0, 'rmse': 0, 'mape': 0, 'total': 0} for model in MODELS.keys()}
    
    for scenario, bests in best_models.items():
        model_wins[bests['best_mae']]['mae'] += 1
        model_wins[bests['best_mae']]['total'] += 1
        
        model_wins[bests['best_rmse']]['rmse'] += 1
        model_wins[bests['best_rmse']]['total'] += 1
        
        model_wins[bests['best_mape']]['mape'] += 1
        model_wins[bests['best_mape']]['total'] += 1
    
    # Create a DataFrame for model wins
    wins_rows = []
    for model, wins in model_wins.items():
        wins_rows.append({
            'model': model,
            'mae_wins': wins['mae'],
            'rmse_wins': wins['rmse'],
            'mape_wins': wins['mape'],
            'total_wins': wins['total']
        })
    
    wins_df = pd.DataFrame(wins_rows)
    wins_df = wins_df.sort_values('total_wins', ascending=False)
    wins_df.to_csv('results/model_wins.csv', index=False)
    
    # Print the overall winner
    if len(wins_df) > 0:
        overall_winner = wins_df.iloc[0]['model']
        logging.info(f"\n\n===== OVERALL WINNER: {overall_winner} =====")
        logging.info(f"Total wins: {wins_df.iloc[0]['total_wins']}")
        logging.info(f"MAE wins: {wins_df.iloc[0]['mae_wins']}")
        logging.info(f"RMSE wins: {wins_df.iloc[0]['rmse_wins']}")
        logging.info(f"MAPE wins: {wins_df.iloc[0]['mape_wins']}")
        
        # Print the top 3 models
        logging.info("\nTop 3 Models:")
        for i in range(min(3, len(wins_df))):
            model = wins_df.iloc[i]
            logging.info(f"{i+1}. {model['model']} - Total wins: {model['total_wins']} (MAE: {model['mae_wins']}, RMSE: {model['rmse_wins']}, MAPE: {model['mape_wins']})")
    
    # Create visualizations
    create_visualizations(results_df, wins_df)

def create_best_model_summary(results_df):
    """Create a summary chart showing the best model for each scenario based on RMSE."""
    
    # Skip if no data
    if results_df.empty:
        logging.warning("Empty results DataFrame - cannot create best model summary")
        return
    
    # Create directory for the summary chart
    os.makedirs("results/summary", exist_ok=True)
    
    # Find the best model for each scenario based on RMSE
    best_models = {}
    for scenario in results_df['scenario'].unique():
        scenario_df = results_df[results_df['scenario'] == scenario]
        if not scenario_df.empty:
            # Make sure the scenario dataframe has valid RMSE values
            valid_rmse = scenario_df['rmse'].dropna()
            if not valid_rmse.empty:
                best_model = scenario_df.loc[valid_rmse.idxmin()]['model']
                best_models[scenario] = best_model
    
    # Skip if no best models were identified
    if not best_models:
        logging.warning("No best models identified - skipping summary chart")
        return
    
    # Convert to DataFrame for easy counting
    best_models_df = pd.DataFrame(list(best_models.items()), columns=['scenario', 'best_model'])
    
    # Count occurrences of each model as best
    model_counts = best_models_df['best_model'].value_counts()
    
    plt.figure(figsize=(12, 8))
    
    if model_counts.empty:
        # If no data, create an empty plot with a message
        plt.text(0.5, 0.5, "Insufficient data for model comparison", 
                ha='center', va='center', fontsize=14)
    else:
        # Create bar chart
        ax = model_counts.plot(kind='bar', color='skyblue')
        
        # Add count labels on top of each bar
        for i, count in enumerate(model_counts):
            ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.title('Best Performing Models by Scenario (Based on RMSE)', fontsize=14)
    plt.xlabel('Model')
    plt.ylabel('Number of Scenarios')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    summary_path = "results/summary/best_models_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Best models summary chart saved to {summary_path}")
    
    # Also create a table with the best model for each scenario
    summary_table = pd.DataFrame({
        'Scenario': list(best_models.keys()),
        'Best Model': list(best_models.values())
    })
    
    # Sort by scenario name
    summary_table = summary_table.sort_values('Scenario')
    
    # Save to CSV
    summary_table.to_csv("results/summary/best_models_by_scenario.csv", index=False)
    logging.info("Best models by scenario saved to results/summary/best_models_by_scenario.csv")

def create_model_radar_chart(results_df):
    """Create a radar chart comparing model performance across metrics."""
    
    # Create directory for the radar chart
    os.makedirs("results/plots", exist_ok=True)
    
    # Get average metrics for each model across all scenarios
    metrics_by_model = results_df.groupby('model')[['rmse', 'mae', 'mape']].mean()
    
    # Skip if too many models (create multiple charts)
    max_models_per_chart = 8
    all_models = metrics_by_model.index.tolist()
    
    # Handle case with too many models by creating multiple charts
    if len(all_models) > max_models_per_chart:
        logging.info(f"Creating multiple radar charts for {len(all_models)} models")
        
        # Create directory for radar charts
        os.makedirs("results/plots/radar_charts", exist_ok=True)
        
        # Create a chart for all models (might be crowded but useful for overview)
        create_single_radar_chart(metrics_by_model, "results/plots/model_radar_chart.png", 
                                 title="All Models Performance Comparison")
        
        # Create multiple charts with fewer models each
        for i in range(0, len(all_models), max_models_per_chart):
            subset_models = all_models[i:i+max_models_per_chart]
            subset_metrics = metrics_by_model.loc[subset_models]
            
            chart_number = i // max_models_per_chart + 1
            filename = f"results/plots/radar_charts/model_radar_chart_group{chart_number}.png"
            
            create_single_radar_chart(subset_metrics, filename, 
                                     title=f"Model Group {chart_number} Performance Comparison")
    else:
        # Just create one chart if we have a reasonable number of models
        create_single_radar_chart(metrics_by_model, "results/plots/model_radar_chart.png")

def create_single_radar_chart(metrics_by_model, output_file, title="Model Performance Comparison Radar Chart"):
    """Helper function to create a single radar chart."""
    
    # Normalize the metrics for radar chart (lower is better, so invert)
    normalized = pd.DataFrame()
    for col in metrics_by_model.columns:
        max_val = metrics_by_model[col].max()
        min_val = metrics_by_model[col].min()
        # Invert the scale so higher is better
        if max_val > min_val:
            normalized[col] = 1 - (metrics_by_model[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 1  # Avoid division by zero
    
    # Radar chart setup
    categories = ['RMSE', 'MAE', 'MAPE']
    N = len(categories)
    
    # Number of models in this chart
    num_models = len(normalized.index)
    
    # Adjust figure size based on number of models (larger for more models)
    figsize = (12, 12) if num_models <= 4 else (15, 15)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=300)  # Higher DPI for better quality
    
    # Color palette for different models - use a colorblind-friendly palette
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    
    # Set angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the radar chart
    ax = plt.subplot(111, polar=True)
    
    # Determine line width based on number of models
    line_width = max(1.5, 3 - (num_models * 0.1))  # Decrease width as models increase
    
    # Draw one line per model with a different color
    for i, model in enumerate(normalized.index):
        values = normalized.loc[model].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot data with adjusted line width
        ax.plot(angles, values, linewidth=line_width, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set radar chart attributes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    plt.xticks(angles[:-1], categories, fontsize=12, fontweight='bold')
    
    # Draw y-axis labels with 0 at center and 1 at edge
    ax.set_rlabel_position(180 / len(categories))
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=10)
    ax.set_ylim(0, 1)
    
    # Adjust legend position and font size based on number of models
    if num_models <= 4:
        legend_loc = 'upper right'
        bbox_to_anchor = (0.1, 0.1)
        font_size = 10
    else:
        legend_loc = 'lower center' 
        bbox_to_anchor = (0.5, -0.15)
        font_size = max(6, 10 - (num_models * 0.4))  # Scale down font size for many models
    
    # Add legend with custom position and font size
    legend = plt.legend(
        loc=legend_loc, 
        bbox_to_anchor=bbox_to_anchor,
        fontsize=font_size,
        ncol=min(3, (num_models + 1) // 2)  # Multiple columns for many models
    )
    
    # Make the legend lines thicker for visibility
    for line in legend.get_lines():
        line.set_linewidth(2.5)
    
    # Add a title with larger font
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    
    # Add grid lines for better readability
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add textual explanation at the bottom
    plt.figtext(0.5, 0.01, 
               "Higher values indicate better performance.\nMetrics are normalized and inverted (since lower original metrics are better).", 
               ha="center", fontsize=10)
    
    # Save with tight layout and high dpi
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Model radar chart saved to {output_file}")

def create_performance_comparison_table(results_df, wins_df=None):
    """Create a detailed performance comparison table for all scenarios and models.
    If there are many scenarios, they will be grouped by scenario type for better readability.
    """
    
    # Create directory for detailed comparisons
    os.makedirs("results/detailed_comparisons", exist_ok=True)
    
    # Prepare a DataFrame for output
    comparison_rows = []
    
    # Check if we should group scenarios (more than 15 scenarios)
    num_scenarios = len(results_df['scenario'].unique())
    should_group_scenarios = num_scenarios > 15
    
    if should_group_scenarios:
        logging.info(f"Grouping {num_scenarios} scenarios by type for better readability in the performance table")
        # Group scenarios by type
        grouped_results = group_scenarios_by_type(results_df)
        
        # Process each group first, then each scenario within the group
        for scenario_type, group_df in grouped_results.items():
            # Add a group header
            comparison_rows.append({
                'rank': f"Scenario Group: {scenario_type.upper()}", 
                'model': "", 
                'rmse': "", 
                'mae': "", 
                'mape': "",
                'section': f"group_{scenario_type}"  # Special section marker for groups
            })
            
            # For each scenario in this group, create a section in the table
            for scenario in sorted(group_df['scenario'].unique()):
                scenario_df = group_df[group_df['scenario'] == scenario].copy()
                
                # Sort by RMSE (best first)
                scenario_df = scenario_df.sort_values('rmse')
                
                # Add a ranking column
                scenario_df['rank'] = range(1, len(scenario_df) + 1)
                
                # Add scenario information
                scenario_df['section'] = scenario
                
                # Extract and simplify scenario description for display
                scenario_params = get_scenario_parameters(scenario)
                scenario_display = f"{scenario_type}_{scenario_params.get('variant', '')}"
                
                # Add important parameters if available
                important_params = []
                for param in ['base_pods', 'burst_magnitude', 'chaos_level', 'cascade_count']:
                    if param in scenario_params:
                        param_name = param.replace('_', ' ')
                        important_params.append(f"{param_name}: {scenario_params[param]}")
                
                if important_params:
                    scenario_display += f" ({', '.join(important_params)})"
                
                # Add to our collection
                comparison_rows.append({
                    'rank': f"Scenario: {scenario_display}", 
                    'model': "", 
                    'rmse': "", 
                    'mae': "", 
                    'mape': "",
                    'section': scenario
                })
                
                for _, row in scenario_df.iterrows():
                    comparison_rows.append({
                        'rank': row['rank'],
                        'model': row['model'],
                        'rmse': round(row['rmse'], 3),
                        'mae': round(row['mae'], 3),
                        'mape': round(row['mape'], 3),
                        'section': scenario
                    })
                
                # Add a blank row after each scenario
                comparison_rows.append({
                    'rank': "", 
                    'model': "", 
                    'rmse': "", 
                    'mae': "", 
                    'mape': "",
                    'section': scenario
                })
    else:
        # Original approach - process scenarios without grouping
        for scenario in sorted(results_df['scenario'].unique()):
            scenario_df = results_df[results_df['scenario'] == scenario].copy()
            
            # Sort by RMSE (best first)
            scenario_df = scenario_df.sort_values('rmse')
            
            # Add a ranking column
            scenario_df['rank'] = range(1, len(scenario_df) + 1)
            
            # Add scenario information
            scenario_df['section'] = scenario
            
            # Add to our collection
            comparison_rows.append({
                'rank': f"Scenario: {scenario}", 
                'model': "", 
                'rmse': "", 
                'mae': "", 
                'mape': "",
                'section': scenario
            })
            
            for _, row in scenario_df.iterrows():
                comparison_rows.append({
                    'rank': row['rank'],
                    'model': row['model'],
                    'rmse': round(row['rmse'], 3),
                    'mae': round(row['mae'], 3),
                    'mape': round(row['mape'], 3),
                    'section': scenario
                })
            
            # Add a blank row after each scenario
            comparison_rows.append({
                'rank': "", 
                'model': "", 
                'rmse': "", 
                'mae': "", 
                'mape': "",
                'section': scenario
            })
    
    # Convert to DataFrame
    styled_df = pd.DataFrame(comparison_rows)
    
    # Save as CSV
    styled_df.to_csv("results/detailed_comparisons/all_models_comparison.csv", index=False)
    
    # Determine if we need pagination (more than 20 scenarios or 200 rows total)
    need_pagination = len(styled_df) > 200 or len(results_df['scenario'].unique()) > 20
    
    # Create an HTML version with styling
    try:
        # Add CSS with responsive design and better handling of large tables
        css = """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        th, td { 
            padding: 6px; 
            text-align: left; 
            border-bottom: 1px solid #ddd; 
        }
        th { 
            background-color: #f2f2f2; 
            position: sticky;
            top: 0;
        }
        tr:hover { background-color: #f5f5f5; }
        .section-header { 
            background-color: #4CAF50; 
            color: white; 
            font-weight: bold; 
            font-size: 1.1em;
            padding: 10px;
        }
        .group-header {
            background-color: #326fa8; 
            color: white; 
            font-weight: bold; 
            font-size: 1.2em;
            padding: 12px;
        }
        .scenario-header { 
            background-color: #e6f2ff; 
            font-weight: bold; 
        }
        .best-model { 
            background-color: #d4edda; 
        }
        .second-best { 
            background-color: #fff3cd; 
        }
        .metrics {
            text-align: right;
            font-family: monospace;
        }
        .search-box {
            padding: 10px;
            margin-bottom: 20px;
            width: 100%;
            box-sizing: border-box;
            font-size: 16px;
        }
        .pagination {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .pagination button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 4px;
        }
        .page-info {
            margin: 10px 0;
            text-align: center;
        }
        .group-toggle {
            cursor: pointer;
            user-select: none;
        }
        .group-content {
            display: block;
        }
        .collapsed .group-content {
            display: none;
        }
        @media print {
            .no-print {
                display: none;
            }
            body {
                padding: 0;
                font-size: 10pt;
            }
            table {
                font-size: 9pt;
            }
        }
        </style>
        """
        
        # Add JavaScript for filtering, pagination, and group collapsing
        javascript = """
        <script>
        function filterTable() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toUpperCase();
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            let visibleRows = 0;
            
            for (let i = 0; i < rows.length; i++) {
                const row = rows[i];
                const cells = row.getElementsByTagName('td');
                
                if (cells.length > 0) {
                    let txtValue = '';
                    
                    // Concat all cell values
                    for (let j = 0; j < cells.length; j++) {
                        txtValue += cells[j].textContent || cells[j].innerText;
                    }
                    
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        row.style.display = '';
                        visibleRows++;
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
            
            document.getElementById('visibleRowCount').textContent = visibleRows;
        }
        
        // Group toggling functionality
        function toggleGroup(element) {
            const groupRow = element.closest('tr');
            const groupName = groupRow.getAttribute('data-group');
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            
            // Find all rows belonging to this group
            const groupRows = [];
            let inGroup = false;
            for (let i = 0; i < rows.length; i++) {
                const row = rows[i];
                if (row === groupRow) {
                    inGroup = true;
                    continue;
                }
                if (inGroup && row.hasAttribute('data-group')) {
                    inGroup = false;
                    break;
                }
                if (inGroup) {
                    groupRows.push(row);
                }
            }
            
            // Toggle visibility of group rows
            if (groupRow.classList.contains('collapsed')) {
                groupRow.classList.remove('collapsed');
                groupRows.forEach(row => row.style.display = '');
            } else {
                groupRow.classList.add('collapsed');
                groupRows.forEach(row => row.style.display = 'none');
            }
        }
        
        // Pagination variables and functions
        let currentPage = 1;
        const rowsPerPage = 50;
        
        function showPage(page) {
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            const totalRows = rows.length;
            const totalPages = Math.ceil((totalRows - 1) / rowsPerPage); // Subtract header row
            
            // Update current page
            currentPage = page;
            
            // Show correct page
            let rowCount = 0;
            for (let i = 1; i < totalRows; i++) { // Skip header row
                if (i > (page-1) * rowsPerPage && i <= page * rowsPerPage) {
                    rows[i].style.display = '';
                    rowCount++;
                } else {
                    rows[i].style.display = 'none';
                }
            }
            
            // Update page info
            document.getElementById('currentPage').textContent = page;
            document.getElementById('totalPages').textContent = totalPages;
            
            // Update button states
            document.getElementById('prevBtn').disabled = page === 1;
            document.getElementById('nextBtn').disabled = page === totalPages;
        }
        
        function nextPage() {
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            const totalRows = rows.length;
            const totalPages = Math.ceil((totalRows - 1) / rowsPerPage);
            
            if (currentPage < totalPages) {
                showPage(currentPage + 1);
            }
        }
        
        function prevPage() {
            if (currentPage > 1) {
                showPage(currentPage - 1);
            }
        }
        
        // Toggle all groups
        function toggleAllGroups(expand) {
            const table = document.getElementById('comparisonTable');
            const groupRows = table.querySelectorAll('tr[data-group]');
            
            groupRows.forEach(row => {
                const isCollapsed = row.classList.contains('collapsed');
                if (expand && isCollapsed) {
                    toggleGroup({closest: function() { return row; }});
                } else if (!expand && !isCollapsed) {
                    toggleGroup({closest: function() { return row; }});
                }
            });
        }
        
        // Initialize when document is loaded
        document.addEventListener('DOMContentLoaded', function() {
            if (document.getElementById('paginationEnabled')) {
                showPage(1);
            }
        });
        </script>
        """
        
        # Build HTML table content
        html_rows = []
        current_section = None
        current_group = None
        
        for _, row in styled_df.iterrows():
            # Check if this is a group header (groups start with "group_")
            if isinstance(row['section'], str) and row['section'].startswith('group_'):
                current_group = row['section'].replace('group_', '')
                group_header = row['rank']
                html_rows.append(f'<tr class="group-header" data-group="{current_group}" onclick="toggleGroup(this)"><td colspan="5">{group_header} <span class="group-toggle">[−]</span></td></tr>')
                continue
                
            if current_section != row['section']:
                current_section = row['section']
                html_rows.append(f'<tr class="section-header"><td colspan="5">Scenario: {current_section}</td></tr>')
                html_rows.append('<tr><th>Rank</th><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th></tr>')
            
            # Style based on rank
            if isinstance(row['rank'], int) and row['rank'] == 1:
                row_class = 'class="best-model"'
            elif isinstance(row['rank'], int) and row['rank'] == 2:
                row_class = 'class="second-best"'
            elif isinstance(row['rank'], str) and "Scenario" in str(row['rank']):
                row_class = 'class="scenario-header"'
            else:
                row_class = ''
            
            # Add the row
            html_rows.append(f'''
            <tr {row_class}>
                <td>{row['rank']}</td>
                <td>{row['model']}</td>
                <td class="metrics">{row['rmse']}</td>
                <td class="metrics">{row['mae']}</td>
                <td class="metrics">{row['mape']}</td>
            </tr>
            ''')
        
        # Build the HTML document
        pagination_controls = ""
        filter_box = ""
        group_controls = ""
        
        if should_group_scenarios:
            group_controls = """
            <div class="group-controls no-print">
                <button onclick="toggleAllGroups(true)">Expand All Groups</button>
                <button onclick="toggleAllGroups(false)">Collapse All Groups</button>
            </div>
            """
        
        if need_pagination:
            pagination_controls = """
            <div id="paginationEnabled" class="no-print"></div>
            <div class="pagination no-print">
                <button id="prevBtn" onclick="prevPage()">Previous</button>
                <div class="page-info">
                    Page <span id="currentPage">1</span> of <span id="totalPages">1</span>
                </div>
                <button id="nextBtn" onclick="nextPage()">Next</button>
            </div>
            """
            
            filter_box = """
            <div class="no-print">
                <input type="text" id="searchInput" class="search-box" onkeyup="filterTable()" placeholder="Search for models, scenarios...">
                <div class="page-info">Showing <span id="visibleRowCount">0</span> rows</div>
            </div>
            """
        
        html_table = f"""<table id="comparisonTable">
        {''.join(html_rows)}
        </table>"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_html = f"""
        <html>
        <head>
            <title>Model Performance Comparison</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            {css}
            {javascript}
        </head>
        <body>
            <h1>Model Performance Comparison</h1>
            <p>Generated on {timestamp}</p>
            {group_controls}
            {filter_box}
            {pagination_controls}
            {html_table}
            {pagination_controls}
        </body>
        </html>"""
        
        with open("results/detailed_comparisons/all_models_comparison.html", "w") as f:
            f.write(full_html)
        
        logging.info("Detailed performance comparison saved to results/detailed_comparisons/all_models_comparison.html")
    except Exception as e:
        logging.error(f"Error creating HTML comparison: {e}")
        # Continue execution even if HTML creation fails
    
    logging.info("Detailed performance comparison table saved to results/detailed_comparisons/all_models_comparison.csv")

# Function to create visualizations
def create_visualizations(results_df, wins_df):
    # Create plots directory
    os.makedirs("results/plots", exist_ok=True)

    # Skip visualization if the results dataframe is empty
    if results_df.empty:
        logging.warning("Results dataframe is empty. Skipping visualizations.")
        return
    
    try:
        # Make sure ensemble model is included in the wins_df if it exists in results_df
        if 'ensemble' in results_df['model'].values and wins_df is not None and 'ensemble' not in wins_df['model'].values:
            # Calculate wins for ensemble model
            ensemble_wins = {'model': 'ensemble', 'total_wins': 0, 'mae_wins': 0, 'rmse_wins': 0, 'mape_wins': 0}
            
            # Count wins for each scenario
            for scenario in results_df['scenario'].unique():
                scenario_df = results_df[results_df['scenario'] == scenario]
                
                # Check if ensemble is the best for each metric
                if scenario_df[scenario_df['mae'] == scenario_df['mae'].min()]['model'].values[0] == 'ensemble':
                    ensemble_wins['mae_wins'] += 1
                    ensemble_wins['total_wins'] += 1
                    
                if scenario_df[scenario_df['rmse'] == scenario_df['rmse'].min()]['model'].values[0] == 'ensemble':
                    ensemble_wins['rmse_wins'] += 1
                    ensemble_wins['total_wins'] += 1
                    
                if scenario_df[scenario_df['mape'] == scenario_df['mape'].min()]['model'].values[0] == 'ensemble':
                    ensemble_wins['mape_wins'] += 1
                    ensemble_wins['total_wins'] += 1
            
            # Add ensemble to wins_df
            wins_df = pd.concat([wins_df, pd.DataFrame([ensemble_wins])], ignore_index=True)
    except Exception as e:
        logging.error(f"Error processing ensemble model: {e}")
    
    try:    
        # Sort wins_df by total_wins for visualization
        if wins_df is not None:
            wins_df = wins_df.sort_values('total_wins', ascending=False)
    except Exception as e:
        logging.error(f"Error sorting wins dataframe: {e}")
        
    # Create best model plots for each scenario
    try:
        create_best_model_plots(results_df)
    except Exception as e:
        logging.error(f"Error creating best model plots: {e}")
    
    # Create a summary chart of the best models
    try:
        create_best_model_summary(results_df)
    except Exception as e:
        logging.error(f"Error creating best model summary: {e}")
    
    # Create a radar chart for model comparison
    try:
        create_model_radar_chart(results_df)
    except Exception as e:
        logging.error(f"Error creating model radar chart: {e}")
    
    # Create detailed performance comparison
    try:
        create_performance_comparison_table(results_df, wins_df)
    except Exception as e:
        logging.error(f"Error creating performance comparison table: {e}")
        
    # Skip the rest of visualization in test mode (with only one scenario)
    if len(results_df['scenario'].unique()) <= 1:
        logging.info("Skipping additional visualizations in test mode (only one scenario)")
        return
    
    # ======= ORIGINAL VISUALIZATIONS =======
    # Plot model wins
    if wins_df is not None:
        plt.figure(figsize=(12, 6))
        wins_df = wins_df.sort_values('total_wins')
        plt.barh(wins_df['model'], wins_df['total_wins'], color='blue', alpha=0.7)
        plt.xlabel('Number of Wins')
        plt.ylabel('Model')
        plt.title('Total Wins by Model Across All Scenarios')
        plt.tight_layout()
        plt.savefig('results/plots/model_wins.png')
        
        # Plot wins by metric type
        plt.figure(figsize=(14, 8))
        bar_width = 0.25
        index = np.arange(len(wins_df))
        
        plt.barh(index, wins_df['mae_wins'], bar_width, color='blue', alpha=0.7, label='MAE Wins')
        plt.barh(index + bar_width, wins_df['rmse_wins'], bar_width, color='green', alpha=0.7, label='RMSE Wins')
        plt.barh(index + 2*bar_width, wins_df['mape_wins'], bar_width, color='red', alpha=0.7, label='MAPE Wins')
        
        plt.xlabel('Number of Wins')
        plt.ylabel('Model')
        plt.title('Wins by Metric Type for Each Model')
        plt.yticks(index + bar_width, wins_df['model'])
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/plots/wins_by_metric.png')
    
    # Plot average metrics by model
    avg_metrics = results_df.groupby('model')[['mae', 'rmse', 'mape']].mean().reset_index()
    
    # Function to create improved bar plots
    def create_bar_chart(metric, title, filename, figsize=(14, 8)):
        # Sort data
        sorted_data = avg_metrics.sort_values(metric)
        
        # Calculate appropriate font size based on number of models
        num_models = len(sorted_data)
        fontsize = max(6, 12 - (0.3 * num_models))
        bar_width = max(0.5, 0.8 - (0.02 * num_models))  # Narrower bars for many models
        
        # Create figure with adjusted size for many models
        plt.figure(figsize=(max(figsize[0], num_models * 0.6), figsize[1]))
        
        # Create bars with gradient color based on performance
        bars = plt.barh(
            sorted_data['model'],
            sorted_data[metric],
            color=plt.cm.viridis(np.linspace(0, 0.8, len(sorted_data))),
            alpha=0.8,
            height=bar_width
        )
        
        # Add value labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            value_fontsize = max(6, fontsize - 1)
            plt.text(
                width + width * 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left',
                va='center',
                fontsize=value_fontsize
            )
        
        # Add a light grid
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Customize chart
        plt.xlabel(f'Average {metric.upper()} (lower is better)', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(title, fontsize=14)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        
        # Add the best model highlight
        best_model = sorted_data.iloc[0]['model']
        best_value = sorted_data.iloc[0][metric]
        plt.text(
            0.98, 0.02,
            f'Best: {best_model} ({best_value:.2f})',
            transform=plt.gca().transAxes,
            ha='right',
            va='bottom',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, pad=5, boxstyle='round')
        )
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create improved bar charts
    create_bar_chart('mae', 'Average MAE by Model Across All Scenarios', 'results/plots/average_mae.png')
    create_bar_chart('rmse', 'Average RMSE by Model Across All Scenarios', 'results/plots/average_rmse.png')
    create_bar_chart('mape', 'Average MAPE by Model Across All Scenarios', 'results/plots/average_mape.png')
    
    # If we have many models, create a consolidated comparison chart
    if len(avg_metrics) > 8:
        # Create a compact consolidated chart showing all three metrics
        plt.figure(figsize=(16, 10))
        
        # Get the top 10 models based on overall performance
        performance_rank = (
            avg_metrics['mae'].rank() + 
            avg_metrics['rmse'].rank() + 
            avg_metrics['mape'].rank()
        ) / 3
        
        avg_metrics['rank'] = performance_rank
        top_models = avg_metrics.sort_values('rank').head(10)['model'].tolist()
        
        # Filter to top models
        top_data = avg_metrics[avg_metrics['model'].isin(top_models)].sort_values('rank')
        
        # Create subplots for each metric
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot each metric
        for i, metric in enumerate(['mae', 'rmse', 'mape']):
            # Normalize for visual comparison
            normalized = top_data[metric] / top_data[metric].max()
            
            # Plot horizontal bars
            bars = axes[i].barh(
                top_data['model'],
                normalized,
                color=plt.cm.viridis(np.linspace(0, 0.8, len(top_data))),
                alpha=0.8
            )
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                raw_value = top_data.iloc[j][metric]
                axes[i].text(
                    width + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{raw_value:.3f}',
                    ha='left',
                    va='center'
                )
            
            # Customize
            axes[i].set_title(f'{metric.upper()} (Normalized)', fontsize=12)
            axes[i].grid(axis='x', linestyle='--', alpha=0.3)
            
            # Only show y-axis labels for the first plot
            if i > 0:
                axes[i].set_ylabel('')
            else:
                axes[i].set_ylabel('Model', fontsize=12)
        
        # Overall chart title
        fig.suptitle('Top 10 Models Performance Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        plt.savefig('results/plots/top_models_consolidated.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ======= ADDITIONAL DETAILED VISUALIZATIONS =======
    
    # 1. Heatmap of model performance across scenarios
    plt.figure(figsize=(16, 10))
    # Pivot the data to get a matrix of model vs scenario with RMSE values
    heatmap_data_rmse = results_df.pivot_table(index='model', columns='scenario', values='rmse')
    
    # Check if we have a large dataset and adjust the heatmap accordingly
    num_models = len(heatmap_data_rmse.index)
    num_scenarios = len(heatmap_data_rmse.columns)
    
    # Create directory for additional heatmaps if needed
    if num_models > 8 or num_scenarios > 12:
        os.makedirs("results/plots/detailed_heatmaps", exist_ok=True)
        logging.info(f"Creating detailed heatmaps for large dataset ({num_models} models x {num_scenarios} scenarios)")
    
    # Normalize values for better visualization (optional)
    normalized_heatmap = (heatmap_data_rmse - heatmap_data_rmse.min()) / (heatmap_data_rmse.max() - heatmap_data_rmse.min())
    
    # Calculate appropriate font size based on dimensions
    fontsize = max(5, 10 - (0.3 * max(num_models, num_scenarios)))
    
    # Create the main heatmap - if too large, don't show annotations
    show_annot = num_models * num_scenarios <= 200  # Only show annotations for smaller heatmaps
    
    # Main heatmap (overview)
    sns.heatmap(
        normalized_heatmap, 
        annot=show_annot, 
        cmap='YlGnBu', 
        fmt='.2f' if show_annot else '', 
        linewidths=.5,
        annot_kws={'size': fontsize} if show_annot else {}
    )
    plt.title('Model Performance Heatmap (RMSE) - Normalized Across Scenarios')
    plt.xticks(rotation=45, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('results/plots/model_scenario_heatmap_rmse.png', dpi=300)
    plt.close()
    
    # If we have a large dataset, create additional split heatmaps
    if num_models > 8 or num_scenarios > 12:
        # Split model groups (rows)
        model_groups = [heatmap_data_rmse.index[i:i+8] for i in range(0, num_models, 8)]
        
        # Split scenario groups (columns)
        scenario_groups = [heatmap_data_rmse.columns[i:i+12] for i in range(0, num_scenarios, 12)]
        
        # Create detailed heatmaps for each combination
        for m_idx, model_group in enumerate(model_groups):
            for s_idx, scenario_group in enumerate(scenario_groups):
                plt.figure(figsize=(min(16, 2 + len(scenario_group)), min(10, 2 + len(model_group))))
                
                # Get the subset data
                subset_data = normalized_heatmap.loc[model_group, scenario_group]
                
                # Calculate appropriate font size for this subset
                subset_fontsize = max(8, 12 - (0.3 * max(len(model_group), len(scenario_group))))
                
                # Create the subset heatmap with annotations
                sns.heatmap(
                    subset_data, 
                    annot=True, 
                    cmap='YlGnBu', 
                    fmt='.2f', 
                    linewidths=.5,
                    annot_kws={'size': subset_fontsize}
                )
                
                plt.title(f'Model Performance Detail - Group {m_idx+1} vs Scenarios {s_idx+1}')
                plt.xticks(rotation=45, fontsize=subset_fontsize)
                plt.yticks(fontsize=subset_fontsize)
                plt.tight_layout()
                
                # Save the detailed heatmap
                plt.savefig(f'results/plots/detailed_heatmaps/heatmap_models{m_idx+1}_scenarios{s_idx+1}.png', dpi=300)
                plt.close()
    
    # 2. Box plots for comparing distribution of each metric across models
    # Check if we have many models and need to adjust the box plots
    num_models = len(results_df['model'].unique())
    
    # Create directory for boxplots
    os.makedirs("results/plots/boxplots", exist_ok=True)
    
    # Create a common function for box plots with better handling of large datasets
    def create_boxplot(metric, title, filename, order=None):
        plt.figure(figsize=(max(14, num_models * 0.8), 8))
        
        # Calculate appropriate font size
        fontsize = max(6, 12 - (0.3 * num_models))
        rotation = min(90, 45 + (num_models // 5 * 15))  # Increase rotation angle as number of models grows
        
        # Create boxplot with custom styling
        ax = sns.boxplot(
            x='model', 
            y=metric, 
            data=results_df,
            order=order,
            palette='viridis',
            width=0.6,
            fliersize=3,
            linewidth=1
        )
        
        # Add a strip plot on top for individual data points if we have few models
        if num_models <= 15:
            sns.stripplot(
                x='model', 
                y=metric, 
                data=results_df,
                order=order,
                size=3, 
                color=".3", 
                linewidth=0,
                alpha=0.5,
                jitter=True
            )
        
        # Add grid lines for easier readability
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add median labels
        medians = results_df.groupby('model')[metric].median()
        pos = range(len(medians))
        
        # Only show labels if we have reasonably few models
        if num_models <= 20:
            for tick, label in zip(pos, ax.get_xticklabels()):
                model = label.get_text()
                if model in medians:
                    ax.text(
                        tick, 
                        medians[model] + 0.02 * results_df[metric].max(), 
                        f'{medians[model]:.2f}', 
                        horizontalalignment='center',
                        size=fontsize,
                        color='black',
                        weight='semibold'
                    )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(f"{metric.upper()} (lower is better)", fontsize=12)
        plt.xticks(rotation=rotation, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Get ordered models by median of each metric
    rmse_order = results_df.groupby('model')['rmse'].median().sort_values().index.tolist()
    mae_order = results_df.groupby('model')['mae'].median().sort_values().index.tolist()
    mape_order = results_df.groupby('model')['mape'].median().sort_values().index.tolist()
    
    # Create the boxplots for each metric with models ordered by performance
    create_boxplot('mae', 'Distribution of MAE Across Models', 'results/plots/mae_distribution_boxplot.png', order=mae_order)
    create_boxplot('rmse', 'Distribution of RMSE Across Models', 'results/plots/rmse_distribution_boxplot.png', order=rmse_order)
    create_boxplot('mape', 'Distribution of MAPE Across Models', 'results/plots/mape_distribution_boxplot.png', order=mape_order)
    
    # For very large datasets, create additional visualizations showing only top and bottom performers
    if num_models > 15:
        logging.info(f"Creating simplified boxplots for large number of models ({num_models})")
        
        # Create top/bottom boxplots for each metric
        for metric, order in [('rmse', rmse_order), ('mae', mae_order), ('mape', mape_order)]:
            # Take top 5 and bottom 5 models
            top_models = order[:5]
            bottom_models = order[-5:]
            selected_models = top_models + bottom_models
            
            # Filter the data
            filtered_df = results_df[results_df['model'].isin(selected_models)]
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(
                x='model', 
                y=metric, 
                data=filtered_df,
                order=top_models + bottom_models,
                palette=['green']*5 + ['red']*5,
                width=0.6
            )
            
            # Add strip plot for individual points
            sns.stripplot(
                x='model', 
                y=metric, 
                data=filtered_df,
                order=top_models + bottom_models,
                size=4, 
                color=".3", 
                linewidth=0,
                alpha=0.5,
                jitter=True
            )
            
            plt.title(f"Top 5 vs Bottom 5 Models - {metric.upper()}", fontsize=14)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel(f"{metric.upper()} (lower is better)", fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'results/plots/boxplots/top_bottom_{metric}_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()

    # 3. Radar chart comparing models on multiple metrics
    # First compute normalized metrics for radar chart
    metrics_for_radar = avg_metrics.copy()
    for metric in ['mae', 'rmse', 'mape']:
        max_val = metrics_for_radar[metric].max()
        min_val = metrics_for_radar[metric].min()
        # Normalize and invert (1 = best, 0 = worst)
        metrics_for_radar[f'{metric}_normalized'] = 1 - (metrics_for_radar[metric] - min_val) / (max_val - min_val)
    
    # Create radar chart
    labels = ['MAE', 'RMSE', 'MAPE']
    num_models = len(metrics_for_radar)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, model in enumerate(metrics_for_radar['model']):
        values = [metrics_for_radar.loc[i, f'{m}_normalized'] for m in ['mae', 'rmse', 'mape']]
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(180 / len(labels))
    ax.set_title("Model Comparison on Multiple Metrics (Higher is Better)", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig('results/plots/model_radar_chart.png')

    # 4. Grouped bar chart for model comparison across all metrics
    metrics_of_interest = ['mae', 'rmse', 'mape']
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(avg_metrics))
    bar_width = 0.25
    opacity = 0.8
    
    colors = ['blue', 'green', 'red']
    
    for i, metric in enumerate(metrics_of_interest):
        # Normalize the metric values for better visualization on the same chart
        normalized_values = avg_metrics[metric] / avg_metrics[metric].max()
        ax.bar(x + i*bar_width, normalized_values, bar_width,
               alpha=opacity, color=colors[i], label=metric.upper())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Normalized Metric Values (lower is better)')
    ax.set_title('Comparison of All Metrics Across Models (Normalized)')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(avg_metrics['model'], rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/all_metrics_comparison.png')

    # 5. Performance vs. Scenario complexity
    # Assuming scenario complexity is implicitly captured in average error across all models
    scenario_complexity = results_df.groupby('scenario')[['mae', 'rmse', 'mape']].mean().reset_index()
    scenario_complexity['complexity_score'] = (
        scenario_complexity['mae'] / scenario_complexity['mae'].max() +
        scenario_complexity['rmse'] / scenario_complexity['rmse'].max() +
        scenario_complexity['mape'] / scenario_complexity['mape'].max()
    ) / 3
    
    # Sort scenarios by complexity
    scenario_complexity = scenario_complexity.sort_values('complexity_score')
    scenario_order = scenario_complexity['scenario'].tolist()
    
    # Plot the performance of each model across scenarios ordered by complexity
    plt.figure(figsize=(16, 10))
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        # Reorder data by scenario complexity
        model_data = pd.merge(
            pd.DataFrame({'scenario': scenario_order}),
            model_data,
            on='scenario',
            how='left'
        )
        plt.plot(model_data['scenario'], model_data['rmse'], marker='o', label=model)
    
    plt.title('Model Performance Across Scenarios of Increasing Complexity')
    plt.xlabel('Scenario (Ordered by Complexity)')
    plt.ylabel('RMSE (lower is better)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/model_performance_vs_complexity.png')

    # 6. Performance Heatmap with Hierarchical Clustering
    # This shows clusters of similar-performing models across scenarios
    if len(results_df['scenario'].unique()) >= 2:  # Need at least 2 scenarios for clustering
        plt.figure(figsize=(16, 10))
        performance_matrix = results_df.pivot_table(index='model', columns='scenario', values='rmse')
        
        # Apply clustering
        row_linkage = hierarchy.linkage(performance_matrix, method='average')
        col_linkage = hierarchy.linkage(performance_matrix.T, method='average')
        
        sns.clustermap(performance_matrix, 
                     figsize=(16, 10),
                     cmap='YlGnBu_r', 
                     annot=True, 
                     fmt='.2f',
                     row_linkage=row_linkage,
                     col_linkage=col_linkage,
                     linewidths=.5)
        
        plt.savefig('results/plots/model_scenario_clustered_heatmap.png')

    # 7. Error distribution analysis
    # Get all scenarios that have results for all models
    complete_scenarios = results_df.groupby('scenario').filter(
        lambda x: len(x) == len(results_df['model'].unique())
    )['scenario'].unique()
    
    if len(complete_scenarios) > 0:
        scenario = complete_scenarios[0]  # Just take the first complete scenario
        scenario_df = results_df[results_df['scenario'] == scenario]
        
        # Load the actual predictions for this scenario for all models
        predictions_by_model = {}
        for model in scenario_df['model']:
            # Find the most recent prediction file
            model_run_id = f"{model}_{scenario.replace('cleaned_data_', '')}"
            pattern = f"results/{model_run_id}_*.json"
            prediction_files = sorted(glob.glob(pattern))
            
            if prediction_files:
                latest_file = prediction_files[-1]
                # Extract timestamp from filename
                timestamp = latest_file.split('_')[-1].split('.')[0]
                
                # Load predictions file
                prediction_file = f"train/models/{model}_model/runs/{model_run_id}/predictions.csv"
                if os.path.exists(prediction_file):
                    predictions = pd.read_csv(prediction_file)
                    predictions_by_model[model] = predictions
        
        # If we have predictions, plot error distributions
        if predictions_by_model:
            plt.figure(figsize=(16, 10))
            for model, predictions in predictions_by_model.items():
                errors = predictions['actual'] - predictions['predicted']
                sns.kdeplot(errors, label=model)
            
            plt.title(f'Error Distribution by Model for Scenario: {scenario}')
            plt.xlabel('Error (Actual - Predicted)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'results/plots/error_distribution_{scenario}.png')

    # 8. Training Time Comparison
    try:
        # Extract training times from results JSON files
        training_times = []
        for index, row in results_df.iterrows():
            model = row['model']
            scenario = row['scenario']
            model_run_id = f"{model}_{scenario.replace('cleaned_data_', '')}"
            metrics_file = f"train/models/{model}_model/runs/{model_run_id}/metrics.json"
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    if 'training_time' in metrics_data:
                        training_times.append({
                            'model': model,
                            'scenario': scenario,
                            'training_time': metrics_data['training_time']
                        })
        
        if training_times:
            training_df = pd.DataFrame(training_times)
            
            # Average training time by model
            avg_training_time = training_df.groupby('model')['training_time'].mean().reset_index()
            avg_training_time = avg_training_time.sort_values('training_time')
            
            plt.figure(figsize=(12, 6))
            plt.barh(avg_training_time['model'], avg_training_time['training_time'], color='purple', alpha=0.7)
            plt.xlabel('Average Training Time (seconds)')
            plt.ylabel('Model')
            plt.title('Average Training Time by Model')
            plt.tight_layout()
            plt.savefig('results/plots/average_training_time.png')
            
            # Training time heatmap
            plt.figure(figsize=(16, 10))
            heatmap_training = training_df.pivot_table(index='model', columns='scenario', values='training_time')
            sns.heatmap(heatmap_training, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
            plt.title('Training Time (seconds) by Model and Scenario')
            plt.tight_layout()
            plt.savefig('results/plots/training_time_heatmap.png')
    except Exception as e:
        logging.warning(f"Error creating training time visualizations: {e}")

    # 9. Create model comparison matrices
    create_model_comparison_matrix(results_df)
    
    # Export a detailed model comparison table
    detailed_table = pd.pivot_table(
        results_df,
        values=['mae', 'rmse', 'mape'],
        index=['model'],
        columns=['scenario'],
        aggfunc='mean'
    )
    
    # Create a more readable format
    detailed_table = detailed_table.round(2)
    detailed_table.to_csv('results/detailed_model_comparison.csv')
    
    # Save figures with higher DPI for better quality
    plt.figure(figsize=(20, 12))
    plt.text(0.5, 0.5, 'Model Comparison Summary', 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=24, fontweight='bold')
    plt.axis('off')
    plt.savefig('results/plots/model_comparison_title.png', dpi=300)

def create_model_comparison_matrix(results_df):
    """Create a matrix of pairwise differences between models."""
    # Get unique models
    models = results_df['model'].unique()
    
    # Create matrices to store average percentage improvements
    rmse_improvement = pd.DataFrame(index=models, columns=models, data=0.0)
    mae_improvement = pd.DataFrame(index=models, columns=models, data=0.0)
    mape_improvement = pd.DataFrame(index=models, columns=models, data=0.0)
    
    # For each scenario, calculate pairwise improvements
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        
        for model1 in models:
            for model2 in models:
                if model1 != model2:
                    # Get metrics for each model
                    metrics1 = scenario_data[scenario_data['model'] == model1]
                    metrics2 = scenario_data[scenario_data['model'] == model2]
                    
                    if not metrics1.empty and not metrics2.empty:
                        # Calculate percentage improvements (positive means model1 is better than model2)
                        rmse_diff = (metrics2['rmse'].values[0] - metrics1['rmse'].values[0]) / metrics2['rmse'].values[0] * 100
                        mae_diff = (metrics2['mae'].values[0] - metrics1['mae'].values[0]) / metrics2['mae'].values[0] * 100
                        mape_diff = (metrics2['mape'].values[0] - metrics1['mape'].values[0]) / metrics2['mape'].values[0] * 100
                        
                        # Add to the cumulative values
                        rmse_improvement.loc[model1, model2] += rmse_diff
                        mae_improvement.loc[model1, model2] += mae_diff
                        mape_improvement.loc[model1, model2] += mape_diff
    
    # Calculate average across all scenarios
    n_scenarios = len(results_df['scenario'].unique())
    rmse_improvement = rmse_improvement / n_scenarios
    mae_improvement = mae_improvement / n_scenarios
    mape_improvement = mape_improvement / n_scenarios
    
    # Create visualizations of the improvement matrices
    plt.figure(figsize=(14, 10))
    sns.heatmap(rmse_improvement, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=0.5)
    plt.title('Average RMSE Improvement (%) - Row Model vs Column Model')
    plt.tight_layout()
    plt.savefig('results/plots/rmse_improvement_matrix.png')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(mae_improvement, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=0.5)
    plt.title('Average MAE Improvement (%) - Row Model vs Column Model')
    plt.tight_layout()
    plt.savefig('results/plots/mae_improvement_matrix.png')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(mape_improvement, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=0.5)
    plt.title('Average MAPE Improvement (%) - Row Model vs Column Model')
    plt.tight_layout()
    plt.savefig('results/plots/mape_improvement_matrix.png')
    
    # Create a summary of overall model ranking
    average_improvement = (rmse_improvement + mae_improvement + mape_improvement) / 3
    model_scores = average_improvement.mean(axis=1).sort_values(ascending=False)
    
    # Plot overall model ranking
    plt.figure(figsize=(12, 8))
    model_scores.plot(kind='bar', color='skyblue')
    plt.title('Average Performance Improvement Across All Metrics and Models')
    plt.xlabel('Model')
    plt.ylabel('Average Improvement %')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/model_overall_ranking.png')
    
    return rmse_improvement, mae_improvement, mape_improvement

def create_best_model_plots(results_df):
    """Create clean plots showing only the best model for each scenario."""
    # Create directory for best model plots
    os.makedirs("results/best_model_plots", exist_ok=True)
    
    # Check if we have many scenarios and need to create a directory for grouped plots
    num_scenarios = len(results_df['scenario'].unique())
    
    if num_scenarios > 20:
        os.makedirs("results/best_model_plots/groups", exist_ok=True)
        logging.info(f"Creating grouped best model plots due to large number of scenarios ({num_scenarios})")
    
    # Keep track of scenarios already processed for summary plot
    processed_scenarios = []
    
    # Loop through each scenario (these should be normalized names now)
    for idx, scenario in enumerate(results_df['scenario'].unique()):
        scenario_df = results_df[results_df['scenario'] == scenario]
        
        # Find the best model based on RMSE (you can change this to MAE or MAPE if preferred)
        best_model = scenario_df.loc[scenario_df['rmse'].idxmin()]['model']
        
        logging.info(f"Creating plot for {scenario} with best model: {best_model}")
        
        # Define data paths using the scenario name directly
        train_data_path = f"{scenarios_path}/{scenario}_train.csv"
        test_data_path = f"{scenarios_path}/{scenario}_test.csv"
        
        if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
            logging.warning(f"Data files for {scenario} not found at {train_data_path}. Skipping best model plot.")
            continue
        
        # Load data - we only need the test data for the plot
        try:
            test_data = pd.read_csv(test_data_path)
            if 'timestamp' in test_data.columns:
                test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
            else:
                logging.warning(f"No timestamp column in test data for {scenario}. Skipping best model plot.")
                continue
            # Check if 'pod_count' column exists, otherwise try 'y'
            if 'pod_count' not in test_data.columns:
                 if 'y' in test_data.columns:
                      test_data = test_data.rename(columns={'y': 'pod_count'})
                 else:
                      logging.warning(f"Neither 'pod_count' nor 'y' column found in test data for {scenario}. Skipping.")
                      continue

        except Exception as e:
            logging.error(f"Error loading test data for {scenario}: {e}")
            continue
        
        # Get model predictions
        # Results folder uses normalized name
        model_predictions_path = f"results/{scenario}/{best_model}_predictions.csv"
        if not os.path.exists(model_predictions_path):
            logging.warning(f"Predictions for {best_model} in {scenario} not found at {model_predictions_path}. Trying model run directory...")
            
            # Try to find predictions in model run directory using normalized scenario name
            run_id = f"{best_model}_{scenario}"
            model_run_predictions = f"train/models/{best_model}_model/runs/{run_id}/predictions.csv"
            
            if os.path.exists(model_run_predictions):
                try:
                    # Copy predictions to the expected location
                    os.makedirs(f"results/{scenario}", exist_ok=True)
                    predictions_df = pd.read_csv(model_run_predictions)
                    
                    # Ensure the predictions DataFrame has the expected columns
                    if "actual" in predictions_df.columns and "predicted" in predictions_df.columns:
                        # Rename columns if needed
                        predictions_df = predictions_df.rename(columns={
                            "predicted": "prediction"
                        })
                    
                    predictions_df.to_csv(model_predictions_path, index=False)
                    logging.info(f"Copied predictions from {model_run_predictions} to {model_predictions_path}")
                except Exception as e:
                    logging.error(f"Error copying predictions file: {e}")
                    continue
            else:
                logging.error(f"No predictions found for {best_model} in {scenario} in model run directory: {model_run_predictions}")
                continue
        
        try:
            model_predictions = pd.read_csv(model_predictions_path)
            if 'timestamp' in model_predictions.columns:
                model_predictions['timestamp'] = pd.to_datetime(model_predictions['timestamp'])
            else:
                logging.warning(f"No timestamp column in predictions for {scenario}. Skipping best model plot.")
                continue
                
            logging.info(f"Loaded predictions from {model_predictions_path}, shape: {model_predictions.shape}")
        except Exception as e:
            logging.error(f"Error reading predictions file {model_predictions_path}: {e}")
            continue
        
        # Create the plot with a clean professional look
        try:
            # Determine if we should handle a large time series (downsample for performance)
            time_points = len(model_predictions)
            downsample = time_points > 500  # Only downsample very large series
            
            # Create a modern, professional plot
            plt.style.use('ggplot')  # Modern style
            fig, ax = plt.subplots(figsize=(15, 8), dpi=300)  # Increased DPI for higher quality
            
            # Apply downsampling if needed
            if downsample:
                logging.info(f"Downsampling time series for {scenario} ({time_points} points)")
                # Calculate appropriate step size
                step = max(1, time_points // 500)
                
                # Plot only test data (actual values) - use blue color
                ax.plot(test_data['timestamp'].iloc[::step], test_data['pod_count'].iloc[::step], 
                        color='#1f77b4', linewidth=2.5, label='Actual Values')
                
                # Plot the best model predictions with orange color
                ax.plot(model_predictions['timestamp'].iloc[::step], model_predictions['prediction'].iloc[::step], 
                        color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'Best Model: {best_model}')
            else:
                # Plot original data without downsampling
                ax.plot(test_data['timestamp'], test_data['pod_count'], 
                        color='#1f77b4', linewidth=2.5, label='Actual Values')
                
                # Plot the best model predictions with orange color
                ax.plot(model_predictions['timestamp'], model_predictions['prediction'], 
                        color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'Best Model: {best_model}')
            
            # Add a legend in the top left corner with better font size
            legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
            
            # Add scenario name as title
            ax.set_title(f"Scenario: {scenario}", fontsize=14, pad=10)
            
            # Format axes better
            # Format date axis when we have a reasonable number of data points
            if not downsample and time_points < 1000:
                # Format date axis nicely
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45, ha='right', fontsize=10)
                ax.set_xlabel("Date", fontsize=12)
            else:
                # For very large datasets or downsampled ones, simplify the x-axis
                ax.set_xlabel("")
                ax.get_xaxis().set_visible(False)
            
            ax.set_ylabel("Value", fontsize=12)
            
            # Keep border (spines) visible for a "main border" appearance
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            
            # Add metrics in a text box
            metrics = scenario_df[scenario_df['model'] == best_model].iloc[0]
            metrics_text = (
                f"RMSE: {metrics['rmse']:.3f}\n"
                f"MAE: {metrics['mae']:.3f}\n"
                f"MAPE: {metrics['mape']:.3f}%"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Add grid but make it subtle
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Save the figure with high resolution - use normalized name for consistent filenames
            normalized_scenario = normalize_scenario_name(scenario)
            plt.tight_layout()
            
            # Main plot saved to the root directory
            plot_path = f"results/best_model_plots/{normalized_scenario}_best_model.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # If we have many scenarios, also save to groups directory
            if num_scenarios > 20:
                group_idx = idx // 20 + 1  # Group number
                group_path = f"results/best_model_plots/groups/group{group_idx}_{normalized_scenario}_best_model.png"
                plt.savefig(group_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            # Add to processed scenarios list
            processed_scenarios.append({
                'scenario': scenario,
                'best_model': best_model,
                'rmse': metrics['rmse']
            })
            
            logging.info(f"Best model plot for {scenario} saved to {plot_path}")
        except Exception as e:
            logging.error(f"Error creating plot for {scenario}: {e}")
            continue
            
    # Create a summary grid of thumbnails if we have many scenarios
    if num_scenarios > 8 and len(processed_scenarios) > 8:
        try:
            create_best_model_grid(processed_scenarios, results_df)
        except Exception as e:
            logging.error(f"Error creating best model grid: {e}")
            
def create_best_model_grid(processed_scenarios, results_df):
    """Create a grid of thumbnails for the best model plots."""
    logging.info("Creating a summary grid of best model thumbnails")
    
    # Sort scenarios by RMSE (best first)
    sorted_scenarios = sorted(processed_scenarios, key=lambda x: x['rmse'])
    
    # Take the top 9 scenarios (or fewer if we have less)
    top_scenarios = sorted_scenarios[:min(9, len(sorted_scenarios))]
    
    # Calculate grid dimensions
    grid_size = len(top_scenarios)
    cols = min(3, grid_size)
    rows = (grid_size + cols - 1) // cols
    
    # Create figure for the grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=150)
    
    # Flatten axes for easier indexing
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols > 1:
        axes = axes.reshape(-1)
    elif rows > 1 and cols == 1:
        axes = axes.reshape(-1)
    else:
        axes = [axes]  # Single axis, make it a list
        
    # Add each thumbnail to the grid
    for i, scenario_info in enumerate(top_scenarios):
        if i >= len(axes):
            break  # Safety check
            
        scenario = scenario_info['scenario']
        best_model = scenario_info['best_model']
        normalized_scenario = normalize_scenario_name(scenario)
        
        # Load the image
        img_path = f"results/best_model_plots/{normalized_scenario}_best_model.png"
        
        if os.path.exists(img_path):
            try:
                img = plt.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{scenario}\nBest: {best_model}")
                axes[i].axis('off')
            except Exception as e:
                logging.error(f"Error adding image to grid: {e}")
                axes[i].text(0.5, 0.5, "Image load error", ha='center', va='center')
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i].axis('off')
    
    # Hide any unused axes
    for i in range(len(top_scenarios), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/best_model_plots/top_models_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Best model grid saved to results/best_model_plots/top_models_grid.png")

def save_summary_metrics(all_results, skip_visualization=False):
    """Save all model results to a CSV file for analysis."""
    
    # Create a list to store all metrics
    all_metrics_list = []
    
    # Iterate through scenarios and models to collect metrics
    for scenario, models in all_results.items():
        # Normalize the scenario name
        scenario_name = normalize_scenario_name(scenario)
        
        for model_name, metrics in models.items():
            # Add scenario and model to metrics dict
            metrics_row = {
                "scenario": scenario_name,
                "model": model_name,
                "mae": metrics.get("mae", None),
                "rmse": metrics.get("rmse", None),
                "mape": metrics.get("mape", None)
            }
            
            # Add any other metrics that might be present
            for key, value in metrics.items():
                if key not in ["mae", "rmse", "mape"]:
                    metrics_row[key] = value
            
            all_metrics_list.append(metrics_row)
    
    # Create a DataFrame with all metrics
    if all_metrics_list:
        results_df = pd.DataFrame(all_metrics_list)
        
        # Save the metrics to CSV
        results_df.to_csv("results/all_results.csv", index=False)
        logging.info(f"Saved all metrics to results/all_results.csv")
        
        # Create additional visualizations only if not skipping visualization
        # This avoids redundant visualization when called from compare_models
        if not skip_visualization:
            try:
                # Note: These visualizations will be created by compare_models in most cases,
                # but we keep this code as a fallback for when save_summary_metrics is called directly
                logging.info("Creating additional visualizations from metrics data")
                create_model_comparison_matrix(results_df)
                create_performance_comparison_table(results_df)
                create_model_radar_chart(results_df)
                create_best_model_summary(results_df)
            except Exception as e:
                logging.error(f"Error creating visualizations: {e}")
        
        return results_df
    else:
        logging.warning("No metrics to save")
        return None

def load_existing_results():
    """Load existing results from CSV files in the results directory."""
    results_df = None
    wins_df = None
    
    # First check if a specific results file was provided
    if args.results_file and os.path.exists(args.results_file):
        try:
            results_df = pd.read_csv(args.results_file)
            logging.info(f"Loaded {len(results_df)} records from {args.results_file}")
            
            # Make sure we have the required columns
            required_columns = ['model', 'scenario', 'rmse', 'mae', 'mape']
            missing_columns = [col for col in required_columns if col not in results_df.columns]
            
            if missing_columns:
                logging.warning(f"Results file is missing required columns: {', '.join(missing_columns)}")
                
                # Try to rename columns if we can match them
                for col in missing_columns:
                    # Look for similar columns (case insensitive)
                    potential_matches = [c for c in results_df.columns if c.lower() == col.lower()]
                    if potential_matches:
                        results_df = results_df.rename(columns={potential_matches[0]: col})
                        logging.info(f"Renamed column {potential_matches[0]} to {col}")
                        missing_columns.remove(col)
                
                # If we still have missing columns, we can't proceed
                if missing_columns:
                    logging.error(f"Cannot proceed with missing columns: {', '.join(missing_columns)}")
                    results_df = None
        except Exception as e:
            logging.error(f"Error loading specified results file {args.results_file}: {e}")
    
    # If no custom file or it failed, try the default locations
    if results_df is None:
        # Try to load the main results file
        main_results_path = os.path.join(args.output_dir, "all_results.csv")
        if os.path.exists(main_results_path):
            try:
                results_df = pd.read_csv(main_results_path)
                logging.info(f"Loaded {len(results_df)} records from {main_results_path}")
            except Exception as e:
                logging.error(f"Error loading {main_results_path}: {e}")
        
        # If that fails, try alternative files
        if results_df is None or results_df.empty:
            alt_results_path = os.path.join(args.output_dir, "all_model_comparisons.csv")
            if os.path.exists(alt_results_path):
                try:
                    results_df = pd.read_csv(alt_results_path)
                    logging.info(f"Loaded {len(results_df)} records from {alt_results_path}")
                except Exception as e:
                    logging.error(f"Error loading {alt_results_path}: {e}")
    
    # If all direct files fail, try to build results from individual result files
    if results_df is None or results_df.empty:
        logging.info("Trying to build results from individual JSON files...")
        json_pattern = os.path.join(args.output_dir, "*_*.json")
        json_files = glob.glob(json_pattern)
        
        if json_files:
            all_metrics = []
            for json_file in tqdm(json_files, desc="Loading JSON results", colour="blue"):
                try:
                    with open(json_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Parse model and scenario from filename
                    filename = os.path.basename(json_file)
                    parts = filename.split('_')
                    if len(parts) > 2:
                        model = parts[0]
                        # Scenario is everything after model but before timestamp
                        timestamp_pattern = r"_\d{8}_\d{6}"
                        match = re.search(timestamp_pattern, filename)
                        if match:
                            timestamp_start = match.start()
                            scenario = filename[len(model)+1:timestamp_start]
                        else:
                            # If no timestamp, assume scenario is everything up to .json
                            scenario = '_'.join(parts[1:]).replace('.json', '')
                        
                        # Add to metrics list
                        metrics['model'] = model
                        metrics['scenario'] = scenario
                        all_metrics.append(metrics)
                except Exception as e:
                    logging.warning(f"Error processing {json_file}: {e}")
            
            if all_metrics:
                results_df = pd.DataFrame(all_metrics)
                logging.info(f"Built results dataframe with {len(results_df)} records from JSON files")
    
    # Try to load wins data
    wins_path = os.path.join(args.output_dir, "model_wins.csv")
    if os.path.exists(wins_path):
        try:
            wins_df = pd.read_csv(wins_path)
            logging.info(f"Loaded model wins data from {wins_path}")
        except Exception as e:
            logging.error(f"Error loading {wins_path}: {e}")
    
    # If no wins data but we have results, calculate wins
    if (wins_df is None or wins_df.empty) and results_df is not None and not results_df.empty:
        logging.info("Calculating model wins from results data...")
        try:
            # Count how many times each model was the best
            model_wins = {}
            
            for model in results_df['model'].unique():
                model_wins[model] = {'mae': 0, 'rmse': 0, 'mape': 0, 'total': 0}
            
            for scenario in results_df['scenario'].unique():
                scenario_df = results_df[results_df['scenario'] == scenario]
                
                # Skip if missing metrics
                if any(col not in scenario_df.columns for col in ['mae', 'rmse', 'mape']):
                    continue
                    
                # Find best models
                best_mae = scenario_df.loc[scenario_df['mae'].idxmin()]['model']
                best_rmse = scenario_df.loc[scenario_df['rmse'].idxmin()]['model']
                best_mape = scenario_df.loc[scenario_df['mape'].idxmin()]['model']
                
                # Update wins counts
                model_wins[best_mae]['mae'] += 1
                model_wins[best_mae]['total'] += 1
                
                model_wins[best_rmse]['rmse'] += 1
                model_wins[best_rmse]['total'] += 1
                
                model_wins[best_mape]['mape'] += 1
                model_wins[best_mape]['total'] += 1
            
            # Create DataFrame
            wins_rows = []
            for model, wins in model_wins.items():
                wins_rows.append({
                    'model': model,
                    'mae_wins': wins['mae'],
                    'rmse_wins': wins['rmse'],
                    'mape_wins': wins['mape'],
                    'total_wins': wins['total']
                })
            
            wins_df = pd.DataFrame(wins_rows)
            wins_df = wins_df.sort_values('total_wins', ascending=False)
            
            # Save for future use
            wins_output_path = os.path.join(args.output_dir, "model_wins.csv")
            os.makedirs(os.path.dirname(wins_output_path), exist_ok=True)
            wins_df.to_csv(wins_output_path, index=False)
            logging.info(f"Calculated and saved model wins data to {wins_output_path}")
        except Exception as e:
            logging.error(f"Error calculating wins: {e}")
    
    return results_df, wins_df

def setup_visualization_params():
    """Set up parameters for visualizations based on command line arguments"""
    # Set matplotlib figure sizes based on whether we're creating an HTML report
    if args.html_report:
        matplotlib.rcParams['figure.figsize'] = [10, 6]  # Larger figures for HTML report
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['axes.labelsize'] = 12
    else:
        matplotlib.rcParams['figure.figsize'] = [8, 5]
        # Use default font sizes

def get_base_scenario_type(scenario_name):
    """
    Extract the base scenario type from the full scenario name.
    Examples: 
        'stepped_3_min_pods5_max_pods25_step_count4_step_duration48' -> 'stepped'
        'burst_1_base_pods15_burst_frequency0.01_burst_magnitude25_burst_duration24' -> 'burst'
    """
    # Handle the case with or without _arima suffix
    if "_arima" in scenario_name:
        scenario_name = scenario_name.replace("_arima", "")
    
    # Get the base type (everything before the first underscore)
    parts = scenario_name.split('_')
    if len(parts) > 0:
        return parts[0]
    
    return scenario_name

def get_scenario_parameters(scenario_name):
    """
    Extract the parameters from a scenario name.
    Example: 'burst_1_base_pods15_burst_frequency0.01_burst_magnitude25_burst_duration24_burst_shapedecay' 
    -> {'variant': '1', 'base_pods': '15', 'burst_frequency': '0.01', 'burst_magnitude': '25', 
        'burst_duration': '24', 'burst_shape': 'decay'}
    """
    # Remove _arima suffix if present
    clean_name = scenario_name.replace("_arima", "")
    
    # Extract the base type and variant
    parts = clean_name.split('_')
    base_type = parts[0] if len(parts) > 0 else ""
    variant = parts[1] if len(parts) > 1 else ""
    
    # Initialize params with variant
    params = {'variant': variant} if variant.isdigit() else {}
    
    # Process parameter pairs
    current_param = None
    for i in range(2, len(parts)):
        part = parts[i]
        
        # If this part starts with a pattern type prefix, it's a new parameter name
        if part.startswith(base_type + "_") or part == "base":
            # If part starts with base_type, remove the prefix
            if part.startswith(base_type + "_"):
                part = part[len(base_type)+1:]
            
            # Start of a new parameter
            current_param = part
            continue
            
        # If we have a current parameter and this part contains digits
        if current_param and any(c.isdigit() for c in part):
            # Try to extract the value - could be numeric or mixed (like 'decay')
            value = ""
            param_name = current_param
            
            # Handle case where the value is mixed with alpha (like decay, spike)
            digit_part = ''.join(c for c in part if c.isdigit() or c == '.')
            alpha_part = ''.join(c for c in part if c.isalpha())
            
            if digit_part:
                value = digit_part
                
            if alpha_part:
                # If alpha part exists, it might be the value or the next param
                if not value:
                    value = alpha_part
                else:
                    # This is likely the next parameter name
                    params[param_name] = value
                    current_param = alpha_part
                    continue
            
            # Store the parameter and value
            if param_name and value:
                params[param_name] = value
                current_param = None
    
    return params

def group_scenarios_by_type(results_df):
    """
    Group scenarios by their base type.
    Returns a dictionary of dataframes grouped by scenario type.
    """
    # Extract the base scenario type
    results_df['scenario_type'] = results_df['scenario'].apply(get_base_scenario_type)
    
    # Create a dictionary to hold grouped dataframes
    grouped_results = {}
    
    # Group by scenario_type
    for scenario_type, group in results_df.groupby('scenario_type'):
        grouped_results[scenario_type] = group
    
    return grouped_results

def create_performance_comparison_table(results_df, wins_df=None):
    """Create a detailed performance comparison table for all scenarios and models.
    If there are many scenarios, they will be grouped by scenario type for better readability.
    """
    
    # Create directory for detailed comparisons
    os.makedirs("results/detailed_comparisons", exist_ok=True)
    
    # Prepare a DataFrame for output
    comparison_rows = []
    
    # Check if we should group scenarios (more than 15 scenarios)
    num_scenarios = len(results_df['scenario'].unique())
    should_group_scenarios = num_scenarios > 15
    
    if should_group_scenarios:
        logging.info(f"Grouping {num_scenarios} scenarios by type for better readability in the performance table")
        # Group scenarios by type
        grouped_results = group_scenarios_by_type(results_df)
        
        # Process each group first, then each scenario within the group
        for scenario_type, group_df in grouped_results.items():
            # Add a group header
            comparison_rows.append({
                'rank': f"Scenario Group: {scenario_type.upper()}", 
                'model': "", 
                'rmse': "", 
                'mae': "", 
                'mape': "",
                'section': f"group_{scenario_type}"  # Special section marker for groups
            })
            
            # For each scenario in this group, create a section in the table
            for scenario in sorted(group_df['scenario'].unique()):
                scenario_df = group_df[group_df['scenario'] == scenario].copy()
                
                # Sort by RMSE (best first)
                scenario_df = scenario_df.sort_values('rmse')
                
                # Add a ranking column
                scenario_df['rank'] = range(1, len(scenario_df) + 1)
                
                # Add scenario information
                scenario_df['section'] = scenario
                
                # Extract and simplify scenario description for display
                scenario_params = get_scenario_parameters(scenario)
                scenario_display = f"{scenario_type}_{scenario_params.get('variant', '')}"
                
                # Add important parameters if available
                important_params = []
                for param in ['base_pods', 'burst_magnitude', 'chaos_level', 'cascade_count']:
                    if param in scenario_params:
                        param_name = param.replace('_', ' ')
                        important_params.append(f"{param_name}: {scenario_params[param]}")
                
                if important_params:
                    scenario_display += f" ({', '.join(important_params)})"
                
                # Add to our collection
                comparison_rows.append({
                    'rank': f"Scenario: {scenario_display}", 
                    'model': "", 
                    'rmse': "", 
                    'mae': "", 
                    'mape': "",
                    'section': scenario
                })
                
                for _, row in scenario_df.iterrows():
                    comparison_rows.append({
                        'rank': row['rank'],
                        'model': row['model'],
                        'rmse': round(row['rmse'], 3),
                        'mae': round(row['mae'], 3),
                        'mape': round(row['mape'], 3),
                        'section': scenario
                    })
                
                # Add a blank row after each scenario
                comparison_rows.append({
                    'rank': "", 
                    'model': "", 
                    'rmse': "", 
                    'mae': "", 
                    'mape': "",
                    'section': scenario
                })
    else:
        # Original approach - process scenarios without grouping
        for scenario in sorted(results_df['scenario'].unique()):
            scenario_df = results_df[results_df['scenario'] == scenario].copy()
            
            # Sort by RMSE (best first)
            scenario_df = scenario_df.sort_values('rmse')
            
            # Add a ranking column
            scenario_df['rank'] = range(1, len(scenario_df) + 1)
            
            # Add scenario information
            scenario_df['section'] = scenario
            
            # Add to our collection
            comparison_rows.append({
                'rank': f"Scenario: {scenario}", 
                'model': "", 
                'rmse': "", 
                'mae': "", 
                'mape': "",
                'section': scenario
            })
            
            for _, row in scenario_df.iterrows():
                comparison_rows.append({
                    'rank': row['rank'],
                    'model': row['model'],
                    'rmse': round(row['rmse'], 3),
                    'mae': round(row['mae'], 3),
                    'mape': round(row['mape'], 3),
                    'section': scenario
                })
            
            # Add a blank row after each scenario
            comparison_rows.append({
                'rank': "", 
                'model': "", 
                'rmse': "", 
                'mae': "", 
                'mape': "",
                'section': scenario
            })
    
    # Convert to DataFrame
    styled_df = pd.DataFrame(comparison_rows)
    
    # Save as CSV
    styled_df.to_csv("results/detailed_comparisons/all_models_comparison.csv", index=False)
    
    # Determine if we need pagination (more than 20 scenarios or 200 rows total)
    need_pagination = len(styled_df) > 200 or len(results_df['scenario'].unique()) > 20
    
    # Create an HTML version with styling
    try:
        # Add CSS with responsive design and better handling of large tables
        css = """
        <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        th, td { 
            padding: 6px; 
            text-align: left; 
            border-bottom: 1px solid #ddd; 
        }
        th { 
            background-color: #f2f2f2; 
            position: sticky;
            top: 0;
        }
        tr:hover { background-color: #f5f5f5; }
        .section-header { 
            background-color: #4CAF50; 
            color: white; 
            font-weight: bold; 
            font-size: 1.1em;
            padding: 10px;
        }
        .group-header {
            background-color: #326fa8; 
            color: white; 
            font-weight: bold; 
            font-size: 1.2em;
            padding: 12px;
        }
        .scenario-header { 
            background-color: #e6f2ff; 
            font-weight: bold; 
        }
        .best-model { 
            background-color: #d4edda; 
        }
        .second-best { 
            background-color: #fff3cd; 
        }
        .metrics {
            text-align: right;
            font-family: monospace;
        }
        .search-box {
            padding: 10px;
            margin-bottom: 20px;
            width: 100%;
            box-sizing: border-box;
            font-size: 16px;
        }
        .pagination {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .pagination button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 4px;
        }
        .page-info {
            margin: 10px 0;
            text-align: center;
        }
        .group-toggle {
            cursor: pointer;
            user-select: none;
        }
        .group-content {
            display: block;
        }
        .collapsed .group-content {
            display: none;
        }
        @media print {
            .no-print {
                display: none;
            }
            body {
                padding: 0;
                font-size: 10pt;
            }
            table {
                font-size: 9pt;
            }
        }
        </style>
        """
        
        # Add JavaScript for filtering, pagination, and group collapsing
        javascript = """
        <script>
        function filterTable() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toUpperCase();
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            let visibleRows = 0;
            
            for (let i = 0; i < rows.length; i++) {
                const row = rows[i];
                const cells = row.getElementsByTagName('td');
                
                if (cells.length > 0) {
                    let txtValue = '';
                    
                    // Concat all cell values
                    for (let j = 0; j < cells.length; j++) {
                        txtValue += cells[j].textContent || cells[j].innerText;
                    }
                    
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        row.style.display = '';
                        visibleRows++;
                    } else {
                        row.style.display = 'none';
                    }
                }
            }
            
            document.getElementById('visibleRowCount').textContent = visibleRows;
        }
        
        // Group toggling functionality
        function toggleGroup(element) {
            const groupRow = element.closest('tr');
            const groupName = groupRow.getAttribute('data-group');
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            
            // Find all rows belonging to this group
            const groupRows = [];
            let inGroup = false;
            for (let i = 0; i < rows.length; i++) {
                const row = rows[i];
                if (row === groupRow) {
                    inGroup = true;
                    continue;
                }
                if (inGroup && row.hasAttribute('data-group')) {
                    inGroup = false;
                    break;
                }
                if (inGroup) {
                    groupRows.push(row);
                }
            }
            
            // Toggle visibility of group rows
            if (groupRow.classList.contains('collapsed')) {
                groupRow.classList.remove('collapsed');
                groupRows.forEach(row => row.style.display = '');
            } else {
                groupRow.classList.add('collapsed');
                groupRows.forEach(row => row.style.display = 'none');
            }
        }
        
        // Pagination variables and functions
        let currentPage = 1;
        const rowsPerPage = 50;
        
        function showPage(page) {
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            const totalRows = rows.length;
            const totalPages = Math.ceil((totalRows - 1) / rowsPerPage); // Subtract header row
            
            // Update current page
            currentPage = page;
            
            // Show correct page
            let rowCount = 0;
            for (let i = 1; i < totalRows; i++) { // Skip header row
                if (i > (page-1) * rowsPerPage && i <= page * rowsPerPage) {
                    rows[i].style.display = '';
                    rowCount++;
                } else {
                    rows[i].style.display = 'none';
                }
            }
            
            // Update page info
            document.getElementById('currentPage').textContent = page;
            document.getElementById('totalPages').textContent = totalPages;
            
            // Update button states
            document.getElementById('prevBtn').disabled = page === 1;
            document.getElementById('nextBtn').disabled = page === totalPages;
        }
        
        function nextPage() {
            const table = document.getElementById('comparisonTable');
            const rows = table.getElementsByTagName('tr');
            const totalRows = rows.length;
            const totalPages = Math.ceil((totalRows - 1) / rowsPerPage);
            
            if (currentPage < totalPages) {
                showPage(currentPage + 1);
            }
        }
        
        function prevPage() {
            if (currentPage > 1) {
                showPage(currentPage - 1);
            }
        }
        
        // Toggle all groups
        function toggleAllGroups(expand) {
            const table = document.getElementById('comparisonTable');
            const groupRows = table.querySelectorAll('tr[data-group]');
            
            groupRows.forEach(row => {
                const isCollapsed = row.classList.contains('collapsed');
                if (expand && isCollapsed) {
                    toggleGroup({closest: function() { return row; }});
                } else if (!expand && !isCollapsed) {
                    toggleGroup({closest: function() { return row; }});
                }
            });
        }
        
        // Initialize when document is loaded
        document.addEventListener('DOMContentLoaded', function() {
            if (document.getElementById('paginationEnabled')) {
                showPage(1);
            }
        });
        </script>
        """
        
        # Build HTML table content
        html_rows = []
        current_section = None
        current_group = None
        
        for _, row in styled_df.iterrows():
            # Check if this is a group header (groups start with "group_")
            if isinstance(row['section'], str) and row['section'].startswith('group_'):
                current_group = row['section'].replace('group_', '')
                group_header = row['rank']
                html_rows.append(f'<tr class="group-header" data-group="{current_group}" onclick="toggleGroup(this)"><td colspan="5">{group_header} <span class="group-toggle">[−]</span></td></tr>')
                continue
                
            if current_section != row['section']:
                current_section = row['section']
                html_rows.append(f'<tr class="section-header"><td colspan="5">Scenario: {current_section}</td></tr>')
                html_rows.append('<tr><th>Rank</th><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th></tr>')
            
            # Style based on rank
            if isinstance(row['rank'], int) and row['rank'] == 1:
                row_class = 'class="best-model"'
            elif isinstance(row['rank'], int) and row['rank'] == 2:
                row_class = 'class="second-best"'
            elif isinstance(row['rank'], str) and "Scenario" in str(row['rank']):
                row_class = 'class="scenario-header"'
            else:
                row_class = ''
            
            # Add the row
            html_rows.append(f'''
            <tr {row_class}>
                <td>{row['rank']}</td>
                <td>{row['model']}</td>
                <td class="metrics">{row['rmse']}</td>
                <td class="metrics">{row['mae']}</td>
                <td class="metrics">{row['mape']}</td>
            </tr>
            ''')
        
        # Build the HTML document
        pagination_controls = ""
        filter_box = ""
        group_controls = ""
        
        if should_group_scenarios:
            group_controls = """
            <div class="group-controls no-print">
                <button onclick="toggleAllGroups(true)">Expand All Groups</button>
                <button onclick="toggleAllGroups(false)">Collapse All Groups</button>
            </div>
            """
        
        if need_pagination:
            pagination_controls = """
            <div id="paginationEnabled" class="no-print"></div>
            <div class="pagination no-print">
                <button id="prevBtn" onclick="prevPage()">Previous</button>
                <div class="page-info">
                    Page <span id="currentPage">1</span> of <span id="totalPages">1</span>
                </div>
                <button id="nextBtn" onclick="nextPage()">Next</button>
            </div>
            """
            
            filter_box = """
            <div class="no-print">
                <input type="text" id="searchInput" class="search-box" onkeyup="filterTable()" placeholder="Search for models, scenarios...">
                <div class="page-info">Showing <span id="visibleRowCount">0</span> rows</div>
            </div>
            """
        
        html_table = f"""<table id="comparisonTable">
        {''.join(html_rows)}
        </table>"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_html = f"""
        <html>
        <head>
            <title>Model Performance Comparison</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            {css}
            {javascript}
        </head>
        <body>
            <h1>Model Performance Comparison</h1>
            <p>Generated on {timestamp}</p>
            {group_controls}
            {filter_box}
            {pagination_controls}
            {html_table}
            {pagination_controls}
        </body>
        </html>"""
        
        with open("results/detailed_comparisons/all_models_comparison.html", "w") as f:
            f.write(full_html)
        
        logging.info("Detailed performance comparison saved to results/detailed_comparisons/all_models_comparison.html")
    except Exception as e:
        logging.error(f"Error creating HTML comparison: {e}")
        # Continue execution even if HTML creation fails
    
    logging.info("Detailed performance comparison table saved to results/detailed_comparisons/all_models_comparison.csv")

def create_model_comparison_matrix(results_df):
    """Create a matrix of pairwise differences between models."""
    # Get unique models
    models = results_df['model'].unique()
    
    # Create matrices to store average percentage improvements
    rmse_improvement = pd.DataFrame(index=models, columns=models, data=0.0)
    mae_improvement = pd.DataFrame(index=models, columns=models, data=0.0)
    mape_improvement = pd.DataFrame(index=models, columns=models, data=0.0)
    
    # For each scenario, calculate pairwise improvements
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        
        for model1 in models:
            for model2 in models:
                if model1 != model2:
                    # Get metrics for each model
                    metrics1 = scenario_data[scenario_data['model'] == model1]
                    metrics2 = scenario_data[scenario_data['model'] == model2]
                    
                    if not metrics1.empty and not metrics2.empty:
                        # Calculate percentage improvements (positive means model1 is better than model2)
                        rmse_diff = (metrics2['rmse'].values[0] - metrics1['rmse'].values[0]) / metrics2['rmse'].values[0] * 100
                        mae_diff = (metrics2['mae'].values[0] - metrics1['mae'].values[0]) / metrics2['mae'].values[0] * 100
                        mape_diff = (metrics2['mape'].values[0] - metrics1['mape'].values[0]) / metrics2['mape'].values[0] * 100
                        
                        # Add to the cumulative values
                        rmse_improvement.loc[model1, model2] += rmse_diff
                        mae_improvement.loc[model1, model2] += mae_diff
                        mape_improvement.loc[model1, model2] += mape_diff
    
    # Calculate average across all scenarios
    n_scenarios = len(results_df['scenario'].unique())
    rmse_improvement = rmse_improvement / n_scenarios
    mae_improvement = mae_improvement / n_scenarios
    mape_improvement = mape_improvement / n_scenarios
    
    # Create visualizations of the improvement matrices
    plt.figure(figsize=(14, 10))
    sns.heatmap(rmse_improvement, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=0.5)
    plt.title('Average RMSE Improvement (%) - Row Model vs Column Model')
    plt.tight_layout()
    plt.savefig('results/plots/rmse_improvement_matrix.png')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(mae_improvement, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=0.5)
    plt.title('Average MAE Improvement (%) - Row Model vs Column Model')
    plt.tight_layout()
    plt.savefig('results/plots/mae_improvement_matrix.png')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(mape_improvement, annot=True, cmap='RdYlGn', center=0, fmt='.1f', linewidths=0.5)
    plt.title('Average MAPE Improvement (%) - Row Model vs Column Model')
    plt.tight_layout()
    plt.savefig('results/plots/mape_improvement_matrix.png')
    
    # Create a summary of overall model ranking
    average_improvement = (rmse_improvement + mae_improvement + mape_improvement) / 3
    model_scores = average_improvement.mean(axis=1).sort_values(ascending=False)
    
    # Plot overall model ranking
    plt.figure(figsize=(12, 8))
    model_scores.plot(kind='bar', color='skyblue')
    plt.title('Average Performance Improvement Across All Metrics and Models')
    plt.xlabel('Model')
    plt.ylabel('Average Improvement %')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/model_overall_ranking.png')
    
    return rmse_improvement, mae_improvement, mape_improvement

def create_best_model_plots(results_df):
    """Create clean plots showing only the best model for each scenario."""
    # Create directory for best model plots
    os.makedirs("results/best_model_plots", exist_ok=True)
    
    # Check if we have many scenarios and need to create a directory for grouped plots
    num_scenarios = len(results_df['scenario'].unique())
    
    if num_scenarios > 20:
        os.makedirs("results/best_model_plots/groups", exist_ok=True)
        logging.info(f"Creating grouped best model plots due to large number of scenarios ({num_scenarios})")
    
    # Keep track of scenarios already processed for summary plot
    processed_scenarios = []
    
    # Loop through each scenario (these should be normalized names now)
    for idx, scenario in enumerate(results_df['scenario'].unique()):
        scenario_df = results_df[results_df['scenario'] == scenario]
        
        # Find the best model based on RMSE (you can change this to MAE or MAPE if preferred)
        best_model = scenario_df.loc[scenario_df['rmse'].idxmin()]['model']
        
        logging.info(f"Creating plot for {scenario} with best model: {best_model}")
        
        # Define data paths using the scenario name directly
        train_data_path = f"{scenarios_path}/{scenario}_train.csv"
        test_data_path = f"{scenarios_path}/{scenario}_test.csv"
        
        if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
            logging.warning(f"Data files for {scenario} not found at {train_data_path}. Skipping best model plot.")
            continue
        
        # Load data - we only need the test data for the plot
        try:
            test_data = pd.read_csv(test_data_path)
            if 'timestamp' in test_data.columns:
                test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
            else:
                logging.warning(f"No timestamp column in test data for {scenario}. Skipping best model plot.")
                continue
            # Check if 'pod_count' column exists, otherwise try 'y'
            if 'pod_count' not in test_data.columns:
                 if 'y' in test_data.columns:
                      test_data = test_data.rename(columns={'y': 'pod_count'})
                 else:
                      logging.warning(f"Neither 'pod_count' nor 'y' column found in test data for {scenario}. Skipping.")
                      continue

        except Exception as e:
            logging.error(f"Error loading test data for {scenario}: {e}")
            continue
        
        # Get model predictions
        # Results folder uses normalized name
        model_predictions_path = f"results/{scenario}/{best_model}_predictions.csv"
        if not os.path.exists(model_predictions_path):
            logging.warning(f"Predictions for {best_model} in {scenario} not found at {model_predictions_path}. Trying model run directory...")
            
            # Try to find predictions in model run directory using normalized scenario name
            run_id = f"{best_model}_{scenario}"
            model_run_predictions = f"train/models/{best_model}_model/runs/{run_id}/predictions.csv"
            
            if os.path.exists(model_run_predictions):
                try:
                    # Copy predictions to the expected location
                    os.makedirs(f"results/{scenario}", exist_ok=True)
                    predictions_df = pd.read_csv(model_run_predictions)
                    
                    # Ensure the predictions DataFrame has the expected columns
                    if "actual" in predictions_df.columns and "predicted" in predictions_df.columns:
                        # Rename columns if needed
                        predictions_df = predictions_df.rename(columns={
                            "predicted": "prediction"
                        })
                    
                    predictions_df.to_csv(model_predictions_path, index=False)
                    logging.info(f"Copied predictions from {model_run_predictions} to {model_predictions_path}")
                except Exception as e:
                    logging.error(f"Error copying predictions file: {e}")
                    continue
            else:
                logging.error(f"No predictions found for {best_model} in {scenario} in model run directory: {model_run_predictions}")
                continue
        
        try:
            model_predictions = pd.read_csv(model_predictions_path)
            if 'timestamp' in model_predictions.columns:
                model_predictions['timestamp'] = pd.to_datetime(model_predictions['timestamp'])
            else:
                logging.warning(f"No timestamp column in predictions for {scenario}. Skipping best model plot.")
                continue
                
            logging.info(f"Loaded predictions from {model_predictions_path}, shape: {model_predictions.shape}")
        except Exception as e:
            logging.error(f"Error reading predictions file {model_predictions_path}: {e}")
            continue
        
        # Create the plot with a clean professional look
        try:
            # Determine if we should handle a large time series (downsample for performance)
            time_points = len(model_predictions)
            downsample = time_points > 500  # Only downsample very large series
            
            # Create a modern, professional plot
            plt.style.use('ggplot')  # Modern style
            fig, ax = plt.subplots(figsize=(15, 8), dpi=300)  # Increased DPI for higher quality
            
            # Apply downsampling if needed
            if downsample:
                logging.info(f"Downsampling time series for {scenario} ({time_points} points)")
                # Calculate appropriate step size
                step = max(1, time_points // 500)
                
                # Plot only test data (actual values) - use blue color
                ax.plot(test_data['timestamp'].iloc[::step], test_data['pod_count'].iloc[::step], 
                        color='#1f77b4', linewidth=2.5, label='Actual Values')
                
                # Plot the best model predictions with orange color
                ax.plot(model_predictions['timestamp'].iloc[::step], model_predictions['prediction'].iloc[::step], 
                        color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'Best Model: {best_model}')
            else:
                # Plot original data without downsampling
                ax.plot(test_data['timestamp'], test_data['pod_count'], 
                        color='#1f77b4', linewidth=2.5, label='Actual Values')
                
                # Plot the best model predictions with orange color
                ax.plot(model_predictions['timestamp'], model_predictions['prediction'], 
                        color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'Best Model: {best_model}')
            
            # Add a legend in the top left corner with better font size
            legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
            
            # Add scenario name as title
            ax.set_title(f"Scenario: {scenario}", fontsize=14, pad=10)
            
            # Format axes better
            # Format date axis when we have a reasonable number of data points
            if not downsample and time_points < 1000:
                # Format date axis nicely
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45, ha='right', fontsize=10)
                ax.set_xlabel("Date", fontsize=12)
            else:
                # For very large datasets or downsampled ones, simplify the x-axis
                ax.set_xlabel("")
                ax.get_xaxis().set_visible(False)
            
            ax.set_ylabel("Value", fontsize=12)
            
            # Keep border (spines) visible for a "main border" appearance
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            
            # Add metrics in a text box
            metrics = scenario_df[scenario_df['model'] == best_model].iloc[0]
            metrics_text = (
                f"RMSE: {metrics['rmse']:.3f}\n"
                f"MAE: {metrics['mae']:.3f}\n"
                f"MAPE: {metrics['mape']:.3f}%"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Add grid but make it subtle
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Save the figure with high resolution - use normalized name for consistent filenames
            normalized_scenario = normalize_scenario_name(scenario)
            plt.tight_layout()
            
            # Main plot saved to the root directory
            plot_path = f"results/best_model_plots/{normalized_scenario}_best_model.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # If we have many scenarios, also save to groups directory
            if num_scenarios > 20:
                group_idx = idx // 20 + 1  # Group number
                group_path = f"results/best_model_plots/groups/group{group_idx}_{normalized_scenario}_best_model.png"
                plt.savefig(group_path, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            # Add to processed scenarios list
            processed_scenarios.append({
                'scenario': scenario,
                'best_model': best_model,
                'rmse': metrics['rmse']
            })
            
            logging.info(f"Best model plot for {scenario} saved to {plot_path}")
        except Exception as e:
            logging.error(f"Error creating plot for {scenario}: {e}")
            continue
            
    # Create a summary grid of thumbnails if we have many scenarios
    if num_scenarios > 8 and len(processed_scenarios) > 8:
        try:
            create_best_model_grid(processed_scenarios, results_df)
        except Exception as e:
            logging.error(f"Error creating best model grid: {e}")
            
def create_best_model_grid(processed_scenarios, results_df):
    """Create a grid of thumbnails for the best model plots."""
    logging.info("Creating a summary grid of best model thumbnails")
    
    # Sort scenarios by RMSE (best first)
    sorted_scenarios = sorted(processed_scenarios, key=lambda x: x['rmse'])
    
    # Take the top 9 scenarios (or fewer if we have less)
    top_scenarios = sorted_scenarios[:min(9, len(sorted_scenarios))]
    
    # Calculate grid dimensions
    grid_size = len(top_scenarios)
    cols = min(3, grid_size)
    rows = (grid_size + cols - 1) // cols
    
    # Create figure for the grid
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=150)
    
    # Flatten axes for easier indexing
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1 and cols > 1:
        axes = axes.reshape(-1)
    elif rows > 1 and cols == 1:
        axes = axes.reshape(-1)
    else:
        axes = [axes]  # Single axis, make it a list
        
    # Add each thumbnail to the grid
    for i, scenario_info in enumerate(top_scenarios):
        if i >= len(axes):
            break  # Safety check
            
        scenario = scenario_info['scenario']
        best_model = scenario_info['best_model']
        normalized_scenario = normalize_scenario_name(scenario)
        
        # Load the image
        img_path = f"results/best_model_plots/{normalized_scenario}_best_model.png"
        
        if os.path.exists(img_path):
            try:
                img = plt.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{scenario}\nBest: {best_model}")
                axes[i].axis('off')
            except Exception as e:
                logging.error(f"Error adding image to grid: {e}")
                axes[i].text(0.5, 0.5, "Image load error", ha='center', va='center')
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, "Image not found", ha='center', va='center')
            axes[i].axis('off')
    
    # Hide any unused axes
    for i in range(len(top_scenarios), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/best_model_plots/top_models_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Best model grid saved to results/best_model_plots/top_models_grid.png")

def save_summary_metrics(all_results, skip_visualization=False):
    """Save all model results to a CSV file for analysis."""
    
    # Create a list to store all metrics
    all_metrics_list = []
    
    # Iterate through scenarios and models to collect metrics
    for scenario, models in all_results.items():
        # Normalize the scenario name
        scenario_name = normalize_scenario_name(scenario)
        
        for model_name, metrics in models.items():
            # Add scenario and model to metrics dict
            metrics_row = {
                "scenario": scenario_name,
                "model": model_name,
                "mae": metrics.get("mae", None),
                "rmse": metrics.get("rmse", None),
                "mape": metrics.get("mape", None)
            }
            
            # Add any other metrics that might be present
            for key, value in metrics.items():
                if key not in ["mae", "rmse", "mape"]:
                    metrics_row[key] = value
            
            all_metrics_list.append(metrics_row)
    
    # Create a DataFrame with all metrics
    if all_metrics_list:
        results_df = pd.DataFrame(all_metrics_list)
        
        # Save the metrics to CSV
        results_df.to_csv("results/all_results.csv", index=False)
        logging.info(f"Saved all metrics to results/all_results.csv")
        
        # Create additional visualizations only if not skipping visualization
        # This avoids redundant visualization when called from compare_models
        if not skip_visualization:
            try:
                # Note: These visualizations will be created by compare_models in most cases,
                # but we keep this code as a fallback for when save_summary_metrics is called directly
                logging.info("Creating additional visualizations from metrics data")
                create_model_comparison_matrix(results_df)
                create_performance_comparison_table(results_df)
                create_model_radar_chart(results_df)
                create_best_model_summary(results_df)
            except Exception as e:
                logging.error(f"Error creating visualizations: {e}")
        
        return results_df
    else:
        logging.warning("No metrics to save")
        return None

def load_existing_results():
    """Load existing results from CSV files in the results directory."""
    results_df = None
    wins_df = None
    
    # First check if a specific results file was provided
    if args.results_file and os.path.exists(args.results_file):
        try:
            results_df = pd.read_csv(args.results_file)
            logging.info(f"Loaded {len(results_df)} records from {args.results_file}")
            
            # Make sure we have the required columns
            required_columns = ['model', 'scenario', 'rmse', 'mae', 'mape']
            missing_columns = [col for col in required_columns if col not in results_df.columns]
            
            if missing_columns:
                logging.warning(f"Results file is missing required columns: {', '.join(missing_columns)}")
                
                # Try to rename columns if we can match them
                for col in missing_columns:
                    # Look for similar columns (case insensitive)
                    potential_matches = [c for c in results_df.columns if c.lower() == col.lower()]
                    if potential_matches:
                        results_df = results_df.rename(columns={potential_matches[0]: col})
                        logging.info(f"Renamed column {potential_matches[0]} to {col}")
                        missing_columns.remove(col)
                
                # If we still have missing columns, we can't proceed
                if missing_columns:
                    logging.error(f"Cannot proceed with missing columns: {', '.join(missing_columns)}")
                    results_df = None
        except Exception as e:
            logging.error(f"Error loading specified results file {args.results_file}: {e}")
    
    # If no custom file or it failed, try the default locations
    if results_df is None:
        # Try to load the main results file
        main_results_path = os.path.join(args.output_dir, "all_results.csv")
        if os.path.exists(main_results_path):
            try:
                results_df = pd.read_csv(main_results_path)
                logging.info(f"Loaded {len(results_df)} records from {main_results_path}")
            except Exception as e:
                logging.error(f"Error loading {main_results_path}: {e}")
        
        # If that fails, try alternative files
        if results_df is None or results_df.empty:
            alt_results_path = os.path.join(args.output_dir, "all_model_comparisons.csv")
            if os.path.exists(alt_results_path):
                try:
                    results_df = pd.read_csv(alt_results_path)
                    logging.info(f"Loaded {len(results_df)} records from {alt_results_path}")
                except Exception as e:
                    logging.error(f"Error loading {alt_results_path}: {e}")
    
    # If all direct files fail, try to build results from individual result files
    if results_df is None or results_df.empty:
        logging.info("Trying to build results from individual JSON files...")
        json_pattern = os.path.join(args.output_dir, "*_*.json")
        json_files = glob.glob(json_pattern)
        
        if json_files:
            all_metrics = []
            for json_file in tqdm(json_files, desc="Loading JSON results", colour="blue"):
                try:
                    with open(json_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Parse model and scenario from filename
                    filename = os.path.basename(json_file)
                    parts = filename.split('_')
                    if len(parts) > 2:
                        model = parts[0]
                        # Scenario is everything after model but before timestamp
                        timestamp_pattern = r"_\d{8}_\d{6}"
                        match = re.search(timestamp_pattern, filename)
                        if match:
                            timestamp_start = match.start()
                            scenario = filename[len(model)+1:timestamp_start]
                        else:
                            # If no timestamp, assume scenario is everything up to .json
                            scenario = '_'.join(parts[1:]).replace('.json', '')
                        
                        # Add to metrics list
                        metrics['model'] = model
                        metrics['scenario'] = scenario
                        all_metrics.append(metrics)
                except Exception as e:
                    logging.warning(f"Error processing {json_file}: {e}")
            
            if all_metrics:
                results_df = pd.DataFrame(all_metrics)
                logging.info(f"Built results dataframe with {len(results_df)} records from JSON files")
    
    # Try to load wins data
    wins_path = os.path.join(args.output_dir, "model_wins.csv")
    if os.path.exists(wins_path):
        try:
            wins_df = pd.read_csv(wins_path)
            logging.info(f"Loaded model wins data from {wins_path}")
        except Exception as e:
            logging.error(f"Error loading {wins_path}: {e}")
    
    # If no wins data but we have results, calculate wins
    if (wins_df is None or wins_df.empty) and results_df is not None and not results_df.empty:
        logging.info("Calculating model wins from results data...")
        try:
            # Count how many times each model was the best
            model_wins = {}
            
            for model in results_df['model'].unique():
                model_wins[model] = {'mae': 0, 'rmse': 0, 'mape': 0, 'total': 0}
            
            for scenario in results_df['scenario'].unique():
                scenario_df = results_df[results_df['scenario'] == scenario]
                
                # Skip if missing metrics
                if any(col not in scenario_df.columns for col in ['mae', 'rmse', 'mape']):
                    continue
                    
                # Find best models
                best_mae = scenario_df.loc[scenario_df['mae'].idxmin()]['model']
                best_rmse = scenario_df.loc[scenario_df['rmse'].idxmin()]['model']
                best_mape = scenario_df.loc[scenario_df['mape'].idxmin()]['model']
                
                # Update wins counts
                model_wins[best_mae]['mae'] += 1
                model_wins[best_mae]['total'] += 1
                
                model_wins[best_rmse]['rmse'] += 1
                model_wins[best_rmse]['total'] += 1
                
                model_wins[best_mape]['mape'] += 1
                model_wins[best_mape]['total'] += 1
            
            # Create DataFrame
            wins_rows = []
            for model, wins in model_wins.items():
                wins_rows.append({
                    'model': model,
                    'mae_wins': wins['mae'],
                    'rmse_wins': wins['rmse'],
                    'mape_wins': wins['mape'],
                    'total_wins': wins['total']
                })
            
            wins_df = pd.DataFrame(wins_rows)
            wins_df = wins_df.sort_values('total_wins', ascending=False)
            
            # Save for future use
            wins_output_path = os.path.join(args.output_dir, "model_wins.csv")
            os.makedirs(os.path.dirname(wins_output_path), exist_ok=True)
            wins_df.to_csv(wins_output_path, index=False)
            logging.info(f"Calculated and saved model wins data to {wins_output_path}")
        except Exception as e:
            logging.error(f"Error calculating wins: {e}")
    
    return results_df, wins_df

def setup_visualization_params():
    """Set up parameters for visualizations based on command line arguments"""
    # Set matplotlib figure sizes based on whether we're creating an HTML report
    if args.html_report:
        matplotlib.rcParams['figure.figsize'] = [10, 6]  # Larger figures for HTML report
        matplotlib.rcParams['axes.titlesize'] = 14
        matplotlib.rcParams['axes.labelsize'] = 12
    else:
        matplotlib.rcParams['figure.figsize'] = [8, 5]
        # Use default font sizes

def get_base_scenario_type(scenario_name):
    """
    Extract the base scenario type from the full scenario name.
    Examples: 
        'stepped_3_min_pods5_max_pods25_step_count4_step_duration48' -> 'stepped'
        'burst_1_base_pods15_burst_frequency0.01_burst_magnitude25_burst_duration24' -> 'burst'
    """
    # Handle the case with or without _arima suffix
    if "_arima" in scenario_name:
        scenario_name = scenario_name.replace("_arima", "")
    
    # Get the base type (everything before the first underscore)
    parts = scenario_name.split('_')
    if len(parts) > 0:
        return parts[0]
    
    return scenario_name

def get_scenario_parameters(scenario_name):
    """
    Extract the parameters from a scenario name.
    Example: 'burst_1_base_pods15_burst_frequency0.01_burst_magnitude25_burst_duration24_burst_shapedecay' 
    -> {'variant': '1', 'base_pods': '15', 'burst_frequency': '0.01', 'burst_magnitude': '25', 
        'burst_duration': '24', 'burst_shape': 'decay'}
    """
    # Remove _arima suffix if present
    clean_name = scenario_name.replace("_arima", "")
    
    # Extract the base type and variant
    parts = clean_name.split('_')
    base_type = parts[0] if len(parts) > 0 else ""
    variant = parts[1] if len(parts) > 1 else ""
    
    # Initialize params with variant
    params = {'variant': variant} if variant.isdigit() else {}
    
    # Process parameter pairs
    current_param = None
    for i in range(2, len(parts)):
        part = parts[i]
        
        # If this part starts with a pattern type prefix, it's a new parameter name
        if part.startswith(base_type + "_") or part == "base":
            # If part starts with base_type, remove the prefix
            if part.startswith(base_type + "_"):
                part = part[len(base_type)+1:]
            
            # Start of a new parameter
            current_param = part
            continue
            
        # If we have a current parameter and this part contains digits
        if current_param and any(c.isdigit() for c in part):
            # Try to extract the value - could be numeric or mixed (like 'decay')
            value = ""
            param_name = current_param
            
            # Handle case where the value is mixed with alpha (like decay, spike)
            digit_part = ''.join(c for c in part if c.isdigit() or c == '.')
            alpha_part = ''.join(c for c in part if c.isalpha())
            
            if digit_part:
                value = digit_part
                
            if alpha_part:
                # If alpha part exists, it might be the value or the next param
                if not value:
                    value = alpha_part
                else:
                    # This is likely the next parameter name
                    params[param_name] = value
                    current_param = alpha_part
                    continue
            
            # Store the parameter and value
            if param_name and value:
                params[param_name] = value
                current_param = None
    
    return params

def group_scenarios_by_type(results_df):
    """
    Group scenarios by their base type.
    Returns a dictionary of dataframes grouped by scenario type.
    """
    # Extract the base scenario type
    results_df['scenario_type'] = results_df['scenario'].apply(get_base_scenario_type)
    
    # Create a dictionary to hold grouped dataframes
    grouped_results = {}
    
    # Group by scenario_type
    for scenario_type, group in results_df.groupby('scenario_type'):
        grouped_results[scenario_type] = group
    
    return grouped_results

def create_scenario_type_comparison(results_df):
    """
    Create visualizations comparing performance across different scenario types.
    """
    logging.info("Creating scenario type comparison visualizations...")
    
    # Make sure output directory exists
    os.makedirs(f"{args.output_dir}/scenario_comparisons", exist_ok=True)
    
    # Group by scenario type
    grouped_results = group_scenarios_by_type(results_df)
    
    # Create a dataframe with average performance by scenario type and model
    avg_by_type = []
    
    for scenario_type, group in grouped_results.items():
        # Calculate average metrics by model for this scenario type
        for model, model_data in group.groupby('model'):
            avg_by_type.append({
                'scenario_type': scenario_type,
                'model': model,
                'rmse_avg': model_data['rmse'].mean(),
                'mae_avg': model_data['mae'].mean(),
                'mape_avg': model_data['mape'].mean(),
                'training_time_avg': model_data['training_time'].mean(),
                'inference_time_avg': model_data['inference_time'].mean(),
                'variant_count': len(model_data)
            })
    
    avg_by_type_df = pd.DataFrame(avg_by_type)
    
    # Save to CSV
    avg_by_type_df.to_csv(f"{args.output_dir}/scenario_comparisons/avg_by_scenario_type.csv", index=False)
    
    # Create heatmaps for each metric
    metrics = [
        ('rmse_avg', 'Average RMSE by Scenario Type'),
        ('mae_avg', 'Average MAE by Scenario Type'),
        ('mape_avg', 'Average MAPE by Scenario Type')
    ]
    
    # Calculate best model for each scenario type
    best_models = avg_by_type_df.loc[avg_by_type_df.groupby('scenario_type')['rmse_avg'].idxmin()]
    best_models = best_models[['scenario_type', 'model', 'rmse_avg']]
    best_models.columns = ['scenario_type', 'best_model', 'best_rmse']
    best_models.to_csv(f"{args.output_dir}/scenario_comparisons/best_models_by_type.csv", index=False)
    
    # Create comparison charts
    for metric, title in metrics:
        plt.figure(figsize=FIGURE_SIZES['heatmap'])
        
        # Create pivot table
        pivot_data = avg_by_type_df.pivot(index='model', columns='scenario_type', values=metric)
        
        # Generate heatmap
        ax = sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title(title, fontsize=FONT_SIZES['title'])
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{args.output_dir}/scenario_comparisons/{metric}_heatmap.png", dpi=args.dpi, bbox_inches='tight')
        plt.close()
    
    # Create summary bar chart of best models by scenario type
    plt.figure(figsize=FIGURE_SIZES['bar_chart'])
    
    # Count best models for each scenario type
    model_counts = best_models['best_model'].value_counts()
    ax = sns.barplot(x=model_counts.index, y=model_counts.values)
    
    # Add value labels on top of bars
    for i, v in enumerate(model_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.title('Best Performing Models by Scenario Type Count', fontsize=FONT_SIZES['title'])
    plt.ylabel('Count', fontsize=FONT_SIZES['axis_label'])
    plt.xlabel('Model', fontsize=FONT_SIZES['axis_label'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{args.output_dir}/scenario_comparisons/best_models_count.png", dpi=args.dpi, bbox_inches='tight')
    plt.close()
    
    return grouped_results, avg_by_type_df

def create_parameter_impact_analysis(results_df):
    """
    Analyze how different parameters affect model performance within each scenario type.
    """
    logging.info("Creating parameter impact analysis...")
    
    # Make sure output directory exists
    os.makedirs(f"{args.output_dir}/parameter_analysis", exist_ok=True)
    
    # Group scenarios by type
    grouped_results = group_scenarios_by_type(results_df)
    
    parameter_impacts = {}
    
    # Analyze each scenario type separately
    for scenario_type, group in grouped_results.items():
        # Extract parameters for each scenario in this group
        scenarios_with_params = []
        
        for scenario in group['scenario'].unique():
            params = get_scenario_parameters(scenario)
            # Add scenario name
            params['scenario'] = scenario
            scenarios_with_params.append(params)
        
        # Convert to DataFrame
        params_df = pd.DataFrame(scenarios_with_params)
        
        # If there are no parameters other than variant, skip
        if len(params_df.columns) <= 2:  # 'scenario' and 'variant' only
            continue
        
        # Create a DataFrame with scenarios, parameters, and performance metrics
        scenario_params_metrics = pd.merge(
            params_df, 
            group[['scenario', 'model', 'rmse', 'mae', 'mape']], 
            on='scenario'
        )
        
        # Save this data for the scenario type
        parameter_impacts[scenario_type] = scenario_params_metrics
        
        # Save to CSV
        scenario_params_metrics.to_csv(
            f"{args.output_dir}/parameter_analysis/{scenario_type}_parameter_impact.csv", 
            index=False
        )
        
        # For each parameter, create a chart showing its impact on RMSE
        param_columns = [col for col in params_df.columns 
                        if col not in ['scenario', 'variant']]
        
        for param in param_columns:
            # For categorical parameters, use boxplot
            if scenario_params_metrics[param].dtype == 'object':
                plt.figure(figsize=FIGURE_SIZES['box_plot'])
                sns.boxplot(x=param, y='rmse', data=scenario_params_metrics)
                plt.title(f'Impact of {param} on RMSE - {scenario_type}', 
                          fontsize=FONT_SIZES['title'])
                plt.ylabel('RMSE', fontsize=FONT_SIZES['axis_label'])
                plt.xlabel(param, fontsize=FONT_SIZES['axis_label'])
                plt.xticks(rotation=45)
                plt.tight_layout()
                
            # For numerical parameters, use scatterplot
            else:
                # Convert param to float for proper sorting
                scenario_params_metrics[param] = scenario_params_metrics[param].astype(float)
                
                plt.figure(figsize=FIGURE_SIZES['scatter_plot'])
                sns.scatterplot(
                    x=param, 
                    y='rmse', 
                    hue='model', 
                    data=scenario_params_metrics,
                    alpha=0.7
                )
                plt.title(f'Impact of {param} on RMSE - {scenario_type}', 
                          fontsize=FONT_SIZES['title'])
                plt.ylabel('RMSE', fontsize=FONT_SIZES['axis_label'])
                plt.xlabel(param, fontsize=FONT_SIZES['axis_label'])
                plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
            
            # Save figure
            plt.savefig(
                f"{args.output_dir}/parameter_analysis/{scenario_type}_{param}_impact.png", 
                dpi=args.dpi, 
                bbox_inches='tight'
            )
            plt.close()
    
    return parameter_impacts

def create_html_scenario_report(results_df):
    """
    Create detailed HTML reports for each scenario type with interactive elements.
    """
    logging.info("Creating HTML scenario reports...")
    
    # Make sure output directory exists
    os.makedirs(f"{args.output_dir}/html_reports", exist_ok=True)
    
    # Group scenarios by type
    grouped_results = group_scenarios_by_type(results_df)
    
    # For each scenario type, create a detailed HTML report
    for scenario_type, group in grouped_results.items():
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{scenario_type.capitalize()} Scenario Analysis</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin: 20px 0;
                }}
                .card {{
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                    flex: 1 1 300px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metric {{
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                .value {{
                    font-size: 1.8em;
                    color: #3498db;
                }}
                .highlight {{
                    background-color: #fffde7;
                }}
                .plot-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .selector {{
                    margin: 10px 0;
                    padding: 8px;
                    width: 200px;
                }}
                .tab {{
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                }}
                .tab button:hover {{
                    background-color: #ddd;
                }}
                .tab button.active {{
                    background-color: #3498db;
                    color: white;
                }}
                .tabcontent {{
                    display: none;
                    padding: 6px 12px;
                    border: 1px solid #ccc;
                    border-top: none;
                }}
                .visible {{
                    display: block;
                }}
            </style>
        </head>
        <body>
            <h1>{scenario_type.capitalize()} Scenario Analysis</h1>
            
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'Overview')">Overview</button>
                <button class="tablinks" onclick="openTab(event, 'DetailedComparison')">Detailed Comparison</button>
                <button class="tablinks" onclick="openTab(event, 'VariantAnalysis')">Variant Analysis</button>
            </div>
            
            <div id="Overview" class="tabcontent visible">
                <h2>Overview</h2>
                <p>This report provides a detailed analysis of the {scenario_type} scenario type across different variants and models.</p>
                
                <div class="container">
                    <div class="card">
                        <h3>Variants Analyzed</h3>
                        <div class="metric">Number of Variants</div>
                        <div class="value">{len(group['scenario'].unique())}</div>
                    </div>
                    <div class="card">
                        <h3>Models Compared</h3>
                        <div class="metric">Number of Models</div>
                        <div class="value">{len(group['model'].unique())}</div>
                    </div>
                    <div class="card">
                        <h3>Best Model Overall</h3>
                        <div class="metric">Model</div>
                        <div class="value">{group.loc[group['rmse'].idxmin()]['model']}</div>
                        <div class="metric">RMSE</div>
                        <div class="value">{group['rmse'].min():.4f}</div>
                    </div>
                </div>
                
                <h3>Performance Summary by Model</h3>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Avg RMSE</th>
                        <th>Avg MAE</th>
                        <th>Avg MAPE</th>
                        <th>Avg Training Time (s)</th>
                        <th>Win Count</th>
                    </tr>
        """
        
        # Add rows for each model's performance
        model_summary = []
        for model, model_data in group.groupby('model'):
            wins = sum(1 for _, row in model_data.iterrows() 
                      if row['rmse'] == group.loc[group['scenario'] == row['scenario'], 'rmse'].min())
            
            model_summary.append({
                'model': model,
                'avg_rmse': model_data['rmse'].mean(),
                'avg_mae': model_data['mae'].mean(),
                'avg_mape': model_data['mape'].mean(),
                'avg_training_time': model_data['training_time'].mean(),
                'win_count': wins
            })
        
        # Sort by average RMSE
        model_summary_sorted = sorted(model_summary, key=lambda x: x['avg_rmse'])
        
        for model_data in model_summary_sorted:
            html_content += f"""
                <tr>
                    <td>{model_data['model']}</td>
                    <td>{model_data['avg_rmse']:.4f}</td>
                    <td>{model_data['avg_mae']:.4f}</td>
                    <td>{model_data['avg_mape']:.4f}</td>
                    <td>{model_data['avg_training_time']:.4f}</td>
                    <td>{model_data['win_count']}</td>
                </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div id="DetailedComparison" class="tabcontent">
                <h2>Detailed Model Comparison</h2>
                
                <h3>Performance by Variant</h3>
                <table>
                    <tr>
                        <th>Scenario</th>
                        <th>Best Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>MAPE</th>
                        <th>Training Time (s)</th>
                    </tr>
        """
        
        # Add rows for each scenario
        for scenario in sorted(group['scenario'].unique()):
            scenario_data = group[group['scenario'] == scenario]
            best_idx = scenario_data['rmse'].idxmin()
            best_row = scenario_data.loc[best_idx]
            
            html_content += f"""
                <tr>
                    <td>{scenario}</td>
                    <td>{best_row['model']}</td>
                    <td>{best_row['rmse']:.4f}</td>
                    <td>{best_row['mae']:.4f}</td>
                    <td>{best_row['mape']:.4f}</td>
                    <td>{best_row['training_time']:.4f}</td>
                </tr>
            """
        
        html_content += """
                </table>
                
                <h3>All Results</h3>
                <div class="plot-container">
                    <img src="../plots/rmse_by_model_and_scenario_{}.png" alt="RMSE Comparison" width="100%">
                </div>
            </div>
            
            <div id="VariantAnalysis" class="tabcontent">
                <h2>Variant Analysis</h2>
                
                <h3>Parameter Comparison</h3>
        """.format(scenario_type)
        
        # Extract and display parameters for variants if available
        scenarios_with_params = []
        for scenario in group['scenario'].unique():
            params = get_scenario_parameters(scenario)
            params['scenario'] = scenario
            scenarios_with_params.append(params)
        
        if scenarios_with_params:
            params_df = pd.DataFrame(scenarios_with_params)
            
            html_content += """
                <table>
                    <tr>
                        <th>Variant</th>
            """
            
            # Add parameter column headers
            param_cols = [col for col in params_df.columns if col not in ['scenario', 'variant']]
            for param in param_cols:
                html_content += f"<th>{param}</th>"
            
            html_content += """
                    </tr>
            """
            
            # Add rows for each variant
            for _, row in params_df.iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['variant']}</td>
                """
                
                for param in param_cols:
                    if param in row:
                        html_content += f"<td>{row[param]}</td>"
                    else:
                        html_content += "<td>-</td>"
                
                html_content += """
                    </tr>
                """
            
            html_content += """
                </table>
            """
        
        # Complete the HTML
        html_content += """
            </div>
            
            <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].className = tabcontent[i].className.replace(" visible", "");
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).className += " visible";
                evt.currentTarget.className += " active";
            }
            </script>
        </body>
        </html>
        """
        
        # Write the HTML to a file
        with open(f"{args.output_dir}/html_reports/{scenario_type}_report.html", 'w') as f:
            f.write(html_content)
    
    # Create index.html that links to all scenario reports
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Time Series Forecasting Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2 {
                color: #2c3e50;
            }
            h1 {
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }
            .card {
                background: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 15px;
                flex: 1 1 300px;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                margin-bottom: 10px;
                padding: 8px;
                background-color: #f8f8f8;
                border-radius: 4px;
            }
            li:hover {
                background-color: #f0f0f0;
            }
        </style>
    </head>
    <body>
        <h1>Time Series Forecasting Analysis</h1>
        
        <div class="container">
            <div class="card">
                <h2>Scenario Reports</h2>
                <ul>
    """
    
    # Add links to all scenario reports
    for scenario_type in grouped_results.keys():
        index_html += f"""
                    <li><a href="{scenario_type}_report.html">{scenario_type.capitalize()} Scenarios</a></li>
        """
    
    index_html += """
                </ul>
            </div>
            
            <div class="card">
                <h2>Overall Analysis</h2>
                <ul>
                    <li><a href="../detailed_comparisons/all_models_comparison.html">Complete Model Comparison</a></li>
                    <li><a href="../scenario_comparisons/avg_by_scenario_type.csv">Scenario Type Averages (CSV)</a></li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the index.html
    with open(f"{args.output_dir}/html_reports/index.html", 'w') as f:
        f.write(index_html)
    
    logging.info(f"Created HTML reports in {args.output_dir}/html_reports/")
    
    return True

def create_enhanced_visualizations(results_df, wins_df):
    """
    Create enhanced visualizations including grouped scenarios and HTML report.
    
    Args:
        results_df: DataFrame containing all model results
        wins_df: DataFrame containing win counts for each model
    """
    logging.info("Creating enhanced visualizations...")
    
    # Create standard visualizations first
    create_visualizations(results_df, wins_df)
    
    # Group scenarios if requested
    if args.group_scenarios:
        create_scenario_group_visualizations(results_df)
    
    # Create HTML report if requested
    if args.html_report:
        create_performance_comparison_table(results_df, wins_df)
        create_html_scenario_report(results_df)
        
    logging.info("Enhanced visualizations created successfully")

def create_scenario_group_visualizations(results_df):
    """
    Create visualizations that group similar scenarios together.
    
    Args:
        results_df: DataFrame containing all model results
    """
    logging.info("Creating scenario group visualizations...")
    
    # Create necessary directories
    scenario_groups_dir = os.path.join(args.output_dir, "scenario_groups")
    os.makedirs(scenario_groups_dir, exist_ok=True)
    
    # Extract the base scenario name from each scenario (before the variant number)
    results_df['scenario_group'] = results_df['scenario'].apply(lambda x: extract_scenario_base(x))
    
    # Group metrics by scenario group and model
    grouped_metrics = results_df.groupby(['scenario_group', 'model']).agg({
        'rmse': 'mean',
        'mae': 'mean',
        'mape': 'mean'
    }).reset_index()
    
    # Save grouped metrics to CSV
    grouped_metrics.to_csv(os.path.join(scenario_groups_dir, "grouped_metrics.csv"), index=False)
    
    # Create bar charts for each metric by scenario group
    for metric in ['rmse', 'mae', 'mape']:
        create_grouped_bar_chart(grouped_metrics, metric, scenario_groups_dir)
    
    # Create heatmap of all metrics across scenario groups
    create_scenario_heatmap(grouped_metrics, scenario_groups_dir)
    
    # Create scenario group comparison matrix
    create_scenario_comparison_matrix(results_df, scenario_groups_dir)
    
    logging.info(f"Scenario group visualizations saved to {scenario_groups_dir}")

def extract_scenario_base(scenario_name):
    """
    Extract the base scenario name from the full scenario name.
    Example: 'stepped_1_param_x_1' -> 'stepped'
    
    Args:
        scenario_name: The full scenario name
        
    Returns:
        The base scenario name
    """
    # First try to split by underscores and get the base name
    parts = scenario_name.split('_')
    
    # Check if it follows pattern like 'stepped_1_...'
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0]
    
    # For more complex cases, use regex to extract alphabetic prefix
    import re
    match = re.match(r'^([a-zA-Z]+)', scenario_name)
    if match:
        return match.group(1)
    
    # Fallback: return the original name
    return scenario_name

def create_grouped_bar_chart(grouped_metrics, metric, output_dir):
    """
    Create a grouped bar chart for a specific metric across scenario groups.
    
    Args:
        grouped_metrics: DataFrame containing grouped metrics
        metric: The metric to visualize (rmse, mae, or mape)
        output_dir: Directory to save the output
    """
    plt.figure(figsize=(10, 6))
    
    # Create a pivot table for easier plotting
    pivot_df = grouped_metrics.pivot(index='model', columns='scenario_group', values=metric)
    
    # Plot the bar chart
    ax = pivot_df.plot(kind='bar', figsize=(12, 7))
    
    # Add labels and title
    metric_upper = metric.upper()
    plt.title(f'Average {metric_upper} by Model and Scenario Group')
    plt.xlabel('Model')
    plt.ylabel(metric_upper)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend with better positioning
    plt.legend(title='Scenario Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'grouped_{metric}_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

def create_scenario_heatmap(grouped_metrics, output_dir):
    """
    Create a heatmap showing model performance across scenario groups.
    
    Args:
        grouped_metrics: DataFrame containing grouped metrics
        output_dir: Directory to save the output
    """
    # Create separate heatmaps for each metric
    for metric in ['rmse', 'mae', 'mape']:
        plt.figure(figsize=(10, 8))
        
        # Create a pivot table for the heatmap
        pivot_df = grouped_metrics.pivot(index='model', columns='scenario_group', values=metric)
        
        # Normalize the data for better visualization
        # For each scenario group, scale the values between 0 and 1
        normalized_df = pivot_df.copy()
        for col in normalized_df.columns:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        # Create the heatmap
        sns.heatmap(normalized_df, annot=pivot_df.round(4), fmt='.4f', cmap='YlGnBu_r', 
                    linewidths=0.5, cbar_kws={'label': f'Normalized {metric.upper()}'})
        
        # Add title and labels
        plt.title(f'Model Performance Heatmap - {metric.upper()} by Scenario Group')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'), bbox_inches='tight', dpi=300)
        plt.close()

def create_scenario_comparison_matrix(results_df, output_dir):
    """
    Create a comparison matrix showing which models perform best in each scenario group.
    
    Args:
        results_df: DataFrame containing all model results
        output_dir: Directory to save the output
    """
    # Group by scenario group and find the best model for each
    results_df['scenario_group'] = results_df['scenario'].apply(lambda x: extract_scenario_base(x))
    
    # For each scenario group and individual scenario, find the best model based on RMSE
    best_models = results_df.loc[results_df.groupby(['scenario_group', 'scenario'])['rmse'].idxmin()]
    best_models = best_models[['scenario_group', 'scenario', 'model', 'rmse']]
    
    # Count occurrences of each model being the best in each scenario group
    best_count = best_models.groupby(['scenario_group', 'model']).size().reset_index(name='count')
    
    # Create a pivot table for the matrix
    matrix_df = best_count.pivot(index='scenario_group', columns='model', values='count').fillna(0)
    
    # Calculate the total number of scenarios in each group for percentage calculation
    scenario_counts = best_models.groupby('scenario_group').size()
    
    # Convert counts to percentages
    for idx in matrix_df.index:
        matrix_df.loc[idx] = matrix_df.loc[idx] / scenario_counts[idx] * 100
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix_df, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5,
                cbar_kws={'label': 'Percentage of Scenarios (%)'})
    
    # Add title and labels
    plt.title('Best Model Frequency by Scenario Group (%)')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'best_model_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also save the raw data
    matrix_df.to_csv(os.path.join(output_dir, 'best_model_matrix.csv'))

def compare_models_from_df(results_df):
    """Create a wins dataframe from results DataFrame."""
    logging.info("Calculating model wins from results data...")
    
    # Count how many times each model was the best
    model_wins = {}
    
    for model in results_df['model'].unique():
        model_wins[model] = {'mae': 0, 'rmse': 0, 'mape': 0, 'total': 0}
    
    for scenario in results_df['scenario'].unique():
        scenario_df = results_df[results_df['scenario'] == scenario]
        
        # Skip if missing metrics
        if any(col not in scenario_df.columns for col in ['mae', 'rmse', 'mape']):
            continue
            
        # Find best models
        best_mae = scenario_df.loc[scenario_df['mae'].idxmin()]['model']
        best_rmse = scenario_df.loc[scenario_df['rmse'].idxmin()]['model']
        best_mape = scenario_df.loc[scenario_df['mape'].idxmin()]['model']
        
        # Update wins counts
        model_wins[best_mae]['mae'] += 1
        model_wins[best_mae]['total'] += 1
        
        model_wins[best_rmse]['rmse'] += 1
        model_wins[best_rmse]['total'] += 1
        
        model_wins[best_mape]['mape'] += 1
        model_wins[best_mape]['total'] += 1
    
    # Create DataFrame
    wins_rows = []
    for model, wins in model_wins.items():
        wins_rows.append({
            'model': model,
            'mae_wins': wins['mae'],
            'rmse_wins': wins['rmse'],
            'mape_wins': wins['mape'],
            'total_wins': wins['total']
        })
    
    wins_df = pd.DataFrame(wins_rows)
    wins_df = wins_df.sort_values('total_wins', ascending=False)
    
    # Save for future use
    wins_output_path = os.path.join(args.output_dir, "model_wins.csv")
    os.makedirs(os.path.dirname(wins_output_path), exist_ok=True)
    wins_df.to_csv(wins_output_path, index=False)
    logging.info(f"Calculated and saved model wins data to {wins_output_path}")
    
    return wins_df

def create_pattern_type_analysis(results_df):
    """Create visualizations that compare model performance across different pattern types."""
    logging.info("Creating pattern type analysis visualizations")
    
    # Add pattern type column
    results_df['pattern_type'] = results_df['scenario'].apply(get_base_scenario_type)
    
    # Get unique pattern types
    pattern_types = results_df['pattern_type'].unique()
    
    if len(pattern_types) <= 1:
        logging.info("Only one pattern type found, skipping pattern type analysis")
        return
    
    # Create directory for pattern analysis
    os.makedirs("results/pattern_analysis", exist_ok=True)
    
    # 1. Overall performance by pattern type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # For each metric (RMSE, MAE, MAPE)
    metrics = ['rmse', 'mae', 'mape']
    titles = ['RMSE by Pattern Type', 'MAE by Pattern Type', 'MAPE by Pattern Type']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Get mean of metric by pattern type and model
        pivot_df = results_df.pivot_table(
            index='pattern_type', 
            columns='model', 
            values=metric,
            aggfunc='mean'
        )
        
        # Plot
        pivot_df.plot(kind='bar', ax=axes[i])
        axes[i].set_title(title)
        axes[i].set_ylabel(metric.upper())
        axes[i].legend(title='Model')
        
        # Rotate x-axis labels
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("results/pattern_analysis/pattern_type_metrics.png")
    plt.close()
    
    # 2. Heatmap of best models by pattern type
    plt.figure(figsize=(12, 8))
    
    # Create a matrix of counts (pattern_type x model)
    best_model_counts = pd.DataFrame(index=pattern_types, columns=results_df['model'].unique(), data=0)
    
    # Find the best model for each scenario based on RMSE
    for pattern_type in pattern_types:
        pattern_scenarios = results_df[results_df['pattern_type'] == pattern_type]['scenario'].unique()
        
        for scenario in pattern_scenarios:
            scenario_df = results_df[results_df['scenario'] == scenario]
            best_model = scenario_df.loc[scenario_df['rmse'].idxmin()]['model']
            best_model_counts.loc[pattern_type, best_model] += 1
    
    # Create heatmap
    sns.heatmap(best_model_counts, annot=True, cmap='YlGnBu', fmt='g')
    plt.title('Best Model Count by Pattern Type (Based on RMSE)')
    plt.ylabel('Pattern Type')
    plt.xlabel('Model')
    plt.tight_layout()
    plt.savefig("results/pattern_analysis/best_model_by_pattern.png")
    plt.close()
    
    # 3. Create a performance comparison across pattern types for each model
    for model in results_df['model'].unique():
        try:
            model_df = results_df[results_df['model'] == model]
            
            # Skip if we have too few data points
            if len(model_df) < 3:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by pattern type and calculate mean metrics
            pattern_metrics = model_df.groupby('pattern_type').agg({
                'rmse': 'mean',
                'mae': 'mean',
                'mape': 'mean'
            }).reset_index()
            
            # Normalize metrics for radar chart (0-1 scale)
            for metric in ['rmse', 'mae', 'mape']:
                max_val = pattern_metrics[metric].max()
                if max_val > 0:
                    pattern_metrics[f'{metric}_norm'] = 1 - (pattern_metrics[metric] / max_val)
                else:
                    pattern_metrics[f'{metric}_norm'] = 1  # Avoid division by zero
            
            # Prepare data for radar chart
            categories = pattern_metrics['pattern_type'].tolist()
            N = len(categories)
            
            if N < 3:
                logging.info(f"Not enough pattern types for radar chart for model {model}")
                continue
                
            # Create angles for radar chart
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Set up the axes
            ax = plt.subplot(111, polar=True)
            
            # Plot each metric
            for metric, color, label in zip(
                ['rmse_norm', 'mae_norm', 'mape_norm'],
                ['#1f77b4', '#ff7f0e', '#2ca02c'],
                ['RMSE (Inverted)', 'MAE (Inverted)', 'MAPE (Inverted)']
            ):
                values = pattern_metrics[metric].tolist()
                values += values[:1]  # Close the loop
                
                # Plot the metric
                ax.plot(angles, values, 'o-', linewidth=2, color=color, label=label)
                ax.fill(angles, values, color=color, alpha=0.1)
            
            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines for each label and label them
            plt.xticks(angles[:-1], categories)
            
            # Draw y axis labels (0.2, 0.4, etc)
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            plt.ylim(0, 1)
            
            # Add legend and title
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title(f"{model} Performance Across Pattern Types\n(Higher is better - metrics inverted)")
            
            plt.tight_layout()
            plt.savefig(f"results/pattern_analysis/{model}_pattern_radar.png")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating radar chart for model {model}: {e}")
    
    # 4. Pattern parameter sensitivity analysis
    for pattern_type in pattern_types:
        try:
            # Filter to this pattern type
            pattern_df = results_df[results_df['pattern_type'] == pattern_type].copy()
            
            # Get scenarios for this pattern
            pattern_scenarios = pattern_df['scenario'].unique()
            
            # Extract parameters for each scenario
            parameters_by_scenario = {}
            for scenario in pattern_scenarios:
                parameters_by_scenario[scenario] = get_scenario_parameters(scenario)
            
            # Check if we have enough scenarios and parameters
            if len(parameters_by_scenario) < 3:
                logging.info(f"Not enough scenarios for parameter analysis of pattern type {pattern_type}")
                continue
                
            # Find common parameters across scenarios
            parameter_names = set()
            for params in parameters_by_scenario.values():
                parameter_names.update(params.keys())
            
            # Remove 'variant' as it's not a real parameter
            if 'variant' in parameter_names:
                parameter_names.remove('variant')
                
            # Skip if we have no parameters
            if not parameter_names:
                logging.info(f"No parameters found for pattern type {pattern_type}")
                continue
                
            # For each parameter, create a plot showing its relationship with error
            for param_name in parameter_names:
                # Check if parameter exists for at least 3 scenarios
                scenarios_with_param = [s for s, p in parameters_by_scenario.items() if param_name in p]
                if len(scenarios_with_param) < 3:
                    continue
                    
                # Create a plot for this parameter
                plt.figure(figsize=(12, 8))
                
                # Extract parameter values and performance metrics
                param_data = []
                for scenario in scenarios_with_param:
                    params = parameters_by_scenario[scenario]
                    if param_name not in params:
                        continue
                        
                    try:
                        # Convert parameter value to float
                        param_value = float(params[param_name])
                        
                        # Get best model performance for this scenario
                        scenario_results = pattern_df[pattern_df['scenario'] == scenario]
                        best_rmse = scenario_results['rmse'].min()
                        
                        param_data.append({
                            'param_value': param_value,
                            'rmse': best_rmse,
                            'scenario': scenario
                        })
                    except ValueError:
                        # Skip if parameter can't be converted to float
                        continue
                
                # Skip if we have too few data points
                if len(param_data) < 3:
                    continue
                
                # Create DataFrame and sort by parameter value
                param_df = pd.DataFrame(param_data)
                param_df = param_df.sort_values('param_value')
                
                # Create scatter plot
                plt.scatter(param_df['param_value'], param_df['rmse'], s=80, alpha=0.7)
                
                # Try to fit a trend line if we have enough points
                try:
                    if len(param_df) >= 3:
                        # Simple linear regression
                        x = param_df['param_value'].values.reshape(-1, 1)
                        y = param_df['rmse'].values
                        
                        model = LinearRegression()
                        model.fit(x, y)
                        
                        # Plot trend line
                        x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                        plt.plot(x_range, model.predict(x_range), 'r--', alpha=0.7)
                        
                        # Calculate correlation coefficient
                        corr, p_value = pearsonr(param_df['param_value'], param_df['rmse'])
                        plt.title(f"Impact of '{param_name}' on RMSE for {pattern_type} patterns\nCorrelation: {corr:.2f} (p-value: {p_value:.3f})")
                    else:
                        plt.title(f"Impact of '{param_name}' on RMSE for {pattern_type} patterns")
                except Exception as e:
                    logging.error(f"Error fitting trend line: {e}")
                    plt.title(f"Impact of '{param_name}' on RMSE for {pattern_type} patterns")
                
                # Add labels for each point
                for i, row in param_df.iterrows():
                    plt.annotate(
                        row['scenario'].split('_')[1],  # Just show variant number
                        (row['param_value'], row['rmse']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center'
                    )
                
                plt.xlabel(f"{param_name}")
                plt.ylabel("Best RMSE")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                os.makedirs(f"results/pattern_analysis/{pattern_type}", exist_ok=True)
                plt.savefig(f"results/pattern_analysis/{pattern_type}/{param_name}_sensitivity.png")
                plt.close()
                
        except Exception as e:
            logging.error(f"Error analyzing parameters for pattern type {pattern_type}: {e}")
    
    logging.info("Pattern type analysis visualizations created")

# Create a detailed analysis of model performance by pattern characteristics
def analyze_pattern_characteristics(results_df):
    """
    Analyze how different models perform against various pattern features
    by loading the feature files and correlating with model performance.
    """
    logging.info("Analyzing pattern characteristics and model performance")
    
    # Create output directory
    os.makedirs("results/pattern_characteristics", exist_ok=True)
    
    # Load pattern features for each scenario that has them
    pattern_features = {}
    feature_files = glob.glob(f"{scenarios_path}/*_features.json")
    
    for feature_file in feature_files:
        try:
            # Extract scenario name from filename
            filename = os.path.basename(feature_file)
            scenario_name = filename.replace("_features.json", "")
            
            # Load features
            with open(feature_file, 'r') as f:
                features = json.load(f)
                
            pattern_features[scenario_name] = features
        except Exception as e:
            logging.warning(f"Error loading features from {feature_file}: {e}")
    
    if not pattern_features:
        logging.warning("No pattern features found, skipping pattern characteristics analysis")
        return
        
    logging.info(f"Loaded features for {len(pattern_features)} patterns")
    
    # Extract common numerical features across all patterns
    numerical_features = set()
    
    for features in pattern_features.values():
        for key, value in features.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numerical_features.add(key)
    
    # Remove any non-useful features
    exclude_features = {'min', 'max'}
    numerical_features = numerical_features - exclude_features
    
    if not numerical_features:
        logging.warning("No usable numerical features found")
        return
    
    logging.info(f"Found {len(numerical_features)} numerical features for analysis")
    
    # Create a consolidated features DataFrame
    feature_rows = []
    
    for scenario, features in pattern_features.items():
        row = {'scenario': scenario, 'pattern_type': get_base_scenario_type(scenario)}
        
        for feature in numerical_features:
            if feature in features and isinstance(features[feature], (int, float)):
                row[feature] = features[feature]
        
        feature_rows.append(row)
    
    features_df = pd.DataFrame(feature_rows)
    
    # Merge with model performance data
    merged_data = pd.merge(
        results_df[['scenario', 'model', 'rmse', 'mae', 'mape']],
        features_df,
        on='scenario',
        how='inner'
    )
    
    if merged_data.empty:
        logging.warning("No data matches between features and results")
        return
    
    logging.info(f"Created merged dataset with {len(merged_data)} records")
    
    # 1. Correlation matrix between features and model performance
    for model in merged_data['model'].unique():
        try:
            model_data = merged_data[merged_data['model'] == model]
            
            # Skip if too few data points
            if len(model_data) < 5:
                logging.info(f"Not enough data points for model {model}, skipping correlation analysis")
                continue
            
            # Select metrics and features
            corr_columns = ['rmse', 'mae', 'mape'] + list(numerical_features.intersection(model_data.columns))
            
            # Calculate correlation matrix
            corr_matrix = model_data[corr_columns].corr()
            
            # Extract just the correlations with performance metrics
            metric_correlations = corr_matrix.loc[['rmse', 'mae', 'mape'], :].drop(['rmse', 'mae', 'mape'], axis=1)
            
            # Create a heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(metric_correlations, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title(f"{model}: Correlation between Pattern Features and Performance Metrics")
            plt.tight_layout()
            plt.savefig(f"results/pattern_characteristics/{model}_feature_correlations.png")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating correlation matrix for model {model}: {e}")
    
    # 2. Model performance by feature value
    top_features = ['cv', 'mean_abs_change', 'r2_linear', 'autocorr_daily']
    
    # Filter to features that are actually available
    available_top_features = [f for f in top_features if f in merged_data.columns]
    
    if not available_top_features:
        logging.warning("None of the top features are available in the dataset")
        return
    
    for feature in available_top_features:
        try:
            plt.figure(figsize=(12, 8))
            
            # Group data into bins by feature value
            merged_data['feature_bin'] = pd.qcut(
                merged_data[feature], 
                q=min(4, len(merged_data[feature].unique())), 
                duplicates='drop'
            )
            
            # Group by bin and model, calculate mean RMSE
            grouped = merged_data.groupby(['feature_bin', 'model'])['rmse'].mean().reset_index()
            
            # Create grouped bar chart
            sns.barplot(x='feature_bin', y='rmse', hue='model', data=grouped)
            
            plt.title(f"Model RMSE by {feature} value")
            plt.xlabel(f"{feature} range")
            plt.ylabel("Mean RMSE")
            plt.legend(title="Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"results/pattern_characteristics/{feature}_model_performance.png")
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating analysis for feature {feature}: {e}")
    
    # 3. Feature importance for model performance
    # Create a combined "error" metric by normalizing and averaging RMSE, MAE, MAPE
    try:
        # Create copy of merged data for this analysis
        analysis_data = merged_data.copy()
        
        # Select only numerical features
        feature_columns = list(numerical_features.intersection(analysis_data.columns))
        
        # Normalize metrics (0-1 scale where 0 is best)
        for metric in ['rmse', 'mae', 'mape']:
            max_val = analysis_data[metric].max()
            min_val = analysis_data[metric].min()
            
            if max_val > min_val:
                analysis_data[f'{metric}_norm'] = (analysis_data[metric] - min_val) / (max_val - min_val)
            else:
                analysis_data[f'{metric}_norm'] = 0
            
        # Create combined error metric
        analysis_data['combined_error'] = (
            analysis_data['rmse_norm'] + 
            analysis_data['mae_norm'] + 
            analysis_data['mape_norm']
        ) / 3
        
        # For each model, create a feature importance plot
        for model in analysis_data['model'].unique():
            model_data = analysis_data[analysis_data['model'] == model]
            
            # Skip if too few data points
            if len(model_data) < max(5, len(feature_columns) + 1):
                continue
                
            # Create feature importance using Random Forest
            try:
                X = model_data[feature_columns]
                y = model_data['combined_error']
                
                # Skip if we have any NaNs
                if X.isna().any().any() or y.isna().any():
                    continue
                
                # Fit a random forest to determine feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importances
                importances = rf.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Create bar chart of feature importance
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title(f"{model}: Feature Importance for Prediction Error")
                plt.tight_layout()
                plt.savefig(f"results/pattern_characteristics/{model}_feature_importance.png")
                plt.close()
                
            except Exception as e:
                logging.error(f"Error calculating feature importance for model {model}: {e}")
                continue
            
    except Exception as e:
        logging.error(f"Error in feature importance analysis: {e}")
        
    logging.info("Pattern characteristics analysis completed")

def main():
    """Main function to run the training and evaluation pipeline."""
    # Parse command line arguments
    global args
    args = parse_args()

    # Time the execution
    start_time = time.time()
    
    # Define log file path
    log_file = os.path.join("logs", "train_models.log")
    os.makedirs("logs", exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting training and evaluation process")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    
    # Get list of available models
    models = get_available_models()
    logging.info(f"Available models: {', '.join(models)}")
    
    # Filter to requested models if specified
    if args.models:
        requested_models = [m.strip() for m in args.models.split(',')]
        models = [m for m in models if m in requested_models]
        logging.info(f"Running with requested models: {', '.join(models)}")
    
    # Use the scenarios from command line args if provided, otherwise list them
    if args.scenarios:
        SCENARIOS = [s.strip() for s in args.scenarios.split(',')]
        logging.info(f"Using {len(SCENARIOS)} scenarios from command line arguments")
    elif args.scenarios_list:
        try:
            with open(args.scenarios_list, 'r') as f:
                SCENARIOS = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(SCENARIOS)} scenarios from {args.scenarios_list}")
        except Exception as e:
            logging.error(f"Failed to load scenarios from {args.scenarios_list}: {e}")
            return
    else:
        # Updated: Find scenarios by looking for _train.csv files
        SCENARIOS = []
        if os.path.exists(scenarios_path):
            for f in os.listdir(scenarios_path):
                if f.endswith("_train.csv"):
                    # Extract scenario name (remove _train.csv suffix)
                    scenario_name = f.replace("_train.csv", "")
                    SCENARIOS.append(scenario_name)
        else:
            logging.error(f"Scenarios path '{scenarios_path}' does not exist.")
            return
    
    # Sort scenarios for consistent ordering
    SCENARIOS.sort()
    logging.info(f"Found {len(SCENARIOS)} scenarios to evaluate")
    
    if args.limit:
        SCENARIOS = SCENARIOS[:args.limit]
        logging.info(f"Limited to first {args.limit} scenarios")
    
    if not SCENARIOS:
        logging.error("No scenarios found or specified.")
        return
    
    if args.visualize_only:
        logging.info("Running in visualize-only mode, skipping model training...")
        results_df, wins_df = load_existing_results()
        
        # Check if we have any results
        if results_df is None or results_df.empty:
            logging.error("No results found in results directory. Run training first or specify results file.")
            return
        
        logging.info(f"Loaded results for {len(results_df['model'].unique())} models and {len(results_df['scenario'].unique())} scenarios")
        
        # Create the wins dataframe if not loaded
        if wins_df is None:
            try:
                wins_df = create_wins_summary(results_df)
            except Exception as e:
                logging.error(f"Error creating wins summary: {e}")
        
        # Create visualizations
        setup_visualization_params()
        
        try:
            create_model_comparison_matrix(results_df)
            logging.info("Created model comparison matrix")
        except Exception as e:
            logging.error(f"Error creating model comparison matrix: {e}")
        
        try:
            create_performance_comparison_table(results_df, wins_df)
            logging.info("Created performance comparison table")
        except Exception as e:
            logging.error(f"Error creating performance table: {e}")
        
        try:
            create_model_radar_chart(results_df)
            logging.info("Created model radar chart")
        except Exception as e:
            logging.error(f"Error creating radar chart: {e}")
        
        try:
            create_best_model_summary(results_df)
            logging.info("Created best model summary")
        except Exception as e:
            logging.error(f"Error creating best model summary: {e}")
        
        try:
            create_best_model_plots(results_df)
            logging.info("Created best model plots")
        except Exception as e:
            logging.error(f"Error creating best model plots: {e}")
            
        # Add the pattern-specific analyses
        try:
            create_pattern_type_analysis(results_df)
            logging.info("Created pattern type analysis")
        except Exception as e:
            logging.error(f"Error creating pattern type analysis: {e}")
            
        try:
            analyze_pattern_characteristics(results_df)
            logging.info("Created pattern characteristics analysis")
        except Exception as e:
            logging.error(f"Error creating pattern characteristics analysis: {e}")
        
        if args.html_report:
            try:
                create_html_report(results_df, "results/model_report.html")
                logging.info("Created HTML report")
            except Exception as e:
                logging.error(f"Error creating HTML report: {e}")
        
        logging.info("Visualization complete")
        return
    
    # Train and evaluate each model
    all_results = {}
    executed_models = []
    
    for scenario in SCENARIOS:
        logging.info(f"Processing scenario: {scenario}")
        all_results[scenario] = {}
        
        # Create directory for this scenario's results
        os.makedirs(f"results/{scenario}", exist_ok=True)
        
        for model_name in models:
            logging.info(f"Training {model_name} on {scenario}")
            train_start = time.time()
            
            if args.load_existing:
                # Try to load existing results
                try:
                    if model_name not in executed_models:
                        logging.info(f"Looking for existing results for {model_name}")
                    
                    # Format string for JSON filename
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_json = f"results/{scenario}/{model_name}_metrics.json"
                    
                    # Try to load model metrics
                    metrics = None
                    if os.path.exists(model_json):
                        with open(model_json, 'r') as f:
                            metrics = json.load(f)
                        logging.info(f"Loaded existing metrics for {model_name} on {scenario}")
                    
                    if metrics:
                        all_results[scenario][model_name] = metrics
                        continue
                    else:
                        logging.info(f"No existing metrics found for {model_name} on {scenario}, proceeding with training")
                except Exception as e:
                    logging.warning(f"Error loading existing metrics: {e}")
            
            try:
                # Get the model module
                model_class = get_model_class(model_name)
                
                # Skip if model is unavailable
                if model_class is None:
                    logging.warning(f"Model {model_name} is unavailable, skipping")
                    continue
                
                # Format string for JSON filename
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_json = f"results/{scenario}/{model_name}_metrics.json"
                
                # Define file paths for this scenario
                train_file = f"{scenarios_path}/{scenario}_train.csv"
                test_file = f"{scenarios_path}/{scenario}_test.csv"
                
                if not (os.path.exists(train_file) and os.path.exists(test_file)):
                    logging.error(f"Training or test data not found for {scenario}")
                    continue
                
                # Train the model and get metrics
                metrics = train_and_evaluate_model(
                    model_class, model_name, scenario, train_file, test_file
                )
                
                if metrics is None:
                    logging.error(f"Failed to get metrics for {model_name} on {scenario}")
                    continue
                
                # Save metrics to results dictionary
                all_results[scenario][model_name] = metrics
                
                # Save metrics to JSON
                with open(model_json, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Add to list of executed models
                if model_name not in executed_models:
                    executed_models.append(model_name)
                
                train_end = time.time()
                logging.info(f"Completed {model_name} in {train_end - train_start:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Error training {model_name} on {scenario}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save all metrics to a comprehensive results dataframe
    logging.info("Saving comprehensive results")
    results_df = save_summary_metrics(all_results)
    
    if results_df is not None and not results_df.empty:
        # Set up visualization parameters
        setup_visualization_params()
        
        # Create visualizations
        try:
            create_model_comparison_matrix(results_df)
            logging.info("Created model comparison matrix")
        except Exception as e:
            logging.error(f"Error creating model comparison matrix: {e}")
        
        try:
            create_performance_comparison_table(results_df)
            logging.info("Created performance comparison table")
        except Exception as e:
            logging.error(f"Error creating performance table: {e}")
        
        try:
            create_best_model_plots(results_df)
            logging.info("Created best model plots")
        except Exception as e:
            logging.error(f"Error creating best model plots: {e}")
        
        try:
            create_model_radar_chart(results_df)
            logging.info("Created model radar chart")
        except Exception as e:
            logging.error(f"Error creating radar chart: {e}")
        
        try:
            create_best_model_summary(results_df)
            logging.info("Created best model summary")
        except Exception as e:
            logging.error(f"Error creating best model summary: {e}")
            
        # Add the pattern-specific analyses
        try:
            create_pattern_type_analysis(results_df)
            logging.info("Created pattern type analysis")
        except Exception as e:
            logging.error(f"Error creating pattern type analysis: {e}")
            
        try:
            analyze_pattern_characteristics(results_df)
            logging.info("Created pattern characteristics analysis")
        except Exception as e:
            logging.error(f"Error creating pattern characteristics analysis: {e}")
        
        # Create HTML report if requested
        if args.html_report:
            try:
                create_html_report(results_df, "results/model_report.html")
                logging.info("Created HTML report")
            except Exception as e:
                logging.error(f"Error creating HTML report: {e}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")

# Add this after the MODELS definition
def get_available_models():
    """Return a list of available model names."""
    return list(MODELS.keys())

def get_model_class(model_name):
    """Get the model class for a given model name."""
    if model_name not in MODELS:
        logging.error(f"Model {model_name} is not defined in MODELS dictionary")
        return None
    
    model_info = MODELS[model_name]
    script_path = model_info["script"]
    
    if not os.path.exists(script_path):
        logging.error(f"Model script not found: {script_path}")
        return None
    
    return model_name  # Return the model name as a placeholder for the actual class

def train_and_evaluate_model(model_class, model_name, scenario, train_file, test_file):
    """Train and evaluate a model by running the model's script."""
    logging.info(f"Training {model_name} on {scenario}")
    
    # Use the run_model function to actually run the model training
    model_name, scenario, metrics, execution_time = run_model(model_name, scenario, measure_time=True)
    
    if metrics is None:
        logging.error(f"Failed to get metrics for {model_name} on {scenario}")
        return None
        
    # Return the metrics from the actual model run
    return metrics

def create_wins_summary(results_df):
    """Create a summary of model wins."""
    # Placeholder for wins summary
    return pd.DataFrame()

if __name__ == "__main__":
    main()
