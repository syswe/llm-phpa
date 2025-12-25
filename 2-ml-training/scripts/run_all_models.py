#!/usr/bin/env python3
"""
Unified Model Training Script
Runs all 6 CPU models on a given dataset and reports MAE, RMSE, MAPE.
Compatible with Python 3.12+
"""

import subprocess
import sys
import os
import json
import argparse
from datetime import datetime

# Set libomp environment variables for XGBoost/LightGBM on macOS
os.environ["LDFLAGS"] = "-L/opt/homebrew/opt/libomp/lib"
os.environ["CPPFLAGS"] = "-I/opt/homebrew/opt/libomp/include"
os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Model scripts relative to this file's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CPU_MODELS_DIR = os.path.join(SCRIPT_DIR, "cpu-models")

MODELS = {
    "catboost": "train_catboost.py",
    "gbdt": "train_gbdt.py",
    "xgboost": "train_xgboost.py",
    "lightgbm": "train_lightgbm.py",
    "var": "train_var.py",
    "baseline": "train_baseline.py",
}

def run_model(model_name, script_name, train_file, test_file, run_id):
    """Run a single model and return metrics."""
    script_path = os.path.join(CPU_MODELS_DIR, script_name)
    
    if not os.path.exists(script_path):
        return {"error": f"Script not found: {script_path}"}
    
    cmd = [
        sys.executable,
        script_path,
        "--train-file", train_file,
        "--test-file", test_file,
        "--run-id", run_id
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.returncode != 0:
            return {"error": result.stderr[:500]}
        
        # Try to parse JSON from stdout
        output = result.stdout.strip()
        # Find the JSON part (last line usually)
        for line in output.split('\n'):
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        
        return {"error": "Could not parse output"}
        
    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Run all models on a dataset")
    parser.add_argument("--train-file", required=True, help="Path to training CSV")
    parser.add_argument("--test-file", required=True, help="Path to test CSV")
    parser.add_argument("--run-id", default="run", help="Run identifier")
    parser.add_argument("--models", default="all", help="Comma-separated list of models or 'all'")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()
    
    # Determine which models to run
    if args.models == "all":
        models_to_run = MODELS
    else:
        model_list = [m.strip() for m in args.models.split(",")]
        models_to_run = {k: v for k, v in MODELS.items() if k in model_list}
    
    print("=" * 70)
    print(f"Unified Model Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Train file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    print(f"Run ID: {args.run_id}")
    print(f"Models: {', '.join(models_to_run.keys())}")
    print("=" * 70)
    
    results = {}
    
    for model_name, script_name in models_to_run.items():
        print(f"\n▶ Running {model_name.upper()}...", end=" ", flush=True)
        
        metrics = run_model(
            model_name, 
            script_name, 
            args.train_file, 
            args.test_file, 
            args.run_id
        )
        
        if "error" in metrics:
            print(f"❌ Error: {metrics['error'][:100]}")
        else:
            mae = metrics.get("mae", "N/A")
            rmse = metrics.get("rmse", "N/A")
            mape = metrics.get("mape", "N/A")
            time = metrics.get("training_time", "N/A")
            print(f"✅ MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, Time={time:.2f}s")
        
        results[model_name] = metrics
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE':>12} {'Time':>10}")
    print("-" * 70)
    
    # Sort by MAE for ranking
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if "mae" in v],
        key=lambda x: x[1].get("mae", float("inf"))
    )
    
    for i, (model, metrics) in enumerate(sorted_results):
        rank = "🏆" if i == 0 else f"{i+1}."
        mae = f"{metrics.get('mae', 0):.4f}"
        rmse = f"{metrics.get('rmse', 0):.4f}"
        mape = f"{metrics.get('mape', 0):.2f}%"
        time = f"{metrics.get('training_time', 0):.2f}s"
        print(f"{rank} {model:<10} {mae:>10} {rmse:>10} {mape:>12} {time:>10}")
    
    # Add failed models
    for model, metrics in results.items():
        if "error" in metrics:
            print(f"❌ {model:<10} {'ERROR':>10} {'-':>10} {'-':>12} {'-':>10}")
    
    print("=" * 70)
    
    # Save results to JSON if requested
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "train_file": args.train_file,
                "test_file": args.test_file,
                "run_id": args.run_id,
                "results": results
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results

if __name__ == "__main__":
    main()
