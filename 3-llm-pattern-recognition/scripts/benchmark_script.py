#!/usr/bin/env python
"""
PHPA Pattern Detection and Model Recommendation Tool

This script uses existing datasets from the 1-dataset-generation module to:
1. Load example datasets for 6 main patterns (burst, chaotic, growing, onoff, seasonal, stepped)
2. Analyze time-series data to detect patterns and recommend appropriate forecasting models

Prerequisites:
  - Run the 1-dataset-generation module first to create datasets
  - Datasets should be located in ../../1-dataset-generation/scripts/test_output_all/

Usage:
  - Basic benchmark: python benchmark_script.py
    (Loads existing datasets and runs benchmark against them)
  
  - Test a specific CSV file: python benchmark_script.py --test-file path/to/your/data.csv
    (Analyzes your custom CSV file and recommends a model)

The CSV file should have 'timestamp' and 'pod_count' columns (or similar metric).

Based on performance analysis across multiple scenarios, the best models per pattern are:
- Burst Pattern: gbdt (44.0% win rate)
- Chaotic Pattern: gbdt (67.0% win rate)
- Growing Pattern: var (96.0% win rate)
- Onoff Pattern: catboost (62.0% win rate)
- Seasonal Pattern: gbdt (48.0% win rate)
- Stepped Pattern: gbdt (54.0% win rate)
"""

import csv
import os
import requests
import json
from collections import Counter, defaultdict
import glob
import random # Import random module
import time # Import time module
import re # For pattern matching in filenames
import matplotlib.pyplot as plt
import pandas as pd
import io # To read CSV string into pandas
import datetime
import argparse # For command line arguments
import numpy as np
from datetime import datetime, timedelta # For timestamps

# --- Configuration ---
API_KEY = "AIzaSyAdm87mpwBJkSus6gSkbSnRr2L9ABIRCgc"  # IMPORTANT: Replace with your actual Gemini API Key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
CSV_FILE_PATH = "results/summary/best_models_by_scenario.csv"
EXISTING_DATASET_DIR = "../../1-dataset-generation/scripts/test_output_all" # Directory for existing datasets
# DATASET_LINES_TO_SEND = 2018 # No longer needed, send full small file
MAX_ROWS_PER_EXAMPLE = 500 # Max rows for generated example files

# --- Test Flag ---
# Set to a pattern name (e.g., "burst", "seasonal") to test only that pattern.
# Set to None to test all 6 patterns.
TEST_SINGLE_PATTERN = None

# --- Wait Time ---
# Delay in seconds between API calls when testing multiple patterns
DELAY_BETWEEN_CALLS = 10

# --- Number of samples per pattern ---
SAMPLES_PER_PATTERN = 1

# --- Functions ---

def analyze_best_models(csv_path):
    """
    Updated to use hardcoded best models based on provided performance data.
    """
    # Hardcoded best models from the performance data
    best_model_per_pattern = {
        "burst": "gbdt",
        "chaotic": "gbdt",
        "growing": "var",
        "onoff": "catboost",
        "seasonal": "gbdt",
        "stepped": "gbdt"
    }
    
    print("Using hardcoded best models from performance analysis.")
    return best_model_per_pattern

def generate_system_prompt(best_models):
    """Generates the system prompt for the Gemini API."""
    prompt = """You are an expert assistant specializing in Predictive Horizontal Pod Autoscaling (PHPA).
Your task is to analyze a given time-series dataset representing pod usage or a similar metric.
Based on the characteristics and patterns observed in the **full data sample provided in the user prompt**, you must identify the underlying dataset generation pattern from the list below and recommend the most suitable forecasting model for PHPA based *only* on the provided list.

**Analysis Process:**
1. Examine the **entire provided CSV data sample**.
2. Analyze the data to infer its key characteristics (Trend, Seasonality, Events, Noise). Look for:
    - **Trend:** Is there a clear upward or downward direction over time? (e.g., growing)
    - **Seasonality:** Are there regular, repeating cycles? (e.g., seasonal)
    - **Events:** Are there sudden spikes, drops, or level shifts? Are they transient or sustained? (e.g., burst, onoff, stepped)
    - **Noise/Irregularity:** Is the data very stable, somewhat noisy, or highly unpredictable? (e.g., chaotic)
3. Explicitly describe the likely characteristics (Trend, Seasonality, Events, Noise) inferred *solely* from the **provided data**.
4. Compare these inferred characteristics against the pattern descriptions below.
5. Conclude the most likely pattern name from the list, ensuring consistency with the observed data characteristics.
6. State the recommended model associated ONLY with your identified pattern from the list.

**Known Dataset Patterns & Characteristics:**

"""
    
    # Define detailed descriptions based on generation logic
    pattern_descriptions = {
        "seasonal": "**Characteristics:** Regular, predictable cycles (e.g., daily, weekly). Data follows smooth oscillations around a relatively stable baseline. Look for repeating peaks and troughs at consistent intervals.",
        "growing": "**Characteristics:** Clear overall upward trend over time. May also include superimposed seasonal oscillations, but the increasing trend is dominant.",
        "burst": "**Characteristics:** Mostly stable baseline with sudden, short-lived, high-magnitude spikes occurring at irregular intervals. Spikes typically decay relatively quickly back towards the baseline.",
        "onoff": "**Characteristics:** Alternates sharply between two distinct, relatively flat levels (a low 'off' state and a high 'on' state). Resembles a square wave with abrupt transitions.",
        "chaotic": "**Characteristics:** Highly irregular and unpredictable. Lacks clear repeating patterns or trends. May exhibit sudden, random changes in direction, level, and variance. Often appears very noisy.",
        "stepped": "**Characteristics:** Features distinct, discrete, abrupt jumps between different flat levels (steps). The system holds a specific level for a duration before transitioning sharply to another level." 
    }

    # Sort items for consistent prompt generation
    for pattern, model in sorted(best_models.items()):
        description = pattern_descriptions.get(pattern, "No specific description available.")
        # Use pattern name directly as key, ensure description has key details
        prompt += f"- **{pattern.replace('_', ' ').title()}** ({pattern}):\n"
        prompt += f"  - {description}\n"
        prompt += f"  - *Recommended Model*: '{model}'\n\n"

    prompt += """Analyze the provided dataset sample based **only** on its full content. The filename indicates the true pattern but should primarily be used for context if needed; focus your analysis on the data itself.
Follow the **Analysis Process** described above. Provide your reasoning (inferred characteristics) before stating the final pattern.

**Output format (strict):**
Reasoning: [Brief description of inferred trend, seasonality, events, noise based on the provided data]
Pattern: [Identified Pattern Name Exactly as Listed Above, e.g., seasonal]
Recommended Model: [Recommended Model Name from the List]
"""
    return prompt

def call_gemini_api(system_prompt, user_prompt):
    """Calls the Gemini API to get the model recommendation."""
    if API_KEY == "YOUR_GEMINI_API_KEY" or not API_KEY:
        print("Error: API_KEY is not set. Please set 'YOUR_GEMINI_API_KEY' in the script.")
        return None

    headers = {
        'Content-Type': 'application/json',
    }
    params = {
        'key': API_KEY,
    }
    data = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "generationConfig": {
             "temperature": 0.1,
             "topP": 0.8,
             "topK": 40,
             "maxOutputTokens": 500  # Increased from 150 to handle longer responses
        }
    }

    response = None # Initialize response to None
    try:
        # Add a timeout to the request
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        # More robust text extraction
        if 'candidates' in result and result['candidates']:
             candidate = result['candidates'][0]
             if 'content' in candidate and 'parts' in candidate['content']:
                 full_text = "".join(part['text'] for part in candidate['content']['parts'] if 'text' in part)
                 return full_text.strip()

        # If text extraction fails, print warning and response
        print("Warning: Could not extract text from Gemini response.")
        try:
            print("Response JSON:", json.dumps(result, indent=2))
        except Exception:
            print("Response Content (non-JSON):", result)
        return None

    except requests.exceptions.Timeout:
        print("Error: API request timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        if response is not None:
            print("Response status code:", response.status_code)
            try:
                print("Response content:", response.json())
            except json.JSONDecodeError:
                 print("Response content (raw text):", response.text)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return None

def parse_llm_response(response_text):
    """Parses the Pattern, Recommended Model, and Reasoning from the LLM's response."""
    pattern = None
    model = None
    reasoning = "N/A" # Default if not found
    if not response_text:
        return pattern, model, reasoning

    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_lower = line.lower()

        if line_lower.startswith("reasoning:"):
            try:
                 # Capture potentially multi-line reasoning if needed, though unlikely with current tokens
                 reasoning = line.split(":", 1)[1].strip()
            except IndexError:
                 print(f"Warning: Could not parse reasoning from line: '{line}'")
        elif line_lower.startswith("pattern:"):
            try:
                pattern_text = line.split(":", 1)[1].strip()
                # Handle potential markdown formatting like **pattern** and normalize spaces to underscores
                pattern = pattern_text.lower().strip("* ").replace(" ", "_")
            except IndexError:
                print(f"Warning: Could not parse pattern from line: '{line}'")

        elif line_lower.startswith("recommended model:"):
            try:
                model_text = line.split(":", 1)[1].strip()
                 # Handle potential markdown or quotes
                model = model_text.lower().strip("'\"`* ")
            except IndexError:
                 print(f"Warning: Could not parse model from line: '{line}'")

    return pattern, model, reasoning

def generate_plots_and_report(results_summary):
    """
    Generates plots and an HTML report based on the benchmark results.
    """
    if not results_summary:
        print("No results to plot or report.")
        return

    # Create a timestamp for the report files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories if they don't exist
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Convert results to pandas DataFrame for easier manipulation
    results_df = pd.DataFrame(results_summary)
    
    # Filter only successful results
    success_df = results_df[results_df['status'] == 'Success'].copy()
    
    if success_df.empty:
        print("No successful results to plot or report.")
        return
    
    # Count overall pattern and model match accuracy
    pattern_accuracy = success_df['pattern_match'].mean() * 100
    model_accuracy = success_df['model_match'].mean() * 100
    
    # Group by expected pattern and calculate accuracy per pattern
    pattern_group = success_df.groupby('expected_pattern').agg({
        'pattern_match': 'mean',
        'model_match': 'mean',
        'file': 'count'
    }).reset_index()
    
    pattern_group['pattern_accuracy'] = pattern_group['pattern_match'] * 100
    pattern_group['model_accuracy'] = pattern_group['model_match'] * 100
    pattern_group.rename(columns={'file': 'samples_count'}, inplace=True)
    
    # ------- Create Plots -------
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Overall accuracy
    plt.subplot(2, 1, 1)
    metrics = ['Pattern Recognition', 'Model Recommendation']
    values = [pattern_accuracy, model_accuracy]
    plt.bar(metrics, values, color=['#3498db', '#2ecc71'])
    plt.title('Overall LLM Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)  # Set y-axis limit to 105 to leave room for text
    
    # Add percentage text on bars
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Plot 2: Accuracy by pattern
    plt.subplot(2, 1, 2)
    
    # Use different colors for each pattern
    patterns = pattern_group['expected_pattern'].tolist()
    x = range(len(patterns))
    
    # Plot pattern accuracy
    pattern_acc = pattern_group['pattern_accuracy'].tolist()
    model_acc = pattern_group['model_accuracy'].tolist()
    
    width = 0.35
    plt.bar([i - width/2 for i in x], pattern_acc, width=width, color='#3498db', label='Pattern Recognition')
    plt.bar([i + width/2 for i in x], model_acc, width=width, color='#2ecc71', label='Model Recommendation')
    
    plt.xlabel('Pattern Type')
    plt.ylabel('Accuracy (%)')
    plt.title('LLM Accuracy by Pattern')
    plt.xticks(x, patterns, rotation=45)
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(report_dir, f"accuracy_plot_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"\nPlot saved to: {plot_file}")
    
    # ------- Create HTML Report -------
    # Create a more detailed HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PHPA Benchmark Results - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .pattern-summary {{ margin-top: 30px; }}
            .accuracy-good {{ color: green; }}
            .accuracy-medium {{ color: orange; }}
            .accuracy-bad {{ color: red; }}
            .plot-container {{ margin: 30px 0; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>PHPA Benchmark Results</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <h2>Overall Summary</h2>
            <p>Total datasets tested: {len(results_df)}</p>
            <p>Successful tests: {len(success_df)} ({len(success_df)/len(results_df)*100:.1f}%)</p>
            <p>Pattern recognition accuracy: <span class="{'accuracy-good' if pattern_accuracy >= 80 else 'accuracy-medium' if pattern_accuracy >= 60 else 'accuracy-bad'}">{pattern_accuracy:.1f}%</span></p>
            <p>Model recommendation accuracy: <span class="{'accuracy-good' if model_accuracy >= 80 else 'accuracy-medium' if model_accuracy >= 60 else 'accuracy-bad'}">{model_accuracy:.1f}%</span></p>
        </div>
        
        <div class="plot-container">
            <img src="{os.path.basename(plot_file)}" alt="Accuracy Plot" style="max-width: 100%;">
        </div>
        
        <div class="pattern-summary">
            <h2>Results by Pattern</h2>
            <table>
                <tr>
                    <th>Pattern</th>
                    <th>Samples</th>
                    <th>Pattern Recognition</th>
                    <th>Model Recommendation</th>
                </tr>
    """
    
    for _, row in pattern_group.iterrows():
        pattern = row['expected_pattern']
        samples = int(row['samples_count'])
        pattern_acc = row['pattern_accuracy']
        model_acc = row['model_accuracy']
        
        html_content += f"""
                <tr>
                    <td>{pattern}</td>
                    <td>{samples}</td>
                    <td class="{'accuracy-good' if pattern_acc >= 80 else 'accuracy-medium' if pattern_acc >= 60 else 'accuracy-bad'}">{pattern_acc:.1f}%</td>
                    <td class="{'accuracy-good' if model_acc >= 80 else 'accuracy-medium' if model_acc >= 60 else 'accuracy-bad'}">{model_acc:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Status</th>
                <th>Expected Pattern</th>
                <th>LLM Pattern</th>
                <th>Pattern Match</th>
                <th>Expected Model</th>
                <th>LLM Model</th>
                <th>Model Match</th>
            </tr>
    """
    
    for _, row in results_df.iterrows():
        file = row['file']
        status = row['status']
        
        html_content += f"""
            <tr>
                <td>{file}</td>
                <td>{status}</td>
        """
        
        if status == 'Success':
            expected_pattern = row['expected_pattern']
            llm_pattern = row.get('llm_pattern', 'N/A')
            pattern_match = 'YES' if row['pattern_match'] else 'NO'
            expected_model = row['expected_model']
            llm_model = row.get('llm_model', 'N/A')
            model_match = 'YES' if row['model_match'] else 'NO'
            
            html_content += f"""
                <td>{expected_pattern}</td>
                <td>{llm_pattern}</td>
                <td>{pattern_match}</td>
                <td>{expected_model}</td>
                <td>{llm_model}</td>
                <td>{model_match}</td>
            """
        else:
            html_content += """
                <td colspan="6">N/A</td>
            """
        
        html_content += """
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Save the HTML report
    html_file = os.path.join(report_dir, f"benchmark_report_{timestamp}.html")
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {html_file}")

def preprocess_csv_content(csv_content):
    """
    Preprocess CSV content to reduce token usage:
    1. Remove "2024-" year prefix from timestamps
    2. Remove seconds portion from time
    Format: "MM-DD HH:MM, pod_counts"
    """
    lines = csv_content.strip().split('\n')
    processed_lines = []
    
    # Keep the header
    processed_lines.append(lines[0])
    
    for i in range(1, len(lines)):
        line = lines[i]
        parts = line.split(',')
        if len(parts) >= 2:
            # Process timestamp: remove year and seconds
            timestamp = parts[0]
            if timestamp.startswith('2024-'):
                # Extract date-time components
                date_time_parts = timestamp.split(' ')
                if len(date_time_parts) == 2:
                    # Remove year from date part
                    date_part = date_time_parts[0].replace('2024-', '')
                    # Remove seconds from time part (keep only HH:MM)
                    time_part = ':'.join(date_time_parts[1].split(':')[:2])
                    # Format: "MM-DD HH:MM"
                    formatted_timestamp = f"{date_part} {time_part}"
                    # Keep the value (pod_count)
                    processed_line = f"{formatted_timestamp},{parts[1]}"
                    processed_lines.append(processed_line)
                else:
                    processed_lines.append(line)
            else:
                # If not expected format, keep original
                processed_lines.append(line)
        else:
            # If line format unexpected, keep original
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

# --- Dataset Loading Functions ---

def load_existing_datasets():
    """Load existing datasets from the 1-dataset-generation directory."""
    if not os.path.exists(EXISTING_DATASET_DIR):
        print(f"Error: Existing dataset directory not found: {EXISTING_DATASET_DIR}")
        print("Please ensure you have generated datasets using the 1-dataset-generation module first.")
        return {}
    
    # Pattern mapping from filename patterns to standardized names
    pattern_mapping = {
        'seasonal': 'seasonal',
        'growing': 'growing', 
        'burst': 'burst',
        'onoff': 'onoff',
        'chaotic': 'chaotic',
        'stepped': 'stepped'
    }
    
    available_datasets = {}
    
    # Find datasets for each pattern type
    for pattern_name in pattern_mapping.keys():
        # Look for full CSV files for this pattern
        pattern_files = glob.glob(os.path.join(EXISTING_DATASET_DIR, f"{pattern_name}_*_full.csv"))
        
        if pattern_files:
            # Use the first available file for each pattern
            selected_file = pattern_files[0]
            available_datasets[pattern_name] = selected_file
            print(f"  Found {pattern_name} dataset: {os.path.basename(selected_file)}")
        else:
            print(f"  Warning: No {pattern_name} dataset found")
    
    return available_datasets

# --- End Dataset Loading Functions ---

# --- Main Execution ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PHPA Pattern Detection and Model Recommendation Benchmark')
    parser.add_argument('--generate-only', action='store_true', 
                        help='Only generate example datasets without running the benchmark')
    parser.add_argument('--test-file', type=str,
                        help='Path to a specific CSV file to test (skips example generation)')
    args = parser.parse_args()

    # If a specific test file is provided, use that instead of generating examples
    if args.test_file:
        if not os.path.exists(args.test_file):
            print(f"Error: Specified test file not found: {args.test_file}")
            exit(1)
            
        print(f"Testing specific CSV file: {args.test_file}")
        # Get the best models for all patterns
        best_models_map = analyze_best_models(CSV_FILE_PATH)
        system_prompt = generate_system_prompt(best_models_map)
        
        # Read the file content
        try:
            with open(args.test_file, 'r') as f:
                csv_content = f.read()
            if not csv_content.strip():
                print("Error: The CSV file is empty.")
                exit(1)
                
            # Preprocess CSV content to reduce token usage
            processed_csv_content = preprocess_csv_content(csv_content)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            exit(1)
            
        # Create user prompt with the CSV content
        user_prompt = f"""Analyze the following time-series data sample and recommend a PHPA model based on the system instructions.
Dataset Filename: {os.path.basename(args.test_file)}

**Full Dataset Content:**
```csv
{processed_csv_content}
```

Identify the pattern and state the recommended model.
"""

        print("Calling Gemini API...")
        llm_response = call_gemini_api(system_prompt, user_prompt)
        
        if llm_response:
            llm_pattern, llm_model, llm_reasoning = parse_llm_response(llm_response)
            
            print("\n--- Pattern Detection Results ---")
            print(f"LLM Identified Pattern: {llm_pattern}")
            print(f"LLM Recommended Model: {llm_model}")
            print(f"LLM Reasoning: {llm_reasoning}")
            
            # You can optionally plot the data here
            try:
                df = pd.read_csv(args.test_file)
                if 'timestamp' in df.columns and len(df) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(df['pod_count'])
                    plt.title(f"Pattern: {llm_pattern} - Recommended Model: {llm_model}")
                    plt.xlabel("Time")
                    plt.ylabel("Pod Count")
                    plt.grid(True)
                    
                    # Save the plot
                    plot_file = f"results/plots/{os.path.basename(args.test_file).replace('.csv', '_analysis.png')}"
                    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
                    plt.savefig(plot_file)
                    print(f"\nPlot saved to: {plot_file}")
            except Exception as e:
                print(f"Could not generate plot: {e}")
                
        else:
            print("Error: Failed to get response from Gemini API")
            
        # Exit after processing the single file
        exit(0)
    
    # Load best models for pattern analysis
    if not args.generate_only:
        print("Analyzing patterns and loading existing datasets...")
        best_models_map = analyze_best_models(CSV_FILE_PATH)
        
        if not best_models_map:
            print("Exiting due to errors in best models analysis.")
            exit(1)
    
    # If only generating examples, exit here (this option is no longer relevant)
    if args.generate_only:
        print("Note: --generate-only flag is deprecated. Using existing datasets from 1-dataset-generation module.")
    
    print("\nGenerating System Prompt...")
    system_prompt = generate_system_prompt(best_models_map)

    # Load existing datasets instead of generating new ones
    print(f"\nLoading existing datasets from {EXISTING_DATASET_DIR}...")
    available_datasets = load_existing_datasets()
    
    if not available_datasets:
        print("Error: No datasets found. Please run the 1-dataset-generation module first.")
        exit(1)
    
    # Convert to list of file paths for compatibility with existing code
    sample_files = list(available_datasets.values())
    print(f"Found {len(sample_files)} dataset files.")

    # Normal benchmark mode continues from here
    print(f"\nRunning benchmark on {len(sample_files)} generated example dataset(s)..." )
    print("-" * 50)

    correct_pattern_count = 0
    correct_model_count = 0
    results_summary = []

    for i, dataset_path in enumerate(sample_files):
        dataset_basename = os.path.basename(dataset_path)
        print(f"Processing ({i+1}/{len(sample_files)}): {dataset_basename}")

        # Extract expected pattern from filename (format: pattern_xxx_full.csv)
        expected_pattern = dataset_basename.split('_')[0]

        # We get the expected model from the potentially filtered map
        expected_model = best_models_map.get(expected_pattern)

        if not expected_model:
            print(f"  Internal Warning: No best model found for pattern '{expected_pattern}' in '{CSV_FILE_PATH}'. Skipping.")
            results_summary.append({'file': dataset_basename, 'status': 'Skipped (No expected model)'})
            continue

        # Read the *entire* content of the small example CSV
        try:
            with open(dataset_path, 'r') as f:
                csv_content = f.read()
            if not csv_content.strip():
                 print("  Warning: Sample content is empty. Skipping.")
                 results_summary.append({'file': dataset_basename, 'status': 'Skipped (Empty content)'})
                 continue
                 
            # Preprocess CSV content to reduce token usage
            processed_csv_content = preprocess_csv_content(csv_content)
        except Exception as e:
            print(f"  Error reading sample content from {dataset_path}: {e}. Skipping.")
            results_summary.append({'file': dataset_basename, 'status': 'Skipped (Read error)'})
            continue
            
        # Create the user prompt with the processed CSV content
        user_prompt = f"""Analyze the following time-series data sample and recommend a PHPA model based on the system instructions.
Dataset Filename: {dataset_basename}

**Full Dataset Content:**
```csv
{processed_csv_content}
```

Identify the pattern and state the recommended model.
"""

        print("  Calling Gemini API...")
        llm_response = call_gemini_api(system_prompt, user_prompt)

        llm_pattern = None
        llm_model = None
        pattern_match = False
        model_match = False
        llm_reasoning = "N/A" # Initialize reasoning variable

        if llm_response:
            print(f"  LLM Raw Response: ~{len(llm_response)} chars")
            # Capture reasoning along with pattern and model
            llm_pattern, llm_model, llm_reasoning = parse_llm_response(llm_response)

            print(f"  Expected Pattern: {expected_pattern.capitalize()}")
            if llm_pattern:
                print(f"  LLM Identified Pattern: {llm_pattern.capitalize()}")
                if llm_pattern == expected_pattern:
                    print("    Pattern Match: YES")
                    correct_pattern_count += 1
                    pattern_match = True
                else:
                    print("    Pattern Match: NO")
                    # Log reasoning on mismatch
                    print(f"      LLM Reasoning: {llm_reasoning}") 
            else:
                print("  LLM Identified Pattern: Not Found in Response")
                print(f"      LLM Reasoning: {llm_reasoning}") # Log if pattern missing

            print(f"\n  Expected Model: {expected_model}")
            if llm_model:
                print(f"  LLM Recommended Model: {llm_model}")
                if llm_model == expected_model:
                    print("    Model Match: YES")
                    correct_model_count += 1
                    model_match = True
                else:
                    print("    Model Match: NO")
            else:
                print("  LLM Recommended Model: Not Found in Response")

            results_summary.append({
                'file': dataset_basename,
                'status': 'Success',
                'expected_pattern': expected_pattern,
                'llm_pattern': llm_pattern,
                'pattern_match': pattern_match,
                'expected_model': expected_model,
                'llm_model': llm_model,
                'model_match': model_match
            })

        else:
            print("  LLM Response: Failed or Empty")
            results_summary.append({'file': dataset_basename, 'status': 'API Error/Empty'})

        print("-" * 50)

        # Add delay if testing multiple patterns and it's not the last file
        if TEST_SINGLE_PATTERN is None and i < len(sample_files) - 1:
            print(f"Waiting for {DELAY_BETWEEN_CALLS} seconds before next API call...")
            time.sleep(DELAY_BETWEEN_CALLS)

    # --- Final Summary and Reports ---
    print("\n--- Benchmark Summary ---")
    tested_files_count = len(sample_files)
    if tested_files_count > 0:
         # Calculate accuracy safely, avoiding division by zero
         pattern_accuracy = (correct_pattern_count / tested_files_count) * 100 if tested_files_count > 0 else 0
         model_accuracy = (correct_model_count / tested_files_count) * 100 if tested_files_count > 0 else 0
         print(f"Total Datasets Tested: {tested_files_count}")
         print(f"Correct Pattern Identifications: {correct_pattern_count} ({pattern_accuracy:.2f}%)")
         print(f"Correct Model Recommendations: {correct_model_count} ({model_accuracy:.2f}%)")

         print("\nDetailed Results:")
         for result in results_summary:
             print(f"- File: {result['file']}")
             if result['status'] == 'Success':
                 p_match_str = "YES" if result['pattern_match'] else "NO"
                 m_match_str = "YES" if result['model_match'] else "NO"
                 # Ensure these fields exist before accessing
                 llm_p = result.get('llm_pattern', 'N/A')
                 llm_m = result.get('llm_model', 'N/A')
                 print(f"    Status: {result['status']}")
                 print(f"    Pattern: Expected='{result['expected_pattern']}', LLM='{llm_p}' (Match: {p_match_str})")
                 print(f"    Model:   Expected='{result['expected_model']}', LLM='{llm_m}' (Match: {m_match_str})")
             else:
                 print(f"    Status: {result['status']}") # Print status for non-success cases

         # Generate plots and report
         print("\nGenerating plots and HTML report...")
         generate_plots_and_report(results_summary)

    else:
        print("No datasets were successfully tested.")
    print("-" * 25) 