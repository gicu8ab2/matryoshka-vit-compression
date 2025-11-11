#!/usr/bin/env python3
"""
Extract accuracy and parameter count from TensorBoard event files.
"""
import os
import re
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_dim_from_dirname(dirname):
    """Extract embedding dimension from directory name."""
    match = re.search(r'dim(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None

def extract_metrics_from_events(event_dir):
    """Extract accuracy and parameters from TensorBoard events."""
    ea = event_accumulator.EventAccumulator(event_dir)
    ea.Reload()
    
    results = {}
    
    # Get text summaries which contain the model info
    if 'Model_Info/text_summary' in ea.Tags()['tensors']:
        text_events = ea.Tensors('Model_Info/text_summary')
        if text_events:
            # Get the latest event
            latest_event = text_events[-1]
            text_content = latest_event.tensor_proto.string_val[0].decode('utf-8')
            
            # Parse accuracy
            acc_match = re.search(r'Accuracy:\s*([\d.]+)%', text_content)
            if acc_match:
                results['accuracy'] = float(acc_match.group(1))
            
            # Parse parameters
            param_match = re.search(r'Parameters:\s*([\d,]+)', text_content)
            if param_match:
                param_str = param_match.group(1).replace(',', '')
                results['parameters'] = int(param_str)
    
    return results

def main():
    # Base directory containing the eval subdirectories
    base_dir = Path('/home/ubuntu/robert.taylor/vision-transformers-cifar10/runs/MRL_evals')
    
    # Collect results
    data = []
    
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and 'eval_' in subdir.name:
            dim = extract_dim_from_dirname(subdir.name)
            if dim is None:
                continue
            
            print(f"Processing {subdir.name}...")
            
            try:
                metrics = extract_metrics_from_events(str(subdir))
                if metrics:
                    data.append({
                        'Embedding Dimension': dim,
                        'Accuracy (%)': metrics.get('accuracy', 'N/A'),
                        'Parameters': metrics.get('parameters', 'N/A')
                    })
            except Exception as e:
                print(f"  Error processing {subdir.name}: {e}")
    
    # Create DataFrame and sort by embedding dimension
    df = pd.DataFrame(data)
    df = df.sort_values('Embedding Dimension')
    
    # Format parameters with commas
    if 'Parameters' in df.columns:
        df['Parameters'] = df['Parameters'].apply(
            lambda x: f"{x:,}" if isinstance(x, int) else x
        )
    
    # Print table
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    output_file = base_dir / 'results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
