#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def extract_dim_from_dirname(dirname):
    match = re.search(r'dim(\d+)', dirname)
    if match:
        return int(match.group(1))
    return None

def extract_metrics_from_events(event_dir):
    ea = event_accumulator.EventAccumulator(event_dir)
    ea.Reload()
    
    results = {}
    
    if 'Model_Info/text_summary' in ea.Tags()['tensors']:
        text_events = ea.Tensors('Model_Info/text_summary')
        if text_events:

            latest_event = text_events[-1]
            text_content = latest_event.tensor_proto.string_val[0].decode('utf-8')
            
            acc_match = re.search(r'Accuracy:\s*([\d.]+)%', text_content)
            if acc_match:
                results['accuracy'] = float(acc_match.group(1))
            
            param_match = re.search(r'Parameters:\s*([\d,]+)', text_content)
            if param_match:
                param_str = param_match.group(1).replace(',', '')
                results['parameters'] = int(param_str)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard results for Matryoshka-ViT runs")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], help="dataset whose runs to read (cifar10 or cifar100)")
    parser.add_argument("--nested", action="store_true", help="use nested MRL eval runs directory instead of the default")
    args = parser.parse_args()

    if args.nested:
        base_dir = Path(f'/home/ubuntu/robert.taylor/matryoshka-vit-compression/runs/MRL_nested_evals_{args.dataset}')
    else:
        base_dir = Path(f'/home/ubuntu/robert.taylor/matryoshka-vit-compression/runs/MRL_evals_{args.dataset}')
    
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
    
    if not data:
        print("No evaluation results found in", base_dir)
        return

    df = pd.DataFrame(data)
    df = df.sort_values('Embedding Dimension')
    
    if 'Parameters' in df.columns:
        df['Parameters'] = df['Parameters'].apply(
            lambda x: f"{x:,}" if isinstance(x, int) else x
        )
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    output_file = base_dir / 'results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
