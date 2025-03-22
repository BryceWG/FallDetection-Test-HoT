#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_lstm import FallDetectionLSTM

def load_model_config(model_dir):
    """Load model configuration and parameters"""
    summary_file = os.path.join(model_dir, 'training_summary.json')
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Model configuration file not found: {summary_file}")
        
    with open(summary_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Extract model parameters
    model_params = config['parameters']['model_params']
    norm_params = config['parameters']['normalization_params']
    
    return model_params, norm_params

def load_trained_model(model_dir, model_params, device):
    """Load trained model"""
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model instance
    model = FallDetectionLSTM(
        input_dim=51,  # 17 keypoints * 3 coordinates
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def load_pose_data(pose_file):
    """Load 3D pose data"""
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    
    return np.load(pose_file, allow_pickle=True)['reconstruction']

def create_sequences(pose_data, seq_length=30, stride=10):
    """Split pose data into sequences"""
    sequences = []
    frame_indices = []
    total_frames = len(pose_data)
    
    for start_idx in range(0, total_frames - seq_length + 1, stride):
        end_idx = start_idx + seq_length
        if end_idx > total_frames:
            break
            
        # Get sequence data and flatten
        sequence = pose_data[start_idx:end_idx]
        flattened_sequence = sequence.reshape(seq_length, -1)
        sequences.append(flattened_sequence)
        
        # Record corresponding frame indices
        frame_indices.append((start_idx, end_idx))
    
    return np.array(sequences), frame_indices

def normalize_sequences(sequences, mean, std):
    """Normalize sequence data"""
    return (sequences - mean) / std

def predict_sequences(model, sequences, device, batch_size=32):
    """Make predictions on sequences"""
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            outputs = model(batch_tensor)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)

def plot_predictions(frame_indices, predictions, save_path=None):
    """Plot prediction results"""
    # Create complete frame index list
    all_frames = list(range(frame_indices[-1][1]))
    all_predictions = np.zeros(len(all_frames))
    frame_counts = np.zeros(len(all_frames))
    
    # Accumulate predictions for each sequence
    for (start, end), pred in zip(frame_indices, predictions):
        all_predictions[start:end] += pred
        frame_counts[start:end] += 1
    
    # Calculate average predictions
    mask = frame_counts > 0
    all_predictions[mask] /= frame_counts[mask]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(all_frames, all_predictions, '-b', label='Fall Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    plt.fill_between(all_frames, all_predictions, 0.5,
                    where=(all_predictions >= 0.5),
                    color='red', alpha=0.3, label='Fall Region')
    plt.xlabel('Frame Number')
    plt.ylabel('Fall Probability')
    plt.title('Fall Detection Results')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D pose data using trained model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Model directory containing best_model.pth and training_summary.json')
    parser.add_argument('--pose_file', type=str, required=True,
                       help='Path to 3D pose file (.npz format)')
    parser.add_argument('--output_dir', type=str, default='./train/evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--seq_length', type=int, default=30,
                       help='Sequence length')
    parser.add_argument('--stride', type=int, default=10,
                       help='Sliding window stride')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID')
    args = parser.parse_args()
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model configuration and parameters
        print("Loading model configuration...")
        model_params, norm_params = load_model_config(args.model_dir)
        
        # Load trained model
        print("Loading model...")
        model = load_trained_model(args.model_dir, model_params, device)
        
        # Load pose data
        print("Loading pose data...")
        pose_data = load_pose_data(args.pose_file)
        
        # Create sequences
        print("Creating sequences...")
        sequences, frame_indices = create_sequences(
            pose_data,
            seq_length=args.seq_length,
            stride=args.stride
        )
        
        # Normalize data
        print("Normalizing data...")
        normalized_sequences = normalize_sequences(
            sequences,
            np.array(norm_params['mean']),
            np.array(norm_params['std'])
        )
        
        # Make predictions
        print("Making predictions...")
        predictions = predict_sequences(model, normalized_sequences, device)
        
        # Save prediction results
        output_base = os.path.splitext(os.path.basename(args.pose_file))[0]
        results = {
            'frame_indices': frame_indices,
            'predictions': predictions.tolist()
        }
        
        results_file = os.path.join(args.output_dir, f'{output_base}_predictions.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot predictions
        plot_path = os.path.join(args.output_dir, f'{output_base}_predictions.png')
        plot_predictions(frame_indices, predictions, save_path=plot_path)
        
        print(f"Evaluation complete! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()