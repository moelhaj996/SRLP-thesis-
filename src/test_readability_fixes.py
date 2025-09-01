#!/usr/bin/env python3
"""Test readability fixes for problematic figures."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set headless backend
os.environ['MPLBACKEND'] = 'Agg'

# Create realistic test data
np.random.seed(42)
test_data = []

for provider in ['gpt4', 'claude3', 'gemini']:
    for strategy in ['srlp', 'cot', 'tot', 'react']:
        for i in range(25):  # 25 samples per combination = 300 total
            # SRLP gets higher PQS
            if strategy == 'srlp':
                pqs = np.random.normal(85, 8)
                cost = np.random.uniform(0.3, 0.7)
                exec_time = np.random.uniform(3.8, 4.2)
            else:
                pqs = np.random.normal(75, 10)
                cost = np.random.uniform(0.1, 0.5)
                exec_time = np.random.uniform(3.9, 4.1)
            
            test_data.append({
                'provider': provider,
                'strategy': strategy,
                'pqs': max(0, min(100, pqs)),
                'cost_usd': max(0.01, cost),
                'execution_time': max(0.1, exec_time),
            })

df = pd.DataFrame(test_data)

# Output directory
output_dir = Path('test_readability_fixes')
output_dir.mkdir(exist_ok=True)

print(f"📊 Generated {len(df)} test samples")

# ===== FIX 1: PROVIDER PERFORMANCE BAR CHART =====
def test_provider_performance():
    """Test the improved provider performance figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    providers = ['gpt4', 'claude3', 'gemini']
    provider_labels = ['GPT-4', 'Claude-3', 'Gemini']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Time analysis
    times = []
    time_stds = []
    for provider in providers:
        provider_data = df[df['provider'] == provider]
        times.append(provider_data['execution_time'].mean())
        time_stds.append(provider_data['execution_time'].std())
    
    bars1 = ax1.bar(provider_labels, times, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=0.8)
    
    ax1.errorbar(provider_labels, times, yerr=time_stds, fmt='none', 
                color='black', capsize=5, capthick=1.2)
    
    # READABILITY FIX: Better rotation and larger fonts
    ax1.set_ylabel('Average Execution Time (s)', fontweight='bold', fontsize=18)
    ax1.set_xlabel('AI Provider', fontweight='bold', fontsize=18)
    ax1.set_title('Average Execution Time by Provider', fontweight='bold')
    ax1.tick_params(axis='x', labelsize=18, rotation=60)  # 60° rotation + larger font
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylim(0, max(times) * 1.5)  # Extra space for rotated labels
    
    # Add value labels
    for bar, time, std in zip(bars1, times, time_stds):
        height = bar.get_height()
        label_y = height + std + (max(times) * 0.02)
        ax1.text(bar.get_x() + bar.get_width()/2, label_y,
                f'{time:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Cost analysis
    costs = []
    cost_stds = []
    for provider in providers:
        provider_data = df[df['provider'] == provider]
        costs.append(provider_data['cost_usd'].sum())  # Total cost
        cost_stds.append(provider_data['cost_usd'].std())
    
    bars2 = ax2.bar(provider_labels, costs, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.8)
    
    ax2.errorbar(provider_labels, costs, yerr=cost_stds, fmt='none',
                color='black', capsize=5, capthick=1.2)
    
    # READABILITY FIX: Better rotation and larger fonts
    ax2.set_ylabel('Total Cost (USD)', fontweight='bold', fontsize=18)
    ax2.set_xlabel('AI Provider', fontweight='bold', fontsize=18)
    ax2.set_title('Total Cost by Provider', fontweight='bold')
    ax2.tick_params(axis='x', labelsize=18, rotation=60)  # 60° rotation + larger font
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_ylim(0, max(costs) * 1.5)  # Extra space for rotated labels
    
    # Add value labels
    for bar, cost, std in zip(bars2, costs, cost_stds):
        height = bar.get_height()
        label_y = height + std + (max(costs) * 0.02)
        ax2.text(bar.get_x() + bar.get_width()/2, label_y,
                f'${cost:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_provider_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fixed_provider_performance.pdf", bbox_inches='tight')
    plt.close()
    
    print("✅ Fixed provider performance figure saved")

# Run tests
print("🔧 Testing readability fixes...")
test_provider_performance()

print(f"\n📁 Check improved figures in: {output_dir.absolute()}")

# ===== FIX 2: COST-QUALITY SCATTER PLOT =====
def test_cost_quality():
    """Test the improved cost-quality scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Aggregate by provider and strategy
    aggregated = []
    for provider in df['provider'].unique():
        for strategy in df['strategy'].unique():
            subset = df[(df['provider'] == provider) & (df['strategy'] == strategy)]
            if len(subset) > 0:
                aggregated.append({
                    'provider': provider,
                    'strategy': strategy,
                    'pqs_mean': subset['pqs'].mean(),
                    'cost_mean': subset['cost_usd'].mean(),
                    'n': len(subset)
                })
    
    # Extract data for plotting
    x_costs = [d['cost_mean'] for d in aggregated]
    y_pqs = [d['pqs_mean'] for d in aggregated]
    sizes = [50 + d['n'] * 2 for d in aggregated]  # Larger dots
    
    # Colors
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
    
    # Calculate medians for smart positioning
    median_cost = np.median(x_costs)
    median_pqs = np.median(y_pqs)
    
    for i, data in enumerate(aggregated):
        color = colors[i % len(colors)]
        ax.scatter(data['cost_mean'], data['pqs_mean'], s=sizes[i], 
                  color=color, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # READABILITY FIX: Shorter abbreviations
        provider_abbrev = {'gpt4': 'G4', 'claude3': 'C3', 'gemini': 'GM'}
        strategy_abbrev = {'srlp': 'SRLP', 'cot': 'CoT', 'tot': 'ToT', 'react': 'ReAct'}
        
        provider_short = provider_abbrev.get(data['provider'].lower(), data['provider'][:2].upper())
        strategy_short = strategy_abbrev.get(data['strategy'].lower(), data['strategy'][:3].upper())
        label_text = f"{provider_short}-{strategy_short}"
        
        # READABILITY FIX: Smart quadrant-based positioning
        x_pos = data['cost_mean']
        y_pos = data['pqs_mean']
        
        if x_pos <= median_cost and y_pos >= median_pqs:  # Top-left
            offset_x, offset_y = -60, 20
            ha = 'right'
        elif x_pos > median_cost and y_pos >= median_pqs:  # Top-right
            offset_x, offset_y = 30, 20
            ha = 'left'
        elif x_pos <= median_cost and y_pos < median_pqs:  # Bottom-left
            offset_x, offset_y = -60, -30
            ha = 'right'
        else:  # Bottom-right
            offset_x, offset_y = 30, -30
            ha = 'left'
        
        # Add small variation to prevent exact overlap
        offset_x += (i % 3) * 8
        offset_y += (i % 3) * 8
        
        ax.annotate(label_text, 
                   (data['cost_mean'], data['pqs_mean']),
                   xytext=(offset_x, offset_y), textcoords='offset points', 
                   fontsize=14, fontweight='bold', ha=ha,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, 
                            edgecolor='gray', linewidth=0.8),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.9))
    
    # Reference lines
    ax.axvline(x=median_cost, color='gray', linestyle=':', alpha=0.6, label='Median Cost')
    ax.axhline(y=median_pqs, color='gray', linestyle=':', alpha=0.6, label='Median PQS')
    
    # READABILITY FIX: Larger fonts and thicker Pareto frontier
    ax.set_xlabel('Mean Cost (USD)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean PQS', fontsize=16, fontweight='bold')
    ax.set_title('Cost–Quality Trade-off by Provider and Strategy', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(x_costs) * 1.1)
    ax.grid(True, axis='y', alpha=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_cost_quality.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fixed_cost_quality.pdf", bbox_inches='tight')
    plt.close()
    
    print("✅ Fixed cost-quality scatter plot saved")

# Add the cost-quality test to the execution
test_cost_quality()

print("🎯 Key improvements:")
print("   • Provider labels: 60° rotation + 18pt font")
print("   • Cost-quality: Smart quadrant positioning + shorter labels") 
print("   • All: Larger fonts, better spacing, cleaner layout")
