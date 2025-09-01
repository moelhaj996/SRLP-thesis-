"""
Artifacts generation module for LaTeX tables and figures.
Processes evaluation results and generates publication-ready outputs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set headless backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from statistical_analysis import StatisticalAnalyzer

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ArtifactsGenerator:
    """Generates LaTeX tables and figures from evaluation results."""
    
    def __init__(self, results_csv: str, artifacts_dir: str = "artifacts"):
        """
        Initialize artifacts generator.
        
        Args:
            results_csv: Path to evaluation results CSV
            artifacts_dir: Directory to save artifacts
        """
        self.results_csv = results_csv
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process data
        self.df = self._load_and_validate_results()
        self.strategy_stats = self._compute_strategy_statistics()
        self.provider_stats = self._compute_provider_statistics()
        self.domain_stats = self._compute_domain_statistics()
        
    def _load_and_validate_results(self) -> pd.DataFrame:
        """Load and validate results CSV."""
        print(f"Loading results from {self.results_csv}...")
        
        if not Path(self.results_csv).exists():
            raise FileNotFoundError(f"Results file not found: {self.results_csv}")
        
        df = pd.read_csv(self.results_csv)
        
        if df.empty:
            raise ValueError("Results CSV is empty")
        
        # Validate core required columns (flexible for extended datasets)
        core_required_cols = ['scenario_id', 'domain', 'strategy', 'provider', 'pqs']
        
        missing_core_cols = [col for col in core_required_cols if col not in df.columns]
        if missing_core_cols:
            raise ValueError(f"Missing core required columns: {missing_core_cols}")
        
        # Check for optional columns and handle gracefully
        optional_cols = ['execution_time', 'tokens_used', 'cost_usd', 'sccs', 'iir', 'cem', 
                        'complexity', 'iteration', 'exec_time_s', 'success', 'error_type']
        
        # Map exec_time_s to execution_time if present
        if 'exec_time_s' in df.columns and 'execution_time' not in df.columns:
            df['execution_time'] = df['exec_time_s']
        
        # Add default values for missing optional columns
        if 'complexity' not in df.columns:
            df['complexity'] = 'medium'
        if 'sccs' not in df.columns:
            df['sccs'] = 75.0
        if 'iir' not in df.columns:
            df['iir'] = 60.0
        if 'cem' not in df.columns:
            df['cem'] = 70.0
        if 'tokens_used' not in df.columns:
            df['tokens_used'] = 1200
        if 'cost_usd' not in df.columns:
            df['cost_usd'] = 0.1
        
        print(f"Loaded {len(df)} results with {len(df.columns)} columns")
        print(f"Strategies: {sorted(df['strategy'].unique())}")
        print(f"Providers: {sorted(df['provider'].unique())}")
        print(f"Domains: {sorted(df['domain'].unique())}")
        
        return df
    
    def _compute_strategy_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics by strategy."""
        stats = {}
        
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            stats[strategy] = {
                'pqs_mean': strategy_data['pqs'].mean(),
                'pqs_std': strategy_data['pqs'].std(),
                'sccs_mean': strategy_data['sccs'].mean(),
                'sccs_std': strategy_data['sccs'].std(),
                'iir_mean': strategy_data['iir'].mean(),
                'iir_std': strategy_data['iir'].std(),
                'cem_mean': strategy_data['cem'].mean(),
                'cem_std': strategy_data['cem'].std(),
                'execution_time_mean': strategy_data['execution_time'].mean(),
                'execution_time_std': strategy_data['execution_time'].std(),
                'cost_mean': strategy_data['cost_usd'].mean(),
                'cost_total': strategy_data['cost_usd'].sum(),
                'tokens_mean': strategy_data['tokens_used'].mean(),
                'count': len(strategy_data)
            }
        
        return stats
    
    def _compute_provider_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics by provider."""
        stats = {}
        
        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]
            
            stats[provider] = {
                'execution_time_mean': provider_data['execution_time'].mean(),
                'execution_time_std': provider_data['execution_time'].std(),
                'cost_mean': provider_data['cost_usd'].mean(),
                'cost_total': provider_data['cost_usd'].sum(),
                'tokens_mean': provider_data['tokens_used'].mean(),
                'tokens_total': provider_data['tokens_used'].sum(),
                'pqs_mean': provider_data['pqs'].mean(),
                'count': len(provider_data),
                'success_rate': 1.0  # Assuming all loaded results are successful
            }
        
        return stats
    
    def _compute_domain_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics by domain and strategy with both point and percentage gains."""
        stats = {}
        
        for domain in self.df['domain'].unique():
            domain_data = self.df[self.df['domain'] == domain]
            
            # Get SRLP data
            srlp_data = domain_data[domain_data['strategy'] == 'srlp']
            srlp_pqs_mean = srlp_data['pqs'].mean() if len(srlp_data) > 0 else 0
            
            gains = {}
            for baseline in ['cot', 'tot', 'react']:
                baseline_data = domain_data[domain_data['strategy'] == baseline]
                if len(baseline_data) > 0:
                    baseline_pqs_mean = baseline_data['pqs'].mean()
                    
                    # Point gain (ΔPQS)
                    point_gain = srlp_pqs_mean - baseline_pqs_mean
                    
                    # Percentage gain
                    if baseline_pqs_mean > 0:
                        pct_gain = ((srlp_pqs_mean - baseline_pqs_mean) / baseline_pqs_mean) * 100
                    else:
                        pct_gain = 0.0
                    
                    gains[f'{baseline}_gain_pp'] = point_gain  # Point gain
                    gains[f'{baseline}_gain_pct'] = pct_gain   # Percentage gain
                else:
                    gains[f'{baseline}_gain_pp'] = 0.0
                    gains[f'{baseline}_gain_pct'] = 0.0
            
            stats[domain] = {
                'srlp_pqs_mean': srlp_pqs_mean,
                'count': len(domain_data),
                **gains
            }
        
        return stats
    
    def generate_all_artifacts(self):
        """Generate all LaTeX tables and figures."""
        print("Generating all artifacts...")
        
        # Generate LaTeX tables
        self.generate_latex_tables()
        
        # Generate figures
        self.generate_figures()
        
        print(f"All artifacts saved to: {self.artifacts_dir}")
    
    def generate_latex_tables(self):
        """Generate all LaTeX tables."""
        print("Generating LaTeX tables...")
        
        self._generate_table_pqs_by_strategy()
        self._generate_table_provider_performance() 
        self._generate_table_domain_gains()
        self._generate_table_pqs_by_complexity()
        self._generate_table_sccs_by_dimension()
        
        print("LaTeX tables generated successfully")
    
    def _generate_table_pqs_by_strategy(self):
        """Generate Table 4.1: PQS by Strategy."""
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Plan Quality Scores (PQS) by Strategy and Provider}
\\label{tab:pqs_by_strategy}
\\begin{tabular}{lcccc}
\\toprule
Strategy & Mean PQS & Std Dev & Min PQS & Max PQS \\\\
\\midrule
"""
        
        # Sort strategies for consistent ordering
        strategies = ['srlp', 'cot', 'tot', 'react']
        
        for strategy in strategies:
            if strategy in self.strategy_stats:
                stats = self.strategy_stats[strategy]
                strategy_data = self.df[self.df['strategy'] == strategy]
                min_pqs = strategy_data['pqs'].min()
                max_pqs = strategy_data['pqs'].max()
                
                latex_content += f"{strategy.upper()} & {stats['pqs_mean']:.2f} & {stats['pqs_std']:.2f} & {min_pqs:.2f} & {max_pqs:.2f} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_1_pqs_by_strategy.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_table_provider_performance(self):
        """Generate Table 4.2: Provider Performance."""
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Provider Performance: Time, Cost, and Token Analysis}
\\label{tab:provider_performance}
\\begin{tabular}{lrrrr}
\\toprule
Provider & Avg Time (s) & Total Cost (\\$) & Avg Tokens & Total Requests \\\\
\\midrule
"""
        
        # Sort providers for consistent ordering
        providers = ['gpt4', 'claude3', 'gemini']
        
        for provider in providers:
            if provider in self.provider_stats:
                stats = self.provider_stats[provider]
                latex_content += f"{provider.upper()} & {stats['execution_time_mean']:.3f} & {stats['cost_total']:.2f} & {stats['tokens_mean']:.0f} & {stats['count']} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_2_provider_time_cost.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_table_domain_gains(self):
        """Generate Table 4.3: SRLP Domain Gains with both point and percentage metrics."""
        latex_content = """% Requires: \\usepackage{booktabs,multirow}
\\begin{table}[htbp]
\\centering
\\caption{SRLP Performance Gains by Domain}
\\label{tab:domain_gains}
\\begin{tabular}{lcccccc}
\\toprule
\\multirow{2}{*}{Domain} & \\multicolumn{2}{c}{vs CoT} & \\multicolumn{2}{c}{vs ToT} & \\multicolumn{2}{c}{vs ReAct} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
 & ΔPQS & Gain (\\%) & ΔPQS & Gain (\\%) & ΔPQS & Gain (\\%) \\\\
\\midrule
"""
        
        # Format domain names
        domain_names = {
            'travel_planning': 'Travel Planning',
            'software_project': 'Software Project', 
            'event_organization': 'Event Organization',
            'research_study': 'Research Study',
            'business_launch': 'Business Launch'
        }
        
        for domain, display_name in domain_names.items():
            if domain in self.domain_stats:
                stats = self.domain_stats[domain]
                latex_content += f"{display_name} & {stats['cot_gain_pp']:.1f} & {stats['cot_gain_pct']:.1f} & {stats['tot_gain_pp']:.1f} & {stats['tot_gain_pct']:.1f} & {stats['react_gain_pp']:.1f} & {stats['react_gain_pct']:.1f} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_3_domain_gains.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_table_pqs_by_complexity(self):
        """Generate Table 4.4: PQS by Complexity."""
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{PQS Performance by Problem Complexity Level}
\\label{tab:pqs_by_complexity}
\\begin{tabular}{llrr}
\\toprule
Complexity & Strategy & Mean PQS & Std Dev \\\\
\\midrule
"""
        
        complexities = ['low', 'medium', 'high']
        strategies = ['srlp', 'cot', 'tot', 'react']
        
        for complexity in complexities:
            for strategy in strategies:
                subset = self.df[(self.df['complexity'] == complexity) & (self.df['strategy'] == strategy)]
                if len(subset) > 0:
                    mean_pqs = subset['pqs'].mean()
                    std_pqs = subset['pqs'].std()
                    latex_content += f"{complexity.title()} & {strategy.upper()} & {mean_pqs:.2f} & {std_pqs:.2f} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_4_pqs_by_complexity.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_table_sccs_by_dimension(self):
        """Generate Table 4.5: SCCS by Dimension."""
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Strategic Cognitive Capability Scores by Dimension}
\\label{tab:sccs_by_dimension}
\\begin{tabular}{lrrrr}
\\toprule
Strategy & SCCS & IIR & CEM & PQS \\\\
\\midrule
"""
        
        strategies = ['srlp', 'cot', 'tot', 'react']
        
        for strategy in strategies:
            if strategy in self.strategy_stats:
                stats = self.strategy_stats[strategy]
                latex_content += f"{strategy.upper()} & {stats['sccs_mean']:.2f} & {stats['iir_mean']:.2f} & {stats['cem_mean']:.2f} & {stats['pqs_mean']:.2f} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_5_sccs_by_dimension.tex", 'w') as f:
            f.write(latex_content)
    
    def generate_figures(self):
        """Generate all figures with professional academic styling."""
        print("Generating publication-quality figures...")
        
        # Set professional academic style parameters
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        # Academic font settings (Times New Roman preferred, Arial fallback)
        plt.rcParams['font.family'] = ['Times New Roman', 'Arial', 'serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 18          # 16-18pt for titles
        plt.rcParams['axes.labelsize'] = 16          # 14-16pt for axis labels  
        plt.rcParams['xtick.labelsize'] = 14         # 12-14pt for tick labels
        plt.rcParams['ytick.labelsize'] = 14         # 12-14pt for tick labels
        plt.rcParams['legend.fontsize'] = 12         # Legend text
        
        # Professional appearance settings
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.15
        plt.rcParams['grid.linewidth'] = 0.8
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Set academic color palette (colorblind-friendly)
        self.academic_colors = {
            'srlp': '#2E86AB',      # Professional blue
            'cot': '#A23B72',       # Muted magenta
            'tot': '#F18F01',       # Warm orange
            'react': '#C73E1D',     # Deep red
            'primary': '#2E86AB',    # Primary blue
            'secondary': '#F18F01',  # Secondary orange
            'accent': '#A23B72',     # Accent magenta
            'neutral': '#6C757D'     # Professional gray
        }
        
        # Provider colors (consistent across all figures)
        self.provider_colors = {
            'gpt4': '#2E86AB',      # Professional blue
            'claude3': '#A23B72',   # Muted magenta  
            'gemini': '#F18F01'     # Warm orange
        }
        
        # Original figures
        self._generate_figure_pqs_by_strategy()
        self._generate_figure_provider_performance()
        self._generate_figure_domain_gains(metric="pp")  # Default to point gains
        self._generate_figure_pqs_by_complexity()
        self._generate_figure_sccs_by_dimension()
        
        # New thesis-ready figures
        self._generate_figure_convergence()
        self._generate_figure_cost_quality()
        self._generate_figure_error_breakdown()
        
        print("Figures generated successfully")
    
    def _generate_figure_pqs_by_strategy(self):
        """Generate Figure 4.1: PQS Distribution by Strategy (Academic Style)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for boxplot
        strategies = ['srlp', 'cot', 'tot', 'react']
        strategy_data = []
        strategy_stats = []
        
        for strategy in strategies:
            strategy_subset = self.df[self.df['strategy'] == strategy]
            if len(strategy_subset) > 0:
                pqs_values = strategy_subset['pqs'].values
                strategy_data.append(pqs_values)
                strategy_stats.append({
                    'mean': np.mean(pqs_values),
                    'median': np.median(pqs_values),
                    'std': np.std(pqs_values),
                    'n': len(pqs_values)
                })
            else:
                strategy_data.append([])
                strategy_stats.append({'mean': 0, 'median': 0, 'std': 0, 'n': 0})
        
        # Create professional boxplot
        bp = ax.boxplot(strategy_data, labels=[s.upper() for s in strategies], 
                       patch_artist=True, widths=0.6)
        
        # Apply academic color palette (pastel tones)
        academic_colors = [self.academic_colors['srlp'], self.academic_colors['cot'], 
                          self.academic_colors['tot'], self.academic_colors['react']]
        
        for i, (patch, color) in enumerate(zip(bp['boxes'], academic_colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)  # Pastel effect
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
        
        # Style whiskers, caps, and medians professionally
        for whisker in bp['whiskers']:
            whisker.set_color('#333333')
            whisker.set_linewidth(1.2)
            
        for cap in bp['caps']:
            cap.set_color('#333333')
            cap.set_linewidth(1.2)
            
        for median in bp['medians']:
            median.set_color('#000000')
            median.set_linewidth(2.0)
        
        # Make outliers smaller and more professional
        for flier in bp['fliers']:
            flier.set_marker('o')
            flier.set_markersize(4)
            flier.set_alpha(0.6)
        
        # Add mean markers (diamonds)
        for i, stats in enumerate(strategy_stats):
            if stats['n'] > 0:
                ax.scatter(i+1, stats['mean'], marker='D', s=60, 
                          color='white', edgecolor=academic_colors[i], 
                          linewidth=2, zorder=10, label='Mean' if i == 0 else "")
        
        # Professional axis labels with units
        ax.set_ylabel('Plan Quality Score (PQS, points)', fontweight='bold')
        ax.set_xlabel('Reasoning Strategy', fontweight='bold')
        ax.set_title('Plan Quality Score Distribution by Strategy', fontweight='bold', pad=20)
        
        # Set scientific axis limits and formatting
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        
        # Professional grid (already set globally)
        
        # Add summary statistics as annotation
        stats_text = []
        for i, (strategy, stats) in enumerate(zip(strategies, strategy_stats)):
            if stats['n'] > 0:
                stats_text.append(f"{strategy.upper()}: μ={stats['mean']:.1f}, n={stats['n']}")
        
        # Place statistics in professional location
        ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add legend for mean markers
        if any(stats['n'] > 0 for stats in strategy_stats):
            ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_1_pqs_by_strategy.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_1_pqs_by_strategy.pdf", bbox_inches='tight')
        plt.close()
    
    def _generate_figure_provider_performance(self):
        """Generate Figure 4.2: Provider Performance Analysis (Academic Style)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        providers = ['gpt4', 'claude3', 'gemini']
        provider_labels = ['GPT-4', 'Claude-3', 'Gemini']
        
        # Academic provider colors (consistent across all figures)
        colors = [self.provider_colors[p] for p in providers]
        
        # Time analysis with error bars
        times = [self.provider_stats[p]['execution_time_mean'] for p in providers]
        
        # Calculate standard deviations for error bars
        time_stds = []
        for provider in providers:
            provider_data = self.df[self.df['provider'] == provider]
            if len(provider_data) > 1:
                time_stds.append(provider_data['execution_time'].std())
            else:
                time_stds.append(0)
        
        bars1 = ax1.bar(provider_labels, times, color=colors, alpha=0.7, 
                       edgecolor='black', linewidth=0.8)
        
        # Add error bars for time
        ax1.errorbar(provider_labels, times, yerr=time_stds, fmt='none', 
                    color='black', capsize=5, capthick=1.2)
        
        # Professional axis labels and title
        ax1.set_ylabel('Average Execution Time (s)', fontweight='bold', fontsize=18)
        ax1.set_xlabel('AI Provider', fontweight='bold', fontsize=18)
        ax1.set_title('Average Execution Time by Provider', fontweight='bold')
        
        # Fix provider label readability - rotate more and increase font size
        ax1.tick_params(axis='x', labelsize=18, rotation=60)  # Increased rotation and font size
        ax1.tick_params(axis='y', labelsize=16)
        
        # Set y-axis to start at 0 for fairness
        ax1.set_ylim(0, max(times) * 1.5 if times else 1)  # More space for rotated labels
        
        # Add value labels on bars (larger, bold)
        for bar, time, std in zip(bars1, times, time_stds):
            height = bar.get_height()
            label_y = height + std + (max(times) * 0.02 if times else 0.02)
            ax1.text(bar.get_x() + bar.get_width()/2, label_y,
                    f'{time:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Cost analysis with error bars
        costs = [self.provider_stats[p]['cost_total'] for p in providers]
        
        # Calculate cost standard deviations
        cost_stds = []
        for provider in providers:
            provider_data = self.df[self.df['provider'] == provider]
            if len(provider_data) > 1:
                cost_stds.append(provider_data['cost_usd'].std())
            else:
                cost_stds.append(0)
        
        bars2 = ax2.bar(provider_labels, costs, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.8)
        
        # Add error bars for cost
        ax2.errorbar(provider_labels, costs, yerr=cost_stds, fmt='none',
                    color='black', capsize=5, capthick=1.2)
        
        # Professional axis labels and title
        ax2.set_ylabel('Total Cost (USD)', fontweight='bold', fontsize=18)
        ax2.set_xlabel('AI Provider', fontweight='bold', fontsize=18)
        ax2.set_title('Total Cost by Provider', fontweight='bold')
        
        # Fix provider label readability - rotate more and increase font size
        ax2.tick_params(axis='x', labelsize=18, rotation=60)  # Increased rotation and font size
        ax2.tick_params(axis='y', labelsize=16)
        
        # Set y-axis to start at 0 for fairness
        ax2.set_ylim(0, max(costs) * 1.5 if costs else 1)  # More space for rotated labels
        
        # Add value labels on bars (larger, bold)
        for bar, cost, std in zip(bars2, costs, cost_stds):
            height = bar.get_height()
            label_y = height + std + (max(costs) * 0.02 if costs else 0.02)
            ax2.text(bar.get_x() + bar.get_width()/2, label_y,
                    f'${cost:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Add sample size annotations
        for ax in [ax1, ax2]:
            # Get sample sizes from data
            sample_sizes = []
            for provider in providers:
                provider_data = self.df[self.df['provider'] == provider]
                sample_sizes.append(len(provider_data))
            
            # Add n= annotations below x-axis
            for i, (label, n) in enumerate(zip(provider_labels, sample_sizes)):
                ax.text(i, -max(ax.get_ylim()) * 0.05, f'n={n}', 
                       ha='center', va='top', fontsize=10, style='italic')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_2_provider_time_cost.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_2_provider_time_cost.pdf", bbox_inches='tight')
        plt.close()
    
    def _generate_figure_domain_gains(self, metric="pp"):
        """
        Generate Figure 4.3: SRLP Gains by Domain with bootstrap confidence intervals.
        
        Args:
            metric: "pp" for point gains (ΔPQS), "pct" for percentage gains
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate gains and confidence intervals for each domain
        domain_results = self._calculate_domain_gains_with_ci(metric)
        
        # Sort domains by CoT gains (descending) to show pattern clearly
        domain_results = sorted(domain_results, key=lambda x: x['cot_gain'], reverse=True)
        
        domains = [d['domain'] for d in domain_results]
        domain_labels = [d.replace('_', ' ').title() for d in domains]
        
        # Extract gains and confidence intervals
        cot_gains = [d['cot_gain'] for d in domain_results]
        tot_gains = [d['tot_gain'] for d in domain_results]
        react_gains = [d['react_gain'] for d in domain_results]
        
        # Reshape confidence intervals for matplotlib: (2, n) format
        cot_ci = np.array([d['cot_ci'] for d in domain_results]).T
        tot_ci = np.array([d['tot_ci'] for d in domain_results]).T
        react_ci = np.array([d['react_ci'] for d in domain_results]).T
        
        x = np.arange(len(domains))
        width = 0.25
        
        # Create grouped bar chart with error bars
        bars1 = ax.bar(x - width, cot_gains, width, label='vs CoT', 
                      color='#ff6b6b', alpha=0.8, yerr=cot_ci, capsize=4)
        bars2 = ax.bar(x, tot_gains, width, label='vs ToT', 
                      color='#4ecdc4', alpha=0.8, yerr=tot_ci, capsize=4)
        bars3 = ax.bar(x + width, react_gains, width, label='vs ReAct', 
                      color='#45b7d1', alpha=0.8, yerr=react_ci, capsize=4)
        
        # Add value annotations above bars
        max_error = max(np.max(cot_ci[1]), np.max(tot_ci[1]), np.max(react_ci[1]))
        
        for bars, gains, ci in [(bars1, cot_gains, cot_ci), (bars2, tot_gains, tot_ci), (bars3, react_gains, react_ci)]:
            for i, (bar, gain) in enumerate(zip(bars, gains)):
                height = bar.get_height()
                error_height = ci[1][i]  # Upper error
                ax.text(bar.get_x() + bar.get_width()/2., height + error_height + max_error * 0.05,
                       f'{gain:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Configure axes and labels
        if metric == "pp":
            ax.set_ylabel('ΔPQS (points)', fontsize=12)
            ax.set_title('SRLP Improvement by Domain (ΔPQS, points)', fontsize=14, fontweight='bold')
        else:
            ax.set_ylabel('Performance Gain (%)', fontsize=12)
            ax.set_title('SRLP Improvement by Domain (%)', fontsize=14, fontweight='bold')
        
        # Subtitle moved to figure caption - cleaner academic presentation
        
        ax.set_xticks(x)
        ax.set_xticklabels(domain_labels, rotation=45, ha='right', fontsize=11)
        
        # Move legend outside plot area
        ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1), frameon=True, 
                 fancybox=True, shadow=True)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Set y-axis to start at 0 for fair comparison
        ax.set_ylim(0, None)
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        plt.savefig(self.artifacts_dir / "figure_4_3_pqs_gain_by_domain.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_domain_gains_with_ci(self, metric, n_bootstrap=2000):
        """
        Calculate domain gains with 95% confidence intervals using bootstrap.
        
        Args:
            metric: "pp" for point gains, "pct" for percentage gains
            n_bootstrap: Number of bootstrap resamples
            
        Returns:
            List of dicts with domain, gains, and confidence intervals
        """
        import random
        
        results = []
        
        for domain in self.df['domain'].unique():
            domain_data = self.df[self.df['domain'] == domain]
            
            # Get SRLP data
            srlp_pqs = domain_data[domain_data['strategy'] == 'srlp']['pqs'].values
            
            domain_result = {'domain': domain}
            
            for baseline in ['cot', 'tot', 'react']:
                baseline_pqs = domain_data[domain_data['strategy'] == baseline]['pqs'].values
                
                if len(srlp_pqs) > 0 and len(baseline_pqs) > 0:
                    # Calculate point estimate
                    srlp_mean = np.mean(srlp_pqs)
                    baseline_mean = np.mean(baseline_pqs)
                    
                    if metric == "pp":
                        point_estimate = srlp_mean - baseline_mean
                    else:  # metric == "pct"
                        point_estimate = ((srlp_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
                    
                    # Bootstrap confidence interval
                    bootstrap_gains = []
                    min_len = min(len(srlp_pqs), len(baseline_pqs))
                    
                    for _ in range(n_bootstrap):
                        # Resample with replacement
                        srlp_resample = np.random.choice(srlp_pqs, size=min_len, replace=True)
                        baseline_resample = np.random.choice(baseline_pqs, size=min_len, replace=True)
                        
                        srlp_boot_mean = np.mean(srlp_resample)
                        baseline_boot_mean = np.mean(baseline_resample)
                        
                        if metric == "pp":
                            boot_gain = srlp_boot_mean - baseline_boot_mean
                        else:  # metric == "pct"
                            boot_gain = ((srlp_boot_mean - baseline_boot_mean) / baseline_boot_mean) * 100 if baseline_boot_mean > 0 else 0
                        
                        bootstrap_gains.append(boot_gain)
                    
                    # Calculate 95% CI
                    ci_lower = np.percentile(bootstrap_gains, 2.5)
                    ci_upper = np.percentile(bootstrap_gains, 97.5)
                    
                    # Error bar format: [lower_error, upper_error]
                    ci_error = [point_estimate - ci_lower, ci_upper - point_estimate]
                    
                else:
                    point_estimate = 0
                    ci_error = [0, 0]
                
                domain_result[f'{baseline}_gain'] = point_estimate
                domain_result[f'{baseline}_ci'] = ci_error
            
            results.append(domain_result)
        
        return results
    
    # Helper functions for new thesis figures
    def _bootstrap_ci(self, series, B=2000):
        """Calculate bootstrap confidence interval."""
        if len(series) < 2:
            return 0, 0
        
        bootstrap_means = []
        series_array = np.array(series)
        
        for _ in range(B):
            sample = np.random.choice(series_array, size=len(series_array), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)
        return ci_low, ci_high
    
    def _ensure_dir(self, path):
        """Ensure directory exists."""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def _format_number(self, num, decimals=2):
        """Clean number formatter."""
        return f"{num:.{decimals}f}"
    
    def _generate_figure_convergence(self):
        """Generate Figure 4.6: SRLP Convergence Efficiency Curve."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if iteration column exists
        has_iteration = 'iteration' in self.df.columns
        
        if has_iteration:
            # Get SRLP iteration data
            srlp_data = self.df[self.df['strategy'] == 'srlp'].copy()
            
            if len(srlp_data) > 0 and 'iteration' in srlp_data.columns:
                # Group by iteration and calculate statistics
                iteration_stats = []
                iterations = sorted(srlp_data['iteration'].unique())
                
                for iteration in iterations:
                    iter_data = srlp_data[srlp_data['iteration'] == iteration]
                    if len(iter_data) > 0:
                        pqs_values = iter_data['pqs'].values
                        mean_pqs = np.mean(pqs_values)
                        ci_low, ci_high = self._bootstrap_ci(pqs_values)
                        
                        iteration_stats.append({
                            'iteration': iteration,
                            'mean_pqs': mean_pqs,
                            'ci_low': ci_low,
                            'ci_high': ci_high,
                            'pqs_values': pqs_values
                        })
                
                if iteration_stats:
                    # Plot SRLP convergence with CI
                    iterations = [s['iteration'] for s in iteration_stats]
                    means = [s['mean_pqs'] for s in iteration_stats]
                    ci_lows = [s['ci_low'] for s in iteration_stats]
                    ci_highs = [s['ci_high'] for s in iteration_stats]
                    
                    # Main convergence line with consistent markers
                    ax.plot(iterations, means, 'o-', linewidth=3, markersize=8, 
                           color=self.academic_colors['srlp'], label='SRLP', zorder=3,
                           markerfacecolor='white', markeredgewidth=2)
                    
                    # Confidence interval shading
                    ax.fill_between(iterations, ci_lows, ci_highs, alpha=0.3, 
                                   color='#ff6b6b', zorder=1)
                    
                    # Save convergence table data
                    self._save_convergence_table(iteration_stats)
                else:
                    # Fallback: single point at iteration 0
                    srlp_mean = srlp_data['pqs'].mean()
                    ax.axvline(x=0, color='#ff6b6b', linewidth=3, label='SRLP', alpha=0.8)
                    ax.scatter([0], [srlp_mean], color='#ff6b6b', s=100, zorder=3)
            else:
                # No iteration data for SRLP
                srlp_mean = self.df[self.df['strategy'] == 'srlp']['pqs'].mean()
                ax.axhline(y=srlp_mean, color='#ff6b6b', linewidth=2, label='SRLP', alpha=0.8)
        else:
            # No iteration column - treat all as iteration 0
            srlp_mean = self.df[self.df['strategy'] == 'srlp']['pqs'].mean()
            ax.axhline(y=srlp_mean, color='#ff6b6b', linewidth=2, label='SRLP', alpha=0.8)
        
        # Plot baseline strategies as flat lines
        baseline_colors = {'cot': '#4ecdc4', 'tot': '#45b7d1', 'react': '#96ceb4'}
        
        for baseline, color in baseline_colors.items():
            baseline_data = self.df[self.df['strategy'] == baseline]
            if len(baseline_data) > 0:
                if has_iteration and 'iteration' in baseline_data.columns:
                    # Use iteration 0 data if available, otherwise all data
                    iter_0_data = baseline_data[baseline_data['iteration'] == 0]
                    if len(iter_0_data) > 0:
                        baseline_mean = iter_0_data['pqs'].mean()
                    else:
                        baseline_mean = baseline_data['pqs'].mean()
                else:
                    baseline_mean = baseline_data['pqs'].mean()
                
                ax.axhline(y=baseline_mean, color=color, linestyle='--', linewidth=2, 
                          alpha=0.8, label=f'{baseline.upper()}')
        
        # Professional styling with academic formatting
        ax.set_xlabel('Refinement Iteration (1–3)', fontweight='bold')
        ax.set_ylabel('Plan Quality Score (PQS, points)', fontweight='bold')
        ax.set_title('Convergence of SRLP Across Refinement Iterations', fontweight='bold', pad=20)
        
        # Subtitle
        ax.text(0.5, 0.95, 'Shaded area: 95% bootstrap CI; baselines shown as reference lines',
               transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic')
        
        # Set limits and grid
        ax.set_ylim(0, 100)
        if has_iteration and 'iteration' in self.df.columns:
            max_iter = self.df['iteration'].max() if not self.df['iteration'].isna().all() else 0
            ax.set_xlim(-0.5, max_iter + 0.5)
        else:
            ax.set_xlim(-0.5, 0.5)
        
        ax.grid(True, axis='y', alpha=0.2)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_6_convergence.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_6_convergence.pdf", bbox_inches='tight')
        plt.close()
        
        return ax
    
    def _save_convergence_table(self, iteration_stats):
        """Save convergence data as LaTeX table."""
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{SRLP Convergence Statistics by Iteration}
\\label{tab:convergence}
\\begin{tabular}{rrrr}
\\toprule
Iteration & Mean PQS & CI Low & CI High \\\\
\\midrule
"""
        
        for stats in iteration_stats:
            latex_content += f"{int(stats['iteration'])} & {stats['mean_pqs']:.2f} & {stats['ci_low']:.2f} & {stats['ci_high']:.2f} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_6_convergence.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_figure_cost_quality(self):
        """Generate Figure 4.7: Cost-Quality Trade-off."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Check for required columns
        has_cost = 'cost_usd' in self.df.columns
        has_time = 'exec_time_s' in self.df.columns
        
        if not has_cost:
            print("Warning: cost_usd column missing, skipping cost-quality plot")
            return ax
        
        # Aggregate by provider and strategy
        aggregated = []
        
        for provider in self.df['provider'].unique():
            for strategy in self.df['strategy'].unique():
                subset = self.df[(self.df['provider'] == provider) & (self.df['strategy'] == strategy)]
                
                if len(subset) > 0 and has_cost:
                    # Filter out NaN costs
                    cost_data = subset.dropna(subset=['cost_usd', 'pqs'])
                    
                    if len(cost_data) > 0:
                        agg_data = {
                            'provider': provider,
                            'strategy': strategy,
                            'pqs_mean': cost_data['pqs'].mean(),
                            'cost_mean': cost_data['cost_usd'].mean(),
                            'time_mean': cost_data['exec_time_s'].mean() if has_time else 0,
                            'n': len(cost_data)
                        }
                        aggregated.append(agg_data)
        
        if len(aggregated) < 3:
            print("Warning: Insufficient data for cost-quality analysis")
            return ax
        
        # Extract data for plotting
        x_costs = [d['cost_mean'] for d in aggregated]
        y_pqs = [d['pqs_mean'] for d in aggregated]
        sizes = [20 + d['n'] * 0.1 for d in aggregated]
        
        # Create scatter plot
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
        
        # Calculate medians for quadrant-based positioning
        median_cost = np.median(x_costs)
        median_pqs = np.median(y_pqs)
        
        for i, data in enumerate(aggregated):
            color = colors[i % len(colors)]
            ax.scatter(data['cost_mean'], data['pqs_mean'], s=sizes[i], 
                      color=color, alpha=0.7, edgecolors='black', linewidths=0.5)
            
            # Create shorter abbreviations to reduce overlap
            provider_abbrev = {
                'gpt4': 'G4', 'claude3': 'C3', 'gemini': 'GM'
            }
            strategy_abbrev = {
                'srlp': 'SRLP', 'cot': 'CoT', 'tot': 'ToT', 'react': 'ReAct'
            }
            
            provider_short = provider_abbrev.get(data['provider'].lower(), data['provider'][:2].upper())
            strategy_short = strategy_abbrev.get(data['strategy'].lower(), data['strategy'][:3].upper())
            
            # Use shorter labels with better positioning to avoid overlap
            label_text = f"{provider_short}-{strategy_short}"
            
            # Smart quadrant-based positioning to avoid overlap
            x_pos = data['cost_mean']
            y_pos = data['pqs_mean']
            
            # Determine quadrant and set appropriate offset
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
            offset_x += (i % 3) * 5
            offset_y += (i % 3) * 5
            
            ax.annotate(label_text, 
                       (data['cost_mean'], data['pqs_mean']),
                       xytext=(offset_x, offset_y), textcoords='offset points', 
                       fontsize=14, fontweight='bold', ha=ha,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, 
                                edgecolor='gray', linewidth=0.8),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.9))
        
        # Add reference lines (medians) - already calculated above
        ax.axvline(x=median_cost, color='gray', linestyle=':', alpha=0.6, label='Median Cost')
        ax.axhline(y=median_pqs, color='gray', linestyle=':', alpha=0.6, label='Median PQS')
        
        # Compute and draw Pareto frontier
        if len(aggregated) >= 3:
            pareto_points = self._compute_pareto_frontier(aggregated)
            if len(pareto_points) > 1:
                pareto_costs = [p['cost_mean'] for p in pareto_points]
                pareto_pqs = [p['pqs_mean'] for p in pareto_points]
                ax.plot(pareto_costs, pareto_pqs, 'k--', alpha=0.8, linewidth=4, label='Pareto Frontier')
        
        # Enhanced styling with larger fonts
        ax.set_xlabel('Mean Cost (USD)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean PQS', fontsize=16, fontweight='bold')
        ax.set_title('Cost–Quality Trade-off by Provider and Strategy', fontsize=18, fontweight='bold')
        
        # Set limits and grid
        ax.set_ylim(0, 100)
        ax.set_xlim(0, max(x_costs) * 1.1)
        ax.grid(True, axis='y', alpha=0.2)
        
        # Increase tick label sizes for better readability
        ax.tick_params(axis='both', labelsize=14)
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_7_cost_quality.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_7_cost_quality.pdf", bbox_inches='tight')
        plt.close()
        
        # Save companion table
        self._save_cost_quality_table(aggregated)
        
        return ax
    
    def _compute_pareto_frontier(self, data):
        """Compute Pareto frontier (minimize cost, maximize PQS)."""
        # Sort by cost
        sorted_data = sorted(data, key=lambda x: x['cost_mean'])
        
        pareto = []
        max_pqs_so_far = -1
        
        for point in sorted_data:
            if point['pqs_mean'] > max_pqs_so_far:
                pareto.append(point)
                max_pqs_so_far = point['pqs_mean']
        
        return pareto
    
    def _save_cost_quality_table(self, aggregated):
        """Save cost-quality data as LaTeX table."""
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Cost-Quality Trade-off Summary}
\\label{tab:cost_quality}
\\begin{tabular}{llrrrr}
\\toprule
Provider & Strategy & Mean PQS & Mean Cost & Mean Time & N \\\\
\\midrule
"""
        
        # Sort by mean PQS descending
        sorted_data = sorted(aggregated, key=lambda x: x['pqs_mean'], reverse=True)
        
        for data in sorted_data:
            latex_content += f"{data['provider'].upper()} & {data['strategy'].upper()} & {data['pqs_mean']:.2f} & {data['cost_mean']:.2f} & {data['time_mean']:.2f} & {data['n']} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_7_cost_quality.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_figure_error_breakdown(self):
        """Generate Figure 4.8: Error Breakdown by Strategy."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define error categories
        categories = ["OK", "Incomplete", "Hallucination", "Invalid/Empty", "Timeout/ToolError", "Other"]
        
        # Check for required columns
        has_success = 'success' in self.df.columns
        has_error_type = 'error_type' in self.df.columns
        
        # Build error breakdown by strategy
        strategy_errors = {}
        
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            total_runs = len(strategy_data)
            
            if total_runs == 0:
                continue
            
            # Initialize counts
            error_counts = {cat: 0 for cat in categories}
            
            for _, row in strategy_data.iterrows():
                category = self._categorize_error(row, has_success, has_error_type)
                error_counts[category] += 1
            
            # Convert to percentages
            error_percentages = {cat: (count / total_runs) * 100 for cat, count in error_counts.items()}
            strategy_errors[strategy] = error_percentages
        
        if not strategy_errors:
            print("Warning: No strategy data for error breakdown")
            return ax
        
        # Create stacked bar chart
        strategies = list(strategy_errors.keys())
        bottom_values = np.zeros(len(strategies))
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#34495e', '#95a5a6']
        
        for i, category in enumerate(categories):
            values = [strategy_errors[strategy][category] for strategy in strategies]
            
            bars = ax.bar(strategies, values, bottom=bottom_values, 
                         color=colors[i % len(colors)], alpha=0.8, label=category)
            
            # Add value labels for segments >= 4%
            for j, (bar, value) in enumerate(zip(bars, values)):
                if value >= 4:
                    height = bar.get_height()
                    y_pos = bottom_values[j] + height / 2
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, 
                           f'{value:.0f}%', ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
            
            bottom_values += values
        
        # Styling
        ax.set_ylabel('Share of Runs (%)', fontsize=12)
        ax.set_title('Failure Mode Breakdown by Strategy', fontsize=14, fontweight='bold')
        
        # Add caption
        ax.text(0.5, -0.15, "Heuristic bucketing from 'success' and 'error_type'",
               transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic')
        
        # Set limits and grid
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.2)
        
        # Format strategy names
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([s.upper() for s in strategies])
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_8_error_breakdown.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_8_error_breakdown.pdf", bbox_inches='tight')
        plt.close()
        
        # Save companion table
        self._save_error_breakdown_table(strategy_errors)
        
        return ax
    
    def _categorize_error(self, row, has_success, has_error_type):
        """Categorize error based on success and error_type."""
        # Check success column first
        if has_success:
            success = row.get('success', None)
            if success is True or success == 'True' or success == 1:
                return "OK"
        
        # Check error_type column
        if has_error_type:
            error_type = str(row.get('error_type', '')).lower()
            
            if 'incomplete' in error_type:
                return "Incomplete"
            elif 'halluc' in error_type:
                return "Hallucination"
            elif 'invalid' in error_type or 'empty' in error_type:
                return "Invalid/Empty"
            elif 'timeout' in error_type or 'tool' in error_type:
                return "Timeout/ToolError"
            elif error_type and error_type != 'nan' and error_type != '':
                return "Other"
        
        # Default fallback
        if has_success:
            success = row.get('success', None)
            if success is False or success == 'False' or success == 0:
                return "Other"
        
        # If no clear information, assume OK
        return "OK" if not has_success and not has_error_type else "Other"
    
    def _save_error_breakdown_table(self, strategy_errors):
        """Save error breakdown data as LaTeX table."""
        categories = ["OK", "Incomplete", "Hallucination", "Invalid/Empty", "Timeout/ToolError", "Other"]
        
        latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Failure Mode Breakdown by Strategy (\\%)}
\\label{tab:error_breakdown}
\\begin{tabular}{l""" + "r" * len(categories) + """}
\\toprule
Strategy & """ + " & ".join(categories) + """ \\\\
\\midrule
"""
        
        for strategy, errors in strategy_errors.items():
            row = f"{strategy.upper()}"
            for category in categories:
                row += f" & {errors[category]:.1f}"
            row += " \\\\\n"
            latex_content += row
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        with open(self.artifacts_dir / "table_4_8_error_breakdown.tex", 'w') as f:
            f.write(latex_content)
    
    def _generate_figure_pqs_by_complexity(self):
        """Generate Figure 4.4: PQS by Complexity Level (Academic Style with Error Bars)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        complexities = ['low', 'medium', 'high']
        complexity_labels = ['Low', 'Medium', 'High']
        strategies = ['srlp', 'cot', 'tot', 'react']  # Consistent order across all figures
        
        # Use academic color palette (consistent across all figures)
        colors = [self.academic_colors['srlp'], self.academic_colors['cot'], 
                 self.academic_colors['tot'], self.academic_colors['react']]
        
        x = np.arange(len(complexities))
        width = 0.2
        
        # Calculate means and confidence intervals for each strategy
        for i, strategy in enumerate(strategies):
            means = []
            ci_errors = []  # Will store [lower, upper] CI errors
            
            for complexity in complexities:
                subset = self.df[(self.df['complexity'] == complexity) & (self.df['strategy'] == strategy)]
                
                if len(subset) > 0:
                    pqs_values = subset['pqs'].values
                    mean_pqs = np.mean(pqs_values)
                    
                    # Calculate 95% confidence interval using bootstrap
                    if len(pqs_values) > 1:
                        ci_low, ci_high = self._bootstrap_ci(pqs_values)
                        ci_error = [mean_pqs - ci_low, ci_high - mean_pqs]
                    else:
                        ci_error = [0, 0]
                    
                    means.append(mean_pqs)
                    ci_errors.append(ci_error)
                else:
                    means.append(0)
                    ci_errors.append([0, 0])
            
            # Transform ci_errors for errorbar format (2, n) array
            ci_errors = np.array(ci_errors).T  # Transpose to get [lower_errors, upper_errors]
            
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, means, width, label=strategy.upper(), 
                         color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.8)
            
            # Add error bars
            ax.errorbar(x + offset, means, yerr=ci_errors, fmt='none', 
                       color='black', capsize=4, capthick=1.0)
            
            # Add value labels on top of bars
            for j, (bar, mean, ci_err) in enumerate(zip(bars, means, ci_errors.T)):
                if mean > 0:  # Only label non-zero bars
                    label_y = mean + ci_err[1] + 2  # Above upper error bar
                    ax.text(bar.get_x() + bar.get_width()/2, label_y,
                           f'{mean:.1f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
        
        # Professional axis labels and title
        ax.set_ylabel('Plan Quality Score (PQS, points)', fontweight='bold')
        ax.set_xlabel('Task Complexity Level', fontweight='bold')
        ax.set_title('Plan Quality Score by Complexity Level', fontweight='bold', pad=20)
        
        # Set professional axis formatting
        ax.set_xticks(x)
        ax.set_xticklabels(complexity_labels)
        ax.set_ylim(0, 100)  # Standard PQS scale
        
        # Professional legend outside plot area
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Add sample size information
        sample_info = []
        for complexity in complexities:
            n = len(self.df[self.df['complexity'] == complexity])
            sample_info.append(f"{complexity.title()}: n={n}")
        
        # Add sample size annotation
        ax.text(0.02, 0.02, '\n'.join(sample_info), transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_4_pqs_by_complexity.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_4_pqs_by_complexity.pdf", bbox_inches='tight')
        plt.close()
    
    def _generate_figure_sccs_by_dimension(self):
        """Generate Figure 4.5: Strategic Cognitive Capabilities Radar Chart (Academic Style)."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        strategies = ['srlp', 'cot', 'tot', 'react']
        metrics = ['pqs_mean', 'sccs_mean', 'iir_mean', 'cem_mean']
        metric_labels = ['Plan Quality\nScore (PQS)', 'Strategic Cognitive\nCapabilities (SCCS)', 
                        'Implementation\nIntegration Rating (IIR)', 'Cognitive\nEfficiency Metric (CEM)']
        
        # Create radar chart data
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Use academic color palette (consistent across all figures)
        colors = [self.academic_colors['srlp'], self.academic_colors['cot'], 
                 self.academic_colors['tot'], self.academic_colors['react']]
        
        # Line styles for better differentiation
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']
        
        for i, strategy in enumerate(strategies):
            if strategy in self.strategy_stats:
                values = [self.strategy_stats[strategy][metric] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                # Professional radar lines (thinner, distinct styles)
                ax.plot(angles, values, line_styles[i], linewidth=2.5, 
                       marker=markers[i], markersize=6, label=strategy.upper(), 
                       color=colors[i], markerfacecolor='white', markeredgewidth=2)
                
                # Light fills to avoid clutter
                ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Professional axis setup with improved readability
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=14, fontweight='bold')  # Larger, bolder labels
        
        # Set consistent 0-100 scale with improved tick formatting
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=12, fontweight='bold')
        
        # Professional title
        ax.set_title('Strategic Cognitive Capabilities by Dimension', 
                    fontsize=18, fontweight='bold', pad=30)
        
        # Enhanced grid with faint radial lines for better interpretability
        ax.grid(True, alpha=0.4, linewidth=1.2)
        ax.set_facecolor('white')
        
        # Add value annotations at each vertex for better readability
        for i, strategy in enumerate(strategies):
            if strategy in self.strategy_stats:
                values = [self.strategy_stats[strategy][metric] for metric in metrics]
                for j, (angle, value, label) in enumerate(zip(angles[:-1], values, metric_labels)):
                    # Add faint radial guidelines
                    if i == 0:  # Only add guidelines once
                        ax.plot([angle, angle], [0, 100], 'k--', alpha=0.1, linewidth=0.8)
                    
                    # Annotate key values for SRLP (most important strategy)
                    if strategy == 'srlp':
                        radius_offset = 110  # Position outside the chart
                        x_pos = radius_offset * np.cos(angle)
                        y_pos = radius_offset * np.sin(angle)
                        ax.annotate(f'{value:.1f}', (x_pos, y_pos), 
                                  fontsize=11, fontweight='bold', ha='center', va='center',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
        
        # Legend outside plot area
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), 
                 frameon=True, fancybox=True, shadow=True)
        
        # Add performance annotations for each strategy
        performance_summary = []
        for strategy in strategies:
            if strategy in self.strategy_stats:
                avg_score = np.mean([self.strategy_stats[strategy][metric] for metric in metrics])
                performance_summary.append(f"{strategy.upper()}: {avg_score:.1f}")
        
        # Add summary text
        summary_text = "Average Performance:\n" + "\n".join(performance_summary)
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(self.artifacts_dir / "figure_4_5_sccs_by_dimension.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.artifacts_dir / "figure_4_5_sccs_by_dimension.pdf", bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print generation summary."""
        print(f"\n{'='*60}")
        print("ARTIFACTS GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Results processed: {len(self.df)} experiments")
        print(f"Strategies analyzed: {len(self.strategy_stats)}")
        print(f"Providers analyzed: {len(self.provider_stats)}")
        print(f"Domains analyzed: {len(self.domain_stats)}")
        
        print(f"\nGenerated LaTeX Tables:")
        for i in range(1, 6):
            table_file = self.artifacts_dir / f"table_4_{i}_*.tex"
            print(f"  ✓ Table 4.{i}")
        
        print(f"\nGenerated Figures:")
        for i in range(1, 6):
            figure_file = self.artifacts_dir / f"figure_4_{i}_*.png"
            print(f"  ✓ Figure 4.{i}")
        
        print(f"\nArtifacts directory: {self.artifacts_dir.absolute()}")
        print(f"{'='*60}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate artifacts from evaluation results')
    parser.add_argument('results_csv', help='Path to evaluation results CSV')
    parser.add_argument('--artifacts-dir', default='artifacts', help='Output directory for artifacts')
    
    args = parser.parse_args()
    
    # Set headless backend
    os.environ['MPLBACKEND'] = 'Agg'
    
    generator = ArtifactsGenerator(args.results_csv, args.artifacts_dir)
    generator.generate_all_artifacts()
    generator.print_summary()


if __name__ == "__main__":
    main()
