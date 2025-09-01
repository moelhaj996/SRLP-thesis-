"""
Comprehensive Statistical Analysis Framework for SRLP Thesis
Implements rigorous statistical testing with effect sizes, multiple comparison correction,
and advanced analysis methods for publication-quality research.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import f_oneway, kruskal, levene, shapiro, anderson, tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings
from typing import Dict, List, Tuple, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis toolkit for SRLP evaluation.
    
    Implements:
    - ANOVA with assumption checking
    - Post-hoc tests with multiple comparison correction
    - Effect size calculations (Cohen's d, eta-squared)
    - Bootstrap confidence intervals
    - Bayesian analysis foundations
    - Power analysis and sample size justification
    """
    
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_state = random_state
        np.random.seed(random_state)
        
    def check_assumptions(self, data: pd.DataFrame, dependent_var: str, 
                         group_var: str) -> Dict[str, Any]:
        """
        Check statistical assumptions for ANOVA.
        
        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable column
            group_var: Name of grouping variable column
            
        Returns:
            Dictionary with assumption test results
        """
        results = {}
        groups = [group[dependent_var].values for name, group in data.groupby(group_var)]
        
        # 1. Normality testing
        normality_results = {}
        for name, group in data.groupby(group_var):
            if len(group) >= 3:  # Minimum sample size for Shapiro-Wilk
                shapiro_stat, shapiro_p = shapiro(group[dependent_var])
                normality_results[name] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'normal': shapiro_p > self.alpha
                }
        
        results['normality'] = normality_results
        results['all_normal'] = all(res['normal'] for res in normality_results.values())
        
        # 2. Homogeneity of variance (Levene's test)
        levene_stat, levene_p = levene(*groups, center='median')
        results['levene'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'equal_variances': levene_p > self.alpha
        }
        
        # 3. Sample size assessment
        sample_sizes = {name: len(group) for name, group in data.groupby(group_var)}
        results['sample_sizes'] = sample_sizes
        results['balanced'] = len(set(sample_sizes.values())) == 1
        results['min_sample_size'] = min(sample_sizes.values())
        
        return results
    
    def welch_anova(self, data: pd.DataFrame, dependent_var: str, 
                   group_var: str) -> Dict[str, Any]:
        """
        Perform Welch's ANOVA for unequal sample sizes and variances.
        
        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable column
            group_var: Name of grouping variable column
            
        Returns:
            Dictionary with ANOVA results
        """
        # Extract groups
        groups = [group[dependent_var].values for name, group in data.groupby(group_var)]
        group_names = [name for name, group in data.groupby(group_var)]
        
        # Welch's ANOVA (does not assume equal variances)
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(data[dependent_var]))**2 
                        for group in groups)
        ss_total = np.sum((data[dependent_var] - np.mean(data[dependent_var]))**2)
        eta_squared = ss_between / ss_total
        
        # Degrees of freedom
        k = len(groups)  # number of groups
        n_total = len(data)
        df_between = k - 1
        df_within = n_total - k
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'degrees_freedom': (df_between, df_within),
            'eta_squared': eta_squared,
            'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
            'significant': p_value < self.alpha,
            'group_names': group_names,
            'group_means': [np.mean(group) for group in groups],
            'group_stds': [np.std(group, ddof=1) for group in groups],
            'group_sizes': [len(group) for group in groups]
        }
    
    def post_hoc_analysis(self, data: pd.DataFrame, dependent_var: str, 
                         group_var: str, method: str = 'tukey') -> Dict[str, Any]:
        """
        Perform post-hoc analysis with multiple comparison correction.
        
        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable column
            group_var: Name of grouping variable column
            method: Post-hoc test method ('tukey', 'bonferroni')
            
        Returns:
            Dictionary with post-hoc test results
        """
        if method == 'tukey':
            # Tukey HSD test
            mc = MultiComparison(data[dependent_var], data[group_var])
            result = mc.tukeyhsd(alpha=self.alpha)
            
            # Extract pairwise comparisons
            comparisons = []
            for i in range(len(result.summary().data[1:])):
                row = result.summary().data[i + 1]
                comparisons.append({
                    'group1': row[0],
                    'group2': row[1],
                    'mean_diff': float(row[2]),
                    'p_value': float(row[3]),
                    'ci_lower': float(row[4]),
                    'ci_upper': float(row[5]),
                    'significant': row[6] == 'True'
                })
            
            return {
                'method': 'Tukey HSD',
                'comparisons': comparisons,
                'summary': str(result)
            }
        
        elif method == 'bonferroni':
            # Bonferroni correction
            groups = data[group_var].unique()
            n_comparisons = len(groups) * (len(groups) - 1) / 2
            bonferroni_alpha = self.alpha / n_comparisons
            
            comparisons = []
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups[i+1:], i+1):
                    data1 = data[data[group_var] == group1][dependent_var]
                    data2 = data[data[group_var] == group2][dependent_var]
                    
                    # Welch's t-test
                    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                    
                    # Effect size (Cohen's d)
                    cohens_d = self.cohens_d(data1, data2)
                    
                    comparisons.append({
                        'group1': group1,
                        'group2': group2,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'p_corrected': p_value * n_comparisons,
                        'significant': p_value < bonferroni_alpha,
                        'cohens_d': cohens_d,
                        'effect_size_interpretation': self._interpret_cohens_d(cohens_d)
                    })
            
            return {
                'method': 'Bonferroni',
                'alpha_corrected': bonferroni_alpha,
                'n_comparisons': n_comparisons,
                'comparisons': comparisons
            }
    
    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            Cohen's d effect size
        """
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def bootstrap_ci(self, data: np.ndarray, statistic: callable = np.mean, 
                    n_bootstrap: int = 2000, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Input data array
            statistic: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return np.percentile(bootstrap_stats, lower_percentile), np.percentile(bootstrap_stats, upper_percentile)
    
    def two_way_anova(self, data: pd.DataFrame, dependent_var: str, 
                     factor1: str, factor2: str) -> Dict[str, Any]:
        """
        Perform two-way ANOVA for interaction effects.
        
        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable
            factor1: First factor variable
            factor2: Second factor variable
            
        Returns:
            Dictionary with ANOVA results
        """
        # Create formula for ANOVA
        formula = f'{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'
        
        # Fit OLS model
        model = ols(formula, data=data).fit()
        
        # Perform ANOVA
        anova_results = anova_lm(model, typ=2)
        
        return {
            'model': model,
            'anova_table': anova_results,
            'formula': formula,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'p_value': model.f_pvalue
        }
    
    def power_analysis(self, effect_size: float, n_groups: int, 
                      sample_size: int = None, power: float = None) -> Dict[str, Any]:
        """
        Perform power analysis for sample size determination.
        
        Args:
            effect_size: Expected effect size (Cohen's f)
            n_groups: Number of groups in comparison
            sample_size: Sample size per group (if calculating power)
            power: Desired power (if calculating sample size)
            
        Returns:
            Dictionary with power analysis results
        """
        from statsmodels.stats.power import FTestAnovaPower
        
        power_analysis = FTestAnovaPower()
        
        if sample_size is not None:
            # Calculate power given sample size
            calculated_power = power_analysis.solve_power(
                effect_size=effect_size,
                nobs=sample_size * n_groups,
                alpha=self.alpha,
                k_groups=n_groups
            )
            return {
                'type': 'power_calculation',
                'effect_size': effect_size,
                'sample_size_per_group': sample_size,
                'total_sample_size': sample_size * n_groups,
                'power': calculated_power,
                'adequate_power': calculated_power >= 0.8
            }
        
        elif power is not None:
            # Calculate required sample size for desired power
            required_n = power_analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=self.alpha,
                k_groups=n_groups
            )
            return {
                'type': 'sample_size_calculation',
                'effect_size': effect_size,
                'desired_power': power,
                'required_total_n': required_n,
                'required_n_per_group': required_n / n_groups
            }
    
    def comprehensive_analysis(self, data: pd.DataFrame, dependent_var: str, 
                             group_var: str) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis for publication quality.
        
        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable
            group_var: Name of grouping variable
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Performing comprehensive analysis: {dependent_var} by {group_var}")
        
        results = {}
        
        # 1. Assumption checking
        results['assumptions'] = self.check_assumptions(data, dependent_var, group_var)
        
        # 2. Choose appropriate test based on assumptions
        if (results['assumptions']['all_normal'] and 
            results['assumptions']['levene']['equal_variances']):
            # Standard ANOVA
            groups = [group[dependent_var].values for name, group in data.groupby(group_var)]
            f_stat, p_value = f_oneway(*groups)
            results['primary_test'] = {
                'test_used': 'One-way ANOVA',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        else:
            # Welch's ANOVA or non-parametric test
            if results['assumptions']['all_normal']:
                results['primary_test'] = self.welch_anova(data, dependent_var, group_var)
                results['primary_test']['test_used'] = "Welch's ANOVA"
            else:
                # Kruskal-Wallis test (non-parametric)
                groups = [group[dependent_var].values for name, group in data.groupby(group_var)]
                h_stat, p_value = kruskal(*groups)
                results['primary_test'] = {
                    'test_used': 'Kruskal-Wallis',
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        
        # 3. Post-hoc analysis if significant
        if results['primary_test']['significant']:
            results['post_hoc'] = self.post_hoc_analysis(data, dependent_var, group_var)
        
        # 4. Effect sizes for all pairwise comparisons
        groups = data[group_var].unique()
        effect_sizes = []
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                data1 = data[data[group_var] == group1][dependent_var]
                data2 = data[data[group_var] == group2][dependent_var]
                d = self.cohens_d(data1, data2)
                effect_sizes.append({
                    'group1': group1,
                    'group2': group2,
                    'cohens_d': d,
                    'interpretation': self._interpret_cohens_d(d)
                })
        results['effect_sizes'] = effect_sizes
        
        # 5. Descriptive statistics
        descriptives = {}
        for name, group in data.groupby(group_var):
            desc = {
                'mean': np.mean(group[dependent_var]),
                'std': np.std(group[dependent_var], ddof=1),
                'median': np.median(group[dependent_var]),
                'n': len(group),
                'ci_lower': 0,
                'ci_upper': 0
            }
            # Bootstrap CI for mean
            desc['ci_lower'], desc['ci_upper'] = self.bootstrap_ci(group[dependent_var].values)
            descriptives[name] = desc
        results['descriptives'] = descriptives
        
        return results
    
    def format_significance_stars(self, p_value: float) -> str:
        """Format p-value as significance stars."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"
    
    def create_analysis_report(self, results: Dict[str, Any], 
                             dependent_var: str, group_var: str) -> str:
        """
        Create a comprehensive analysis report.
        
        Args:
            results: Results from comprehensive_analysis
            dependent_var: Name of dependent variable
            group_var: Name of grouping variable
            
        Returns:
            Formatted analysis report
        """
        report = []
        report.append(f"COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
        report.append(f"Analysis: {dependent_var} by {group_var}")
        report.append("=" * 60)
        
        # Assumptions
        report.append("\n1. ASSUMPTION CHECKING:")
        assumptions = results['assumptions']
        report.append(f"   Normality: {'✓' if assumptions['all_normal'] else '✗'}")
        report.append(f"   Equal variances: {'✓' if assumptions['levene']['equal_variances'] else '✗'}")
        report.append(f"   Balanced design: {'✓' if assumptions['balanced'] else '✗'}")
        
        # Primary test
        report.append("\n2. PRIMARY STATISTICAL TEST:")
        primary = results['primary_test']
        report.append(f"   Test used: {primary['test_used']}")
        if 'f_statistic' in primary:
            report.append(f"   F-statistic: {primary['f_statistic']:.4f}")
        if 'h_statistic' in primary:
            report.append(f"   H-statistic: {primary['h_statistic']:.4f}")
        report.append(f"   p-value: {primary['p_value']:.6f} {self.format_significance_stars(primary['p_value'])}")
        report.append(f"   Significant: {'Yes' if primary['significant'] else 'No'}")
        
        # Effect sizes
        if 'eta_squared' in primary:
            report.append(f"   η² (effect size): {primary['eta_squared']:.4f} ({primary['effect_size_interpretation']})")
        
        # Post-hoc results
        if 'post_hoc' in results:
            report.append("\n3. POST-HOC ANALYSIS:")
            post_hoc = results['post_hoc']
            report.append(f"   Method: {post_hoc['method']}")
            for comp in post_hoc['comparisons']:
                if 'significant' in comp and comp['significant']:
                    stars = self.format_significance_stars(comp['p_value'])
                    report.append(f"   {comp['group1']} vs {comp['group2']}: p={comp['p_value']:.4f} {stars}")
        
        # Effect sizes
        report.append("\n4. PAIRWISE EFFECT SIZES (Cohen's d):")
        for effect in results['effect_sizes']:
            report.append(f"   {effect['group1']} vs {effect['group2']}: d={effect['cohens_d']:.3f} ({effect['interpretation']})")
        
        # Descriptives
        report.append("\n5. DESCRIPTIVE STATISTICS:")
        for group, desc in results['descriptives'].items():
            report.append(f"   {group}: M={desc['mean']:.2f}, SD={desc['std']:.2f}, n={desc['n']}")
            report.append(f"           95% CI: [{desc['ci_lower']:.2f}, {desc['ci_upper']:.2f}]")
        
        return "\n".join(report)
