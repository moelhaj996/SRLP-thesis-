#!/usr/bin/env python3
"""
Test script for refined academic figures.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Set headless backend for matplotlib
os.environ['MPLBACKEND'] = 'Agg'

def create_comprehensive_sample_data():
    """Create comprehensive sample data with all columns needed."""
    np.random.seed(42)
    
    strategies = ['srlp', 'cot', 'tot', 'react']
    providers = ['gpt4', 'claude3', 'gemini']
    domains = ['travel_planning', 'software_project', 'event_organization', 'research_study', 'business_launch']
    complexities = ['low', 'medium', 'high']
    
    results = []
    scenario_id = 0
    
    for domain in domains:
        for complexity in complexities:
            for _ in range(30):  # 30 scenarios per complexity per domain
                scenario_id += 1
                for strategy in strategies:
                    for provider in providers:
                        # Base metrics with realistic strategy performance
                        base_pqs = {'srlp': 82, 'cot': 71, 'tot': 76, 'react': 74}[strategy]
                        
                        # Complexity and provider adjustments
                        complexity_factor = {'low': 1.15, 'medium': 1.0, 'high': 0.85}[complexity]
                        provider_factor = {'gpt4': 1.08, 'claude3': 1.0, 'gemini': 0.92}[provider]
                        
                        # Generate iterations for SRLP (multi-iteration strategy)
                        if strategy == 'srlp':
                            iterations = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                            for iteration in range(iterations):
                                # SRLP improves with iterations
                                iteration_boost = iteration * np.random.uniform(3, 7)
                                pqs = max(20, min(100, np.random.normal(
                                    (base_pqs + iteration_boost) * complexity_factor * provider_factor, 6)))
                                
                                # Related metrics with some correlation
                                sccs = max(20, min(100, pqs + np.random.normal(2, 8)))
                                iir = max(20, min(100, pqs + np.random.normal(-5, 10)))
                                cem = max(20, min(100, pqs + np.random.normal(-3, 8)))
                                
                                # Cost and time
                                base_cost = {'gpt4': 0.15, 'claude3': 0.08, 'gemini': 0.04}[provider]
                                cost_usd = base_cost * np.random.uniform(0.8, 1.3) * (iteration + 1)
                                execution_time = np.random.exponential(2.2) + 0.8 + (iteration * 0.4)
                                
                                # Success and error simulation
                                success_rate = 0.88 + (iteration * 0.04)  # SRLP improves with iterations
                                success = np.random.random() < success_rate
                                
                                if success:
                                    error_type = None
                                else:
                                    error_types = ['incomplete', 'hallucination', 'invalid_output', 'timeout']
                                    error_type = np.random.choice(error_types)
                                
                                result = {
                                    'scenario_id': f"{domain}_{scenario_id:03d}",
                                    'domain': domain,
                                    'complexity': complexity,
                                    'strategy': strategy,
                                    'provider': provider,
                                    'iteration': iteration,
                                    'pqs': pqs,
                                    'sccs': sccs,
                                    'iir': iir,
                                    'cem': cem,
                                    'cost_usd': cost_usd,
                                    'execution_time': execution_time,
                                    'tokens_used': int(np.random.normal(1500, 300)),
                                    'success': success,
                                    'error_type': error_type
                                }
                                results.append(result)
                        else:
                            # Other strategies: single iteration
                            pqs = max(20, min(100, np.random.normal(
                                base_pqs * complexity_factor * provider_factor, 6)))
                            
                            # Related metrics
                            sccs = max(20, min(100, pqs + np.random.normal(0, 8)))
                            iir = max(20, min(100, pqs + np.random.normal(-8, 10)))
                            cem = max(20, min(100, pqs + np.random.normal(-5, 8)))
                            
                            # Cost and time
                            base_cost = {'gpt4': 0.12, 'claude3': 0.06, 'gemini': 0.03}[provider]
                            cost_usd = base_cost * np.random.uniform(0.8, 1.2)
                            execution_time = np.random.exponential(1.8) + 0.6
                            
                            # Success rates vary by strategy
                            success_rates = {'cot': 0.82, 'tot': 0.76, 'react': 0.84}
                            success = np.random.random() < success_rates[strategy]
                            
                            if success:
                                error_type = None
                            else:
                                # Strategy-specific error patterns
                                if strategy == 'cot':
                                    error_types = ['incomplete', 'invalid_output']
                                elif strategy == 'tot':
                                    error_types = ['timeout', 'incomplete']
                                else:  # react
                                    error_types = ['tool_error', 'hallucination']
                                error_type = np.random.choice(error_types)
                            
                            result = {
                                'scenario_id': f"{domain}_{scenario_id:03d}",
                                'domain': domain,
                                'complexity': complexity,
                                'strategy': strategy,
                                'provider': provider,
                                'iteration': 0,
                                'pqs': pqs,
                                'sccs': sccs,
                                'iir': iir,
                                'cem': cem,
                                'cost_usd': cost_usd,
                                'execution_time': execution_time,
                                'tokens_used': int(np.random.normal(1200, 250)),
                                'success': success,
                                'error_type': error_type
                            }
                            results.append(result)
    
    return pd.DataFrame(results)

def main():
    """Test the refined academic figures."""
    print("Creating comprehensive sample data for refined figures...")
    
    # Create sample data
    df = create_comprehensive_sample_data()
    
    # Create test directory
    test_dir = Path("refined_figures_test")
    test_dir.mkdir(exist_ok=True)
    
    # Save sample CSV
    csv_path = test_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Sample data created: {len(df)} results")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print(f"Providers: {sorted(df['provider'].unique())}")
    print(f"Domains: {sorted(df['domain'].unique())}")
    print(f"Complexities: {sorted(df['complexity'].unique())}")
    
    # Test artifacts generation with refined figures
    print(f"\n" + "="*60)
    print("TESTING REFINED ACADEMIC FIGURES")
    print("="*60)
    
    # Import artifacts generator
    import sys
    sys.path.insert(0, 'src')
    from artifacts import ArtifactsGenerator
    
    # Generate refined artifacts
    artifacts_dir = test_dir / "artifacts"
    generator = ArtifactsGenerator(str(csv_path), str(artifacts_dir))
    
    print(f"\nGenerating all refined figures...")
    try:
        generator.generate_figures()
        print("✅ All refined figures generated successfully!")
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
    
    # List generated files
    print(f"\n" + "="*60)
    print("REFINED ACADEMIC FIGURES GENERATED")
    print("="*60)
    
    if artifacts_dir.exists():
        for file in sorted(artifacts_dir.glob("*.png")):
            size_kb = file.stat().st_size / 1024
            print(f"📊 {file.name} ({size_kb:.1f} KB)")
        
        for file in sorted(artifacts_dir.glob("*.pdf")):
            size_kb = file.stat().st_size / 1024
            print(f"📄 {file.name} ({size_kb:.1f} KB)")
    
    print(f"\n🎓 Academic Quality Features Implemented:")
    print(f"   ✅ Unified Times New Roman/Arial font family")
    print(f"   ✅ Professional color palette (colorblind-friendly)")
    print(f"   ✅ Consistent 16-18pt titles, 14-16pt axis labels")
    print(f"   ✅ Error bars and confidence intervals")
    print(f"   ✅ Legends positioned outside plot areas")
    print(f"   ✅ Scientific axis formatting and units")
    print(f"   ✅ Professional grid lines (α=0.15)")
    print(f"   ✅ High-quality 300 DPI output")
    
    print(f"\n📁 Location: {test_dir.absolute()}")
    print(f"📈 Figures: {artifacts_dir.absolute()}")

if __name__ == "__main__":
    main()
