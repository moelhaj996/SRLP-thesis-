#!/usr/bin/env python3
"""
Demonstration script for the refactored SRLP domain gains plot.
Shows both point gains (ΔPQS) and percentage gains with bootstrap CI.
"""

import os
import sys
from pathlib import Path

# Set headless backend
os.environ['MPLBACKEND'] = 'Agg'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Demonstrate both domain gains metrics."""
    
    # Create test data if needed
    print("Creating sample evaluation data...")
    from test_artifacts import create_sample_data
    df = create_sample_data()
    
    # Create test directory
    demo_dir = Path("domain_gains_demo")
    demo_dir.mkdir(exist_ok=True)
    
    # Save sample CSV
    csv_path = demo_dir / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Sample data created: {len(df)} results")
    
    # Import artifacts generator
    from artifacts import ArtifactsGenerator
    
    # Create generator
    generator = ArtifactsGenerator(str(csv_path), str(demo_dir))
    
    print("\n" + "="*60)
    print("GENERATING DOMAIN GAINS WITH BOTH METRICS")
    print("="*60)
    
    # Generate point gains version
    print("\n1. Generating point gains (ΔPQS) version...")
    generator._generate_figure_domain_gains(metric="pp")
    
    # Rename the output
    (demo_dir / "figure_4_3_pqs_gain_by_domain.png").rename(
        demo_dir / "figure_4_3_point_gains.png")
    
    # Generate percentage gains version  
    print("2. Generating percentage gains version...")
    generator._generate_figure_domain_gains(metric="pct")
    
    # Rename the output
    (demo_dir / "figure_4_3_pqs_gain_by_domain.png").rename(
        demo_dir / "figure_4_3_percentage_gains.png")
    
    # Generate the enhanced LaTeX table
    print("3. Generating enhanced LaTeX table...")
    generator._generate_table_domain_gains()
    
    # Show the statistics
    print("\n" + "="*60)
    print("DOMAIN STATISTICS SUMMARY")
    print("="*60)
    
    for domain, stats in generator.domain_stats.items():
        print(f"\n{domain.replace('_', ' ').title()}:")
        print(f"  vs CoT:   ΔPQS={stats['cot_gain_pp']:.1f},   Gain={stats['cot_gain_pct']:.1f}%")
        print(f"  vs ToT:   ΔPQS={stats['tot_gain_pp']:.1f},   Gain={stats['tot_gain_pct']:.1f}%") 
        print(f"  vs ReAct: ΔPQS={stats['react_gain_pp']:.1f}, Gain={stats['react_gain_pct']:.1f}%")
    
    # List generated files
    print(f"\n" + "="*60)
    print("GENERATED FILES")
    print("="*60)
    
    files = [
        "figure_4_3_point_gains.png",
        "figure_4_3_percentage_gains.png", 
        "table_4_3_domain_gains.tex"
    ]
    
    for file in files:
        file_path = demo_dir / file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"✓ {file} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {file} (missing)")
    
    print(f"\nDemo files saved to: {demo_dir.absolute()}")
    
    # Show LaTeX table content
    table_file = demo_dir / "table_4_3_domain_gains.tex"
    if table_file.exists():
        print(f"\n" + "="*60)
        print("ENHANCED LATEX TABLE PREVIEW")
        print("="*60)
        with open(table_file, 'r') as f:
            content = f.read()
            # Show first few lines
            lines = content.split('\n')
            for i, line in enumerate(lines[:15]):
                print(f"{i+1:2d}: {line}")
            if len(lines) > 15:
                print("...")
    
    print(f"\n🎯 Domain gains refactoring completed successfully!")
    print(f"📊 Both point gains (ΔPQS) and percentage gains available")
    print(f"📈 Bootstrap confidence intervals (2000 resamples)")
    print(f"📋 Enhanced LaTeX table with dual metrics")

if __name__ == "__main__":
    main()
