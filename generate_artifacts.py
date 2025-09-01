#!/usr/bin/env python3
"""
Standalone script to generate artifacts (LaTeX tables and figures) from evaluation results.
This script can be run independently to process existing CSV results.
"""

import os
import sys
import argparse
from pathlib import Path

# Set headless backend for matplotlib
os.environ['MPLBACKEND'] = 'Agg'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main function for artifacts generation."""
    parser = argparse.ArgumentParser(
        description='Generate LaTeX tables and PNG figures from evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate artifacts from default results
  python generate_artifacts.py results_full/evaluation_results.csv

  # Generate artifacts with custom output directory
  python generate_artifacts.py results_full/evaluation_results.csv --output artifacts_custom

  # Generate artifacts from test data
  python generate_artifacts.py test_results/evaluation_results.csv
        """
    )
    
    parser.add_argument('results_csv', 
                       help='Path to evaluation results CSV file')
    parser.add_argument('--output', '-o', default='artifacts',
                       help='Output directory for artifacts (default: artifacts)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"❌ Error: Results file not found: {csv_path}")
        sys.exit(1)
    
    if not csv_path.suffix.lower() == '.csv':
        print(f"❌ Error: Input file must be a CSV: {csv_path}")
        sys.exit(1)
    
    # Import artifacts generator
    try:
        from artifacts import ArtifactsGenerator
    except ImportError as e:
        print(f"❌ Error: Failed to import artifacts module: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    print("🚀 Starting artifacts generation...")
    print(f"📁 Input: {csv_path.absolute()}")
    print(f"📁 Output: {Path(args.output).absolute()}")
    
    try:
        # Generate artifacts
        generator = ArtifactsGenerator(str(csv_path), args.output)
        generator.generate_all_artifacts()
        generator.print_summary()
        
        print("\n✅ Artifacts generation completed successfully!")
        
        # Show what was generated
        artifacts_dir = Path(args.output)
        print(f"\n📋 Generated files in {artifacts_dir.absolute()}:")
        
        # List LaTeX tables
        tex_files = list(artifacts_dir.glob("*.tex"))
        if tex_files:
            print("\n📄 LaTeX Tables:")
            for tex_file in sorted(tex_files):
                print(f"  ✓ {tex_file.name}")
        
        # List figures
        png_files = list(artifacts_dir.glob("*.png"))
        if png_files:
            print("\n🖼️  Figures:")
            for png_file in sorted(png_files):
                size_kb = png_file.stat().st_size / 1024
                print(f"  ✓ {png_file.name} ({size_kb:.1f} KB)")
        
        print(f"\n🎯 Ready for thesis Chapter 4!")
        
    except Exception as e:
        print(f"❌ Error during artifacts generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
