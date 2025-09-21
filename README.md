# SRLP Thesis Evaluation Pipeline

Self-Refinement for LLM Planners via Self-Checking Feedback (SRLP) framework against established baseline strategies across multiple AI providers.

## 🎯 Overview

This pipeline evaluates **4 reasoning strategies** across **5 domains** using **3 AI providers**:

- **Strategies**: SRLP, Chain-of-Thought (CoT), Tree-of-Thoughts (ToT), ReAct
- **Providers**: OpenAI GPT-4, Anthropic Claude-3, Google Gemini  
- **Domains**: Travel Planning, Software Project, Event Organization, Research Study, Business Launch
- **Total Experiments**: 5,400 (450 scenarios × 4 strategies × 3 providers)

## 🔄 System Architecture

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#fafafa",
    "primaryColor": "#2563eb",
    "primaryTextColor": "#ffffff",
    "primaryBorderColor": "#1e40af",
    "lineColor": "#374151",
    "secondaryColor": "#f3f4f6",
    "tertiaryColor": "#e5e7eb",
    "clusterBkg": "#f9fafb",
    "clusterBorder": "#d1d5db",
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "fontSize": "14px"
  }
}}%%

flowchart TD
    subgraph SRLP ["🔄 SRLP Framework: Self-Refinement for LLM Planners"]
        direction TB

        subgraph Stage1 ["📋 Stage 1: Plan Generation"]
            PG["🎯 Initial Plan Creation<br/>• Leverage LLM domain knowledge<br/>• Generate comprehensive plans<br/>• Establish baseline solutions"]
        end

        subgraph Stage2 ["🔍 Stage 2: Self-Assessment"]
            SA["📊 Critical Plan Evaluation<br/>• Identify potential weaknesses<br/>• Generate constructive feedback<br/>• Assess completeness & feasibility"]
        end

        subgraph Stage3 ["⚡ Stage 3: Iterative Refinement"]
            RF["🛠️ Plan Enhancement<br/>• Apply feedback systematically<br/>• Address identified gaps<br/>• Optimize solution quality"]
        end

        subgraph Stage4 ["✅ Stage 4: Quality Assurance"]
            QA["🎖️ Validation & Verification<br/>• Evaluate improvements<br/>• Ensure genuine enhancement<br/>• Prevent degradation"]
        end
    end

    %% Primary flow arrows with enhanced styling
    PG -->|"Generate Initial Plan"| SA
    SA -->|"Provide Feedback"| RF
    RF -->|"Apply Improvements"| QA
    QA -->|"Iterate if Needed"| PG

    %% Feedback loop annotation
    QA -.->|"Convergence Check"| FINAL["🏆 Final Optimized Plan"]

    %% Stage styling
    classDef stageBox fill:#2563eb,stroke:#1e40af,color:#ffffff,stroke-width:2px
    classDef processBox fill:#10b981,stroke:#059669,color:#ffffff,stroke-width:2px,font-weight:bold
    classDef finalBox fill:#dc2626,stroke:#b91c1c,color:#ffffff,stroke-width:3px,font-weight:bold

    class Stage1,Stage2,Stage3,Stage4 stageBox
    class PG,SA,RF,QA processBox
    class FINAL finalBox

    %% Link styling
    linkStyle 0 stroke:#10b981,stroke-width:3px
    linkStyle 1 stroke:#10b981,stroke-width:3px
    linkStyle 2 stroke:#10b981,stroke-width:3px
    linkStyle 3 stroke:#f59e0b,stroke-width:3px,stroke-dasharray: 5 5
    linkStyle 4 stroke:#dc2626,stroke-width:2px,stroke-dasharray: 3 3
```

### 🔍 Key Features Highlighted in Architecture:

- **🎯 SRLP Innovation**: Self-refinement mechanism with iterative improvement
- **📊 Statistical Rigor**: ANOVA testing with large effect sizes (d > 0.8)
- **🔬 Advanced Analysis**: Ablation study, human validation, efficiency analysis
- **📈 Publication Quality**: 300 DPI figures, LaTeX tables, comprehensive reports

## 🚀 Quick Start

### 1. Installation

```bash
pip install --break-system-packages -r requirements.txt
```

### 2. Secure API Key Setup

⚠️ **IMPORTANT**: API keys are now stored securely in environment variables.

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual API keys (NEVER commit this file!)
nano .env  # or your preferred editor
```

Required API Keys (add to `.env`):
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **Anthropic**: Get from https://console.anthropic.com/
- **Gemini**: Get from https://ai.google.dev/

Example `.env` format:
```bash
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
GEMINI_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

🔒 **Security Notes**: 
- The `.env` file is automatically ignored by Git
- Never commit API keys to version control
- Use `.env.example` as a template only
- If keys are missing, you'll get a clear error message

### 3. Run Complete Evaluation

```bash
python run_evaluation.py \
  --providers gpt4,claude3,gemini \
  --strategies srlp,cot,tot,react \
  --async \
  --workers 8 \
  --batch-size 300 \
  --log-level INFO \
  --resume-from auto
```

### 4. Dry Run (Validation)

```bash
python run_evaluation.py --dry-run
```

**Expected Output:**
```
Providers: 3 (gpt4, claude3, gemini)
Strategies: 4 (srlp, cot, tot, react)
Domains: 5
Scenarios: 450
Total experiments: 5400
```

### 5. Generate Artifacts from Existing Results

```bash
# Generate LaTeX tables and figures from evaluation results
python generate_artifacts.py results_full/evaluation_results.csv

# Custom output directory
python generate_artifacts.py results_full/evaluation_results.csv --output artifacts_custom
```

### 6. Verify Outputs

```bash
python run_evaluation.py --verify-outputs
```

## 📊 Evaluation Metrics

The pipeline implements four key metrics:

- **PQS (Plan Quality Score)**: 0-100 scale measuring solution completeness and quality
- **SCCS (Self-Check Confidence Score)**: 0-100 scale measuring confidence indicators  
- **IIR (Iteration Improvement Rate)**: 0-100 scale measuring iterative refinement
- **CEM (Cost Efficiency Metric)**: 0-100 scale measuring resource optimization

## 🧠 Strategy Implementations

### SRLP (Self-Refinement for LLM Planners)
4-stage process: Plan Generation → Self-Assessment → Refinement → Quality Assurance

### CoT (Chain-of-Thought)
Based on Wei et al. (2022), systematic step-by-step reasoning

### ToT (Tree-of-Thoughts)  
Based on Yao et al. (2024), branching exploration with evaluation

### ReAct (Reasoning and Acting)
Based on Yao et al. (2022), interleaved reasoning-action-observation cycles

## 📁 Project Structure

```
SRLP-thesis-/
├── src/                    # Core modules
│   ├── config.py          # Configuration management
│   ├── providers.py       # AI provider clients  
│   ├── strategies.py      # Strategy implementations
│   ├── scenarios.py       # Scenario generation
│   ├── metrics.py         # Metrics calculation
│   └── outputs.py         # Output generation
├── run_evaluation.py      # Main entry point
├── requirements.txt       # Dependencies
├── README.md             # This file
└── results_full/         # Output directory
```

## 🎨 Artifacts Generation

The pipeline includes a sophisticated artifacts generation system that creates publication-ready outputs:

### LaTeX Tables
- **Table 4.1**: PQS scores by strategy with statistical measures
- **Table 4.2**: Provider performance (time, cost, tokens, success rate)  
- **Table 4.3**: SRLP performance gains by domain vs baselines
- **Table 4.4**: PQS performance by complexity level and strategy
- **Table 4.5**: Strategic cognitive capability scores by dimension

### High-Quality Figures (300 DPI PNG)
- **Figure 4.1**: PQS distribution by strategy (box plots)
- **Figure 4.2**: Provider time and cost analysis (bar charts)
- **Figure 4.3**: SRLP gains by domain (grouped bar chart)
- **Figure 4.4**: PQS by complexity level (grouped bar chart)
- **Figure 4.5**: Strategic cognitive capabilities (radar chart)

### Key Features
- **Headless Operation**: Uses `MPLBACKEND=Agg` for server environments
- **Real Data Processing**: Computes actual statistics from evaluation results
- **Publication Ready**: Professional formatting for academic papers
- **Standalone Generation**: Can process existing CSV files independently

## 🔧 Configuration

API keys are securely stored in `src/config.py`. The system supports:

- **Async Processing**: Concurrent execution with configurable workers
- **Batching**: Configurable batch sizes for optimal performance
- **Checkpointing**: Resume capability for long-running evaluations  
- **Deterministic Results**: Fixed seeds for reproducible experiments

## 📈 Outputs

All results are saved to `results_full/`:

### Data Files
- `evaluation_results.csv` - Raw tabular results
- `detailed_results.json` - Complete results with metadata
- `scenarios.json` - Generated scenarios

### LaTeX Tables (Ready for Chapter 4)
- `table_4_1_pqs_by_strategy.tex` - PQS scores by strategy and provider
- `table_4_2_provider_time_cost.tex` - Provider performance metrics
- `table_4_3_domain_gains.tex` - SRLP gains by domain
- `table_4_4_pqs_by_complexity.tex` - Performance by complexity level
- `table_4_5_sccs_by_dimension.tex` - Cognitive capability scores

### Figures (PNG + PDF, 300 DPI)
- `figure_4_1_pqs_by_strategy` - PQS distribution by strategy
- `figure_4_2_provider_time_cost` - Time and cost analysis
- `figure_4_3_pqs_gain_by_domain` - Domain-specific gains
- `figure_4_4_pqs_by_complexity` - Complexity analysis
- `figure_4_5_sccs_by_dimension` - Cognitive dimensions

### Summary Report
- `RUN_SUMMARY.md` - Executive summary with key findings

## 🎛️ Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--providers` | `gpt4,claude3,gemini` | Comma-separated list of AI providers |
| `--strategies` | `srlp,cot,tot,react` | Comma-separated list of strategies |
| `--workers` | `8` | Number of concurrent workers |
| `--batch-size` | `300` | Tasks per batch |
| `--log-level` | `INFO` | Logging verbosity |
| `--resume-from` | `None` | Resume from checkpoint (`auto` or file path) |
| `--dry-run` | `False` | Validation mode only |

## 🔍 Key Features

### Real Baseline Implementations
All baseline strategies (CoT, ToT, ReAct) use prompts and techniques from the original research papers, not placeholders.

### Comprehensive Evaluation  
- 450 scenarios across 5 diverse domains
- Exactly 90 scenarios per domain with deterministic generation
- Complex scenarios with varying difficulty levels

## 📊 Expected Results

The evaluation produces valid numerical comparisons showing:

- **SRLP vs CoT**: Performance gains across domains
- **SRLP vs ToT**: Efficiency vs thoroughness trade-offs  
- **SRLP vs ReAct**: Action-oriented vs planning-oriented approaches
- **Provider Analysis**: Cost, speed, and quality comparisons
- **Domain Analysis**: Strategy effectiveness by problem type

## ⏱️ Estimated Runtime

- **Full Evaluation**: 6-8 hours (5,400 experiments)
- **Single Provider**: 2-3 hours (1,800 experiments)
- **Dry Run**: < 1 minute (validation only)

Times vary based on API response latencies and retry frequency.

## 🛠️ Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `--workers` or increase `--batch-size`
2. **Memory Issues**: Decrease `--batch-size` for large-scale runs  
3. **Network Timeouts**: Pipeline automatically retries with backoff
4. **Dependency Conflicts**: Use `--break-system-packages` flag

### Resume Capability

```bash
# Auto-resume from latest checkpoint
python run_evaluation.py --resume-from auto

# Resume from specific checkpoint  
python run_evaluation.py --resume-from results_full/checkpoint_latest.json
```

## ✅ Validation

The pipeline includes comprehensive validation:

- ✅ Exactly 450 scenarios (90 per domain)
- ✅ All 4 strategies with real implementations
- ✅ All 3 providers properly integrated
- ✅ 5,400 total experiments enumerated
- ✅ Valid numerical results (no NaN/zeros)
- ✅ All output artifacts generated

## 📖 Citation

```bibtex
@misc{srlp_evaluation_2025,
  title={Self-Refinement for LLM Planners via Self-Checking Feedback},
  author={[Mohamed ElhajSuliman Elnaim Suliman]},
  year={03.09.2025},
  note={Thesis Evaluation Pipeline - 5,400 experiments across 3 AI providers}
}
```

