# Raw Results Preservation System

## Overview
The evaluation pipeline now includes a robust **Raw Results Preservation System** that ensures your evaluation data is never lost or overwritten, making it safe for academic use, GitHub commits, and university submission.

## Key Features

### 🔒 **Never Overwrite Raw Data**
- ✅ `evaluation_results.csv` is **NEVER** overwritten when `preserve_raw_results=True`
- ✅ `detailed_results.json` is **NEVER** overwritten when `preserve_raw_results=True`  
- ✅ New results are intelligently merged with existing data
- ✅ Original data integrity is always maintained

### 🔄 **Intelligent Merging**
- **Unique Key**: `[scenario_id, strategy, provider]`
- **New Results**: Completely new combinations are added
- **Existing Results**: Only missing fields are updated, never overwritten
- **Conflict Resolution**: Original data takes precedence

### 📦 **Resume Functionality**
- ✅ `--resume-from auto` automatically detects completed tasks
- ✅ Skips already-completed scenario+strategy+provider combinations
- ✅ Only executes remaining tasks
- ✅ Perfect for interrupted runs or incremental evaluation

### 💾 **Backup System**
- ✅ Automatic backups before any save operation
- ✅ Timestamped backup files in `results/tmp/` directory
- ✅ Automatic recovery on save failures
- ✅ Configurable cleanup of old backup files

### 📊 **Academic Integrity**
- ✅ Raw results remain exactly as generated
- ✅ Safe for GitHub version control
- ✅ Suitable for university thesis submission
- ✅ Transparent merge logging for audit trails

## Configuration

### Enable Preservation (Default)
```python
# In src/config.py
preserve_raw_results: bool = True  # Default: ENABLED
```

### Disable Preservation (Legacy Mode)
```python
# In src/config.py  
preserve_raw_results: bool = False  # Use only for testing
```

## Usage Examples

### Fresh Run
```bash
python run_evaluation.py --providers gpt4,claude3 --strategies srlp,cot
# Creates: results/evaluation_results.csv with new data
```

### Resume Interrupted Run
```bash
python run_evaluation.py --providers gpt4,claude3 --strategies srlp,cot --resume-from auto
# Preserves: existing results, only runs missing combinations
```

### Add New Strategy
```bash
# First run: SRLP + CoT
python run_evaluation.py --strategies srlp,cot

# Later: Add ToT strategy  
python run_evaluation.py --strategies srlp,cot,tot --resume-from auto
# Result: Preserves SRLP+CoT data, adds only ToT results
```

### Add New Provider
```bash
# First run: GPT-4 only
python run_evaluation.py --providers gpt4

# Later: Add Claude-3
python run_evaluation.py --providers gpt4,claude3 --resume-from auto  
# Result: Preserves GPT-4 data, adds only Claude-3 results
```

## File Structure

### Raw Results (PRESERVED)
```
results/
├── evaluation_results.csv      [NEVER OVERWRITTEN]
├── detailed_results.json       [NEVER OVERWRITTEN]
└── tmp/                        [Backup files]
    ├── evaluation_results_backup_20240830_150948.csv
    └── detailed_results_backup_20240830_150948.json
```

### Generated Artifacts (Regenerated)
```
results/
├── artifacts/                  [Always regenerated from CSV]
│   ├── figure_4_1_pqs_by_strategy.png
│   ├── table_4_1_pqs_by_strategy.tex
│   └── ... [all other figures/tables]
├── scenarios.json              [Metadata, can be regenerated]
└── RUN_SUMMARY.md             [Summary, can be regenerated]
```

## Merge Logic

### Scenario: Adding New Results

**Existing CSV:**
```csv
scenario_id,strategy,provider,pqs
travel_001,srlp,gpt4,75.5
travel_001,cot,gpt4,68.2
```

**New Results:**
```csv
scenario_id,strategy,provider,pqs
travel_001,tot,gpt4,72.1        # New combination -> ADD
travel_001,srlp,gpt4,77.8       # Existing combination -> PRESERVE original (75.5)
travel_002,srlp,gpt4,80.3       # New scenario -> ADD
```

**Final CSV:**
```csv
scenario_id,strategy,provider,pqs
travel_001,srlp,gpt4,75.5       # PRESERVED original
travel_001,cot,gpt4,68.2        # PRESERVED original  
travel_001,tot,gpt4,72.1        # ADDED new
travel_002,srlp,gpt4,80.3       # ADDED new
```

### Scenario: Filling Missing Data

**Existing CSV:**
```csv
scenario_id,strategy,provider,pqs,cost_usd
travel_001,srlp,gpt4,75.5,      # Missing cost
```

**New Results:**
```csv
scenario_id,strategy,provider,pqs,cost_usd
travel_001,srlp,gpt4,77.8,0.08  # Has cost data
```

**Final CSV:**
```csv
scenario_id,strategy,provider,pqs,cost_usd
travel_001,srlp,gpt4,75.5,0.08  # PQS preserved, cost filled
```

## Resume Detection

### Task Key Generation
```python
# Each task identified by unique key
task_key = f"{scenario_id}|{strategy}|{provider}"

# Examples:
# "travel_001|srlp|gpt4"
# "software_003|cot|claude3"  
# "event_015|tot|gemini"
```

### Resume Process
1. **Load Existing**: Read `evaluation_results.csv`
2. **Generate Keys**: Create task keys from existing results
3. **Filter Tasks**: Skip tasks with matching keys
4. **Execute Remaining**: Run only new/missing combinations

### Resume Output Example
```
Checking for existing results to resume from...
Found 1,250 existing results to resume from

Task generation summary:
  - Total possible tasks: 5,400
  - Already completed: 1,250  
  - Remaining to execute: 4,150

Resume mode: Skipping 1,250 completed tasks
New tasks to execute: 4,150
```

## Logging and Monitoring

### Preservation Status Logging
```
[LOAD] Found existing results: 1,250 rows
[MERGE] Added 150 new rows, updated 23 missing values  
[MERGE] Total rows: 1,250 → 1,400
[SAVE] Preserved evaluation_results.csv with 1,400 rows
```

### Detailed Summary
```
============================================================
RAW RESULTS PRESERVATION SUMMARY
============================================================  
CSV Rows: 1,400
JSON Entries: 1,400
Strategies: ['cot', 'react', 'srlp', 'tot']
Providers: ['claude3', 'gemini', 'gpt4']
Domains: ['business_launch', 'event_organization', 'research_study', 'software_project', 'travel_planning']
Files Location: results_full_clean
Preserve Mode: True
============================================================
```

### Final Status Report
```
All outputs generated in: results_full_clean
Raw results preservation status:
  - evaluation_results.csv: PRESERVED
  - detailed_results.json: PRESERVED
Generated files:
  evaluation_results.csv [PRESERVED]
  detailed_results.json [PRESERVED]
  artifacts/figure_4_1_pqs_by_strategy.png
  artifacts/table_4_1_pqs_by_strategy.tex
  ... [other generated files]
```

## Safety Features

### ✅ **Automatic Backups**
- Created before every save operation
- Timestamped for easy identification
- Stored in `results/tmp/` directory
- Automatic restore on failure

### ✅ **Merge Validation**
- Unique key conflict detection
- Column compatibility checking  
- Data type preservation
- Error handling with rollback

### ✅ **Academic Compliance**
- Never modify original experimental data
- Full audit trail through logging
- Version-control friendly
- Reproducible results

### ✅ **Flexible Recovery**
- Resume from any interruption point
- Partial run completion
- Incremental strategy addition
- Provider expansion support

## Testing

### Comprehensive Test Suite
```bash
python test_preservation.py
```

**Tests Include:**
- ✅ Initial save functionality
- ✅ Merge logic with overlapping data
- ✅ Resume task detection
- ✅ Backup and recovery
- ✅ Legacy mode compatibility
- ✅ Edge case handling

### Expected Test Output
```
✅ Preserved mode: 23 total results
✅ Unique scenarios: 5
✅ Unique strategies: 3  
✅ Unique providers: 3
✅ Temp files: 1 backup files created
✅ Resume detection: 23 completed tasks identified

🎯 Raw results preservation system is working correctly!
🔒 Your evaluation data will never be lost or overwritten.
📊 Safe for GitHub commits and university submission.
```

## Best Practices

### ✅ **DO:**
- Keep `preserve_raw_results=True` (default)
- Use `--resume-from auto` for interrupted runs
- Commit raw CSV files to version control
- Monitor preservation status logs

### ❌ **DON'T:**
- Manually edit `evaluation_results.csv`
- Delete `tmp/` backup files during runs
- Set `preserve_raw_results=False` in production
- Interrupt saves in progress

### 📋 **Academic Workflow:**
1. **Initial Run**: Full evaluation with all strategies
2. **Commit Results**: Add CSV/JSON to Git repository  
3. **Generate Artifacts**: LaTeX tables and figures from preserved data
4. **Incremental Updates**: Add new strategies/providers as needed
5. **Final Submission**: All raw data preserved and auditable

---

**Status**: ✅ **Fully Implemented and Tested**  
**Default Behavior**: Raw results preservation ENABLED  
**Academic Compliance**: Full data integrity and audit trail  
**Recovery**: Automatic backup and resume capabilities
