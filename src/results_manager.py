"""
Results Manager for preserving raw evaluation data with merge capabilities.
Ensures academic integrity by never overwriting existing evaluation results.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultsManager:
    """
    Manages preservation and merging of raw evaluation results.
    
    Key principles:
    1. Never overwrite existing raw results
    2. Merge new results by unique keys
    3. Preserve data integrity for academic use
    4. Log all operations clearly
    """
    
    def __init__(self, results_dir: Path, preserve_raw: bool = True):
        """
        Initialize results manager.
        
        Args:
            results_dir: Directory for results storage
            preserve_raw: If True, preserve existing raw results
        """
        self.results_dir = Path(results_dir)
        self.preserve_raw = preserve_raw
        self.csv_path = self.results_dir / "evaluation_results.csv"
        self.json_path = self.results_dir / "detailed_results.json"
        self.tmp_dir = self.results_dir / "tmp"
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Unique key columns for merging
        self.unique_key_cols = ['scenario_id', 'strategy', 'provider']
    
    def load_existing_results(self) -> pd.DataFrame:
        """Load existing CSV results if they exist."""
        if self.csv_path.exists() and self.preserve_raw:
            try:
                df = pd.read_csv(self.csv_path)
                logger.info(f"[LOAD] Found existing results: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"[LOAD] Failed to load existing CSV: {e}")
                return pd.DataFrame()
        else:
            logger.info("[LOAD] No existing results found, starting fresh")
            return pd.DataFrame()
    
    def load_existing_detailed(self) -> List[Dict[str, Any]]:
        """Load existing detailed JSON results if they exist."""
        if self.json_path.exists() and self.preserve_raw:
            try:
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"[LOAD] Found existing detailed results: {len(data)} entries")
                return data
            except Exception as e:
                logger.warning(f"[LOAD] Failed to load existing JSON: {e}")
                return []
        else:
            logger.info("[LOAD] No existing detailed results found, starting fresh")
            return []
    
    def merge_results(self, existing_df: pd.DataFrame, new_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Merge new results with existing ones.
        
        Strategy:
        1. Convert new results to DataFrame
        2. For existing keys, only update if data is missing
        3. Add completely new rows
        4. Preserve original data integrity
        """
        if not new_results:
            logger.info("[MERGE] No new results to merge")
            return existing_df
        
        # Convert new results to DataFrame
        new_df = pd.DataFrame(new_results)
        
        if existing_df.empty:
            logger.info(f"[MERGE] No existing data, adding {len(new_df)} new rows")
            return new_df
        
        # Ensure both DataFrames have the same columns
        all_columns = list(set(existing_df.columns) | set(new_df.columns))
        for col in all_columns:
            if col not in existing_df.columns:
                existing_df[col] = None
            if col not in new_df.columns:
                new_df[col] = None
        
        # Create unique keys for merging
        if all(col in existing_df.columns for col in self.unique_key_cols):
            existing_df['_merge_key'] = existing_df[self.unique_key_cols].apply(
                lambda x: '|'.join(map(str, x)), axis=1
            )
        else:
            logger.warning(f"[MERGE] Missing key columns, using row-based merge")
            existing_df['_merge_key'] = existing_df.index.astype(str)
        
        if all(col in new_df.columns for col in self.unique_key_cols):
            new_df['_merge_key'] = new_df[self.unique_key_cols].apply(
                lambda x: '|'.join(map(str, x)), axis=1
            )
        else:
            new_df['_merge_key'] = 'new_' + new_df.index.astype(str)
        
        # Identify truly new rows vs. updates
        existing_keys = set(existing_df['_merge_key'])
        new_keys = set(new_df['_merge_key'])
        
        completely_new = new_keys - existing_keys
        potential_updates = new_keys & existing_keys
        
        # Add completely new rows
        new_rows_df = new_df[new_df['_merge_key'].isin(completely_new)]
        merged_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
        
        # Handle potential updates (only fill missing data)
        updates_count = 0
        for key in potential_updates:
            existing_row_idx = existing_df[existing_df['_merge_key'] == key].index[0]
            new_row = new_df[new_df['_merge_key'] == key].iloc[0]
            
            # Only update missing/null values
            for col in new_df.columns:
                if col != '_merge_key':
                    existing_val = merged_df.at[existing_row_idx, col]
                    new_val = new_row[col]
                    
                    # Update if existing is null/missing and new has value
                    if (pd.isna(existing_val) or existing_val is None) and not (pd.isna(new_val) or new_val is None):
                        merged_df.at[existing_row_idx, col] = new_val
                        updates_count += 1
        
        # Clean up merge keys
        merged_df = merged_df.drop(columns=['_merge_key'])
        
        logger.info(f"[MERGE] Added {len(completely_new)} new rows, updated {updates_count} missing values")
        logger.info(f"[MERGE] Total rows: {len(existing_df)} → {len(merged_df)}")
        
        return merged_df
    
    def merge_detailed_results(self, existing_detailed: List[Dict[str, Any]], 
                             new_detailed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge detailed JSON results."""
        if not new_detailed:
            return existing_detailed
        
        # Create lookup by unique key
        existing_lookup = {}
        for item in existing_detailed:
            if all(key in item for key in self.unique_key_cols):
                key = '|'.join(str(item[col]) for col in self.unique_key_cols)
                existing_lookup[key] = item
        
        # Merge new items
        merged_detailed = list(existing_detailed)  # Copy existing
        new_count = 0
        update_count = 0
        
        for new_item in new_detailed:
            if all(key in new_item for key in self.unique_key_cols):
                key = '|'.join(str(new_item[col]) for col in self.unique_key_cols)
                
                if key not in existing_lookup:
                    # Completely new item
                    merged_detailed.append(new_item)
                    new_count += 1
                else:
                    # Update existing item with missing fields
                    existing_item = existing_lookup[key]
                    for field, value in new_item.items():
                        if field not in existing_item or existing_item[field] is None:
                            existing_item[field] = value
                            update_count += 1
            else:
                # No key columns, just append
                merged_detailed.append(new_item)
                new_count += 1
        
        logger.info(f"[MERGE] Detailed: Added {new_count} new entries, updated {update_count} fields")
        
        return merged_detailed
    
    def save_results(self, new_results: List[Dict[str, Any]], 
                    new_detailed: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Save results with preservation logic.
        
        Args:
            new_results: New CSV-compatible results
            new_detailed: Optional detailed results for JSON
        """
        if not self.preserve_raw:
            # Old behavior: overwrite
            self._save_direct(new_results, new_detailed)
            return
        
        # Load existing data
        existing_df = self.load_existing_results()
        existing_detailed = self.load_existing_detailed()
        
        # Merge with new data
        merged_df = self.merge_results(existing_df, new_results)
        
        if new_detailed is not None:
            merged_detailed = self.merge_detailed_results(existing_detailed, new_detailed)
        else:
            merged_detailed = existing_detailed
        
        # Create backup before saving
        self._create_backup()
        
        # Save merged results
        try:
            # Save CSV
            merged_df.to_csv(self.csv_path, index=False)
            logger.info(f"[SAVE] Preserved evaluation_results.csv with {len(merged_df)} rows")
            
            # Save JSON
            if merged_detailed:
                with open(self.json_path, 'w') as f:
                    json.dump(merged_detailed, f, indent=2, default=str)
                logger.info(f"[SAVE] Preserved detailed_results.json with {len(merged_detailed)} entries")
            
            # Log summary
            self._log_save_summary(merged_df, merged_detailed)
            
        except Exception as e:
            logger.error(f"[SAVE] Failed to save results: {e}")
            # Restore backup if available
            self._restore_backup()
            raise
    
    def _save_direct(self, new_results: List[Dict[str, Any]], 
                    new_detailed: Optional[List[Dict[str, Any]]] = None) -> None:
        """Direct save without preservation (legacy mode)."""
        df = pd.DataFrame(new_results)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"[SAVE] Overwrote evaluation_results.csv with {len(df)} rows")
        
        if new_detailed:
            with open(self.json_path, 'w') as f:
                json.dump(new_detailed, f, indent=2, default=str)
            logger.info(f"[SAVE] Overwrote detailed_results.json with {len(new_detailed)} entries")
    
    def _create_backup(self) -> None:
        """Create backup of existing files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.csv_path.exists():
            backup_csv = self.tmp_dir / f"evaluation_results_backup_{timestamp}.csv"
            backup_csv.write_bytes(self.csv_path.read_bytes())
            logger.debug(f"[BACKUP] Created CSV backup: {backup_csv}")
        
        if self.json_path.exists():
            backup_json = self.tmp_dir / f"detailed_results_backup_{timestamp}.json"
            backup_json.write_bytes(self.json_path.read_bytes())
            logger.debug(f"[BACKUP] Created JSON backup: {backup_json}")
    
    def _restore_backup(self) -> None:
        """Restore from most recent backup."""
        # Find most recent backups
        csv_backups = list(self.tmp_dir.glob("evaluation_results_backup_*.csv"))
        json_backups = list(self.tmp_dir.glob("detailed_results_backup_*.json"))
        
        if csv_backups:
            latest_csv = max(csv_backups, key=lambda x: x.stat().st_mtime)
            self.csv_path.write_bytes(latest_csv.read_bytes())
            logger.info(f"[RESTORE] Restored CSV from backup: {latest_csv}")
        
        if json_backups:
            latest_json = max(json_backups, key=lambda x: x.stat().st_mtime)
            self.json_path.write_bytes(latest_json.read_bytes())
            logger.info(f"[RESTORE] Restored JSON from backup: {latest_json}")
    
    def _log_save_summary(self, df: pd.DataFrame, detailed: List[Dict[str, Any]]) -> None:
        """Log detailed summary of saved results."""
        logger.info("="*60)
        logger.info("RAW RESULTS PRESERVATION SUMMARY")
        logger.info("="*60)
        logger.info(f"CSV Rows: {len(df)}")
        logger.info(f"JSON Entries: {len(detailed) if detailed else 0}")
        
        if not df.empty:
            logger.info(f"Strategies: {sorted(df['strategy'].unique())}")
            logger.info(f"Providers: {sorted(df['provider'].unique())}")
            logger.info(f"Domains: {sorted(df['domain'].unique()) if 'domain' in df.columns else 'N/A'}")
        
        logger.info(f"Files Location: {self.results_dir}")
        logger.info(f"Preserve Mode: {self.preserve_raw}")
        logger.info("="*60)
    
    def get_temp_file(self, name: str) -> Path:
        """Get path for temporary processing file."""
        return self.tmp_dir / name
    
    def cleanup_temp_files(self, older_than_hours: int = 24) -> None:
        """Clean up old temporary files."""
        import time
        current_time = time.time()
        
        cleaned_count = 0
        for temp_file in self.tmp_dir.iterdir():
            if temp_file.is_file():
                file_age_hours = (current_time - temp_file.stat().st_mtime) / 3600
                if file_age_hours > older_than_hours:
                    temp_file.unlink()
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"[CLEANUP] Removed {cleaned_count} temporary files older than {older_than_hours}h")
