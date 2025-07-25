#!/usr/bin/env python3
"""
Migration script to consolidate thousands of embedding cache files into SQLite.
This addresses the performance bottleneck of 8,650+ individual .pkl files.
"""

import os
import pickle
import sqlite3
import logging
from pathlib import Path
from tqdm import tqdm

from config.settings import BASE_TMP_DIR
from utils.consolidated_cache import ConsolidatedCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_embeddings_cache():
    """Migrate from individual .pkl files to consolidated SQLite cache."""
    
    old_cache_dir = os.path.join(BASE_TMP_DIR, "embeddings_cache")
    if not os.path.exists(old_cache_dir):
        logger.info("No existing embeddings cache found - nothing to migrate")
        return
    
    # Count existing files
    pkl_files = list(Path(old_cache_dir).glob("*.pkl"))
    if not pkl_files:
        logger.info("No .pkl files found in cache directory")
        return
    
    logger.info(f"Found {len(pkl_files)} embedding cache files to migrate")
    
    # Initialize consolidated cache
    consolidated_cache = ConsolidatedCache(old_cache_dir, max_size_mb=150)
    
    migrated_count = 0
    failed_count = 0
    
    # Migrate each file
    for pkl_file in tqdm(pkl_files, desc="Migrating embeddings"):
        try:
            # Load the pickle file
            with open(pkl_file, 'rb') as f:
                embedding_data = pickle.load(f)
            
            # Generate cache key from filename
            filename = pkl_file.stem
            cache_key = f"embedding:migrated:{filename}"
            
            # Store in consolidated cache
            consolidated_cache.set(cache_key, embedding_data, filename, ttl_hours=168)
            
            migrated_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to migrate {pkl_file}: {e}")
            failed_count += 1
    
    logger.info(f"Migration completed: {migrated_count} migrated, {failed_count} failed")
    
    # Create backup directory for old files
    backup_dir = os.path.join(old_cache_dir, "pkl_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Move old files to backup (don't delete immediately)
    moved_count = 0
    for pkl_file in pkl_files:
        try:
            backup_path = os.path.join(backup_dir, pkl_file.name)
            pkl_file.rename(backup_path)
            moved_count += 1
        except Exception as e:
            logger.warning(f"Failed to move {pkl_file}: {e}")
    
    logger.info(f"Moved {moved_count} old cache files to backup directory")
    
    # Show size comparison
    try:
        old_size = sum(f.stat().st_size for f in Path(backup_dir).glob("*.pkl"))
        new_cache_path = os.path.join(old_cache_dir, "cache.db")
        new_size = os.path.getsize(new_cache_path) if os.path.exists(new_cache_path) else 0
        
        logger.info(f"Cache size comparison:")
        logger.info(f"  Old cache: {old_size / 1024 / 1024:.1f} MB ({len(pkl_files)} files)")
        logger.info(f"  New cache: {new_size / 1024 / 1024:.1f} MB (1 file)")
        logger.info(f"  Space saved: {(old_size - new_size) / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        logger.warning(f"Failed to calculate size comparison: {e}")


if __name__ == "__main__":
    migrate_embeddings_cache()