"""
Initial database migration - Create core tables.

This migration creates the foundational tables for the protein diffusion
design lab: proteins, structures, experiments, and results.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def up(connection) -> None:
    """Apply the migration - create tables."""
    logger.info("Running migration 001: Create initial tables")
    
    # Check database backend for SQL syntax
    backend = connection.config.backend
    
    with connection.get_connection() as conn:
        if backend == "sqlite":
            _create_sqlite_tables(conn)
        elif backend == "postgresql":
            _create_postgresql_tables(conn)
        else:
            raise ValueError(f"Unsupported database backend: {backend}")
        
        conn.commit()
    
    logger.info("Migration 001 completed successfully")


def down(connection) -> None:
    """Rollback the migration - drop tables."""
    logger.info("Rolling back migration 001: Drop initial tables")
    
    with connection.get_connection() as conn:
        # Drop tables in reverse order to handle foreign keys
        tables = ["results", "structures", "experiments", "proteins"]
        
        for table in tables:
            if connection.config.backend == "sqlite":
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            else:  # PostgreSQL
                with conn.cursor() as cursor:
                    cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        
        conn.commit()
    
    logger.info("Migration 001 rollback completed")


def _create_sqlite_tables(conn) -> None:
    """Create tables for SQLite."""
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Proteins table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS proteins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence TEXT NOT NULL,
            sequence_hash TEXT UNIQUE NOT NULL,
            length INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}'
        )
    """)
    
    # Structures table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            protein_id INTEGER NOT NULL,
            pdb_content TEXT DEFAULT '',
            confidence REAL DEFAULT 0.0,
            structure_quality REAL DEFAULT 0.0,
            prediction_method TEXT DEFAULT 'unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (protein_id) REFERENCES proteins (id) ON DELETE CASCADE
        )
    """)
    
    # Experiments table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            parameters TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending'
        )
    """)
    
    # Results table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            protein_id INTEGER NOT NULL,
            binding_affinity REAL,
            composite_score REAL DEFAULT 0.0,
            ranking INTEGER,
            metrics TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE,
            FOREIGN KEY (protein_id) REFERENCES proteins (id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes
    _create_indexes(conn, "sqlite")


def _create_postgresql_tables(conn) -> None:
    """Create tables for PostgreSQL."""
    with conn.cursor() as cursor:
        # Proteins table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proteins (
                id SERIAL PRIMARY KEY,
                sequence TEXT NOT NULL,
                sequence_hash VARCHAR(64) UNIQUE NOT NULL,
                length INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """)
        
        # Structures table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS structures (
                id SERIAL PRIMARY KEY,
                protein_id INTEGER NOT NULL REFERENCES proteins(id) ON DELETE CASCADE,
                pdb_content TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                structure_quality REAL DEFAULT 0.0,
                prediction_method VARCHAR(50) DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """)
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                parameters JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending'
            )
        """)
        
        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id SERIAL PRIMARY KEY,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                protein_id INTEGER NOT NULL REFERENCES proteins(id) ON DELETE CASCADE,
                binding_affinity REAL,
                composite_score REAL DEFAULT 0.0,
                ranking INTEGER,
                metrics JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        _create_indexes(cursor, "postgresql")


def _create_indexes(conn_or_cursor, backend: str) -> None:
    """Create database indexes."""
    indexes = [
        # Proteins indexes
        "CREATE INDEX IF NOT EXISTS idx_proteins_hash ON proteins (sequence_hash)",
        "CREATE INDEX IF NOT EXISTS idx_proteins_length ON proteins (length)",
        "CREATE INDEX IF NOT EXISTS idx_proteins_created_at ON proteins (created_at)",
        
        # Structures indexes
        "CREATE INDEX IF NOT EXISTS idx_structures_protein ON structures (protein_id)",
        "CREATE INDEX IF NOT EXISTS idx_structures_quality ON structures (structure_quality)",
        "CREATE INDEX IF NOT EXISTS idx_structures_method ON structures (prediction_method)",
        
        # Experiments indexes
        "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status)",
        "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at)",
        
        # Results indexes
        "CREATE INDEX IF NOT EXISTS idx_results_experiment ON results (experiment_id)",
        "CREATE INDEX IF NOT EXISTS idx_results_protein ON results (protein_id)",
        "CREATE INDEX IF NOT EXISTS idx_results_ranking ON results (ranking)",
        "CREATE INDEX IF NOT EXISTS idx_results_composite_score ON results (composite_score)",
        "CREATE INDEX IF NOT EXISTS idx_results_binding_affinity ON results (binding_affinity)",
    ]
    
    for index_sql in indexes:
        if backend == "sqlite":
            conn_or_cursor.execute(index_sql)
        else:  # PostgreSQL
            conn_or_cursor.execute(index_sql)


def get_migration_info() -> Dict[str, Any]:
    """Get information about this migration."""
    return {
        "version": "001",
        "name": "create_initial_tables",
        "description": "Create core tables for proteins, structures, experiments, and results",
        "dependencies": [],
        "created_tables": ["proteins", "structures", "experiments", "results"],
        "created_indexes": [
            "idx_proteins_hash", "idx_proteins_length", "idx_proteins_created_at",
            "idx_structures_protein", "idx_structures_quality", "idx_structures_method",
            "idx_experiments_status", "idx_experiments_created_at",
            "idx_results_experiment", "idx_results_protein", "idx_results_ranking",
            "idx_results_composite_score", "idx_results_binding_affinity"
        ]
    }