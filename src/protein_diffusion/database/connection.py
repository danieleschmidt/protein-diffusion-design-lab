"""
Database connection management for protein diffusion design lab.

This module handles database connections, connection pooling, and 
transaction management for different database backends.
"""

import os
import logging
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager
from dataclasses import dataclass
import sqlite3
import json
from pathlib import Path

try:
    import psycopg2
    from psycopg2 import pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    backend: str = "sqlite"  # sqlite, postgresql, redis
    host: str = "localhost"
    port: int = 5432
    database: str = "protein_diffusion"
    username: Optional[str] = None
    password: Optional[str] = None
    
    # SQLite specific
    sqlite_path: str = "./data/protein_diffusion.db"
    
    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 10
    
    # Redis specific
    redis_url: Optional[str] = None
    redis_db: int = 0
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Parse database URL
            if database_url.startswith('postgresql://'):
                # Parse PostgreSQL URL
                # This is a simplified parser
                return cls(
                    backend="postgresql",
                    host=os.getenv('DB_HOST', 'localhost'),
                    port=int(os.getenv('DB_PORT', '5432')),
                    database=os.getenv('DB_NAME', 'protein_diffusion'),
                    username=os.getenv('DB_USER'),
                    password=os.getenv('DB_PASSWORD'),
                )
            elif database_url.startswith('sqlite://'):
                return cls(
                    backend="sqlite",
                    sqlite_path=database_url.replace('sqlite://', ''),
                )
        
        return cls(
            backend=os.getenv('DB_BACKEND', 'sqlite'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'protein_diffusion'),
            username=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            sqlite_path=os.getenv('SQLITE_PATH', './data/protein_diffusion.db'),
            redis_url=os.getenv('REDIS_URL'),
        )


class SQLiteConnection:
    """SQLite database connection manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = Path(config.sqlite_path)
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables."""
        with self.get_connection() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables
            self._create_tables(conn)
            conn.commit()
    
    def _create_tables(self, conn: sqlite3.Connection):
        """Create database tables."""
        # Proteins table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS proteins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence TEXT NOT NULL,
                sequence_hash TEXT UNIQUE NOT NULL,
                length INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON metadata
            )
        """)
        
        # Structures table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protein_id INTEGER NOT NULL,
                pdb_content TEXT,
                confidence REAL,
                structure_quality REAL,
                prediction_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,  -- JSON metadata
                FOREIGN KEY (protein_id) REFERENCES proteins (id)
            )
        """)
        
        # Experiments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                parameters TEXT,  -- JSON parameters
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'running'
            )
        """)
        
        # Results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                protein_id INTEGER NOT NULL,
                binding_affinity REAL,
                composite_score REAL,
                ranking INTEGER,
                metrics TEXT,  -- JSON metrics
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                FOREIGN KEY (protein_id) REFERENCES proteins (id)
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proteins_hash ON proteins (sequence_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proteins_length ON proteins (length)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_structures_protein ON structures (protein_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_experiment ON results (experiment_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_ranking ON results (ranking)")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()


class PostgreSQLConnection:
    """PostgreSQL database connection manager with connection pooling."""
    
    def __init__(self, config: DatabaseConfig):
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
        
        self.config = config
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def close_pool(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connection pool closed")


class RedisConnection:
    """Redis connection manager for caching and session storage."""
    
    def __init__(self, config: DatabaseConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("redis not available. Install with: pip install redis")
        
        self.config = config
        
        if config.redis_url:
            self.client = redis.from_url(config.redis_url)
        else:
            self.client = redis.Redis(
                host=config.host,
                port=config.port,
                db=config.redis_db,
                decode_responses=True,
            )
        
        # Test connection
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL."""
        try:
            return self.client.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False


class DatabaseConnection:
    """
    Main database connection manager that supports multiple backends.
    
    This class provides a unified interface for different database backends
    including SQLite, PostgreSQL, and Redis.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if config is None:
            config = DatabaseConfig.from_env()
        
        self.config = config
        self.primary_db = None
        self.cache_db = None
        
        # Initialize primary database
        if config.backend == "sqlite":
            self.primary_db = SQLiteConnection(config)
        elif config.backend == "postgresql":
            self.primary_db = PostgreSQLConnection(config)
        else:
            raise ValueError(f"Unsupported database backend: {config.backend}")
        
        # Initialize Redis cache if available
        if config.redis_url and REDIS_AVAILABLE:
            try:
                self.cache_db = RedisConnection(config)
            except Exception as e:
                logger.warning(f"Redis cache not available: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get primary database connection."""
        with self.primary_db.get_connection() as conn:
            yield conn
    
    def get_cache(self) -> Optional[RedisConnection]:
        """Get cache connection if available."""
        return self.cache_db
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            if self.config.backend == "sqlite":
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
            else:  # PostgreSQL
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def execute_mutation(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_connection() as conn:
            if self.config.backend == "sqlite":
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
            else:  # PostgreSQL
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    conn.commit()
                    return cursor.rowcount
    
    def close(self):
        """Close database connections."""
        if hasattr(self.primary_db, 'close_pool'):
            self.primary_db.close_pool()
        logger.info("Database connections closed")


# Global connection instance
_connection: Optional[DatabaseConnection] = None


def get_connection() -> DatabaseConnection:
    """Get global database connection instance."""
    global _connection
    if _connection is None:
        _connection = DatabaseConnection()
    return _connection


def init_database(config: Optional[DatabaseConfig] = None):
    """Initialize database with custom configuration."""
    global _connection
    _connection = DatabaseConnection(config)
    return _connection