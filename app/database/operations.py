"""
Database Operations

Provides safe SQL execution and database introspection operations using MCP.
"""

import time
from typing import List, Dict, Any, Optional

from ..database.models import QueryResult, TableSchema, ColumnInfo, ForeignKeyInfo
from ..utils.logging import logger, log_sql_execution
from ..utils.monitoring import track_sql_execution
from ..mcp.mariadb_client import MariaDBClient
from ..mcp.postgres_client import PostgreSQLClient


class DatabaseOperations:
    """Handles safe database operations and schema introspection via MCP."""
    
    def __init__(self, mariadb_client: MariaDBClient, postgres_client: PostgreSQLClient):
        self.mariadb_client = mariadb_client
        self.postgres_client = postgres_client
    
    @track_sql_execution("mariadb")
    async def execute_query(
        self,
        query: str,
        database: str = "mariadb",
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """
        Execute SQL query safely with limits and timeout.
        
        Args:
            query: SQL query to execute
            database: Target database (mariadb/postgres)
            max_rows: Maximum rows to return
            timeout: Query timeout in seconds
        
        Returns:
            QueryResult with data, columns, and metadata
        """
        start_time = time.time()
        
        try:
            # Add LIMIT clause if not present
            limited_query = self._add_limit_clause(query, max_rows)
            
            if database == "mariadb":
                result = await self.mariadb_client.execute_query(
                    limited_query, max_rows=max_rows, timeout=timeout
                )
            elif database == "postgres":
                result = await self.postgres_client.execute_query(
                    limited_query, max_rows=max_rows, timeout=timeout
                )
            else:
                raise ValueError(f"Unsupported database: {database}")
            
            execution_time = time.time() - start_time
            
            # Log execution
            log_sql_execution(
                query=query,
                database=database,
                execution_time=execution_time,
                row_count=len(result.data) if result.data else 0
            )
            
            return QueryResult(
                success=True,
                data=result.data,
                columns=result.columns,
                row_count=len(result.data) if result.data else 0,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Log error
            log_sql_execution(
                query=query,
                database=database,
                execution_time=execution_time,
                row_count=0,
                error=error_msg
            )
            
            return QueryResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    
    
    def _add_limit_clause(self, query: str, max_rows: int) -> str:
        """Add LIMIT clause to query if not present."""
        query_upper = query.upper().strip()
        
        # Check if LIMIT already exists
        if 'LIMIT' in query_upper:
            return query
        
        # Add LIMIT clause
        return f"{query.rstrip(';')} LIMIT {max_rows}"
    
    async def get_schema(
        self,
        database: str = "mariadb",
        table_name: Optional[str] = None
    ) -> List[TableSchema]:
        """
        Get database schema information.
        
        Args:
            database: Target database
            table_name: Specific table (optional)
        
        Returns:
            List of TableSchema objects
        """
        try:
            if database == "mariadb":
                return await self._get_mariadb_schema(table_name)
            elif database == "postgres":
                return await self._get_postgres_schema(table_name)
            else:
                raise ValueError(f"Unsupported database: {database}")
                
        except Exception as e:
            logger.error(f"Failed to get schema for {database}: {e}")
            raise
    
    async def _get_mariadb_schema(self, table_name: Optional[str] = None) -> List[TableSchema]:
        """Get MariaDB schema information."""
        tables = []
        
        # Get table list
        table_query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE()
        """
        
        if table_name:
            table_query += f" AND TABLE_NAME = '{table_name}'"
        
        result = await self.mariadb_client.execute_query(table_query)
        table_names = [row[0] for row in result.data]
        
        # Get detailed info for each table
        for table in table_names:
            columns = await self._get_mariadb_columns(table)
            primary_keys = await self._get_mariadb_primary_keys(table)
            foreign_keys = await self._get_mariadb_foreign_keys(table)
            
            tables.append(TableSchema(
                name=table,
                columns=columns,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys
            ))
        
        return tables
    
    async def _get_mariadb_columns(self, table_name: str) -> List[ColumnInfo]:
        """Get column information for MariaDB table."""
        query = """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT,
               CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        
        columns = []
        result = await self.mariadb_client.execute_query(query)
        
        for row in result.data:
            columns.append(ColumnInfo(
                name=row[0],
                data_type=row[1],
                is_nullable=row[2] == 'YES',
                default_value=row[3],
                max_length=row[4],
                precision=row[5],
                scale=row[6]
            ))
        
        return columns
    
    async def _get_mariadb_primary_keys(self, table_name: str) -> List[str]:
        """Get primary key columns for MariaDB table."""
        query = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND CONSTRAINT_NAME = 'PRIMARY'
        """
        
        result = await self.mariadb_client.execute_query(query)
        return [row[0] for row in result.data]
    
    async def _get_mariadb_foreign_keys(self, table_name: str) -> List[ForeignKeyInfo]:
        """Get foreign key information for MariaDB table."""
        query = """
        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s 
        AND REFERENCED_TABLE_NAME IS NOT NULL
        """
        
        foreign_keys = []
        result = await self.mariadb_client.execute_query(query)
        
        for row in result.data:
            foreign_keys.append(ForeignKeyInfo(
                column=row[0],
                referenced_table=row[1],
                referenced_column=row[2]
            ))
        
        return foreign_keys
    
    async def _get_postgres_schema(self, table_name: Optional[str] = None) -> List[TableSchema]:
        """Get PostgreSQL schema information."""
        # Similar implementation for PostgreSQL
        # For now, return empty list
        return []
    
    async def list_tables(self, database: str = "mariadb") -> List[str]:
        """Get list of table names."""
        try:
            if database == "mariadb":
                query = """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME
                """
                result = await self.mariadb_client.execute_query(query)
                return [row[0] for row in result.data]
            
            elif database == "postgres":
                query = """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
                result = await self.postgres_client.execute_query(query)
                return [row[0] for row in result.data]
            
            else:
                raise ValueError(f"Unsupported database: {database}")
                
        except Exception as e:
            logger.error(f"Failed to list tables for {database}: {e}")
            raise