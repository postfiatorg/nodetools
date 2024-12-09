import pandas as pd
from contextlib import asynccontextmanager
import sqlalchemy
import asyncpg
from typing import Optional
from loguru import logger
from nodetools.utilities.credentials import CredentialManager
import psycopg2
import numpy as np
from nodetools.utilities.credentials import CredentialManager

class AsyncDBManager:
    _instance = None
    _initialized = False
    _pool: Optional[asyncpg.Pool] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self.__class__._initialized:
            self.credential_manager = CredentialManager()
            self.__class__._initialized = True

    async def initialize(self, username: str):
        """Initialize connection pool"""
        if self._pool is None:
            logger.debug(f"AsyncDBManager.intialize: Creating connection pool for {username}")
            connstring = self.credential_manager.get_credential(f"{username}_postgresconnstring")
            self._pool = await asyncpg.create_pool(connstring)

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute(self, query: str, *args):
        """Execute a query"""
        if not self._pool:
            raise RuntimeError("Database not initialized")
        async with self._pool.acquire() as connection:
            return await connection.execute(query, *args)

    @asynccontextmanager
    async def transaction(self):
        """Provide a transaction context"""
        if not self._pool:
            raise RuntimeError("Database not initialized")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                yield connection

class DBConnectionManager:
    ''' supports 1 database for the collective and one for the user'''
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.__class__._initialized:
            self.credential_manager = CredentialManager()
            self.__class__._initialized = True

    def spawn_sqlalchemy_db_connection_for_user(self, username):
        """Create a SQLAlchemy engine for the specified user"""
        db_connstring = self.credential_manager.get_credential(f'{username}_postgresconnstring')
        engine = sqlalchemy.create_engine(db_connstring)
        return engine
    
    def list_sqlalchemy_db_table_names_for_user(self, user_name):
        engine = self.spawn_sqlalchemy_db_connection_for_user(user_name)
        table_names = sqlalchemy.inspect(engine).get_table_names()
        return table_names
    
    def spawn_psycopg2_db_connection(self,user_name):
        
        db_connstring = self.credential_manager.get_credential(f'{user_name}_postgresconnstring')

        db_user = db_connstring.split('://')[1].split(':')[0]
        db_password = db_connstring.split('://')[1].split(':')[1].split('@')[0]
        db_host = db_connstring.split('://')[1].split(':')[1].split('@')[1].split('/')[0]
        db_name = db_connstring.split('/')[-1:][0]
        psycop_conn = psycopg2.connect(user=db_user, password=db_password, 
                                            host=db_host, database=db_name)
        return psycop_conn

