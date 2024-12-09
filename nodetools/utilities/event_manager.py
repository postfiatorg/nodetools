from dataclasses import dataclass
from typing import Optional, Callable, Any
from datetime import datetime
import asyncio
import json
from loguru import logger
from xrpl.clients import WebsocketClient
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models import Subscribe, AccountTx
from enum import Enum
from nodetools.utilities.constants import SystemMemoType, TaskType, MessageType
from nodetools.utilities.db_manager import AsyncDBManager
from nodetools.sql.sql_manager import SQLManager

@dataclass
class Transaction:
    """Normalized transaction data for database storage"""
    hash: str
    close_time_iso: str
    tx_json: dict
    meta: dict
    account: str
    destination: Optional[str]
    memo_data: Optional[str]
    memo_type: Optional[str]
    memo_format: Optional[str]
    processed_at: datetime = datetime.utcnow()

class TransactionStore:
    """Database interface for transaction storage and retrieval"""
    def __init__(self, db_manager):
        self.db = db_manager
        self.sql = SQLManager()
        
    async def initialize(self):
        """Create necessary tables and indices"""
        create_tables_sql = self.sql.load_query('init', 'create_tables')
        create_indices_sql = self.sql.load_query('init', 'create_indices')

        async with self.db.transaction() as conn:
            # Create tables if they don't exist
            await conn.execute(create_tables_sql)
            await conn.execute(create_indices_sql)

    async def store_transaction(self, tx: Transaction):
        """Store a new transaction"""
        logger.debug(f"TransactionStore.store_transaction: Storing transaction {tx.hash}")

        async with self.db.transaction() as conn:
            # First store the transaction
            await conn.execute("""
                INSERT INTO xrpl_transactions 
                (hash, close_time_iso, tx_json, meta, account, destination)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (hash) DO UPDATE 
                SET processed_at = NOW()
            """, tx.hash, tx.close_time_iso, json.dumps(tx.tx_json), 
                json.dumps(tx.meta), tx.account, tx.destination)
            
            # Then store any memos
            if tx.memo_data:
                await conn.execute("""
                    INSERT INTO xrpl_memos 
                    (tx_hash, memo_data, memo_type, memo_format)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (tx_hash) DO UPDATE 
                    SET processed_at = NOW()
                """, tx.hash, tx.memo_data, tx.memo_type, tx.memo_format)

class XRPLEventManager:
    """Manages WebSocket connections and event dispatch"""
    def __init__(self, network_config, node_config, store: TransactionStore):
        self.network_config = network_config
        self.node_address = node_config.node_address
        self.store = store  # Use TransactionStore instead of direct db access
        self.client: Optional[AsyncWebsocketClient] = None
        self.handlers = {}
        
    async def connect(self):
        """Establish WebSocket connection and subscribe to events"""
        logger.debug(f"XRPLEventManager.connect: Connecting to {self.network_config.websockets[0]}")
        
        self.client = WebsocketClient(self.network_config.websockets[0])
        await self.client.connect()
        
        # Subscribe to account transactions
        await self.client.send(Subscribe(
            streams=["transactions"],
            accounts=[self.node_config.node_address]
        ))
        
        logger.debug(f"XRPLEventManager.connect: Successfully connected and subscribed")
        
    async def start(self):
        """Start processing events"""
        logger.debug(f"XRPLEventManager.start: Starting event processing")
        
        while True:
            try:
                message = await self.client.recv()
                await self._handle_message(message)
            except Exception as e:
                logger.error(f"XRPLEventManager.start: Error processing message: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
                
    async def _handle_message(self, message: dict):
        """Process incoming WebSocket messages"""
        if 'transaction' not in message:
            return
        
        tx = message['transaction']
        if not self._is_relevant_transaction(tx):
            return

        # Use TransactionStore for database operations
        await self.store.store_transaction(Transaction(
            hash=tx['hash'],
            close_time_iso=tx.get('close_time_iso'),
            tx_json=tx,
            meta=message.get('meta', {}),
            account=tx.get('Account'),
            destination=tx.get('Destination'),
            **(self._extract_memo_data(tx) or {})
        ))

        # Notify handlers
        for handler in self.handlers.values():
            try:
                await handler(tx)
            except Exception as e:
                logger.error(f"XRPLEventManager._handle_message: Handler error: {e}")

    def _is_relevant_transaction(self, tx: dict) -> bool:
        """Filter for relevant transactions"""
        # First check if it's to/from the node address
        if not (tx.get('Destination') == self.node_address or tx.get('Account') == self.node_address):
            return False
        
        # Check for PFT transfer
        has_pft = False
        if tx.get('DeliverMax', {}).get('currency') == 'PFT':
            has_pft = True
        
        # Check memos
        memos = tx.get('Memos', [])
        if not memos:
            return False
        
        for memo in memos:
            memo_data = memo.get('Memo', {}).get('MemoData')
            if not memo_data:
                continue

            # Check for system memo types
            if any(memo_type.value in memo_data for memo_type in SystemMemoType):
                return True
            
            # If has PFT, check for task/message types
            if has_pft:
                if any(task_type.value in memo_data for task_type in TaskType):
                    return True
                if MessageType.MEMO.value in memo_data:
                    return True
                
        return False
        
    def _extract_memo_data(self, tx: dict) -> Optional[tuple[str, str, str]]:
        """Extract memo data, type, and format from transaction.
        
        Returns:
            tuple: (memo_data, memo_type, memo_format) or None if no memo
        """
        memos = tx.get('Memos', [])
        if not memos:
            return None
            
        memo = memos[0].get('Memo', {})  # Get first memo
        return (
            memo.get('MemoData'),
            memo.get('MemoType'),
            memo.get('MemoFormat')
        )
    
    def register_handler(self, name: str, handler: Callable):
        """Register a handler for transactions"""
        logger.debug(f"XRPLEventManager.register_handler: Registering handler {name}")
        self.handlers[name] = handler

    def remove_handler(self, name: str):
        """Remove a registered handler"""
        logger.debug(f"XRPLEventManager.remove_handler: Removing handler {name}")
        self.handlers.pop(name, None)
