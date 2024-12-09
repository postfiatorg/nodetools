from nodetools.sql.sql_manager import SQLManager

class DiscordStore:
    def __init__(self, db_manager):
        self.db = db_manager
        self.sql = SQLManager()
        
    async def initialize(self):
        """Initialize Discord-specific tables"""
        for category in ['create_tables', 'create_indices', 'create_views']:
            sql = self.sql.load_query(category, module='discord')
            async with self.db.transaction() as conn:
                await conn.execute(sql)
                
    async def store_discord_transaction(self, tx_hash: str, url: str, **kwargs):
        """Store Discord-specific transaction data"""
        async with self.db.transaction() as conn:
            await conn.execute("""
                INSERT INTO discord_transactions 
                (hash, url, account, memo_data, memo_type, memo_format, directional_pft)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (hash) DO UPDATE 
                SET processed_at = NOW()
            """, tx_hash, url, kwargs.get('account'), kwargs.get('memo_data'),
                kwargs.get('memo_type'), kwargs.get('memo_format'),
                kwargs.get('directional_pft'))