-- Discord transaction indices
CREATE INDEX IF NOT EXISTS idx_discord_tx_account ON discord_transactions(account);
CREATE INDEX IF NOT EXISTS idx_discord_tx_datetime ON discord_transactions(datetime DESC);
CREATE INDEX IF NOT EXISTS idx_discord_tx_memo_type ON discord_transactions(memo_type);