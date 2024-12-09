-- Transaction indices
CREATE INDEX IF NOT EXISTS idx_xrpl_tx_accounts ON xrpl_transactions(account, destination);
CREATE INDEX IF NOT EXISTS idx_xrpl_tx_processed ON xrpl_transactions(processed_at DESC);
CREATE INDEX IF NOT EXISTS idx_xrpl_tx_created ON xrpl_transactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_xrpl_tx_ledger ON xrpl_transactions(ledger_index);

-- Memo indices
CREATE INDEX IF NOT EXISTS idx_xrpl_memo_tx_hash ON xrpl_memos(tx_hash);
CREATE INDEX IF NOT EXISTS idx_xrpl_memo_type ON xrpl_memos(memo_type);
CREATE INDEX IF NOT EXISTS idx_xrpl_memo_processed ON xrpl_memos(processed_at DESC);