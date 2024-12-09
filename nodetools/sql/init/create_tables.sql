-- Create main transaction cache table
CREATE TABLE IF NOT EXISTS xrpl_transactions (
    hash VARCHAR(255) PRIMARY KEY,
    ledger_hash VARCHAR(255),
    ledger_index BIGINT,
    close_time_iso VARCHAR(255),
    validated BOOLEAN,
    account VARCHAR(255) NOT NULL,
    destination VARCHAR(255),
    amount_value NUMERIC,
    amount_currency VARCHAR(10),
    delivermax TEXT,
    fee VARCHAR(20),
    flags FLOAT,
    lastledgersequence BIGINT,
    sequence BIGINT,
    signingpubkey TEXT,
    transactiontype VARCHAR(50),
    txnsignature TEXT,
    date BIGINT,
    tx_json JSONB NOT NULL,
    meta JSONB NOT NULL,
    processed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Normalized memo table
CREATE TABLE IF NOT EXISTS xrpl_memos (
    id SERIAL PRIMARY KEY,
    tx_hash VARCHAR(255) REFERENCES xrpl_transactions(hash) ON DELETE CASCADE,
    memo_data TEXT,
    memo_type VARCHAR(255),
    memo_format VARCHAR(255),
    processed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);
