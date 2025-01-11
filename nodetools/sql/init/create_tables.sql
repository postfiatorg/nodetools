CREATE TABLE IF NOT EXISTS postfiat_tx_cache (
    hash VARCHAR(255) PRIMARY KEY,
    ledger_index BIGINT,
    close_time_iso VARCHAR(255),
    meta TEXT,
    tx_json TEXT,
    validated BOOLEAN
);

CREATE TABLE IF NOT EXISTS transaction_memos (
    hash VARCHAR(255) PRIMARY KEY,
    account VARCHAR(255),
    destination VARCHAR(255),
    pft_amount NUMERIC,
    xrp_fee NUMERIC,
    memo_format TEXT DEFAULT '',
    memo_type TEXT DEFAULT '',
    memo_data TEXT DEFAULT '',
    datetime TIMESTAMP,
    transaction_result VARCHAR(50),
    FOREIGN KEY (hash) REFERENCES postfiat_tx_cache(hash)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transaction_processing_results (
    hash VARCHAR(255) PRIMARY KEY,
    processed BOOLEAN NOT NULL,
    rule_name VARCHAR(255),
    response_tx_hash VARCHAR(255),
    notes TEXT,
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (hash) REFERENCES postfiat_tx_cache(hash)
);

CREATE TABLE IF NOT EXISTS pft_holders (
    account VARCHAR(255) PRIMARY KEY,
    balance NUMERIC NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL,
    last_tx_hash VARCHAR(255) NULL
);

CREATE TABLE IF NOT EXISTS authorized_addresses (
    address VARCHAR(255) PRIMARY KEY,
    authorized_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_authorized BOOLEAN DEFAULT TRUE,
    deauthorized_at TIMESTAMP WITH TIME ZONE,
    auth_source VARCHAR(50),  -- e.g., 'discord', 'twitter', etc.
    auth_source_user_id VARCHAR(50),  -- Store external IDs as strings for flexibility
    flag_type VARCHAR(10),                                    -- 'YELLOW' or 'RED' or NULL
    flag_expires_at TIMESTAMP WITH TIME ZONE,                 -- When the flag cool-off period ends
    CONSTRAINT valid_xrp_address CHECK (address ~ '^r[1-9A-HJ-NP-Za-km-z]{25,34}$'),
    CONSTRAINT valid_flag_type CHECK (flag_type IN ('YELLOW', 'RED') OR flag_type IS NULL)
);