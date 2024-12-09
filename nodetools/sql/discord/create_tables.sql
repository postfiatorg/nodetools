-- Foundation Discord transaction view/cache
CREATE TABLE IF NOT EXISTS discord_transactions (
    hash VARCHAR(255) PRIMARY KEY REFERENCES xrpl_transactions(hash) ON DELETE CASCADE,
    memo_data TEXT,
    memo_type VARCHAR(255),
    memo_format VARCHAR(255),
    datetime TIMESTAMP,
    url TEXT,
    directional_pft FLOAT,
    account VARCHAR(255),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);