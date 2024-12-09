-- Create a view that combines transaction and memo data for Discord
CREATE OR REPLACE VIEW foundation_discord AS
SELECT 
    t.hash,
    m.memo_data,
    m.memo_type,
    m.memo_format,
    t.close_time_iso::timestamp as datetime,
    d.url,
    CASE 
        WHEN t.account = t.destination THEN 0
        ELSE COALESCE(t.amount_value, 0)
    END as directional_pft,
    t.account,
    d.processed_at
FROM xrpl_transactions t
JOIN xrpl_memos m ON t.hash = m.tx_hash
LEFT JOIN discord_transactions d ON t.hash = d.hash
WHERE m.memo_type LIKE '%discord%'
   OR m.memo_data LIKE '%discord.com%';