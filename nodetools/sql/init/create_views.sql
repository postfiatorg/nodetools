-- Create memo detail view
DROP VIEW IF EXISTS memo_detail_view;
CREATE VIEW memo_detail_view AS
WITH parsed_json AS (
    SELECT
        *,
        tx_json::jsonb as tx_json_parsed,
        meta::jsonb as meta_parsed
    FROM postfiat_tx_cache
),
memo_base AS (
    SELECT
        *,
        meta_parsed->>'TransactionResult' as transaction_result,
        (tx_json_parsed->'Memos') IS NOT NULL as has_memos,
        (close_time_iso::timestamp) as datetime,
        COALESCE((tx_json_parsed->'DeliverMax'->>'value')::float, 0) as pft_absolute_amount,
        (close_time_iso::timestamp)::date as simple_date,
        (tx_json_parsed->'Memos'->0->'Memo') as main_memo_data
    FROM parsed_json
    WHERE (tx_json_parsed->'Memos') IS NOT NULL
)
SELECT * from memo_base;

-- Maintain compatibility with existing code
CREATE OR REPLACE VIEW transaction_details AS
SELECT 
    t.*,
    json_agg(json_build_object(
        'memo_data', m.memo_data,
        'memo_type', m.memo_type,
        'memo_format', m.memo_format
    )) as memos
FROM transactions t
LEFT JOIN memos m ON t.hash = m.tx_hash
GROUP BY t.hash;

-- First drop the view if it exists (in case we're recreating)
DROP VIEW IF EXISTS postfiat_tx_cache;

-- Create the view with the same name as the original table
-- TODO: Deprecate this view
CREATE VIEW postfiat_tx_cache AS
SELECT 
    t.close_time_iso,
    t.hash,
    t.ledger_hash,
    t.ledger_index,
    t.meta::TEXT as meta,
    t.tx_json::TEXT as tx_json,
    t.validated,
    t.account,
    t.delivermax,
    t.destination,
    t.fee,
    t.flags,
    t.lastledgersequence,
    t.sequence,
    t.signingpubkey,
    t.transactiontype,
    t.txnsignature,
    t.date,
    json_agg(json_build_object(
        'memo_data', m.memo_data,
        'memo_type', m.memo_type,
        'memo_format', m.memo_format
    ))::TEXT as memos
FROM xrpl_transactions t
LEFT JOIN xrpl_memos m ON t.hash = m.tx_hash
GROUP BY t.hash;