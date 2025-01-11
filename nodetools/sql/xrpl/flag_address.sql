WITH user_info AS (
    -- First get the auth source info for the provided address
    SELECT auth_source, auth_source_user_id
    FROM authorized_addresses
    WHERE address = $1
)
UPDATE authorized_addresses
SET 
    is_authorized = FALSE,
    flag_type = $2,                                          -- 'YELLOW' or 'RED'
    flag_expires_at = CASE 
        WHEN $2 = 'YELLOW' THEN CURRENT_TIMESTAMP + INTERVAL '1 day'
        WHEN $2 = 'RED' THEN CURRENT_TIMESTAMP + INTERVAL '10 days'
    END,
    deauthorized_at = CURRENT_TIMESTAMP
FROM user_info
WHERE 
    authorized_addresses.auth_source = user_info.auth_source
    AND authorized_addresses.auth_source_user_id = user_info.auth_source_user_id;