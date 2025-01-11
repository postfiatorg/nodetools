SELECT 
    CASE 
        WHEN flag_type IS NOT NULL AND flag_expires_at > CURRENT_TIMESTAMP
        THEN EXTRACT(EPOCH FROM (flag_expires_at - CURRENT_TIMESTAMP))::INTEGER
        ELSE NULL
    END as cooldown_seconds,
    flag_type
FROM authorized_addresses 
WHERE auth_source = $1 
AND auth_source_user_id = $2
AND flag_type IS NOT NULL
AND flag_expires_at > CURRENT_TIMESTAMP
ORDER BY flag_expires_at DESC
LIMIT 1;