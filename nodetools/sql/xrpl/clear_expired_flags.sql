UPDATE authorized_addresses
SET 
    is_authorized = TRUE,
    flag_type = NULL,
    flag_expires_at = NULL
WHERE 
    flag_type IS NOT NULL 
    AND flag_expires_at < CURRENT_TIMESTAMP;