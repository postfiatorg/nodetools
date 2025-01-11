UPDATE authorized_addresses 
SET 
    is_authorized = FALSE,
    deauthorized_at = CURRENT_TIMESTAMP,
    flag_type = NULL,                -- Clear any flags
    flag_expires_at = NULL           -- Clear flag expiry
WHERE 
    auth_source = $1 
    AND auth_source_user_id = $2;