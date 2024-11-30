from sqlalchemy import create_engine, text, exc
import getpass
import argparse
from nodetools.utilities.credentials import CredentialManager
import sys
import subprocess
import platform

def extract_node_name(postgres_key: str) -> str:
    """Extract node name from PostgreSQL credential key.
    
    Example: 'mynode_postgresconnstring_testnet' -> 'mynode'
            'mynode_postgresconnstring' -> 'mynode'
    """
    # Remove '_testnet' suffix if present
    base_key = postgres_key.replace('_testnet', '')
    # Remove '_postgresconnstring' suffix
    node_name = base_key.replace('_postgresconnstring', '')
    return node_name

def check_prerequisites() -> tuple[bool, list[str]]:
    """Check if all prerequisites are met for database initialization.
    
    Returns:
        tuple: (bool: all prerequisites met, list: error messages)
    """
    errors = []
    
    # Check if PostgreSQL is installed
    try:
        if platform.system() == 'Windows':
            subprocess.run(['where', 'psql'], check=True, capture_output=True)
        else:
            subprocess.run(['which', 'psql'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        errors.append("PostgreSQL is not installed or not in PATH. Please install PostgreSQL first.")
    
    # Check if PostgreSQL service is running
    try:
        if platform.system() == 'Windows':
            subprocess.run(['sc', 'query', 'postgresql'], check=True, capture_output=True)
        else:
            subprocess.run(['systemctl', 'status', 'postgresql'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        errors.append("PostgreSQL service is not running. Please start the PostgreSQL service.")
    
    # Check for psycopg2
    try:
        import psycopg2
    except ImportError:
        errors.append("psycopg2 is not installed. Please install it with: pip install psycopg2-binary")
    
    return len(errors) == 0, errors

def try_fix_permissions(user: str, db_name: str) -> bool:
    """Attempt to fix permissions by granting necessary privileges."""
    try:
        print(f"\nAttempting to grant privileges to user '{user}'...")
        sudo_password = getpass.getpass("Enter sudo password to grant database privileges: ")
        
        commands = [
            f'ALTER USER {user} WITH CREATEDB;',
            f'GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user};',
            # Use -d flag instead of \c
            (f'GRANT ALL PRIVILEGES ON SCHEMA public TO {user};', db_name)
        ]
        
        for cmd in commands:
            if isinstance(cmd, tuple):
                # If command needs specific database
                command, database = cmd
                process = subprocess.Popen(
                    ['sudo', '-S', '-u', 'postgres', 'psql', '-d', database, '-c', command],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                # Regular command
                process = subprocess.Popen(
                    ['sudo', '-S', '-u', 'postgres', 'psql', '-c', cmd],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            stdout, stderr = process.communicate(input=sudo_password + '\n')
            
            if process.returncode != 0:
                print(f"Command failed: {stderr}")
                return False
                
        print(f"Successfully granted privileges to {user}")
        return True
            
    except Exception as e:
        print(f"Failed to fix permissions: {e}")
        print("\nIf automatic privilege granting failed, you can run these commands manually:")
        print(f"sudo -u postgres psql -c 'ALTER USER {user} WITH CREATEDB;'")
        print(f"sudo -u postgres psql -c 'GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user};'")
        print(f"sudo -u postgres psql -d {db_name} -c 'GRANT ALL PRIVILEGES ON SCHEMA public TO {user};'")
        return False

def check_and_create_role(base_conn_string: str) -> tuple[bool, list[str]]:
    """Check database permissions and create postfiat role if needed.
    
    Args:
        base_conn_string: Connection string to postgres database
        
    Returns:
        tuple: (bool: success, list: error messages)
    """
    errors = []
    try:
        engine = create_engine(base_conn_string)
        with engine.connect() as conn:
            # Extract username from connection string
            user = base_conn_string.split('://')[1].split(':')[0]

            # Check if we have superuser privileges
            result = conn.execute(text("SELECT current_setting('is_superuser')")).scalar()
            is_superuser = (result.lower() == 'on')
            
            if not is_superuser:
                print(f"\nUser '{user}' needs superuser privileges for initial setup.")
                if input("Would you like to try to fix this automatically? (y/n): ").lower() == 'y':
                    if try_fix_permissions(user):
                        print("Permissions fixed! Please run this script again.")
                        sys.exit(0)
                
                errors.append(
                    "\nTo fix this manually, you can either:\n"
                    "1. Connect as the postgres superuser by modifying your connection string:\n"
                    "   postgresql://postgres:yourpassword@localhost:5432/postgres\n"
                    "2. Or grant superuser privileges to your current user with:\n"
                    "   sudo -u postgres psql -c 'ALTER USER your_username WITH SUPERUSER;'\n"
                    "\nAfter fixing permissions, run this script again."
                )
                return False, errors
            
            # Check if postfiat role exists, create if it doesn't
            result = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname='postfiat'")).first()
            if not result:
                try:
                    conn.execute(text("CREATE ROLE postfiat WITH LOGIN PASSWORD 'default_password'"))
                    print("Created 'postfiat' role with default password. Please change it!")
                except Exception as e:
                    errors.append(f"Failed to create postfiat role: {str(e)}")
            
            # Check if we can create databases
            try:
                conn.execute(text("CREATE DATABASE test_permissions"))
                conn.execute(text("DROP DATABASE test_permissions"))
            except Exception as e:
                errors.append(f"Connected user cannot create databases: {str(e)}")
                
    except Exception as e:
        errors.append(f"Failed to connect to PostgreSQL: {str(e)}")
    
    return len(errors) == 0, errors

def revoke_all_privileges(db_conn_string: str) -> None:
    """Revoke all privileges from a user for testing purposes."""
    try:
        # Extract username and database from connection string
        user = db_conn_string.split('://')[1].split(':')[0]
        db_name = db_conn_string.split('/')[-1]
        
        print(f"Revoking privileges from user '{user}'...")
        
        # These commands don't require the database to exist
        subprocess.run(['sudo', '-u', 'postgres', 'psql', '-c', 
                       f"ALTER USER {user} NOSUPERUSER NOCREATEDB;"], check=True)
        
        # Check if database exists before trying to revoke its privileges
        result = subprocess.run(['sudo', '-u', 'postgres', 'psql', '-c',
                               f"SELECT 1 FROM pg_database WHERE datname = '{db_name}';"],
                              capture_output=True, text=True)
        
        if "1 row" in result.stdout:
            subprocess.run(['sudo', '-u', 'postgres', 'psql', '-c', 
                          f"REVOKE ALL PRIVILEGES ON DATABASE {db_name} FROM {user};"], check=True)
            subprocess.run(['sudo', '-u', 'postgres', 'psql', '-d', db_name, '-c',
                          f"REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {user};"], check=True)
            print(f"Database privileges revoked for {db_name}")
        else:
            print(f"Database {db_name} does not exist yet - skipping database-specific privileges")
        
        print("User privileges revoked successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error revoking privileges: {e}")

def create_database_if_needed(db_conn_string: str) -> bool:
    """Create the database if it doesn't exist.
    
    Args:
        db_conn_string: Full PostgreSQL connection string
        
    Returns:
        bool: True if database was created, False if it already existed
    """
    try:
        # Extract database name from connection string
        db_name = db_conn_string.split('/')[-1]
        
        # Create a connection string to the default postgres database
        base_conn = db_conn_string.rsplit('/', 1)[0] + '/postgres'
        engine = create_engine(base_conn)
        
        with engine.connect() as conn:
            # Don't recreate if it exists
            exists = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")).first() is not None
            if not exists:
                try:
                    # First try with current user
                    conn.execute(text("COMMIT"))
                    conn.execute(text(f"CREATE DATABASE {db_name}"))
                    print(f"Created database: {db_name}")
                    return True
                except Exception as e:
                    if "permission denied to create database" in str(e):
                        print("Attempting to create database as postgres user...")
                        try:
                            subprocess.run(['sudo', '-u', 'postgres', 'createdb', db_name], check=True)
                            print(f"Successfully created database: {db_name}")
                            return True
                        except subprocess.CalledProcessError as e:
                            print(f"Failed to create database as postgres user: {e}")
                            return False
                    raise e
            return False
            
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def init_database(drop_tables: bool = False, create_db: bool = False):
    """Initialize the PostgreSQL database with required tables and views.
    
    Args:
        drop_tables: If True, drops and recreates tables. If False, only creates if not exist
                    and updates views/indices. Default False for safety.
    """
    try:
        # Check prerequisites first
        print("\nChecking prerequisites...")
        prereqs_met, errors = check_prerequisites()
        if not prereqs_met:
            print("\nPrerequisite checks failed:")
            for error in errors:
                print(f"- {error}")
            print("\nPlease fix these issues and try again.")
            return

        encryption_password = getpass.getpass("Enter your encryption password: ")
        cm = CredentialManager(password=encryption_password)

        # Get all credentials and find the PostgreSQL connection string
        all_creds = cm.list_credentials()
        postgres_keys = [key for key in all_creds if 'postgresconnstring' in key]

        if not postgres_keys:
            print("\nNo PostgreSQL connection strings found!")
            print("Please run setup_credentials.py first to configure your database credentials.")
            return

        for postgres_key in postgres_keys:
            # Extract node name from PostgreSQL credential key
            node_name = extract_node_name(postgres_key)
            network_type = "testnet" if "_testnet" in postgres_key else "mainnet"
        
            print(f"\nInitializing database for {node_name} ({network_type})...")

            db_conn_string = cm.get_credential(postgres_key)

            if create_db:
                created = create_database_if_needed(db_conn_string)
                if created:
                    print("Database created successfully!")
                else:
                    print("Database already exists or couldn't be created.")
                    print("Attempting to continue with initialization...")

            # Create a new engine with the target database
            try:
                engine = create_engine(db_conn_string)
                with engine.connect() as conn:
                    # Test connection
                    conn.execute(text("SELECT 1"))
            except Exception as e:
                print(f"\nError connecting to database: {e}")
                print("Please ensure the database exists and you have proper permissions.")
                return

            if drop_tables:
                confirm = input("WARNING: This will drop existing tables. Are you sure you want to continue? (y/n): ")
                if confirm.lower() != "y":
                    print("Database initialization cancelled.")
                    return

            engine = create_engine(db_conn_string)

            # First, create the tables
            create_tables_sql = {
                "postfiat_tx_cache":
                """
                CREATE TABLE IF NOT EXISTS postfiat_tx_cache (
                    close_time_iso VARCHAR(255),
                    hash VARCHAR(255) PRIMARY KEY,
                    ledger_hash VARCHAR(255),
                    ledger_index BIGINT,
                    meta TEXT,
                    tx_json TEXT,
                    validated BOOLEAN,
                    account VARCHAR(255),
                    delivermax TEXT,
                    destination VARCHAR(255),
                    fee VARCHAR(20),
                    flags FLOAT,
                    lastledgersequence BIGINT,
                    sequence BIGINT,
                    signingpubkey TEXT,
                    transactiontype VARCHAR(50),
                    txnsignature TEXT,
                    date BIGINT,
                    memos TEXT
                );
                """,
                "foundation_discord":
                """
                CREATE TABLE IF NOT EXISTS foundation_discord (
                    hash VARCHAR(255) PRIMARY KEY,
                    memo_data TEXT,
                    memo_type VARCHAR(255),
                    memo_format VARCHAR(255),
                    datetime TIMESTAMP,
                    url TEXT,
                    directional_pft FLOAT,
                    account VARCHAR(255),
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            }

            # Create indices
            create_indices_sql = """
            CREATE INDEX IF NOT EXISTS idx_account_destination
                ON postfiat_tx_cache(account, destination);
            CREATE INDEX IF NOT EXISTS idx_close_time_iso
                ON postfiat_tx_cache(close_time_iso DESC);
            CREATE INDEX IF NOT EXISTS idx_hash
                ON postfiat_tx_cache(hash);
            """

            # Create view
            create_view_sql = """
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
            """

            try:
                with engine.connect() as connection:
                    # Only drop tables if explicitly requested
                    if drop_tables:
                        for table in create_tables_sql.keys():
                            connection.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE;"))
                        connection.commit()
                        print("Dropped existing tables.")

                    # Create the tables if they don't exist
                    for table, create_table_sql in create_tables_sql.items():
                        connection.execute(text(create_table_sql))
                        connection.execute(text(f"GRANT ALL PRIVILEGES ON TABLE {table} to postfiat;"))
                        connection.commit()

                        result = connection.execute(text(f"""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = '{table}';
                        """))
                        columns = result.fetchall()
                        print(f"\nCreated table: {table}")
                        print("Columns:")
                        for col in columns:
                            print(f"- {col[0]}: {col[1]}")

                    connection.commit()

                    # Create or update indices
                    connection.execute(text(create_indices_sql))
                    connection.commit()

                    # Create or replace view 
                    connection.execute(text(create_view_sql))
                    connection.execute(text("GRANT SELECT ON memo_detail_view TO postfiat;"))
                    connection.commit()

            except Exception as e:
                if "permission denied for schema public" in str(e):
                    print("\nPermission denied. The database exists but your user needs additional privileges.")
                    user = db_conn_string.split('://')[1].split(':')[0]
                    db_name = db_conn_string.split('/')[-1]
                    if try_fix_permissions(user, db_name):
                        print("\nPermissions fixed! Retrying initialization...")
                        return init_database(drop_tables=drop_tables, create_db=create_db)
                print(f"Error initializing database: {e}")
                return

            print("Database initialization completed successfully!")
            print("Status:")
            print("- Tables configured (drop_tables={})".format(drop_tables))
            print("- Indices updated")
            print("- Views updated")

    except Exception as e:
        print(f"Error initializing database: {e}")

def print_prerequisites():
    """Print prerequisites information."""
    print("""
        Prerequisites for database initialization:
        ----------------------------------------
        1. PostgreSQL must be installed
        2. PostgreSQL service must be running
        3. psycopg2 must be installed (pip install psycopg2-binary)
        4. User must have superuser privileges in PostgreSQL
        5. Network configuration must be set up

        For Ubuntu/Debian:
        sudo apt-get update
        sudo apt-get install postgresql postgresql-contrib

        For Windows:
        Download and install from: https://www.postgresql.org/download/windows/
        """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the NodeTools database.")
    parser.add_argument("--drop-tables", action="store_true", help="Drop and recreate tables (WARNING: Destructive)")
    parser.add_argument("--create-db", action="store_true", help="Create the database if it doesn't exist")
    parser.add_argument("--help-install", action="store_true", help="Show installation prerequisites")
    parser.add_argument("--revoke-privileges", action="store_true", help="Revoke privileges for testing (WARNING: Destructive)")
    args = parser.parse_args()

    if args.help_install:
        print_prerequisites()
        sys.exit(0)
    
    if args.revoke_privileges:
        # Get credentials and revoke privileges
        encryption_password = getpass.getpass("Enter your encryption password: ")
        cm = CredentialManager(password=encryption_password)
        postgres_keys = [key for key in cm.list_credentials() if 'postgresconnstring' in key]
        
        for key in postgres_keys:
            db_conn_string = cm.get_credential(key)
            revoke_all_privileges(db_conn_string)
        sys.exit(0)

    init_database(drop_tables=args.drop_tables, create_db=args.create_db)