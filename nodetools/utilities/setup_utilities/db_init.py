from sqlalchemy import create_engine, text
from nodetools.utilities.credentials import CredentialManager
import getpass
import argparse
import nodetools.utilities.constants as constants

# Enter the credential key for the database connection string
node_name = constants.get_network_config().node_name
POSTGRES_CREDENTIAL_KEY = f"{node_name}_postgresconnstring"

def init_database(drop_tables: bool = False):
    """Initialize the PostgreSQL database with required tables and views.
    
    Args:
        drop_tables: If True, drops and recreates tables. If False, only creates if not exist
                    and updates views/indices. Default False for safety.
    """
    try:
        encryption_password = getpass.getpass("Enter your encryption password: ")
        cm = CredentialManager(password=encryption_password)
        db_conn_string = cm.get_credential(POSTGRES_CREDENTIAL_KEY)

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

        # Drop the tables if they exist
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

            # Create or update indices
            connection.execute(text(create_indices_sql))

            # Create or replace view 
            connection.execute(text(create_view_sql))
            connection.execute(text("GRANT SELECT ON memo_detail_view TO postfiat;"))
            
            connection.commit()

        print("Database initialization completed successfully!")
        print("Status:")
        print("- Tables configured (drop_tables={})".format(drop_tables))
        print("- Indices updated")
        print("- Views updated")

    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the NodeTools database.")
    parser.add_argument("--drop-tables", action="store_true", help="Drop and recreate tables (WARNING: Destructive)")
    args = parser.parse_args()

    init_database(drop_tables=args.drop_tables)