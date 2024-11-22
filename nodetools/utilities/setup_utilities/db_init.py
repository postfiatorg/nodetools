from sqlalchemy import create_engine, text
from nodetools.utilities.credentials import CredentialManager
import getpass

def init_database():
    """Initialize the PostgreSQL database with required tables."""

    try:
        encryption_password = getpass.getpass("Enter your encryption password: ")

        cm = CredentialManager(password=encryption_password)

        db_conn_string = cm.get_credential("postfiatfoundation_postgresconnstring")

        confirm = input("WARNING: This will drop existing tables. Are you sure you want to continue? (y/n): ")
        if confirm.lower() != "y":
            print("Database initialization cancelled.")
            return

        engine = create_engine(db_conn_string)

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

        # Drop the tables if they exist
        with engine.connect() as connection:
            for create_table in create_tables_sql.keys():
                connection.execute(text(f"DROP TABLE IF EXISTS {create_table} CASCADE;"))
            connection.commit()

        # Create the tables
        with engine.connect() as connection:
            for create_table, create_table_sql in create_tables_sql.items():
                connection.execute(text(create_table_sql))
                connection.execute(text(f"GRANT ALL PRIVILEGES ON TABLE {create_table} to postfiat;"))
            connection.commit()

        print("Database initialized successfully!")
        print("Created tables:")
        print("- postfiat_tx_cache")
        print("- foundation_discord")

    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_database()