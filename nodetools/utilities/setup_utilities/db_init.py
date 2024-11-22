from sqlalchemy import create_engine, text
from nodetools.utilities.credentials import CredentialManager
import getpass

def init_database():
    """Initialize the PostgreSQL database with required tables."""

    cm = CredentialManager()
    try:
        encryption_password = getpass.getpass("Enter your encryption password: ")

        decrypted_cred_map = cm.output_fully_decrypted_cred_map(pw_decryptor=encryption_password)

        db_conn_string = decrypted_cred_map['postfiatfoundation__postgresconnstring']

        engine = create_engine(db_conn_string)

        create_table_sql = """
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
        """

        # Create tables
        with engine.connect() as connection:
            connection.execute(text(create_table_sql))
            connection.execute(text("GRANT ALL PRIVILEGES ON TABLE postfiat_tx_cache to postfiat;"))
            connection.commit()

        print("Database initialized successfully!")
        print("Created tables:")
        print("- postfiat_tx_cache")

    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_database()