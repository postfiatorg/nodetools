from nodetools.utilities.credentials import CredentialManager
import getpass

def setup_credentials():        
    print("\nCredential Setup Script")
    print("======================")
    print("This script will help you set up the required credentials for the PFT Discord bot.")

    # Get network type first
    while True:
        network = input("\nAre you setting up for testnet or mainnet? (testnet/mainnet): ").strip().lower()
        if network in ['testnet', 'mainnet']:
            break
        print("Please enter either 'testnet' or 'mainnet'")

    # Get node name first
    print(f"\nNext, you'll need to specify your node name.")
    print("This will be used to identify your node's credentials.")
    if network == 'testnet':
        print("Since this is testnet, a '_testnet' suffix will automatically be added to your node name. Just enter your node name without the suffix.")
    node_name = input("Enter your node name: ").strip()

    has_remembrancer = input("\nDo you have a remembrancer wallet for your node? (y/n): ").strip().lower() == 'y'
    has_discord = input("Do you want to set up a Discord guild? (y/n): ").strip().lower() == 'y'

    # Define the required credentials
    network_suffix = '_testnet' if network == 'testnet' else ''
    required_credentials = {
        f'{node_name}{network_suffix}__v1xrpsecret': 'Your PFT Foundation XRP Secret',
        'openai': 'Your OpenAI API Key',
        'anthropic': 'Your Anthropic API Key',
        f'{node_name}{network_suffix}_postgresconnstring': 'PostgreSQL connection string (format: postgresql://user:password@host:port/database)'
    }

    # Conditionally add remembrancer credentials
    if has_remembrancer:
        required_credentials[f'{node_name}{network_suffix}_remembrancer__v1xrpsecret'] = 'Your Remembrancer XRP Secret'

    # Conditionally add Discord credentials
    if has_discord:
        required_credentials[f'discordbot{network_suffix}_secret'] = 'Your Discord Bot Token'


    print("\nNow you'll need to enter a password to encrypt your credentials.\n")
    
    # Get encryption password
    while True:
        encryption_password = getpass.getpass("Enter an encryption password (min 8 characters): ")
        if len(encryption_password) >= 8:
            confirm_password = getpass.getpass("Confirm encryption password: ")
            if encryption_password == confirm_password:
                break
            else:
                print("Passwords don't match. Please try again.\n")
        else:
            print("Password must be at least 8 characters long. Please try again.\n")
    
    print("\nNow you'll need to enter each required credential.")
    print("These will be encrypted using your password.\n")

    # Initialize the credential manager
    cm : CredentialManager = CredentialManager(encryption_password)

    # Collect credentials into a dictionary
    credentials_dict = {}
    
    # Collect and encrypt each credential
    for cred_name, description in required_credentials.items():
        print(f"\nSetting up: {cred_name}")
        print(f"Description: {description}")

        while True:
            # Add special instructions for PostgreSQL connection string
            if 'postgresconnstring' in cred_name:
                db_name = 'postfiat_db_testnet' if network == 'testnet' else 'postfiat_db'
                print("\nLet's build your PostgreSQL connection string.")
                print("Default values will be shown in [brackets]. Press Enter to use them.")
                
                user = input("PostgreSQL username [postfiat]: ").strip() or "postfiat"
                password = input("PostgreSQL password: ").strip()
                host = input("Database host [localhost]: ").strip() or "localhost"
                port = input("Database port [5432]: ").strip() or "5432"
                
                credential_value = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                print(f"\nConnection string created with database: {db_name}")
                print(f"Connection string: {credential_value}")
                print("The database will be created automatically when you run db_init.py")
                credentials_dict[cred_name] = credential_value
                break

            else:
                credential_value = input(f"Enter value for {cred_name}: ").strip()

                if credential_value:
                    try:
                        credentials_dict[cred_name] = credential_value
                        break
                    except Exception as e:
                        print(f"Error storing credential: {str(e)}")
                        retry = input("Would you like to try again? (y/n): ")
                        if retry.lower() != 'y':
                            break
                else:
                    print("Credential cannot be empty. Please try again.")
    
    # Store all credentials at once
    try:
        cm.enter_and_encrypt_credential(credentials_dict)
        print("\nCredential setup complete!")
        print(f"Credentials stored in: {cm.db_path}")
        print("\nIMPORTANT: Keep your encryption password safe. You'll need it to run the bot.")
        print("When starting the bot, enter this same encryption password when prompted.")
    except Exception as e:
        print(f"\nError storing credentials: {str(e)}")

if __name__ == "__main__":
    setup_credentials()
