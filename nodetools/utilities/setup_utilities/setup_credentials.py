from nodetools.utilities.settings import CredentialManager
import getpass

def setup_credentials():
    # Initialize the credential manager
    cm = CredentialManager()
    
    # Define the required credentials
    required_credentials = {
        'discordbot_secret': 'Your Discord Bot Token',
        'postfiatfoundation__v1xrpsecret': 'Your PFT Foundation XRP Secret',
        'postfiatremembrancer__v1xrpsecret': 'Your Remembrancer XRP Secret',
        'openai': 'Your OpenAI API Key',
        'anthropic': 'Your Anthropic API Key',
        'postfiatfoundation_postgresconnstring': 'PostgreSQL connection string (format: postgresql://user:password@host:port/database)'
    }
    
    print("\nCredential Setup Script")
    print("======================")
    print("This script will help you set up the required credentials for the PFT Discord bot.")
    print("You will need to enter an encryption password and then provide each credential.\n")
    
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
    
    # Collect and encrypt each credential
    for cred_name, description in required_credentials.items():
        print(f"\nSetting up: {cred_name}")
        print(f"Description: {description}")

        # Add special instructions for PostgreSQL connection string
        if cred_name == 'postfiatfoundation_postgresconnstring':
            print("Format: postgresql://postfiat:your_password@localhost:5432/postfiat_db")
        
        while True:
            credential_value = input(f"Enter value for {cred_name}: ").strip()
            if credential_value:
                try:
                    cm.enter_and_encrypt_credential__variable_based(
                        credential_ref=cred_name,
                        pw_data=credential_value,
                        pw_encryptor=encryption_password
                    )
                    print(f"Successfully stored {cred_name}")
                    break
                except Exception as e:
                    print(f"Error storing credential: {str(e)}")
                    retry = input("Would you like to try again? (y/n): ")
                    if retry.lower() != 'y':
                        break
            else:
                print("Credential cannot be empty. Please try again.")
    
    print("\nCredential setup complete!")
    print(f"Credentials stored in: {cm.credential_file_path}")
    print("\nIMPORTANT: Keep your encryption password safe. You'll need it to run the bot.")
    print("When starting the bot, enter this same encryption password when prompted.")

if __name__ == "__main__":
    setup_credentials()
