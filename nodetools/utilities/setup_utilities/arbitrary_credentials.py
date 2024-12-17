from nodetools.utilities.credentials import CredentialManager
import getpass

def setup_arbitrary_credentials():
    print("\nCredential Entry Tool")
    print("===================")
    print("This tool allows you to enter arbitrary credentials that will be encrypted.")

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

    # Initialize the credential manager
    cm = CredentialManager(encryption_password)

    # Dictionary to store credentials
    credentials_dict = {}

    print("\nEnter your credentials (enter 'q' as the key name to finish):")
    print("Format: You'll be prompted for a key name, then its value.")
    
    while True:
        key = input("\nEnter credential key name (or 'q' to finish): ").strip()
        
        if key.lower() == 'q':
            break
            
        if not key:
            print("Key name cannot be empty. Please try again.")
            continue

        if key in credentials_dict:
            print("This key already exists. Please use a different name.")
            continue

        value = input(f"Enter value for {key}: ").strip()
        
        if not value:
            print("Value cannot be empty. Please try again.")
            continue

        credentials_dict[key] = value

    if credentials_dict:
        try:
            cm.enter_and_encrypt_credential(credentials_dict)
            print("\nCredential setup complete!")
            print(f"Credentials stored in: {cm.db_path}")
            print("\nIMPORTANT: Keep your encryption password safe. You'll need it to decrypt these credentials.")
        except Exception as e:
            print(f"\nError storing credentials: {str(e)}")
    else:
        print("\nNo credentials were entered.")

if __name__ == "__main__":
    try:
        setup_arbitrary_credentials()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")