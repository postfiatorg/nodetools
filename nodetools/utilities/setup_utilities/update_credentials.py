from nodetools.utilities.credentials import CredentialManager
import getpass

def update_credentials():
    print("\nEncrypted Credential Update Script")
    print("=================================")
    print("This script will help you update existing credentials for the PFT Discord bot.")

    # Get encryption password
    try:
        encryption_password = getpass.getpass("\nEnter your encryption password: ")

        # Initialize the credential manager
        cm = CredentialManager(encryption_password)
        
        # Get all existing credentials
        existing_credentials = cm.list_credentials()
        
        if not existing_credentials:
            print("\nNo credentials found. Please run setup_credentials.py first.")
            return

        # Display available credentials with warning
        print("\n⚠️  WARNING: Sensitive information will be displayed in plain text!")
        print("    Make sure no one else can see your screen.")

        try:
            input("Press Enter to continue or Ctrl+C to cancel...\n")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return

        print("\nAvailable credentials:")
        print("=====================")
        for idx, cred_name in enumerate(existing_credentials, 1):
            cred_value = cm.get_credential(cred_name)
            print(f"\n{idx}. {cred_name}")
            print(f"   Current value: {cred_value}")

        # Get user selection
        while True:
            try:
                selection = input("\nEnter the number of the credential you want to update: ")
                
                selection_idx = int(selection) - 1
                if 0 <= selection_idx < len(existing_credentials):
                    selected_credential = existing_credentials[selection_idx]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return
            except ValueError:
                print("Please enter a valid number.")

        # Handle PostgreSQL connection string specially
        if 'postgresconnstring' in selected_credential:
            print("\nUpdating PostgreSQL connection string.")
            print("Let's build your new PostgreSQL connection string.")
            print("Default values will be shown in [brackets]. Press Enter to use them.")
            
            user = input("PostgreSQL username [postfiat]: ").strip() or "postfiat"
            password = input("PostgreSQL password: ").strip()
            host = input("Database host [localhost]: ").strip() or "localhost"
            port = input("Database port [5432]: ").strip() or "5432"
            
            # Determine database name based on credential name
            db_name = 'postfiat_db_testnet' if 'testnet' in selected_credential else 'postfiat_db'
            
            new_value = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            print(f"\nNew connection string created with database: {db_name}")
            print(f"Connection string: {new_value}")
        else:
            print(f"\nUpdating: {selected_credential}")
            new_value = input("Enter new value: ").strip()

        if new_value:
            # Update the credential
            cm.enter_and_encrypt_credential({selected_credential: new_value})
            print(f"\nSuccessfully updated credential: {selected_credential}")
        else:
            print("\nUpdate cancelled - empty value provided.")

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    except Exception as e:
        print(f"\nError: {str(e)}")
        if "MAC check failed" in str(e):
            print("Invalid encryption password.")
        return

if __name__ == "__main__":
    update_credentials()