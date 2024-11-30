from nodetools.utilities.credentials import CredentialManager
import getpass

def check_connection_strings():
    password = getpass.getpass("Enter your encryption password: ")
    cm = CredentialManager(password)
    
    # Replace 'your_node_name' with your actual node name
    mainnet_key = "postfiatfoundation_postgresconnstring"
    testnet_key = "postfiatfoundation_testnet_postgresconnstring"
    
    try:
        mainnet_conn = cm.get_credential(mainnet_key)
        print("\nMainnet connection string:", mainnet_conn)
    except:
        print("\nCouldn't find mainnet connection string")
        
    try:
        testnet_conn = cm.get_credential(testnet_key)
        print("\nTestnet connection string:", testnet_conn)
    except:
        print("\nCouldn't find testnet connection string")

if __name__ == "__main__":
    check_connection_strings()