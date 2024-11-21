import re
import datetime
from platform import system
from pathlib import Path
from nodetools.security import hash_tools as pwl
import getpass

class CredentialManager:
    def __init__(self):
        self.credential_file_path = self.get_credential_file_path()
        self.datadump_directory_path = self.get_datadump_directory_path()

    @staticmethod
    def datetime_current_EST():
        """Returns the current datetime in EST."""
        now = datetime.datetime.now()
        eastern_time = now.astimezone(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=-5)))
        return eastern_time

    @staticmethod
    def get_home_directory():
        """Returns the home directory of the current user."""
        return Path.home()

    def get_credentials_directory(self):
        """Returns the path to the agti directory, creating it if it does not exist."""
        home_dir = self.get_home_directory()
        creds_dir = home_dir / "postfiatcreds"
        creds_dir.mkdir(exist_ok=True)
        return creds_dir

    def get_credential_file_path(self):
        """Returns the path to the credential file, creating it if it does not exist."""
        creds_dir = self.get_credentials_directory()
        cred_file_path = creds_dir / "postfiat_cred_list.txt"
        
        if not cred_file_path.exists():
            cred_file_path.touch()
            print(f"CREATED MANY AS ONE CREDENTIALS FILE AT {cred_file_path}")
        
        return cred_file_path

    def get_datadump_directory_path(self):
        """Returns the path to the datadump directory, creating it if it does not exist."""
        home_dir = self.get_home_directory()
        datadump_dir = home_dir / "datadump"
        data_dir = datadump_dir / "data"
        
        datadump_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        
        print(f"CREATED DATADUMP DIRECTORY AT {datadump_dir}")
        print(f"CREATED DATA DIRECTORY AT {data_dir}")
        
        return datadump_dir

    @staticmethod
    def convert_directory_tuple_to_filename(directory_tuple):
        """Converts a tuple of directory paths to a single path string."""
        string_list = []
        for item in directory_tuple:
            if isinstance(item, list):
                string_list.extend(item)
            else:
                string_list.append(item)
        
        return '/'.join(string_list)

    @staticmethod
    def convert_credential_string_to_map(stringx):
        """Converts a credential string to a map."""
        def convert_string_to_bytes(string):
            if string.startswith("b'"):
                return bytes(string[2:-1], 'utf-8')
            else:
                return string
        
        variables = re.findall(r'variable___\w+', stringx)
        map_constructor = {}
        
        for variable_to_work in variables:
            raw_text = stringx.split(variable_to_work)[1].split('variable___')[0].strip()
            variable_name = variable_to_work.split('variable___')[1]
            map_constructor[variable_name] = convert_string_to_bytes(string=raw_text)
        
        return map_constructor

    def read_creds(self):
        with open(self.credential_file_path, 'r') as f:
            credblock = f.read()
        return credblock

    def output_cred_map(self):
        credblock = self.read_creds()
        cred_map = self.convert_credential_string_to_map(credblock)
        return cred_map

    def enter_and_encrypt_credential(self):
        credential_ref = input('Enter your credential reference (example: aws_key): ')
        existing_cred_map = self.output_cred_map()
        
        if credential_ref in existing_cred_map.keys():
            print('Credential is already loaded')
            print(f'To edit credential file directly go to {self.credential_file_path}')
            return
        
        pw_data = input('Enter your unencrypted credential (will be encrypted next step): ')
        pw_encryptor = getpass.getpass('Enter your encryption password: ')
        
        credential_byte_str = pwl.password_encrypt(message=bytes(pw_data, 'utf-8'), password=pw_encryptor)
        
        fblock = f'''
        variable___{credential_ref}
        {credential_byte_str}'''
        
        with open(self.credential_file_path, 'a') as f:
            f.write(fblock)
        
        print(f"Added credential {credential_ref} to {self.credential_file_path}")

    def enter_and_encrypt_credential__variable_based(self, credential_ref, pw_data, pw_encryptor):
        existing_cred_map = self.output_cred_map()
        
        if credential_ref in existing_cred_map.keys():
            print('Credential is already loaded')
            print(f'To edit credential file directly go to {self.credential_file_path}')
            return
        
        credential_byte_str = pwl.password_encrypt(message=bytes(pw_data, 'utf-8'), password=pw_encryptor)
        
        fblock = f'''
        variable___{credential_ref}
        {credential_byte_str}'''
        
        with open(self.credential_file_path, 'a') as f:
            f.write(fblock)
        
        print(f"Added credential {credential_ref} to {self.credential_file_path}")

    def decrypt_credential(self, credential_ref, pw_decryptor):
        """Decrypts a credential."""
        existing_cred_map = self.output_cred_map()
        
        if credential_ref in existing_cred_map.keys():
            encrypted_cred = existing_cred_map[credential_ref]
            decrypted_cred = pwl.password_decrypt(token=encrypted_cred, password=pw_decryptor)
            return decrypted_cred.decode('utf-8')
        else:
            raise ValueError('Credential not found')

    def output_fully_decrypted_cred_map(self, pw_decryptor):
        """Decrypts all credentials in the file."""
        existing_cred_map = self.output_cred_map()
        decrypted_cred_map = {}
        
        for credential_ref in existing_cred_map.keys():
            decrypted_cred_map[credential_ref] = self.decrypt_credential(credential_ref=credential_ref, pw_decryptor=pw_decryptor)
        
        return decrypted_cred_map

class PasswordMapLoader:
    def __init__(self, password=None):
        if password is None:
            password = getpass.getpass('Enter your decryption password: ')
        self.password = password
        self.credential_manager = CredentialManager()
        self.pw_map = self.credential_manager.output_fully_decrypted_cred_map(pw_decryptor=self.password)
