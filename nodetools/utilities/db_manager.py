import pandas as pd
import sqlalchemy
#from agti.utilities import settings as gset
import psycopg2
import numpy as np
from nodetools.utilities.credentials import CredentialManager

class DBConnectionManager:
    ''' supports 1 database for the collective and one for the user'''
    def __init__(self):
        self.credential_manager = CredentialManager()

    def spawn_sqlalchemy_db_connection_for_user(self, user_name):
        db_connstring = self.credential_manager.get_credential(f'{user_name}_postgresconnstring')
        engine = sqlalchemy.create_engine(db_connstring)
        return engine
    
    def list_sqlalchemy_db_table_names_for_user(self, user_name):
        engine = self.spawn_sqlalchemy_db_connection_for_user(user_name)
        table_names = sqlalchemy.inspect(engine).get_table_names()
        return table_names
    
    def spawn_psycopg2_db_connection(self,user_name):
        
        db_connstring = self.credential_manager.get_credential(f'{user_name}_postgresconnstring')

        db_user = db_connstring.split('://')[1].split(':')[0]
        db_password = db_connstring.split('://')[1].split(':')[1].split('@')[0]
        db_host = db_connstring.split('://')[1].split(':')[1].split('@')[1].split('/')[0]
        db_name = db_connstring.split('/')[-1:][0]
        psycop_conn = psycopg2.connect(user=db_user, password=db_password, 
                                            host=db_host, database=db_name)
        return psycop_conn

