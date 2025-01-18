import argparse
import os

from nodetools.utilities.setup_utilities import (
    arbitrary_credentials,
    db_init,
    db_init_auto,
    setup_node,
    setup_node_auto,
    update_credentials,
)

def main():
    parser = argparse.ArgumentParser(description="NodeTools CLI utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    init_db_parser = subparsers.add_parser('init-db', help='Initialize the database')
    init_db_parser.add_argument("--drop-tables", action="store_true", 
                               help="Drop and recreate tables (WARNING: Destructive)")
    init_db_parser.add_argument("--create-db", action="store_true", 
                               help="Create the database if it doesn't exist")
    init_db_parser.add_argument("--help-install", action="store_true", 
                               help="Show installation prerequisites")
    init_db_parser.add_argument("--revoke-privileges", action="store_true", 
                               help="Revoke privileges for testing (WARNING: Destructive)")

    setup_node_parser = subparsers.add_parser('setup-node', help='Setup a new node')
    update_creds_parser = subparsers.add_parser('update-creds', help='Update credentials')
    arbitrary_creds_parser = subparsers.add_parser('create-arbitrary-creds', 
                                                  help='Create arbitrary credentials')

    args = parser.parse_args()

    if args.command == 'init-db':
        if 'AUTO' not in os.environ:
            db_init.main(
                drop_tables=args.drop_tables,
                create_db=args.create_db,
                help_install=args.help_install,
                revoke_privileges=args.revoke_privileges
            )
        else:
            db_init_auto.main(
                drop_tables=False,
                create_db=args.create_db,
                help_install=False,
                revoke_privileges=False)
    elif args.command == 'setup-node':
        if 'AUTO' not in os.environ:
            setup_node.main()
        else:
            setup_node_auto.main()
    elif args.command == 'update-creds':
        update_credentials.main()
    elif args.command == 'create-arbitrary-creds':
        arbitrary_credentials.main()
    else:
        parser.print_help()
