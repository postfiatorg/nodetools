# Standard Library
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
import traceback
import getpass
import os
import sys
import asyncio

# Third Party
from loguru import logger

# Local
from ..performance.monitor import PerformanceMonitor
from ..configuration.configuration import (
    get_node_config,
    get_network_config,
    RuntimeConfig
)
from ..ai.openrouter import OpenRouterTool
from ..models.models import Dependencies, BusinessLogicProvider
from ..utilities.credentials import CredentialManager
from ..utilities.db_manager import DBConnectionManager
from ..utilities.transaction_repository import TransactionRepository
from ..utilities.generic_pft_utilities import GenericPFTUtilities
from ..utilities.encryption import MessageEncryption
from ..utilities.xrpl_monitor import XRPLWebSocketMonitor
from ..utilities.transaction_orchestrator import TransactionOrchestrator

@dataclass
class ServiceContainer:
    """Container for NodeTools service initialization and management"""
    dependencies: Dependencies
    runtime_config: RuntimeConfig
    xrpl_monitor: XRPLWebSocketMonitor
    transaction_orchestrator: TransactionOrchestrator
    db_connection_manager: DBConnectionManager

    @classmethod
    def initialize(
        cls,
        business_logic: BusinessLogicProvider,
        password_prompt: Optional[Callable[[], str]] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        notifications: bool = False
    ) -> 'ServiceContainer':
        """
        Initialize all NodeTools services with credential management
        
        Args:
            business_logic: Business logic provider
            password_prompt: Optional function to get password (defaults to getpass.getpass)
            performance_monitor: Optional performance monitoring instance
        """
        password_prompt = password_prompt or getpass.getpass

        # Startup phase
        try:
            if 'AUTO' not in os.environ:
                while True:
                    try:
                        password = getpass.getpass("Enter your password: ")
                        credential_manager = CredentialManager(password=password)
                        break
                    except Exception as e:
                        print("Invalid password. Please try again.")
            else:
                password = os.environ['ENCRYPTION_PASSWORD']
                credential_manager = CredentialManager(password=password)

            # Start performance monitoring if provided
            if performance_monitor:
                performance_monitor.start()

            runtime_config = cls.configure_runtime(input_prompt=input)

        except KeyboardInterrupt:
            print("\nStartup cancelled")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            logger.error(traceback.format_exc())
            raise

        try:
            # Retreive network, node, and runtime configurations
            network_config = get_network_config()
            node_config = get_node_config()

            # Initialize core NodeTools services
            openrouter = OpenRouterTool(
                credential_manager=credential_manager
            )

            db_connection_manager = DBConnectionManager(
                credential_manager=credential_manager
            )

            transaction_repository = TransactionRepository(
                db_manager=db_connection_manager,
                username=node_config.node_name,
            )

            pft_utilities = GenericPFTUtilities(
                network_config=network_config,
                node_config=node_config,
                runtime_config=runtime_config,
                credential_manager=credential_manager,
                db_connection_manager=db_connection_manager,
                transaction_repository=transaction_repository,
            )

            message_encryption = MessageEncryption(
                node_config=node_config,
                pft_utilities=pft_utilities,
                transaction_repository=transaction_repository,
            )
            pft_utilities.message_encryption = message_encryption

            xrpl_monitor = XRPLWebSocketMonitor(
                generic_pft_utilities=pft_utilities,
                transaction_repository=transaction_repository,
            )

            transaction_orchestrator = TransactionOrchestrator(
                node_config=node_config,
                network_config=network_config,
                business_logic_provider=business_logic,
                generic_pft_utilities=pft_utilities, 
                transaction_repository=transaction_repository,
                credential_manager=credential_manager,
                message_encryption=message_encryption,
                openrouter=openrouter,
                xrpl_monitor=xrpl_monitor,
                notifications=notifications
            )

            # Create core dependencies container
            deps = Dependencies(
                network_config=network_config,
                node_config=node_config,
                credential_manager=credential_manager,
                generic_pft_utilities=pft_utilities,
                openrouter=openrouter,
                transaction_repository=transaction_repository,
                message_encryption=message_encryption,
            )

            logger.info("All NodeTools services initialized")

            return cls(
                dependencies=deps,
                runtime_config=runtime_config,
                xrpl_monitor=xrpl_monitor,
                transaction_orchestrator=transaction_orchestrator,
                db_connection_manager=db_connection_manager
            )

        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            logger.error(traceback.format_exc())
            raise

    @classmethod
    def configure_runtime(
        cls,
        input_prompt: Callable[[str], str] = input
    ) -> RuntimeConfig:
        """Configure runtime settings based on user inputs or defaults"""

        # Network selection
        if 'AUTO' not in os.environ:
            print(f"Network Configuration:\n1. Mainnet\n2. Testnet")
            network_choice = input_prompt("Select network (1/2) [default=2]: ").strip() or "2"
            use_testnet = network_choice == "2"
        else:
            use_testnet = os.environ['NETWORK'] == "testnet"
        network_config = get_network_config()

        # Initialize runtime config
        runtime_config = RuntimeConfig()
        runtime_config.USE_TESTNET = use_testnet

        # Load network configuration based on selection
        network_config = get_network_config()

        # Local node configuration with network-specific context
        if network_config.local_rpc_url:
            print(f"\nLocal node configuration:")
            print(f"Local {network_config.name} node URL: {network_config.local_rpc_url}")
            use_local = input(
                "Do you have a local node configured? (y/n) [default=n]: "
            ).strip() or "n"
            runtime_config.HAS_LOCAL_NODE = use_local.lower() == "y"
        else:
            print(f"\nNo local node configuration available for {network_config.name}")
            runtime_config.HAS_LOCAL_NODE = False

        logger.debug(f"\nInitializing services for {network_config.name}...")
        logger.info(
            f"Using {'local' if runtime_config.HAS_LOCAL_NODE else 'public'} endpoints..."
        )

        return runtime_config

    @property
    def running(self):
        """Check if the transaction orchestrator is running"""
        return self.transaction_orchestrator.running

    @property
    def node_config(self):
        """Get the node configuration"""
        return self.dependencies.node_config

    @property
    def network_config(self):
        """Get the network configuration"""
        return self.dependencies.network_config

    @property
    def notification_queue(self) -> asyncio.Queue:
        """Access the transaction orchestrator's notification queue"""
        return self.transaction_orchestrator.notification_queue

    def start(self):
        """Start the transaction orchestrator"""
        self.transaction_orchestrator.start()

    def stop(self):
        """Stop the transaction orchestrator"""
        self.transaction_orchestrator.stop()

    def get_credential(self, credential_key: str) -> str:
        """Get a credential from the credential manager"""
        return self.dependencies.credential_manager.get_credential(credential_key)
