from decimal import Decimal
from enum import Enum
from typing import Optional
from nodetools.utilities.configuration import NetworkConfig, NodeConfig
from nodetools.utilities.constants import SystemMemoType

class AddressType(Enum):
    """Types of special addresses"""
    NODE = "Node"   # Each node has an address
    REMEMBRANCER = "Remembrancer"  # Each node may have a separate address for its remembrancer
    ISSUER = "Issuer"  # There's only one PFT issuer per L1 network
    OTHER = "Other"  # Any other address type, including users

class TransactionRequirementService:
    """Service for transaction requirements"""

    def __init__(self, network_config: NetworkConfig, node_config: NodeConfig):
        self.network_config = network_config
        self.node_config = node_config

        # Base PFT requirements by address type
        self.base_pft_requirements = {
            AddressType.NODE: Decimal('1'),
            AddressType.REMEMBRANCER: Decimal('1'),
            AddressType.ISSUER: Decimal('0'),
            AddressType.OTHER: Decimal('0')
        }

    def get_address_type(self, address: str) -> AddressType:
        """Get the type of address."""
        if address == self.node_config.node_address:
            return AddressType.NODE
        elif address == self.node_config.remembrancer_address:
            return AddressType.REMEMBRANCER
        elif address == self.network_config.issuer_address:
            return AddressType.ISSUER
        else:
            return AddressType.OTHER
        
    def get_pft_requirement(self, address: str, memo_type: Optional[str] = None) -> Decimal:
        """Get the PFT requirement for an address.
        
        Args:
            address: XRPL address to check
            memo_type: Optional memo type to consider
            
        Returns:
            Decimal: PFT requirement for the address
        """
        # System memos (like handshakes) don't require PFT
        if memo_type and memo_type in [type.value for type in SystemMemoType]:
            return Decimal('0')
        
        # Otherwise, use base requirements by address type
        return self.base_pft_requirements[self.get_address_type(address)]
    
    def is_node_address(self, address: str) -> bool:
        """Check if address is a node address"""
        return self.get_address_type(address) == AddressType.NODE
    
    def is_remembrancer_address(self, address: str) -> bool:
        """Check if address is a remembrancer address"""
        return self.get_address_type(address) == AddressType.REMEMBRANCER
    
    def is_issuer_address(self, address: str) -> bool:
        """Check if address is the issuer address"""
        return self.get_address_type(address) == AddressType.ISSUER
