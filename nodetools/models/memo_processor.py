# Standard imports
from typing import List, Dict, Any, Optional, Union
import re
import traceback
# Third party imports
import xrpl
from loguru import logger
from cryptography.fernet import InvalidToken
# Local imports
from nodetools.models.models import MemoGroup, MemoStructure, MemoDataStructureType
from nodetools.utilities.compression import compress_data, decompress_data, CompressionError
from nodetools.protocols.encryption import MessageEncryption
from nodetools.protocols.credentials import CredentialManager
from nodetools.utilities.credentials import SecretType
from nodetools.configuration.configuration import NodeConfig

class LegacyMemoProcessor:
    """Handles processing of legacy format memos"""
    
    @staticmethod
    def _determine_secret_type(address: str, node_config: NodeConfig) -> SecretType:
        """Determine the SecretType based on the address"""
        if address == node_config.node_address:
            return SecretType.NODE
        elif address == node_config.remembrancer_address:
            return SecretType.REMEMBRANCER
        else:
            raise ValueError(f"No SecretType found for address: {address}")

    @staticmethod
    async def process_group(
        group: MemoGroup, 
        credential_manager: Optional[CredentialManager] = None,
        message_encryption: Optional[MessageEncryption] = None,
        node_config: Optional[NodeConfig] = None
    ) -> str:
        """
        Process a complete sequence of chunks.
        
        Processing steps:
        1. Sort by chunk index
        2. Join chunks (removing chunk prefixes)
        3. Decompress if first chunk indicates compression (COMPRESSED__ prefix)
        4. Check for encryption after decompression (WHISPER__ prefix)

        For encrypted messages (WHISPER__ prefix), requires:
        - credential_manager: For accessing private keys
        - message_encryption: For ECDH operations

        Raises exception if processing fails.
        """
        if not group:
            raise ValueError("Empty sequence")

        # Sort memos in MemoGroup by chunk index
        sorted_sequence = sorted(
            group.memos,
            key=lambda tx: MemoStructure.from_transaction(tx).chunk_index or 0
        )

        # Join chunks (removing chunk prefixes)
        processed_data = ''
        for tx in sorted_sequence:
            chunk_data = tx['memo_data']
            if chunk_match := re.match(r'^chunk_\d+__', chunk_data):
                chunk_data = chunk_data[len(chunk_match.group(0)):]
            processed_data += chunk_data

        # Handle decompression
        if processed_data.startswith('COMPRESSED__'):
            processed_data = processed_data.replace('COMPRESSED__', '', 1)
            try:
                processed_data = decompress_data(processed_data)
            except CompressionError: 
                # This will happen often with legacy memos since they're processed asynchronously and system may not have all chunks yet
                raise

        # Handle decryption
        if processed_data.startswith('WHISPER__'):
            if not all([credential_manager, message_encryption]):
                logger.warning(
                    "Cannot decrypt message - missing required parameters. "
                    f"Need credential_manager: {bool(credential_manager)}, "
                    f"message_encryption: {bool(message_encryption)}"
                )
                return processed_data
            
            # Get channel details from first transaction
            first_tx = sorted_sequence[0]

            # Channel addresses and channel counterparties vary depending on the direction of the message
            # For example, if the message is from the node to the user, the account is the node's address and the destination is the user's address
            # But the channel address must always be the node's address, and the channel counterparty must always be the user's address
            # node_config.auto_handshake_addresses corresponds to the node's addresses that support encrypted channels
            if first_tx['destination'] in node_config.auto_handshake_addresses:
                channel_address = first_tx['destination']
                channel_counterparty = first_tx['account']
            else:  # The message is from the user to the node
                channel_address = first_tx['account']
                channel_counterparty = first_tx['destination']
            
            try:
                # Determine secret type based on receiving address
                secret_type = LegacyMemoProcessor._determine_secret_type(channel_address, node_config)
            
                # Get handshake keys
                channel_key, counterparty_key = await message_encryption.get_handshake_for_address(
                    channel_address=channel_address,
                    channel_counterparty=channel_counterparty
                )
                if not (channel_key and counterparty_key):
                    logger.warning("LegacyMemoProcessor.process_group: Cannot decrypt message - no handshake found")
                    return processed_data
            
                # Get shared secret using credential manager's API
                shared_secret = credential_manager.get_shared_secret(
                    received_key=counterparty_key, 
                    secret_type=secret_type
                )
                processed_data = message_encryption.process_encrypted_message(
                    processed_data, 
                    shared_secret
                )

            except InvalidToken:
                # This will happen often with legacy memos since they're processed asynchronously and system may not have all chunks yet
                raise

            except Exception as e:
                message = (
                    f"LegacyMemoProcessor.process_group: Error decrypting message {group.group_id} "
                    f"between address {channel_address} and counterparty {channel_counterparty}: {processed_data}"
                )
                logger.error(message)
                logger.error(traceback.format_exc())
                return f"[Decryption Failed] {processed_data}"

        return processed_data

class StandardizedMemoProcessor:
    """Handles processing of new standardized format memos"""

    @staticmethod
    def _determine_secret_type(address: str, node_config: NodeConfig) -> SecretType:
        """Determines SecretType based on address"""
        if address == node_config.node_address:
            return SecretType.NODE
        elif address == node_config.remembrancer_address:
            return SecretType.REMEMBRANCER
        else:
            raise ValueError(f"No SecretType found for address: {address}")
    
    @staticmethod
    async def process_group(
        group: MemoGroup,
        credential_manager: Optional[CredentialManager] = None,
        message_encryption: Optional[MessageEncryption] = None,
        node_config: Optional[NodeConfig] = None
    ) -> str:
        """
        Process a complete group of standardized format memos.
        The memo_format (e.g., "e.b.c1/4") indicates which processing steps are needed:
        - 'c' indicates chunking
        - 'b' indicates brotli compression
        - 'e' indicates ECDH encryption
        
        Processing occurs in a fixed order:
        1. Unchunk (if chunked)
        2. Decompress (if compressed)
        3. Decrypt (if encrypted)
        
        For encrypted messages, requires:
        - credential_manager: For accessing private keys
        - message_encryption: For ECDH operations
        - node_config: For determining secret types
        
        Raises ValueError if group is incomplete or processing fails.
        """
        if not group.memos:
            raise ValueError("Empty group")
        
        structure = MemoStructure.from_transaction(group.memos[0])
        if not structure.is_standardized_format:
            raise ValueError("Not a standardized format group")
        
        # For chunked messages, verify completeness and join
        if structure.is_chunked:
            if not structure.total_chunks:
                raise ValueError("Chunked message missing total_chunks")
                
            # Verify we have all chunks
            chunk_indices = group.chunk_indices
            if len(chunk_indices) != structure.total_chunks:
                raise ValueError(f"Missing chunks. Have {len(chunk_indices)}/{structure.total_chunks}")
                
            # Sort and join chunks
            sorted_msgs = sorted(
                group.memos,
                key=lambda tx: MemoStructure.from_transaction(tx).chunk_index or 0
            )
            
            processed_data = ''
            for tx in sorted_msgs:
                processed_data += tx['memo_data']
                
        else:
            # Single message
            processed_data = group.memos[0]['memo_data']
        
        # Apply decompression if specified
        if structure.compression_type == MemoDataStructureType.BROTLI:
            try:
                processed_data = decompress_data(processed_data)
            except CompressionError as e:
                logger.error(f"Decompression failed for group {group.group_id}: {e}")
                raise
                
        # Handle encryption if specified
        if structure.encryption_type == MemoDataStructureType.ECDH:
            if not all([credential_manager, message_encryption, node_config]):
                logger.warning(
                    "Cannot decrypt message - missing required parameters. "
                    f"Need credential_manager: {bool(credential_manager)}, "
                    f"message_encryption: {bool(message_encryption)}, "
                    f"node_config: {bool(node_config)}"
                )
                return processed_data

            # Get channel details from first transaction
            first_tx = group.memos[0]

            # Channel addresses and channel counterparties vary depending on the direction of the message
            # For example, if the message is from the node to the user, the account is the node's address and the destination is the user's address
            # But the channel address must always be the node's address, and the channel counterparty must always be the user's address
            # node_config.auto_handshake_addresses corresponds to the node's addresses that support encrypted channels
            if first_tx['destination'] in node_config.auto_handshake_addresses:
                channel_address = first_tx['destination']
                channel_counterparty = first_tx['account']
            else:  # The message is from the user to the node
                channel_address = first_tx['account']
                channel_counterparty = first_tx['destination']

            try:
                # Determine secret type based on receiving address
                secret_type = StandardizedMemoProcessor._determine_secret_type(channel_address, node_config)
            
                # Get handshake keys
                channel_key, counterparty_key = await message_encryption.get_handshake_for_address(
                    channel_address=channel_address,
                    channel_counterparty=channel_counterparty
                )
                if not (channel_key and counterparty_key):
                    logger.warning("Cannot decrypt message - no handshake found")
                    return processed_data

                # Get shared secret using credential manager's API
                shared_secret = credential_manager.get_shared_secret(
                    received_key=counterparty_key,
                    secret_type=secret_type
                )
                processed_data = message_encryption.process_encrypted_message(
                    processed_data, 
                    shared_secret
                )
            except Exception as e:
                logger.error(
                    f"StandardizedMemoProcessor.process_group: Error decrypting message {group.group_id} "
                    f"between address {channel_address} and counterparty {channel_counterparty}: {e}"
                )
                logger.error(traceback.format_exc())
                return f"[Decryption Failed] {processed_data}"
            
        return processed_data
    
    @staticmethod
    def validate_group(group: MemoGroup) -> bool:
        """
        Validate that all messages in the group have consistent structure.
        """
        if not group.memos:
            return False
            
        first_structure = MemoStructure.from_transaction(group.memos[0])
        if not first_structure.is_standardized_format:
            return False
            
        # Check all messages have same format
        for msg in group.memos[1:]:
            structure = MemoStructure.from_transaction(msg)
            if not structure.is_standardized_format:
                return False
                
            if (structure.encryption_type != first_structure.encryption_type or
                structure.compression_type != first_structure.compression_type or
                structure.total_chunks != first_structure.total_chunks):
                return False
                
        return True

class MemoProcessor:
    """Entry point for memo processing"""
    
    @staticmethod
    async def process_group(
        group: MemoGroup,
        credential_manager: Optional[CredentialManager] = None,
        message_encryption: Optional[MessageEncryption] = None,
        node_config: Optional[NodeConfig] = None
    ) -> Optional[str]:
        """Process a group of memos based on their format"""
        if not group.memos:
            return None
        
        first_tx = group.memos[0]
        structure = MemoStructure.from_transaction(first_tx)

        try:
            if structure.is_standardized_format:
                if StandardizedMemoProcessor.validate_group(group):
                    return await StandardizedMemoProcessor.process_group(
                        group,
                        credential_manager=credential_manager,
                        message_encryption=message_encryption,
                        node_config=node_config
                    )
                else:
                    logger.warning("Invalid standardized format group")
                    return None
            else:
                return await LegacyMemoProcessor.process_group(
                    group,
                    credential_manager=credential_manager,
                    message_encryption=message_encryption,
                    node_config=node_config
                )
        except Exception as e:
            logger.error(f"Error processing group: {str(e)}")
            return None
