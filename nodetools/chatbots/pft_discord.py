from typing import Dict
from xrpl.wallet import Wallet
import discord
from discord import Object, Interaction, SelectOption, app_commands
from discord.ui import Modal, TextInput, View, Select, Button
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.utilities.credentials import CredentialManager
from nodetools.utilities.generic_pft_utilities import *
from nodetools.task_processing.task_management import PostFiatTaskGenerationSystem
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.performance.monitor import PerformanceMonitor
from nodetools.configuration.configuration import RuntimeConfig, get_network_config
from nodetools.task_processing.user_context_parsing import UserTaskParser
import asyncio
import pytz
import datetime
from datetime import datetime, time, timezone, timedelta
import nodetools.configuration.constants as constants
import nodetools.configuration.configuration as config
import getpass
from loguru import logger
from nodetools.configuration.configure_logger import configure_logger
from pathlib import Path
from dataclasses import dataclass
import traceback
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.chatbots.odv_sprint_planner import ODVSprintPlannerO1
from nodetools.chatbots.odv_context_doc_improvement import ODVContextDocImprover
from nodetools.ai.openrouter import OpenRouterTool
from nodetools.chatbots.corbanu_beta import CorbanuChatBot
from nodetools.chatbots.odv_focus_analyzer import ODVFocusAnalyzer
from nodetools.chatbots.discord_modals import (
    PFTTransactionModal,
    XRPTransactionModal,
    AcceptanceModal,
    RefusalModal,
    InitiationModal,
    UpdateLinkModal,
    CompletionModal,
    VerificationModal
)

@dataclass
class AccountInfo:
    address: str
    username: str = ''
    xrp_balance: float = 0
    pft_balance: float = 0
    transaction_count: int = 0
    monthly_pft_avg: float = 0
    weekly_pft_avg: float = 0
    google_doc_link: Optional[str] = None

@dataclass
class DeathMarchSettings:
    # Configuration
    timezone: str
    start_time: time    # Daily start time
    end_time: time      # Daily end time
    check_interval: int # Minutes between check-ins

    # Session-specific data
    channel_id: Optional[int] = None
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    last_checkin: Optional[datetime] = None

class MyClient(discord.Client):

    NON_EPHEMERAL_USERS = {402536023483088896}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get network configuration and set network-specific attributes
        self.network_config = config.get_network_config()
        self.node_config = config.get_node_config()
        self.remembrancer = self.node_config.remembrancer_address

        # Initialize components
        self.openrouter = OpenRouterTool()
        self.openai_request_tool = OpenAIRequestTool()
        self.generic_pft_utilities = GenericPFTUtilities()
        self.post_fiat_task_generation_system = PostFiatTaskGenerationSystem()
        self.user_task_parser = UserTaskParser(
            task_management_system=self.post_fiat_task_generation_system,
            generic_pft_utilities=self.generic_pft_utilities
        )

        self.default_openai_model = constants.DEFAULT_OPEN_AI_MODEL
        self.conversations = {}
        self.user_seeds = {}
        self.doc_improvers = {}
        self.sprint_planners = {}  # Dictionary: user_id -> ODVSprintPlanner instance
        self.user_steps = {}       # Dictionary: user_id -> current step in the sprint process
        self.user_questions = {}
        self.user_deathmarch_settings: Dict[int, DeathMarchSettings] = {}
        self.death_march_tasks = {}
        self.tree = app_commands.CommandTree(self)

        # Caches for dataframes to enable deferred modals
        self.pending_tasks_cache = {}
        self.refuseable_tasks_cache = {}
        self.accepted_tasks_cache = {}
        self.verification_tasks_cache = {}

    def is_special_user_non_ephemeral(self, interaction: discord.Interaction) -> bool:
        """Return False if the user is not in the NON_EPHEMERAL_USERS set, else True."""
        output = not (interaction.user.id in self.NON_EPHEMERAL_USERS)
        return output

    def chunk_message(self, message, max_length=1900):
            """Split a message into multiple parts to avoid exceeding Discord's 2000-char limit."""
            lines = message.split('\n')
            chunks = []
            current_chunk = ""
            for line in lines:
                if len(current_chunk) + len(line) + 1 > max_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            return chunks

    async def setup_hook(self):
        """Sets up the slash commands for the bot and initiates background tasks."""
        guild_id = self.node_config.discord_guild_id
        guild = Object(id=guild_id)
        self.bg_task = self.loop.create_task(self.transaction_checker())

        @self.tree.command(name="pf_new_wallet", description="Generate a new XRP wallet")
        async def pf_new_wallet(interaction: Interaction):
            # Generate the wallet
            test_wallet = Wallet.create()
            classic_address = test_wallet.classic_address
            wallet_seed = test_wallet.seed

            # Create and send the modal
            class WalletInfoModal(discord.ui.Modal, title='New XRP Wallet'):
                def __init__(self, client):
                    super().__init__()
                    self.client: MyClient = client

                address = discord.ui.TextInput(
                    label='Address (Do not modify)',
                    default=classic_address,
                    style=discord.TextStyle.short,
                    required=True
                )
                seed = discord.ui.TextInput(
                    label='Secret - Submit Stores. Cancel (Exit)',
                    default=wallet_seed,
                    style=discord.TextStyle.short,
                    required=True
                )

                async def on_submit(self, interaction: discord.Interaction):
                    user_id = interaction.user.id
                    logger.debug(f"WalletInfoModal.on_submit: Storing seed for user {interaction.user.name} (ID: {user_id})")
                    self.client.user_seeds[user_id] = self.seed.value

                    await interaction.response.send_message(
                        "Wallet created successfully. You must fund the wallet with 15+ XRP to use as a Post Fiat Wallet. "
                        "The seed is stored to Discord Hot Wallet. To store different seed use /pf_store_seed. "
                        "We recommend you often sweep the wallet to a cold wallet. Use the /pf_send command to do so",
                        ephemeral=True
                    )

            # Create the modal with the client reference and send it
            modal = WalletInfoModal(interaction.client)
            await interaction.response.send_modal(modal)

        @self.tree.command(name="pf_show_seed", description="Show your stored seed")
        async def pf_show_seed(interaction: discord.Interaction):
            user_id = interaction.user.id
            
            # Check if the user has a stored seed
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                # Create and send an ephemeral message with the seed
                await interaction.response.send_message(
                    f"Your stored seed is: {seed}\n"
                    "This message will be deleted in 30 seconds for security reasons.",
                    ephemeral=True,
                    delete_after=30
                )
            else:
                await interaction.response.send_message(
                    "No seed found for your account. Use /pf_store_seed to store a seed first.",
                    ephemeral=True
                )
        
        @self.tree.command(name="pf_guide", description="Show a guide of all available commands")
        async def pf_guide(interaction: discord.Interaction):
            guide_text = """
# Post Fiat Discord Bot Guide

### Info Commands
1. /pf_guide: Show this guide
2. /pf_my_wallet: Show information about your stored wallet.
3. /wallet_info: Get information about a specific wallet address.
4. /pf_show_seed: Display your stored seed 
5. /pf_rewards: Show recent PFT rewards.
6. /pf_outstanding: Show your outstanding tasks and verification tasks.

### Initiation
1. /pf_new_wallet: Generate a new XRP wallet. You need to fund via Coinbase etc to continue
2. /pf_store_seed: Securely store your wallet seed.
3. /pf_initiate: Initiate your commitment to the Post Fiat system, get access to PFT and initial grant
4. /pf_update_link: Update your Google Doc link

### Task Request
1. /pf_request_task: Request a new Post Fiat task.
2. /pf_accept: View and accept available tasks.
3. /pf_refuse: View and refuse available tasks.
4. /pf_initial_verification: Submit a completed task for verification.
5. /pf_final_verification: Submit final verification for a task to receive reward

### Transaction
1. /xrp_send: Send XRP to a destination address with a memo.
2. /pf_send: Open a transaction form to send PFT tokens with a memo.
3. /pf_log: take notes re your workflows, with optional encryption

## Post Fiat operates on a Google Document.
1. Set your Document to be shared (File/Share/Share With Others/Anyone With Link)
2. The PF Initiate Function requires a document and a verbal committment
3. Place the following section in your document:
___x TASK VERIFICATION SECTION START x___ 
task verification details are here 
___x TASK VERIFICATION SECTION END x___

## Local Version
You can run a local version of the wallet. Please reference the Post Fiat Github
https://github.com/postfiatorg/pftpyclient

Note: XRP wallets need 1 XRP to transact. We recommend you fund your wallet with a bit more to start.
"""
            
            await interaction.response.send_message(guide_text, ephemeral=True)

        @self.tree.command(name="pf_my_wallet", description="Show your wallet information")
        async def pf_my_wallet(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            # Defer the response to avoid timeout
            await interaction.response.defer(ephemeral=ephemeral_setting)
            
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "No seed found for your account. Use /pf_store_seed to store a seed first.",
                    ephemeral=True
                )
                return

            try:
                seed = self.user_seeds[user_id]
                logger.debug(f"MyClient.setup_hook.pf_my_wallet: Spawning wallet to fetch info for {interaction.user.name}")
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
                wallet_address = wallet.classic_address

                # Get account info
                account_info = self.generate_basic_balance_info_string(address=wallet.address)
                
                # Get recent messages
                incoming_messages, outgoing_messages = generic_pft_utilities.get_recent_messages(wallet_address)

                # Split long strings if they exceed Discord's limit
                def truncate_field(content, max_length=1024):
                    if len(content) > max_length:
                        return content[:max_length-3] + "..."
                    return content

                # Create multiple embeds if needed
                embeds = []
                
                # First embed with basic info
                embed = discord.Embed(title="Your Wallet Information", color=0x00ff00)
                embed.add_field(name="Wallet Address", value=wallet_address, inline=False)
                
                # Split account info into multiple fields if needed
                if len(account_info) > 1024:
                    parts = [account_info[i:i+1024] for i in range(0, len(account_info), 1024)]
                    for i, part in enumerate(parts):
                        embed.add_field(name=f"Balance Information {i+1}", value=part, inline=False)
                else:
                    embed.add_field(name="Balance Information", value=account_info, inline=False)
                
                embeds.append(embed)

                if incoming_messages or outgoing_messages:
                    embed2 = discord.Embed(title="Recent Transactions", color=0x00ff00)
                    
                    if incoming_messages:
                        incoming = truncate_field(incoming_messages)
                        embed2.add_field(name="Most Recent Incoming Transaction", value=incoming, inline=False)
                    
                    if outgoing_messages:
                        outgoing = truncate_field(outgoing_messages)
                        embed2.add_field(name="Most Recent Outgoing Transaction", value=outgoing, inline=False)
                    
                    embeds.append(embed2)

                # Send all embeds
                await interaction.followup.send(embeds=embeds, ephemeral=ephemeral_setting)
            
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}. Please try again later or contact support if the issue persists."
                logger.error(f"Full traceback: {traceback.format_exc()}")
                await interaction.followup.send(error_message, ephemeral=True)

        @self.tree.command(name="wallet_info", description="Get information about a wallet")
        async def wallet_info(interaction: discord.Interaction, wallet_address: str):
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            try:
                account_info = self.generate_basic_balance_info_string(address=wallet_address, owns_wallet=False)
                
                # Create an embed for better formatting
                embed = discord.Embed(title="Wallet Information", color=0x00ff00)
                embed.add_field(name="Wallet Address", value=wallet_address, inline=False)
                embed.add_field(name="Account Info", value=account_info, inline=False)
                
                await interaction.response.send_message(embed=embed, ephemeral=ephemeral_setting)
            except Exception as e:
                await interaction.response.send_message(f"An error occurred: {str(e)}", ephemeral=ephemeral_setting)

        @self.tree.command(name="admin_debug_full_user_context", description="Return the full user context")
        async def admin_debug_full_user_context(interaction: discord.Interaction, wallet_address: str):
            # Check if the user has permission (matches the specific ID)
            if interaction.user.id not in constants.DISCORD_SUPER_USER_IDS:
                await interaction.response.send_message(
                    "You don't have permission to use this command.", 
                    ephemeral=True
                )
                return

            try:
                await interaction.response.defer(ephemeral=True)

                full_user_context = self.user_task_parser.get_full_user_context_string(
                    account_address=wallet_address,
                    n_memos_in_context=20
                )

                await self.send_long_interaction_response(
                        interaction, 
                        f"\n{full_user_context}", 
                        ephemeral=True
                    )
            except Exception as e:
                logger.error(f"Error in pd_debug_full_user_context: {str(e)}")
                await interaction.followup.send(
                    f"An error occurred while fetching the full user context: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(name="admin_change_ephemeral_setting", description="Change the ephemeral setting for self")
        async def admin_change_ephemeral_setting(interaction: discord.Interaction, public: bool):
            # Check if the user has permission (matches the specific ID)
            if interaction.user.id not in constants.DISCORD_SUPER_USER_IDS:
                await interaction.response.send_message(
                    "You don't have permission to use this command.", 
                    ephemeral=True
                )
                return
            
            user_id = interaction.user.id
            if public:
                self.NON_EPHEMERAL_USERS.add(user_id)
                setting = "PUBLIC"
            else:
                self.NON_EPHEMERAL_USERS.discard(user_id)
                setting = "PRIVATE"
            
            await interaction.response.send_message(
                f"Your messages will now be {setting}",
                ephemeral=True
            )

        @self.tree.command(name="pf_send", description="Open a transaction form")
        async def pf_send(interaction: Interaction):
            user_id = interaction.user.id

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /store_seed before initiating a transaction.", 
                    ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Pass the user's wallet to the modal
            await interaction.response.send_modal(
                PFTTransactionModal(
                    wallet=wallet,
                    generic_pft_utilities=self.generic_pft_utilities
                )
            )

        @self.tree.command(name="xrp_send", description="Send XRP to a destination address")
        async def xrp_send(interaction: discord.Interaction):
            user_id = interaction.user.id
            
            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed before initiating a transaction.", ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Pass the user's wallet to the modal
            await interaction.response.send_modal(
                XRPTransactionModal(
                    wallet=wallet,
                    generic_pft_utilities=self.generic_pft_utilities
                )
            )

        @self.tree.command(name="pf_store_seed", description="Store a seed")
        async def store_seed(interaction: discord.Interaction):
            # Define the modal with a reference to the client
            class SeedModal(discord.ui.Modal, title='Store Your Seed'):
                seed = discord.ui.TextInput(label='Seed', style=discord.TextStyle.long)

                def __init__(self, client, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.client = client  # Save the client reference

                async def on_submit(self, interaction: discord.Interaction):
                    user_id = interaction.user.id
                    # Test seed for validity
                    try:
                        generic_pft_utilities.spawn_wallet_from_seed(self.seed.value.strip())
                    except Exception as e:
                        await interaction.response.send_message(f"An error occurred while storing your seed: {str(e)}", ephemeral=True)
                        return
                    
                    self.client.user_seeds[user_id] = self.seed.value.strip()  # Store the seed
                    await interaction.response.send_message(f'Seed stored successfully for user {interaction.user.name}.', ephemeral=True)

            # Pass the client instance to the modal
            await interaction.response.send_modal(SeedModal(client=self))
            logger.debug(f"MyClient.setup_hook.store_seed: Seed storage command executed by {interaction.user.name}")

        @self.tree.command(name="pf_initiate", description="Initiate your commitment")
        async def pf_initiate(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /store_seed before initiating.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:
                # Spawn the user's wallet
                logger.debug(f"MyClient.setup_hook.pf_initiate: Spawning wallet to initiate for {interaction.user.name}")
                username = interaction.user.name
                seed = self.user_seeds[user_id]
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)

                # Check initiation status
                initiation_check_success = await self._check_initiation_rite(
                    interaction=interaction,
                    wallet_address=wallet.address,
                    require_initiation=False  # this means block re-initiations unless on testnet with ENABLE_REINITIATIONS = True
                )
                if not initiation_check_success:
                    await interaction.followup.send(
                        "You've already completed an initiation rite. Re-initiation is not allowed.", 
                        ephemeral=ephemeral_setting
                    )
                    return

                # Create a button to trigger the modal
                async def button_callback(button_interaction: discord.Interaction):
                    await button_interaction.response.send_modal(
                        InitiationModal(
                            seed=seed,
                            username=username,
                            client_instance=self,
                            post_fiat_task_generation_system=post_fiat_task_generation_system,
                            ephemeral_setting=ephemeral_setting
                        )
                    )

                button = Button(label="Begin Initiation", style=discord.ButtonStyle.primary)
                button.callback = button_callback

                view = View()
                view.add_item(button)
                await interaction.followup.send(
                    "Click the button below to begin your initiation:", 
                    view=view, 
                    ephemeral=ephemeral_setting
                )

            except Exception as e:
                logger.error(f"MyClient.setup_hook.pf_initiate: Error during initiation: {str(e)}")
                await interaction.followup.send(f"An error occurred during initiation: {str(e)}", ephemeral=ephemeral_setting)

        @self.tree.command(name="pf_update_link", description="Update your Google Doc link")
        async def pf_update_link(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed first.",
                    ephemeral=ephemeral_setting
                )
                return
            
            try:
                logger.debug(f"MyClient.pf_update_link: Spawning wallet for {interaction.user.name} to update google doc link")
                seed = self.user_seeds[user_id]
                username = interaction.user.name
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)

                # Check initiation status
                initiation_check_success = await self._check_initiation_rite(
                    interaction=interaction,
                    wallet_address=wallet.address
                )
                if not initiation_check_success:
                    await interaction.followup.send(
                        "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                        ephemeral=ephemeral_setting
                    )
                    return

                # Create a button to trigger the modal
                async def button_callback(button_interaction: discord.Interaction):
                    await button_interaction.response.send_modal(
                        UpdateLinkModal(
                            seed=seed,
                            username=username,
                            client_instance=self,
                            post_fiat_task_generation_system=post_fiat_task_generation_system,
                            ephemeral_setting=ephemeral_setting
                        )
                    )

                button = Button(label="Update Google Doc Link", style=discord.ButtonStyle.primary)
                button.callback = button_callback

                view = View()
                view.add_item(button)
                await interaction.followup.send(
                    "Click the button below to update your Google Doc link:", 
                    view=view, 
                    ephemeral=ephemeral_setting
                )

            except Exception as e:
                logger.error(f"MyClient.pf_update_link: Error during update: {str(e)}")
                await interaction.followup.send(f"An error occurred during update: {str(e)}", ephemeral=True)

        @self.tree.command(name="odv_sprint", description="Start an ODV sprint planning session")
        async def odv_sprint(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before starting an ODV sprint planning session.", 
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:
                odv_planner = ODVSprintPlannerO1(
                    account_address=wallet.classic_address,
                    openrouter=self.openrouter,
                    user_context_parser=self.user_task_parser,
                    pft_utils=self.generic_pft_utilities
                )
                self.sprint_planners[user_id] = odv_planner
                logger.debug(f"MyClient.odv_sprint: Initialized ODV sprint planner for {interaction.user.name}")

                # Potentially long operation
                logger.debug(f"MyClient.odv_sprint: Getting initial response for {interaction.user.name}")
                initial_response = await odv_planner.get_response_async("Please provide your context analysis.")

                # Use the helper function to send the possibly long response
                logger.debug(f"MyClient.odv_sprint: Sending initial response for {interaction.user.name}")
                await self.send_long_interaction_response(
                    interaction, 
                    f"**ODV Sprint Planning Initialized**\n\n{initial_response}", 
                    ephemeral=ephemeral_setting
                )
            except Exception as e:
                logger.error(f"Error in odv_sprint: {str(e)}")
                await interaction.followup.send(
                    f"An error occurred while initializing the ODV sprint planning session: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(name="odv_sprint_reply", description="Continue the ODV sprint planning session")
        @app_commands.describe(message="Your next input to ODV")
        async def odv_sprint_reply(interaction: discord.Interaction, message: str):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            if user_id not in self.sprint_planners:
                await interaction.followup.send(
                    "No active ODV sprint planning session. Start one with /odv_sprint.", 
                    ephemeral=ephemeral_setting
                )
                return

            odv_planner: ODVSprintPlannerO1 = self.sprint_planners[user_id]
            logger.debug(f"MyClient.odv_sprint_reply: Continuing ODV sprint planning session for {interaction.user.name}")

            try:
                # Now using async version
                logger.debug(f"MyClient.odv_sprint_reply: Getting response for {interaction.user.name}")
                response = await odv_planner.get_response_async(message)
                logger.debug(f"MyClient.odv_sprint_reply: Response received for {interaction.user.name}")
                await self.send_long_interaction_response(interaction, response, ephemeral=ephemeral_setting)
            except Exception as e:
                logger.error(f"Error in odv_sprint_reply: {str(e)}")
                await interaction.followup.send(
                    f"An error occurred: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(name="pf_configure_deathmarch", description="Configure your death march")
        async def pf_configure_deathmarch(interaction: discord.Interaction):
            # Common timezone options
            timezone_options = [
                SelectOption(label="US/Pacific", description="Los Angeles, Seattle, Vancouver (UTC-7/8)"),
                SelectOption(label="US/Mountain", description="Denver, Phoenix (UTC-6/7)"),
                SelectOption(label="US/Central", description="Chicago, Mexico City (UTC-5/6)"),
                SelectOption(label="US/Eastern", description="New York, Toronto, Miami (UTC-4/5)"),
                SelectOption(label="Europe/London", description="London, Dublin, Lisbon (UTC+0/1)"),
                SelectOption(label="Europe/Paris", description="Paris, Berlin, Rome (UTC+1/2)"),
                SelectOption(label="Asia/Tokyo", description="Tokyo, Seoul (UTC+9)"),
                SelectOption(label="Australia/Sydney", description="Sydney, Melbourne (UTC+10/11)"),
                SelectOption(label="Pacific/Auckland", description="Auckland, Wellington (UTC+12/13)")
            ]

            # Time options vary based on environment
            if config.RuntimeConfig.USE_TESTNET:
                # Testing: Allow any hour
                start_time_options = [
                    SelectOption(label=f"{hour:02d}:00", value=f"{hour:02d}:00") 
                    for hour in range(0, 24)  # 0-23 hours
                ]
                end_time_options = [
                    SelectOption(label=f"{hour:02d}:00", value=f"{hour:02d}:00") 
                    for hour in range(0, 24)  # 0-23 hours
                ]
                # Add shorter intervals for testing
                interval_options = [
                    SelectOption(label="1 minute", value="1", description="⚠️ Testing only"),
                    SelectOption(label="5 minutes", value="5", description="⚠️ Testing only"),
                    SelectOption(label="15 minutes", value="15", description="⚠️ Testing only"),
                    SelectOption(label="30 minutes", value="30"),
                    SelectOption(label="1 hour", value="60"),
                    SelectOption(label="2 hours", value="120")
                ]
            else:
                # Production: Restricted hours
                start_time_options = [
                    SelectOption(label=f"{hour:02d}:00", value=f"{hour:02d}:00") 
                    for hour in range(5, 13)  # 5 AM to 12 PM
                ]
                end_time_options = [
                    SelectOption(label=f"{hour:02d}:00", value=f"{hour:02d}:00") 
                    for hour in range(16, 24)  # 4 PM to 11 PM
                ]
                # Production intervals
                interval_options = [
                    SelectOption(label="30 minutes", value="30"),
                    SelectOption(label="1 hour", value="60"),
                    SelectOption(label="2 hours", value="120"),
                    SelectOption(label="3 hours", value="180"),
                    SelectOption(label="4 hours", value="240")
                ]

            user_id = interaction.user.id

            # 1. Check user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed first.", 
                    ephemeral=True
                )
                return

            # Create the Select menus
            timezone_select = Select(
                custom_id="timezone",
                placeholder="Choose your timezone",
                options=timezone_options,
                row=0
            )
            
            start_time_select = Select(
                custom_id="start_time",
                placeholder="Choose start time",
                options=start_time_options,
                row=1
            )
            
            end_time_select = Select(
                custom_id="end_time",
                placeholder="Choose end time",
                options=end_time_options,
                row=2
            )
            
            interval_select = Select(
                custom_id="interval",
                placeholder="Choose check-in interval",
                options=interval_options,
                row=3
            )

            user_choices = {}
            
            async def select_callback(interaction: discord.Interaction):
                select_id = interaction.data["custom_id"]
                selected_value = interaction.data["values"][0]
                user_choices[select_id] = selected_value
                
                # Check if all selections have been made
                if len(user_choices) == 4:  # All selections made
                    try:
                        # Convert time strings to time objects
                        start_time = datetime.strptime(user_choices["start_time"], "%H:%M").time()
                        end_time = datetime.strptime(user_choices["end_time"], "%H:%M").time()
                        
                        # Create or update DeathMarchSettings
                        settings = DeathMarchSettings(
                            timezone=user_choices["timezone"],
                            start_time=start_time,
                            end_time=end_time,
                            check_interval=int(user_choices["interval"])
                        )

                        # Calculate costs
                        checks_per_day, daily_cost = self._calculate_death_march_costs(settings)
                        
                        # Store settings
                        self.user_deathmarch_settings[interaction.user.id] = settings
                        
                        settings_msg = (
                            f"Settings saved:\n"
                            f"• Timezone: {settings.timezone}\n"
                            f"• Focus window: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}\n"
                            f"• Check-in interval: {settings.check_interval} minutes\n\n"
                            f"📊 Cost Analysis:\n"
                            f"• Check-ins per day: {checks_per_day}\n"
                            f"• Daily cost: {daily_cost} PFT\n"
                            f"• Weekly cost: {daily_cost * 7} PFT\n"
                            f"• Monthly cost: {daily_cost * 30} PFT\n\n"
                            "Use /pf_death_march_start to begin your death march."
                        )
                        
                        ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
                        await interaction.response.send_message(
                            settings_msg,
                            ephemeral=ephemeral_setting
                        )
                    except Exception as e:
                        await interaction.response.send_message(
                            f"An error occurred: {str(e)}",
                            ephemeral=True
                        )
                else:
                    await interaction.response.defer()

            # Attach callbacks
            timezone_select.callback = select_callback
            start_time_select.callback = select_callback
            end_time_select.callback = select_callback
            interval_select.callback = select_callback

            # Create view and add all selects
            view = discord.ui.View()
            view.add_item(timezone_select)
            view.add_item(start_time_select)
            view.add_item(end_time_select)
            view.add_item(interval_select)

            # Send the message with all dropdowns
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.send_message(
                "Please set your preferences:",
                view=view,
                ephemeral=ephemeral_setting
            )

        @self.tree.command(name="pf_death_march_start", description="Kick off a death march.")
        @app_commands.describe(days="Number of days to continue the death march")
        async def pf_death_march_start(interaction: discord.Interaction, days: int):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # 1. Check user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed first.", 
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            user_address = user_wallet.classic_address

            # 2. Check initiation
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=user_address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first ( /pf_initiate ).", 
                    ephemeral=ephemeral_setting
                )
                return

            # 3. Check user has configured death march settings
            if user_id not in self.user_deathmarch_settings:
                await interaction.followup.send(
                    "You must set your death march configuration using /pf_configure_deathmarch first.", 
                    ephemeral=ephemeral_setting
                )
                return
            
            # 4. Check if user is already in a death march
            if user_id in self.user_deathmarch_settings and self.user_deathmarch_settings[user_id].session_end is not None:
                await interaction.followup.send(
                    "You are already in an active death march. Use /pf_death_march_end to end it first.", 
                    ephemeral=ephemeral_setting
                )
                return

            # Calculate cost based on check-in frequency
            settings = self.user_deathmarch_settings[user_id]
            checks_per_day, cost = self._calculate_death_march_costs(settings, days)

            # 5. Check user PFT balance
            try:
                user_pft_balance = self.generic_pft_utilities.get_pft_balance(user_address)
            except:
                await interaction.followup.send("Error fetching your PFT balance. Try again later.", ephemeral=True)
                return
            
            if user_pft_balance < cost:
                await interaction.followup.send(
                    f"You need {cost} PFT but only have {user_pft_balance} PFT.\n"
                    f"This cost is based on {checks_per_day} check-ins per day for {days} days.\n"
                    "Please acquire more PFT first.", 
                    ephemeral=ephemeral_setting
                )
                return

            # 6. Process payment
            memo_data = f"DEATH_MARCH Payment: {days} days, {checks_per_day} checks/day"
            
            try:
                response = self.generic_pft_utilities.send_memo(
                    wallet_seed_or_wallet=user_wallet,
                    destination=self.node_config.remembrancer_address,  # Or wherever you want the PFT to go
                    memo=memo_data,
                    username=interaction.user.name,
                    chunk=False,
                    compress=False,
                    encrypt=False,
                    pft_amount=Decimal(cost)
                )
                # Verify
                if not self.generic_pft_utilities.verify_transaction_response(response):
                    await interaction.followup.send("Transaction for Death March payment failed.", ephemeral=ephemeral_setting)
                    return
            except Exception as e:
                await interaction.followup.send(f"An error occurred: {str(e)}", ephemeral=True)
                return
            
            # 7. Update death march settings
            session_start = datetime.now(timezone.utc)
            session_end = session_start + timedelta(days=days)
            
            settings.channel_id = interaction.channel_id
            settings.session_start = session_start
            settings.session_end = session_end
            settings.last_checkin = None

            # Create a new task for this user's death march
            task = self.loop.create_task(
                self.death_march_checker_for_user(user_id),
                name=f"death_march_{user_id}"
            )
            self.death_march_tasks[user_id] = task

            await interaction.followup.send(
                f"Death March started for {days} day(s).\n"
                f"• Cost: {cost} PFT ({checks_per_day} check-ins per day)\n"
                f"• Check-in window: {settings.start_time.strftime('%H:%M')} - {settings.end_time.strftime('%H:%M')} "
                f"({settings.timezone})\n"
                f"• Check-in interval: Every {settings.check_interval} minutes\n"
                f"• Session ends: {session_end} UTC\n\n"
                "Use /pf_death_march_end to stop it sooner (no refunds).",
                ephemeral=ephemeral_setting
            )

        @self.tree.command(name="pf_death_march_end", description="End your Death March session early (no refunds).")
        async def pf_death_march_end(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)

            # Check if user has settings and an active session
            if (user_id in self.user_deathmarch_settings and 
                    self.user_deathmarch_settings[user_id].session_end is not None):

                # Cancel the death march task
                if user_id in self.death_march_tasks:
                    self.death_march_tasks[user_id].cancel()
                    del self.death_march_tasks[user_id]

                settings = self.user_deathmarch_settings[user_id]
                # Clear session data but keep configuration
                settings.session_start = None
                settings.session_end = None
                settings.channel_id = None
                settings.last_checkin = None
                
                await interaction.response.send_message(
                    "Your Death March session has ended. Configuration saved for future use.",
                    ephemeral=ephemeral_setting
                )
            else:
                await interaction.response.send_message(
                    "You do not currently have a Death March session active.",
                    ephemeral=ephemeral_setting
                )

        # Inside MyClient.setup_hook or a similar initialization section in your MyClient class
        @self.tree.command(name="odv_context_doc", description="Start an ODV context document improvement session")
        async def odv_context_doc(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before starting an ODV context document improvement session.", 
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:
                # Initialize the ODVContextDocImprover
                doc_improver = ODVContextDocImprover(
                    account_address=wallet.classic_address,
                    openrouter=self.openrouter,
                    user_context_parser=self.user_task_parser,
                    pft_utils=self.generic_pft_utilities
                )

                # Ensure you have a dictionary to store doc improvers per user
                if not hasattr(self, 'doc_improvers'):
                    self.doc_improvers = {}

                self.doc_improvers[user_id] = doc_improver
                logger.debug(f"MyClient.odv_context_doc: Initialized ODV context document improver for {interaction.user.name}")

                # Potentially long operation: getting the initial suggestion
                logger.debug(f"MyClient.odv_context_doc: Getting initial response for {interaction.user.name}")
                initial_response = await doc_improver.get_response_async("Please provide your first improvement suggestion.")
                logger.debug(f"MyClient.odv_context_doc: Sending initial response for {interaction.user.name}")

                # Use the helper function to send the possibly long response
                await self.send_long_interaction_response(
                    interaction, 
                    f"**ODV Context Document Improvement Initialized**\n\n{initial_response}", 
                    ephemeral=ephemeral_setting
                )
            except Exception as e:
                logger.error(f"Error in odv_context_doc: {str(e)}")
                await interaction.followup.send(
                    f"An error occurred while initializing the ODV context document improvement session: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(name="odv_context_doc_reply", description="Continue the ODV context document improvement session")
        @app_commands.describe(message="Your next input to ODV")
        async def odv_context_doc_reply(interaction: discord.Interaction, message: str):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)

            # Check if we have a doc improver session in progress
            if not hasattr(self, 'doc_improvers') or user_id not in self.doc_improvers:
                await interaction.response.send_message(
                    "No active ODV context document improvement session. Start one with /odv_context_doc.", 
                    ephemeral=ephemeral_setting
                )
                return

            doc_improver: ODVContextDocImprover = self.doc_improvers[user_id]
            logger.debug(f"MyClient.odv_context_doc_reply: Continuing ODV context document improvement session for {interaction.user.name}")
            await interaction.response.defer(ephemeral=ephemeral_setting)

            try:
                logger.debug(f"MyClient.odv_context_doc_reply: Getting response for {interaction.user.name}")
                response = await doc_improver.get_response_async(message)
                logger.debug(f"MyClient.odv_context_doc_reply: Response received for {interaction.user.name}")
                await self.send_long_interaction_response(interaction, response, ephemeral=ephemeral_setting)
            except Exception as e:
                logger.error(f"Error in odv_context_doc_reply: {str(e)}")
                await interaction.followup.send(
                    f"An error occurred: {str(e)}", 
                    ephemeral=True
                )

        # In the corbanu_offering command:
        @self.tree.command(name="corbanu_offering", description="Generate a Corbanu offering")
        async def corbanu_offering(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before using this command.",
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.classic_address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return
            
            # Return the existing question if the user has one
            if user_id in self.user_questions:
                await interaction.followup.send(
                    f"Corbanu Offering:\n\n{self.user_questions[user_id]}",
                    ephemeral=ephemeral_setting
                )
                return            

            try:
                # Initialize the CorbanuChatBot instance
                logger.debug(f"MyClient.corbanu_offering: {interaction.user.name} has requested a Corbanu offering. Initializing CorbanuChatBot instance.")
                corbanu = CorbanuChatBot(
                    account_address=wallet.classic_address,
                    openrouter=self.openrouter,
                    user_context_parser=self.user_task_parser,
                    pft_utils=self.generic_pft_utilities
                )

                # Generate a question as the Corbanu offering 
                question = await corbanu.generate_question()
                logger.debug(f"MyClient.corbanu_offering: Question generated for {interaction.user.name}: {question}")

                # Store the question so we can use it in /corbanu_reply
                self.user_questions[user_id] = question

                await interaction.followup.send(
                    f"Corbanu Offering:\n\n{question}",
                    ephemeral=ephemeral_setting
                )

            except Exception as e:
                logger.error(f"Error in corbanu_offering: {str(e)}")
                logger.error(traceback.format_exc())
                await interaction.followup.send(
                    f"An error occurred while generating Corbanu offering: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(name="corbanu_reply", description="Reply to the last Corbanu offering")
        @app_commands.describe(answer="Your answer to the last Corbanu question")
        async def corbanu_reply(interaction: discord.Interaction, answer: str):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before using this command.",
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.classic_address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.",
                    ephemeral=ephemeral_setting
                )
                return

            if user_id not in self.user_questions:
                await interaction.followup.send(
                    "No Corbanu question found. Please use /corbanu_offering first.",
                    ephemeral=ephemeral_setting
                )
                return

            try:
                question = self.user_questions[user_id]
                logger.debug(f"MyClient.corbanu_reply: Received user answer for {interaction.user.name}.\nQuestion:\n{question}\nAnswer: \n{answer}")
                
                corbanu = CorbanuChatBot(
                    account_address=wallet.classic_address,
                    openrouter=self.openrouter,
                    user_context_parser=self.user_task_parser,
                    pft_utils=self.generic_pft_utilities
                )

                scoring = await corbanu.generate_user_question_scoring_output(
                    original_question=question,
                    user_answer=answer,
                    account_address=wallet.classic_address
                )

                reward_value = scoring.get('reward_value', 0)
                reward_description = scoring.get('reward_description', 'No description')

                full_message = (f"CORBANU_OFFERING\n"
                                f"Q: {question}\n\n"
                                f"A: {answer}\n\n"
                                f"Reward: {reward_value} PFT\n"
                                f"{reward_description}")

                # Send a short summary to the user in Discord
                summary_chunks = self.chunk_message(f"**Corbanu Summary**\n{full_message}")
                for chunk in summary_chunks:
                    await interaction.followup.send(chunk, ephemeral=ephemeral_setting)

                # Now the user will send the Q&A to the remembrancer
                # Similar to pf_log, we need to ensure a handshake if encrypt=True
                encrypt = True  # We assume we always encrypt the Q&A to the remembrancer.
                user_name = interaction.user.name
                message_obj = await interaction.followup.send(
                    "Preparing to send Q&A to the remembrancer...",
                    ephemeral=ephemeral_setting,
                    wait=True
                )

                handshake_success, user_key, counterparty_key, message_obj = await self._ensure_handshake(
                    interaction=interaction,
                    seed=seed,
                    counterparty=self.remembrancer,
                    username=user_name,
                    command_name="corbanu_reply",
                    message_obj=message_obj
                )
                if not handshake_success:
                    logger.error(f"MyClient.corbanu_reply: Handshake failed for {interaction.user.name}.")
                    await message_obj.edit(content="Handshake failed. Aborting operation.")
                    return
                
                await message_obj.edit(content="Handshake verified. Proceeding to send memo...")

                # Send Q&A from user wallet to remembrancer
                response = self.generic_pft_utilities.send_memo(
                    wallet_seed_or_wallet=wallet,
                    username="Corbanu",  # This is memo_format
                    destination=self.remembrancer,
                    memo=full_message,
                    chunk=True,
                    compress=True,
                    encrypt=encrypt,
                    pft_amount=Decimal(0)  # No PFT here, just sending the message
                )

                # Verify that the large message was successfully sent
                if not self.generic_pft_utilities.verify_transaction_response(response):
                    await message_obj.edit(content="Failed to send Q&A message to remembrancer.")
                    return

                last_response = response[-1] if isinstance(response, list) else response
                transaction_info = self.generic_pft_utilities.extract_transaction_info_from_response_object(
                    response=last_response
                )
                clean_string = transaction_info['clean_string']

                await message_obj.edit(
                    content=f"Q&A message sent to remembrancer successfully. Last chunk details:\n{clean_string}"
                )

                # Now send the reward from the node to the user
                foundation_seed = self.generic_pft_utilities.credential_manager.get_credential(
                    f"{self.node_config.node_name}__v1xrpsecret"
                )
                foundation_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(foundation_seed)

                short_reward_message = "Corbanu Reward"

                # Check daily reward limit
                remaining_daily_limit = corbanu.check_daily_reward_limit(account_address=wallet.classic_address)
                reward_value = min(reward_value, remaining_daily_limit)

                # Check per-offering reward limit
                reward_value = min(reward_value, corbanu.MAX_PER_OFFERING_REWARD_VALUE)

                logger.debug(f"MyClient.corbanu_reply: Sending reward of {reward_value} PFT to {wallet.classic_address}")
                
                reward_tx = self.generic_pft_utilities.send_memo(
                    wallet_seed_or_wallet=foundation_wallet,
                    destination=wallet.classic_address,
                    memo=short_reward_message,
                    username="Corbanu",
                    chunk=False,
                    compress=False,
                    encrypt=False,
                    pft_amount=Decimal(reward_value)
                )

                if not self.generic_pft_utilities.verify_transaction_response(reward_tx):
                    await interaction.followup.send("Failed to send reward transaction.", ephemeral=True)
                    return

                # Confirm reward sent
                reward_info = self.generic_pft_utilities.extract_transaction_info_from_response_object(
                    reward_tx
                )
                reward_clean_string = reward_info['clean_string']

                reward_chunks = self.chunk_message(f"Reward transaction sent successfully:\n{reward_clean_string}")
                for chunk in reward_chunks:
                    await interaction.followup.send(chunk, ephemeral=ephemeral_setting)

                # Clear stored question
                del self.user_questions[user_id]

            except Exception as e:
                logger.error(f"Error in corbanu_reply: {str(e)}")
                logger.error(traceback.format_exc())
                error_chunks = self.chunk_message(f"An error occurred while processing your reply: {str(e)}")
                for chunk in error_chunks:
                    await interaction.followup.send(chunk, ephemeral=True)

        @self.tree.command(name="corbanu_request", description="Send a request to Corbanu and get a response from Angron or Fulgrim")
        @app_commands.describe(message="Your message to Corbanu")
        async def corbanu_request(interaction: discord.Interaction, message: str):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before using this command.",
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.classic_address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:
                # Add the user message to their conversation history
                if user_id not in self.conversations:
                    self.conversations[user_id] = []

                self.conversations[user_id].append({"role": "user", "content": message})

                # Create CorbanuChatBot instance
                corbanu = CorbanuChatBot(
                    account_address=wallet.classic_address,
                    openrouter=self.openrouter,
                    user_context_parser=self.user_task_parser,
                    pft_utils=self.generic_pft_utilities
                )

                # Get Corbanu's response asynchronously
                response = await corbanu.get_response_async(message)

                # Chunk the response if needed and send to user
                response_chunks = self.chunk_message(response)
                for chunk in response_chunks:
                    await interaction.followup.send(chunk, ephemeral=ephemeral_setting)

                # Append Corbanu's response to conversation history
                self.conversations[user_id].append({"role": "assistant", "content": response})

                # Combine USER MESSAGE + CORBANU RESPONSE
                combined_message = f"USER MESSAGE:\n{message}\n\nCORBANU RESPONSE:\n{response}"

                # Summarize the combined message before sending to remembrancer
                summarized_message = await corbanu.summarize_text(combined_message, max_length=900)

                encrypt = True
                user_name = interaction.user.name

                # Notify user we're sending to remembrancer
                message_obj = await interaction.followup.send(
                    "Sending the Q&A record (summarized) to the remembrancer...",
                    ephemeral=ephemeral_setting,
                    wait=True
                )

                # Ensure handshake
                if encrypt:
                    handshake_success, user_key, counterparty_key, message_obj = await self._ensure_handshake(
                        interaction=interaction,
                        seed=seed,
                        counterparty=self.remembrancer,
                        username=user_name,
                        command_name="corbanu_request",
                        message_obj=message_obj
                    )
                    if not handshake_success:
                        return
                    
                    await message_obj.edit(content="Handshake verified. Proceeding to send memo...")

                # Send summarized message from user's wallet to remembrancer
                send_response = self.generic_pft_utilities.send_memo(
                    wallet_seed_or_wallet=wallet,
                    username=user_name,
                    destination=self.remembrancer,
                    memo=summarized_message,
                    chunk=True,
                    compress=True,
                    encrypt=encrypt,
                    pft_amount=Decimal(0)
                )

                # Verify transaction
                if not self.generic_pft_utilities.verify_transaction_response(send_response):
                    await message_obj.edit(content="Failed to send the summarized Q&A record to remembrancer.")
                    return

                last_response = send_response[-1] if isinstance(send_response, list) else send_response
                transaction_info = self.generic_pft_utilities.extract_transaction_info_from_response_object(last_response)
                clean_string = transaction_info['clean_string']

                await message_obj.edit(
                    content=f"Summarized Q&A record sent to remembrancer successfully. Last chunk details:\n{clean_string}"
                )

            except Exception as e:
                logger.error(f"Error in corbanu_request: {str(e)}")
                logger.error(traceback.format_exc())
                error_chunks = self.chunk_message(f"An error occurred: {str(e)}")
                for chunk in error_chunks:
                    await interaction.followup.send(chunk, ephemeral=True)

        @self.tree.command(name="pf_outstanding", description="Show your outstanding tasks and verification tasks")
        async def pf_outstanding(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed before viewing outstanding tasks.", 
                    ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]
            logger.debug(f"MyClient.setup_hook.pf_outstanding: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.response.send_message("You must perform the initiation rite first. Run /pf_initiate to do so.", ephemeral=True)
                return

            # Defer the response to avoid timeout for longer operations
            await interaction.response.defer(ephemeral=ephemeral_setting)

            try:
                # Get the unformatted output message
                output_message = self.create_full_outstanding_pft_string(account_address=wallet.address)

                logger.debug(f"MyClient.pf_outstanding: Output message: {output_message}")
                
                # Format the message using the new formatting function
                formatted_chunks = self.format_tasks_for_discord(output_message)
                
                # Send the first chunk
                await interaction.followup.send(formatted_chunks[0], ephemeral=ephemeral_setting)

                # Send the rest of the chunks
                for chunk in formatted_chunks[1:]:
                    await interaction.followup.send(chunk, ephemeral=ephemeral_setting)

            except Exception as e:
                logger.error(f"MyClient.pf_outstanding: Error fetching outstanding tasks: {str(e)}")
                logger.error(traceback.format_exc())
                await interaction.followup.send(
                    f"An error occurred while fetching your outstanding tasks: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(name="pf_request_task", description="Request a Post Fiat task")
        async def pf_task_slash(interaction: discord.Interaction, task_request: str):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)
            
            # Check if the user has stored a seed
            if user_id not in self.user_seeds:
                await interaction.followup.send("You must store a seed using /pf_store_seed before generating a task.", ephemeral=ephemeral_setting)
                return

            # Get the user's seed and other necessary information
            seed = self.user_seeds[user_id]
            user_name = interaction.user.name
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            
            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send("You must perform the initiation rite first. Run /pf_initiate to do so.", ephemeral=ephemeral_setting)
                return
            
            try:
                # Send the Post Fiat request
                response = post_fiat_task_generation_system.discord__send_postfiat_request(
                    user_request=task_request,
                    user_name=user_name,
                    user_seed=seed  # TODO: change to wallet
                )
                
                # Extract transaction information
                transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object(response=response)
                clean_string = transaction_info['clean_string']
                
                # Send the response
                await interaction.followup.send(f"Task Requested with Details: {clean_string}", ephemeral=ephemeral_setting)
            except Exception as e:
                await interaction.followup.send(f"An error occurred while processing your request: {str(e)}", ephemeral=True)
            
        @self.tree.command(name="pf_accept", description="Accept tasks")
        async def pf_accept_menu(interaction: discord.Interaction):
            # Fetch the user's seed
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            if user_id not in self.user_seeds:
                await interaction.followup.send("You must store a seed using /store_seed before accepting tasks.", ephemeral=ephemeral_setting)
                return

            seed = self.user_seeds[user_id]

            logger.debug(f"MyClient.setup_hook.pf_accept_menu: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send("You must perform the initiation rite first. Run /pf_initiate to do so.", ephemeral=ephemeral_setting)
                return

            # Fetch proposal acceptance pairs
            memo_history = self.generic_pft_utilities.get_account_memo_history(account_address=wallet.address).copy()

            # Get pending proposals
            pending_tasks = post_fiat_task_generation_system.get_pending_proposals(account=memo_history)

            # Return if proposal acceptance pairs are empty
            if pending_tasks.empty:
                await interaction.followup.send("You have no tasks to accept.", ephemeral=ephemeral_setting)
                return
            
            self.pending_tasks_cache[user_id] = pending_tasks

            # Create dropdown options based on the non-accepted tasks
            options = [
                SelectOption(
                    label=task_id, 
                    description=str(pending_tasks.loc[task_id, 'proposal'])[:100],  # get just the proposal text
                    value=task_id
                )
                for task_id in pending_tasks.index
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to accept", options=options)

            # Create the Select menu with its callback
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = str(self.pending_tasks_cache[user_id].loc[selected_task_id, 'proposal'])
                self.pending_tasks_cache.pop(user_id, None)

                await interaction.response.send_modal(
                    AcceptanceModal(
                        task_id=selected_task_id,
                        task_text=task_text,
                        seed=seed,
                        user_name=interaction.user.name,
                        post_fiat_task_generation_system=post_fiat_task_generation_system,
                        ephemeral_setting=ephemeral_setting
                    )
                )

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.followup.send("Please choose a task to accept:", view=view, ephemeral=ephemeral_setting)
        
        @self.tree.command(name="pf_refuse", description="Refuse tasks")
        async def pf_refuse_menu(interaction: discord.Interaction):
            # Fetch the user's seed
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            if user_id not in self.user_seeds:
                await interaction.followup.send("You must store a seed using /store_seed before refusing tasks.", ephemeral=ephemeral_setting)
                return

            seed = self.user_seeds[user_id]

            logger.debug(f"MyClient.setup_hook.pf_refuse_menu: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send("You must perform the initiation rite first. Run /pf_initiate to do so.", ephemeral=ephemeral_setting)
                return
            
            # Fetch account history
            memo_history = generic_pft_utilities.get_account_memo_history(account_address=wallet.address).copy()

            # Get refuseable proposals
            refuseable_tasks = post_fiat_task_generation_system.get_refuseable_proposals(account=memo_history)

            # Return if proposal refusal pairs are empty
            if refuseable_tasks.empty:
                await interaction.followup.send("You have no tasks to refuse.", ephemeral=ephemeral_setting)
                return

            self.refuseable_tasks_cache[user_id] = refuseable_tasks

            # Create dropdown options based on the non-accepted tasks
            options = [
                SelectOption(
                    label=task_id, 
                    description=str(refuseable_tasks.loc[task_id, 'proposal'])[:100], 
                    value=task_id
                )
                for task_id in refuseable_tasks.index
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to refuse", options=options)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = str(self.refuseable_tasks_cache[user_id].loc[selected_task_id, 'proposal'])
                self.refuseable_tasks_cache.pop(user_id, None)
    
                await interaction.response.send_modal(
                    RefusalModal(
                        task_id=selected_task_id,
                        task_text=task_text,
                        seed=seed,
                        user_name=interaction.user.name,
                        post_fiat_task_generation_system=post_fiat_task_generation_system,
                        ephemeral_setting=ephemeral_setting
                    )
                )

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.followup.send("Please choose a task to refuse:", view=view, ephemeral=ephemeral_setting)

        @self.tree.command(name="pf_initial_verification", description="Submit a task for verification")
        async def pf_submit_for_verification(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /store_seed before submitting a task for verification.", 
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]

            logger.debug(f"MyClient.setup_hook.pf_initial_verification: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            # Fetch account history
            memo_history = generic_pft_utilities.get_account_memo_history(wallet.address).copy()

            # Fetch accepted tasks
            accepted_tasks = post_fiat_task_generation_system.get_accepted_proposals(account=memo_history)

            # Return if no accepted tasks
            if accepted_tasks.empty:
                await interaction.followup.send("You have no tasks to submit for verification.", ephemeral=ephemeral_setting)
                return

            self.accepted_tasks_cache[user_id] = accepted_tasks

            # Create dropdown options based on the accepted tasks
            options = [
                SelectOption(
                    label=task_id, 
                    description=str(accepted_tasks.loc[task_id, 'proposal'])[:100], 
                    value=task_id
                )
                for task_id in accepted_tasks.index
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to submit for verification", options=options)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = str(self.accepted_tasks_cache[user_id].loc[selected_task_id, 'proposal'])
                self.accepted_tasks_cache.pop(user_id, None)

                await interaction.response.send_modal(
                    CompletionModal(
                        task_id=selected_task_id,
                        task_text=task_text,
                        seed=seed,
                        user_name=interaction.user.name,
                        post_fiat_task_generation_system=post_fiat_task_generation_system,
                        ephemeral_setting=ephemeral_setting
                    )
                )

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)
            await interaction.followup.send("Please choose a task to submit for verification:", view=view, ephemeral=ephemeral_setting)

        @self.tree.command(name="pf_final_verification", description="Submit final verification for a task")
        async def pf_final_verification(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /store_seed before submitting final verification.", 
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            logger.debug(f"MyClient.setup_hook.pf_final_verification: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            # Fetch account history
            memo_history = generic_pft_utilities.get_account_memo_history(wallet.address).copy()

            # Fetch verification tasks
            verification_tasks = post_fiat_task_generation_system.get_verification_proposals(account=memo_history)
            
            # If there are no tasks in the verification queue, notify the user
            if verification_tasks.empty:
                await interaction.followup.send("You have no tasks pending final verification.", ephemeral=ephemeral_setting)
                return

            self.verification_tasks_cache[user_id] = verification_tasks

            # Create dropdown options based on the tasks in the verification queue
            options = [
                SelectOption(
                    label=task_id, 
                    description=str(verification_tasks.loc[task_id, 'verification'])[:100], 
                    value=task_id
                )
                for task_id in verification_tasks.index
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to submit for final verification", options=options)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = str(self.verification_tasks_cache[user_id].loc[selected_task_id, 'verification'])
                self.verification_tasks_cache.pop(user_id, None)

                # Open the modal to get the verification justification with the task text pre-populated
                await interaction.response.send_modal(
                    VerificationModal(
                        task_id=selected_task_id,
                        task_text=task_text,
                        seed=seed,
                        user_name=interaction.user.name,
                        post_fiat_task_generation_system=post_fiat_task_generation_system,
                        ephemeral_setting=ephemeral_setting
                    )
                )

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.followup.send("Please choose a task for final verification:", view=view, ephemeral=ephemeral_setting)

        @self.tree.command(name="pf_rewards", description="Show your recent Post Fiat rewards")
        async def pf_rewards(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before viewing rewards.",
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            logger.debug(f"MyClient.setup_hook.pf_rewards: Spawning wallet to fetch rewards for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:
                memo_history = generic_pft_utilities.get_account_memo_history(wallet.address).copy().sort_values('datetime')

                # Return immediately if memo history is empty
                if memo_history.empty:
                    await interaction.followup.send("You have no rewards to show.", ephemeral=ephemeral_setting)
                    return

                reward_summary_map = self.get_reward_data(all_account_info=memo_history)
                recent_rewards = self.format_reward_summary(reward_summary_map['reward_summaries'].tail(10))

                await self.send_long_interaction_response(
                    interaction=interaction,
                    content=recent_rewards,
                    ephemeral=ephemeral_setting
                )

            except Exception as e:
                await interaction.followup.send(f"An error occurred while fetching your rewards: {str(e)}", ephemeral=True)

        @self.tree.command(name="pf_log", description="Send a long message to the remembrancer wallet")
        async def pf_remembrancer(interaction: discord.Interaction, message: str, encrypt: bool = False):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before using this command.",
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            user_name = interaction.user.name
            logger.debug(f"MyClient.pf_remembrancer: Spawning wallet to send message to remembrancer for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:

                message_obj = await interaction.followup.send(
                    "Sending message to remembrancer...",
                    ephemeral=ephemeral_setting,
                    wait=True  # returns message object
                )

                if encrypt:
                    handshake_success, user_key, counterparty_key, message_obj = await self._ensure_handshake(
                        interaction=interaction,
                        seed=seed,
                        counterparty=self.remembrancer,
                        username=user_name,
                        command_name="pf_remembrancer",
                        message_obj=message_obj
                    )
                    if not handshake_success:
                        return
                    
                    await message_obj.edit(content=f"Handshake verified. Proceeding to send message\n{message}...")

                response = generic_pft_utilities.send_memo(
                    wallet_seed_or_wallet=wallet,
                    username=user_name,
                    destination=self.remembrancer,
                    memo=message,
                    chunk=True,
                    compress=True,
                    encrypt=encrypt
                )
                response = response[-1] if isinstance(response, list) else response

                transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object(
                    response=response
                )
                clean_string = transaction_info['clean_string']

                mode = "Encrypted message" if encrypt else "Message"
                await message_obj.edit(
                    content=f"Post Fiat Log: {message}\n{mode} sent to remembrancer successfully. Last chunk details:\n{clean_string}"
                )

            except Exception as e:
                await interaction.followup.send(f"An error occurred while sending the message: {str(e)}", ephemeral=ephemeral_setting)
        
        @self.tree.command(name="pf_chart", description="Generate a chart of your PFT rewards and metrics")
        async def pf_chart(interaction: discord.Interaction):
            user_id = interaction.user.id
            ephemeral_setting = self.is_special_user_non_ephemeral(interaction)
            await interaction.response.defer(ephemeral=ephemeral_setting)

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.followup.send(
                    "You must store a seed using /pf_store_seed before generating a chart.", 
                    ephemeral=ephemeral_setting
                )
                return

            seed = self.user_seeds[user_id]
            logger.debug(f"MyClient.setup_hook.pf_chart: Spawning wallet to generate chart for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)

            # Check initiation status
            initiation_check_success = await self._check_initiation_rite(
                interaction=interaction,
                wallet_address=wallet.address
            )
            if not initiation_check_success:
                await interaction.followup.send(
                    "You must perform the initiation rite first. Run /pf_initiate to do so.", 
                    ephemeral=ephemeral_setting
                )
                return

            try:
                # Call the charting function
                post_fiat_task_generation_system.output_pft_KPI_graph_for_address(user_wallet=wallet.address)
                
                # Create the file object from the saved image
                chart_file = discord.File(f'pft_rewards__{wallet.address}.png', filename='pft_chart.png')
                
                # Create an embed for better formatting
                embed = discord.Embed(
                    title="PFT Rewards Analysis",
                    color=discord.Color.blue()
                )
                
                # Add the chart image to the embed
                embed.set_image(url="attachment://pft_chart.png")
                
                # Send the embed with the chart
                await interaction.followup.send(
                    file=chart_file,
                    embed=embed,
                    ephemeral=ephemeral_setting
                )
                
                # Clean up the file after sending
                import os
                os.remove(f'pft_rewards__{wallet.address}.png')

            except Exception as e:
                await interaction.followup.send(
                    f"An error occurred while generating your PFT chart: {str(e)}", 
                    ephemeral=True
                )

        @self.tree.command(
            name="pf_leaderboard", 
            description="Display the Post Fiat Foundation Node Leaderboard"
        )
        async def pf_leaderboard(interaction: discord.Interaction):
            # Check if the user has permission (matches the specific ID)
            if interaction.user.id not in constants.DISCORD_SUPER_USER_IDS:
                await interaction.response.send_message(
                    "You don't have permission to use this command.", 
                    ephemeral=True
                )
                return
                
            # Proceed with the command for authorized user
            await interaction.response.defer(ephemeral=False)
            
            try:
                # Generate and format the leaderboard
                leaderboard_df = generic_pft_utilities.output_postfiat_foundation_node_leaderboard_df()
                generic_pft_utilities.format_and_write_leaderboard()
                
                embed = discord.Embed(
                    title="Post Fiat Foundation Node Leaderboard 🏆",
                    description=f"Current Post Fiat Leaderboard",
                    color=0x00ff00
                )
                
                file = discord.File("test_leaderboard.png", filename="leaderboard.png")
                embed.set_image(url="attachment://leaderboard.png")
                
                await interaction.followup.send(
                    embed=embed, 
                    file=file
                )
                
                # Clean up
                import os
                os.remove("test_leaderboard.png")
                
            except Exception as e:
                await interaction.followup.send(
                    f"An error occurred while generating the leaderboard: {str(e)}"
                )

        # Sync the commands to the guild
        # self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.debug(f"MyClient.setup_hook: Slash commands synced to guild ID: {guild_id}")

    async def _check_initiation_rite(
        self,
        interaction: discord.Interaction,
        wallet_address: str,
        require_initiation: bool = True
    ) -> bool:
        """
        Check if a wallet has completed the initiation rite.
        
        Args:
            wallet_address: The wallet address to check
            interaction: Discord interaction object for sending responses
            require_initiation: If True, requires initiation to proceed. If False, blocks re-initiation.
        
        Returns:
            bool: True if check passes (can proceed), False if should block
        """
        memo_history = generic_pft_utilities.get_account_memo_history(
            account_address=wallet_address, 
            pft_only=False
        )
        existing_initiations = memo_history[
            (memo_history['memo_type'] == 'INITIATION_RITE') & 
            (memo_history['transaction_result'] == 'tesSUCCESS')
        ]

        has_initiated = not existing_initiations.empty

        if require_initiation:
            # Block if not initiated
            if not has_initiated: 
                return False
        else:
            # Block re-initiation unless on testnet with ENABLE_REINITIATIONS
            if has_initiated and not (config.RuntimeConfig.USE_TESTNET and config.RuntimeConfig.ENABLE_REINITIATIONS):
                logger.debug(f"MyClient._check_initiation_status: Blocking re-initiation for {interaction.user.name} ({wallet_address})")
                return False
        return True

    async def _ensure_handshake(
        self,
        interaction: discord.Interaction,
        seed: str,
        counterparty: str,
        username: str,
        command_name: str,
        message_obj: Optional[discord.Message] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Ensures handshake protocol is established between wallet and counterparty

        Args: 
            interaction: Discord interaction object
            wallet_address: Wallet address of the user
            counterparty: Counterparty address
            seed: Seed of the user
            username: Username of the user
            command_name: Name of the command that requires the handshake protocol
            message_obj: Message object to edit (optional)

        Returns:
            tuple[bool, str, str, discord.Message]: (success, user_key, counterparty_key, message_obj)
        """
        try:
            # Send message if we don't have a message object
            if not message_obj:
                message_obj = await interaction.followup.send(
                    "Checking encryption handshake status...",
                    ephemeral=True,
                    wait=True  # returns message object
                )

            # Check handshake status
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            user_key, counterparty_key = generic_pft_utilities.get_handshake_for_address(
                channel_address=wallet.classic_address,
                channel_counterparty=counterparty
            )

            if not user_key:
                # Send handshake if we haven't yet
                logger.debug(f"MyClient.{command_name}: Initiating handshake for {username} with {counterparty}")
                generic_pft_utilities.send_handshake(
                    wallet_seed=seed,
                    destination=counterparty,
                    username=username
                )
                await message_obj.edit(content="Encryption handshake initiated. Waiting for onchain confirmation...")

                # Verify handshake completion and response from counterparty (node or remembrancer)
                for attempt in range(constants.NODE_HANDSHAKE_RESPONSE_USER_VERIFICATION_ATTEMPTS):
                    logger.debug(f"MyClient.{command_name}: Checking handshake status for {username} with {counterparty} (attempt {attempt+1})")

                    user_key, counterparty_key = generic_pft_utilities.get_handshake_for_address(
                        channel_address=wallet.classic_address,
                        channel_counterparty=counterparty
                    )

                    if counterparty_key:
                        logger.debug(f"MyClient.{command_name}: Handshake confirmed for {username} with {counterparty}")
                        break

                    if user_key:
                        await message_obj.edit(content="Handshake sent. Waiting for node to process...")

                    await asyncio.sleep(constants.NODE_HANDSHAKE_RESPONSE_USER_VERIFICATION_INTERVAL)

            if not user_key:
                await message_obj.edit(content="Encryption handshake failed to send. Please reach out to support.")
                return False, None, None, message_obj
            
            if not counterparty_key:
                await message_obj.edit(content="Encryption handshake sent but not yet processed. Please wait and try again later.")
                return False, user_key, None, message_obj
            
            await message_obj.edit(content="Handshake verified. Proceeding with operation...")
            return True, user_key, counterparty_key, message_obj

        except Exception as e:
            logger.error(f"MyClient.{command_name}: An error occurred while ensuring handshake: {str(e)}")
            await message_obj.edit(content=f"An error occurred during handshake setup: {str(e)}")
            return False, None, None, message_obj

    async def on_ready(self):
        logger.debug(f'MyClient.on_ready: Logged in as {self.user} (ID: {self.user.id})')
        logger.debug('MyClient.on_ready: ------------------------------')
        logger.debug('MyClient.on_ready: Connected to the following guilds:')
        for guild in self.guilds:
            logger.debug(f'- {guild.name} (ID: {guild.id})')

        # # Optionally, re-sync slash commands in all guilds
        # await self.tree.sync()
        # logger.debug('MyClient.on_ready: Slash commands synced across all guilds.')

    async def _split_message_into_chunks(self, content: str, max_chunk_size: int = 1900) -> list[str]:
        """Split a message into chunks that fit within Discord's message limit.
        
        Args:
            content: The message content to split
            max_chunk_size: Maximum size for each chunk (default: 1900 to leave room for formatting)
            
        Returns:
            List of message chunks
        """
        chunks = []
        while content:
            if len(content) <= max_chunk_size:
                chunks.append(content)
                break
                
            # Find the last space within the limit to avoid splitting words
            split_index = content[:max_chunk_size].rfind(' ')
            if split_index == -1:  # No space found, force split at max length
                split_index = max_chunk_size
                
            chunks.append(content[:split_index])
            content = content[split_index:].lstrip()  # Remove leading whitespace
            
        return chunks
    
    async def _format_chunk(self, chunk: str, code_block: bool = False) -> str:
        """Format a message chunk with optional code block formatting.
        
        Args:
            chunk: The message chunk to format
            code_block: Whether to wrap the chunk in a code block
            
        Returns:
            Formatted message chunk
        """
        if code_block:
            return f"```\n{chunk}\n```"
        return chunk
    
    async def _send_long_message(
        self,
        content: str,
        *,
        channel: Optional[discord.abc.GuildChannel] = None,
        message: Optional[discord.Message] = None,
        interaction: Optional[discord.Interaction] = None,
        code_block: bool = False,
        ephemeral: bool = True,
        mention_author: bool = True,
        delete_after: Optional[int] = None
    ) -> list[discord.Message]:
        """Send a long message, splitting it into chunks if necessary.
        
        Args:
            content: The message content to send
            channel: Discord channel to send to (optional)
            message: Original message to reply to (optional) 
            interaction: Discord interaction to respond to (optional)
            code_block: Whether to wrap chunks in code blocks
            ephemeral: Whether interaction responses should be ephemeral
            mention_author: Whether to mention author in replies
            delete_after: Number of seconds after which to delete messages
            
        Returns:
            List of sent messages
        """
        sent_messages = []
        chunks = await self._split_message_into_chunks(content)
        
        for chunk in chunks:
            formatted_chunk = await self._format_chunk(chunk, code_block)
            
            try:
                if interaction:
                    # For slash commands
                    await interaction.followup.send(formatted_chunk, ephemeral=ephemeral)
                elif channel:
                    # For direct channel messages
                    sent = await channel.send(formatted_chunk)
                    sent_messages.append(sent)
                elif message:
                    # For message replies
                    sent = await message.reply(formatted_chunk, mention_author=mention_author)
                    sent_messages.append(sent)
                else:
                    raise ValueError("Must provide either channel, message, or interaction")
                    
            except discord.errors.HTTPException as e:
                logger.error(f"Error sending message chunk: {e}")
                continue
                
        if delete_after and sent_messages:
            await asyncio.sleep(delete_after)
            for sent in sent_messages:
                try:
                    await sent.delete()
                except discord.errors.NotFound:
                    pass  # Message already deleted
                    
            if message:
                try:
                    await message.delete()
                except discord.errors.NotFound:
                    pass  # Original message already deleted
                    
        return sent_messages

    async def send_long_message_to_channel(self, channel, long_message):
        return await self._send_long_message(long_message, channel=channel)

    async def send_long_interaction_response(self, interaction: discord.Interaction, content: str, ephemeral: bool = True):
        return await self._send_long_message(
            content,
            interaction=interaction,
            code_block=True,
            ephemeral=ephemeral
        )
        
    async def send_long_message(self, message, long_message):
        return await self._send_long_message(
            content=long_message,
            message=message,
            mention_author=True
        )

    async def send_long_message_then_delete(self, message, long_message, delete_after):
        return await self._send_long_message(
            long_message,
            message=message,
            delete_after=delete_after
        )

    async def send_long_escaped_message(self, message, long_message):
        return await self._send_long_message(
            long_message,
            message=message,
            code_block=True,
            mention_author=True
        )

    async def check_and_notify_new_transactions(self):
        CHANNEL_ID = self.node_config.discord_activity_channel_id
        channel = self.get_channel(CHANNEL_ID)
        
        if not channel:
            logger.error(f"MyClient.check_and_notify_new_transactions: ERROR: Channel with ID {CHANNEL_ID} not found.")
            return

        # Call the function to get new messages and update the database
        messages_to_send = post_fiat_task_generation_system.sync_and_format_new_transactions()

        # DEBUGGING
        len_messages_to_send = len(messages_to_send)
        if len_messages_to_send > 0:
            logger.debug(f"MyClient.check_and_notify_new_transactions: Sending {len_messages_to_send} messages to the Discord channel") 

        # Send each new message to the Discord channel
        for message in messages_to_send:
            await channel.send(message)

    async def transaction_checker(self):
        await self.wait_until_ready()
        while not self.is_closed():
            await self.check_and_notify_new_transactions()
            await asyncio.sleep(15)  # Check every 60 seconds

    async def death_march_checker_for_user(self, user_id: int):
        """Individual death march checker for a single user."""
        settings = self.user_deathmarch_settings[user_id]

        while not self.is_closed():
            now_utc = datetime.now(timezone.utc)

            # Check if death march has ended
            if now_utc >= settings.session_end:
                logger.debug(f"death_march_checker: user_id={user_id} - Death March ended; clearing session.")
                settings.session_start = None
                settings.session_end = None
                settings.channel_id = None
                settings.last_checkin = None
                break

            # Check local time window
            user_tz = pytz.timezone(settings.timezone)
            now_local = datetime.now(user_tz).time()

            if settings.start_time <= now_local <= settings.end_time:
                # Check if enough time has passed since last check-in
                if settings.last_checkin:
                    time_since_last = (now_utc - settings.last_checkin).total_seconds() / 60
                    if time_since_last < settings.check_interval:
                        await asyncio.sleep((settings.check_interval - time_since_last) * 60)
                        continue
                logger.debug(f"death_march_checker: user_id={user_id} is within time window and due for check-in.")
                try:
                    seed = self.user_seeds.get(user_id)
                    if not seed:
                        logger.debug(f"death_march_checker: user_id={user_id} has no stored seed; ending death march.")
                        break

                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    analyzer = ODVFocusAnalyzer(
                        account_address=user_wallet.classic_address,
                        openrouter=self.openrouter,
                        user_context_parser=self.user_task_parser,
                        pft_utils=self.generic_pft_utilities
                    )
                    focus_text = await analyzer.get_response_async("Death March Check-In")

                    channel = self.get_channel(settings.channel_id)
                    if channel:
                        # Mention the user
                        mention_string = f"<@{user_id}>"
                        await channel.send(f"{mention_string} **Death March Check-In**\n{focus_text}")
                        settings.last_checkin = now_utc
                    else:
                        logger.warning(
                            f"death_march_checker: Channel {settings.channel_id} not found for user_id={user_id}."
                        )
                        break
                except Exception as e:
                    logger.error(
                        f"death_march_checker: Error processing user_id={user_id}: {str(e)}"
                    )

            # Sleep until next interval
            await asyncio.sleep(settings.check_interval * 60)

        logger.info(f"death_march_checker: Ending death march loop for user_id={user_id}")

    async def death_march_reminder(self):
        await self.wait_until_ready()
        
        # Wait for 10 seconds after server start (for testing, change back to 30 minutes in production)
        await asyncio.sleep(30)
        
        target_user_id = 402536023483088896  # The specific user ID
        channel_id = 1229917290254827521  # The specific channel ID
        
        est_tz = pytz.timezone('US/Eastern')
        start_time = time(0, 30)  # 6:30 AM
        end_time = time(23, 59)  # 9:00 PM
        
        while not self.is_closed():
            try:
                now = datetime.now(est_tz).time()
                if start_time <= now <= end_time:
                    channel = self.get_channel(channel_id)
                    if channel:
                        if target_user_id in self.user_seeds:
                            seed = self.user_seeds[target_user_id]
                            logger.debug(f"MyClient.death_march_reminder: Spawning wallet to fetch info for {target_user_id}")
                            user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                            user_address = user_wallet.classic_address
                            tactical_string = self.post_fiat_task_generation_system.get_o1_coaching_string_for_account(user_address)
                            
                            # Send the message to the channel
                            await self.send_long_message_to_channel(channel, f"<@{target_user_id}> Death March Update:\n{tactical_string}")
                        else:
                            logger.debug(f"MyClient.death_march_reminder: No seed found for user {target_user_id}")
                    else:
                        logger.debug(f"MyClient.death_march_reminder: Channel with ID {channel_id} not found")
                else:
                    logger.debug("MyClient.death_march_reminder: Outside of allowed time range. Skipping Death March reminder.")
            except Exception as e:
                logger.error(f"MyClient.death_march_reminder: An error occurred: {str(e)}")

            # Wait for 30 minutes before the next reminder (10 seconds for testing)
            await asyncio.sleep(30*60)  # Change to 1800 (30 minutes) for production


            
    async def on_message(self, message):
        if message.author.id == self.user.id:
            return

        user_id = message.author.id
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        self.conversations[user_id].append({
            "role": "user",
            "content": message.content})

        conversation = self.conversations[user_id]
        if len(self.conversations[user_id]) > constants.MAX_HISTORY:
            del self.conversations[user_id][0]  # Remove the oldest message

        if message.content.startswith('!odv'):
            
            system_content_message = [{"role": "system", "content": odv_system_prompt}]
            ref_convo = system_content_message + conversation
            api_args = {
            "model": self.default_openai_model,
            "messages": ref_convo}
            op_df = self.openai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
            content = op_df['choices__message__content'][0]
            gpt_response = content
            self.conversations[user_id].append({
                "role": 'system',
                "content": gpt_response})
        
            await self.send_long_message(message, gpt_response)

        if message.content.startswith('!tactics'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    logger.debug(f"MyClient.tactics: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    memo_history = self.generic_pft_utilities.get_account_memo_history(user_wallet.classic_address)
                    full_user_context = self.user_task_parser.get_full_user_context_string(user_wallet.classic_address, memo_history=memo_history)
                    
                    openai_request_tool = OpenAIRequestTool()
                    
                    user_prompt = f"""You are ODV Tactician module.
                    The User has the following transaction context as well as strategic context
                    they have uploaded here
                    <FULL USER CONTEXT STARTS HERE>
                    {full_user_context}
                    <FULL USER CONTEXT ENDS HERE>
                    Your job is to read through this and to interogate the future AI as to the best, very short-term use of the user's time.
                    You are to condense this short term use of the user's time down to a couple paragraphs at most and provide it
                    to the user
                    """
                    
                    api_args = {
                        "model": constants.DEFAULT_OPEN_AI_MODEL,
                        "messages": [
                            {"role": "system", "content": odv_system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    }
                    
                    writable_df = openai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
                    tactical_string = writable_df['choices__message__content'][0]
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)

# Add this in the on_message handler section of your Discord bot

        if message.content.startswith('!coach'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    # Get user's wallet address
                    logger.debug(f"MyClient.coach: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    wallet_address = user_wallet.classic_address

                    # Check PFT balance
                    pft_balance = self.generic_pft_utilities.get_account_pft_balance(wallet_address)
                    logger.debug(f"MyClient.coach: PFT balance for {message.author.name} is {pft_balance}")
                    if not (config.RuntimeConfig.USE_TESTNET and config.RuntimeConfig.DISABLE_PFT_REQUIREMENTS):
                        if pft_balance < 25000:
                            await message.reply(
                                f"You need at least 25,000 PFT to use the coach command. Your current balance is {pft_balance:,.2f} PFT.", 
                                mention_author=True
                            )
                            return

                    # Get user's full context
                    memo_history = self.generic_pft_utilities.get_account_memo_history(account_address=wallet_address)
                    full_context = self.user_task_parser.get_full_user_context_string(account_address=wallet_address, memo_history=memo_history)
                    
                    # Get chat history
                    chat_history = []
                    if user_id in self.conversations:
                        chat_history = [
                            f"{msg['role'].upper()}: {msg['content']}"
                            for msg in self.conversations[user_id][-10:]  # Get last 10 messages
                        ]
                    formatted_chat = "\n".join(chat_history)

                    # Get the user's specific question/request
                    user_query = message.content.replace('!coach', '').strip()
                    if not user_query:
                        user_query = "Please provide coaching based on my current context and history."

                    # Create the user prompt
                    user_prompt = f"""Based on the following context about me, please provide coaching and guidance.
Rules of engagement:
1. Take the role of a Tony Robbins Type highly paid executive coach while also fulfilling the ODV mandate
2. The goal is to deliver a high NLP score to the user, or to neurolinguistically program them to be likely to fulfill the mandate
provided
3. Keep your advice succinct enough to get the job done but long enough to fully respond to the advice
4. Have the frame that the user is paying 10% or more of their annual earnings to you so your goal is to MAXIMIZE 
the user's earnings and therefore ability to pay you for advice

FULL USER CONTEXT:
{full_context}

RECENT CHAT HISTORY:
{formatted_chat}

My specific question/request is: {user_query}"""

                    # Add reaction to show processing
                    await message.add_reaction('⏳')

                    # Make the API call using o1_preview_simulated_request
                    response = await self.openai_request_tool.o1_preview_simulated_request_async(
                        system_prompt=odv_system_prompt,
                        user_prompt=user_prompt
                    )
                    
                    # Extract content from response
                    content = response.choices[0].message.content
                    
                    # Store the response in conversation history
                    self.conversations[user_id].append({
                        "role": 'assistant',
                        "content": content
                    })
                    
                    # Remove the processing reaction
                    await message.remove_reaction('⏳', self.user)
                    
                    # Send the response
                    await self.send_long_message(message, content)
                    
                except Exception as e:
                    await message.remove_reaction('⏳', self.user)
                    logger.error(f"MyClient.coach: An error occurred while processing your request: {str(e)}")
                    logger.error(traceback.format_exc())
                    error_msg = f"An error occurred while processing your request: {str(e)}"
                    await message.reply(error_msg, mention_author=True)
            else:
                await message.reply("You must store a seed using !store_seed before using the coach.", mention_author=True)

        if message.content.startswith('!blackprint'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    logger.debug(f"MyClient.blackprint: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    tactical_string = self.post_fiat_task_generation_system.generate_coaching_string_for_account(user_address)
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)

        if message.content.startswith('!deathmarch'):
            user_id = message.author.id
            if user_id not in self.user_seeds:
                await message.reply("You must store a seed using /pf_store_seed first.", mention_author=True)
                return

            try:
                seed = self.user_seeds[user_id]
                user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                
                # Create the analyzer inline
                analyzer = ODVFocusAnalyzer(
                    account_address=user_wallet.classic_address,
                    openrouter=self.openrouter,
                    user_context_parser=self.user_task_parser,
                    pft_utils=self.generic_pft_utilities
                )
                # If you want the same exact prompt
                focus_text = analyzer.get_response("Death March Check-In")

                #await message.channel.send(f"**Death March Check-In**\n{focus_text}")
                await self.send_long_message(message, focus_text)

            except Exception as e:
                logger.error(f"!deathmarch: An error occurred for user {user_id}: {str(e)}")
                await message.reply(f"An error occurred: {str(e)}", mention_author=True)


        if message.content.startswith('!redpill'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    logger.debug(f"MyClient.redpill: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    tactical_string = self.post_fiat_task_generation_system.o1_redpill(user_address)
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)

        if message.content.startswith('!docrewrite'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    logger.debug(f"MyClient.docrewrite: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    tactical_string = self.post_fiat_task_generation_system.generate_document_rewrite_instructions(user_address)
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)

    def generate_basic_balance_info_string(self, address: str, owns_wallet: bool = True) -> str:
        """Generate account information summary including balances and stats.
        
        Args:
            wallet: Either an XRPL wallet object (for full access including encrypted docs) 
                or an address string (for public info only)
                
        Returns:
            str: Formatted account information string
        """
        account_info = AccountInfo(address=address)

        try:
            memo_history = generic_pft_utilities.get_account_memo_history(account_address=address)

            if not memo_history.empty:

                # transaction count
                account_info.transaction_count = len(memo_history)

                # Likely username
                account_info.username = list(memo_history[memo_history['direction']=='OUTGOING']['memo_format'].mode())[0]

                # Reward statistics
                reward_data = self.get_reward_data(all_account_info=memo_history)
                if not reward_data['reward_ts'].empty:
                    account_info.monthly_pft_avg = float(reward_data['reward_ts'].tail(4).mean().iloc[0])
                    account_info.weekly_pft_avg = float(reward_data['reward_ts'].tail(1).mean().iloc[0])

            # Get balances
            try:
                account_info.xrp_balance = self.generic_pft_utilities.get_xrp_balance(address)
                account_info.pft_balance = self.generic_pft_utilities.get_pft_balance(address)
            except Exception as e:
                # Account probably not activated yet
                account_info.xrp_balance = 0
                account_info.pft_balance = 0

            # Get google doc link
            if owns_wallet:
                account_info.google_doc_link = self.generic_pft_utilities.get_latest_outgoing_context_doc_link(address)

        except Exception as e:
            logger.error(f"Error generating account info for {address}: {e}")

        return self._format_account_info(account_info)
    
    def _format_account_info(self, info: AccountInfo) -> str:
        """Format AccountInfo into readable string."""
        output = f"""ACCOUNT INFO for {info.address}
                    LIKELY ALIAS:     {info.username}
                    XRP BALANCE:      {info.xrp_balance}
                    PFT BALANCE:      {info.pft_balance}
                    NUM PFT MEMO TX:  {info.transaction_count}
                    PFT MONTHLY AVG:  {info.monthly_pft_avg}
                    PFT WEEKLY AVG:   {info.weekly_pft_avg}"""
        
        if info.google_doc_link:
            output += f"\n\nCONTEXT DOC:      {info.google_doc_link}"
        
        return output
    
    def format_tasks_for_discord(self, input_text: str):
        """
        Format task list for Discord with proper formatting and emoji indicators.
        Handles three sections: NEW TASKS, ACCEPTED TASKS, and TASKS PENDING VERIFICATION.
        Returns a list of formatted chunks ready for Discord sending.
        """
        # Handle empty input
        if not input_text or input_text.strip() == "":
            return ["```ansi\n\u001b[1;33m=== TASK STATUS ===\u001b[0m\n\u001b[0;37mNo tasks found.\u001b[0m\n```"]

        # Split into sections
        sections = input_text.split('\n')
        current_section = None
        formatted_parts = []
        current_chunk = ["```ansi"]
        current_chunk_size = len(current_chunk[0])

        def add_to_chunks(content):
            nonlocal current_chunk, current_chunk_size
            content_size = len(content) + 1  # +1 for newline
            
            if current_chunk_size + content_size > 1900:
                current_chunk.append("```")
                formatted_parts.append("\n".join(current_chunk))
                current_chunk = ["```ansi"]
                current_chunk_size = len(current_chunk[0])
                
            current_chunk.append(content)
            current_chunk_size += content_size

        def format_task_id(task_id: str) -> tuple[str, str]:
            """Format task ID and extract date"""
            try:
                datetime_str = task_id.split('__')[0]
                date_obj = datetime.strptime(datetime_str, '%Y-%m-%d_%H:%M')
                formatted_date = date_obj.strftime('%d %b %Y %H:%M')
                return task_id, formatted_date
            except (ValueError, IndexError):
                return task_id, task_id
        
        # Process input text line by line
        task_data = {}
        for line in sections:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if line in ["NEW TASKS", "ACCEPTED TASKS", "TASKS PENDING VERIFICATION"]:
                if current_section:  # Add spacing between sections
                    add_to_chunks("")
                current_section = line
                add_to_chunks(f"\u001b[1;33m=== {current_section} ===\u001b[0m")
                continue

            # Process task information
            if line.startswith("Task ID: "):
                task_id = line.replace("Task ID: ", "").strip()
                task_data = {"id": task_id}
                task_id, formatted_date = format_task_id(task_id)
                add_to_chunks(f"\u001b[1;36m📌 Task {task_id}\u001b[0m")
                add_to_chunks(f"\u001b[0;37mDate: {formatted_date}\u001b[0m")
                continue

            if line.startswith("Proposal: "):
                proposal = line.replace("Proposal: ", "").strip()
                proposal = proposal.replace("PROPOSED PF ___", "").strip()
                priority_match = re.search(r'\.\. (\d+)$', proposal)
                if priority_match:
                    priority = priority_match.group(1)
                    proposal = proposal.replace(f".. {priority}", "").strip()
                    add_to_chunks(f"\u001b[0;32mPriority: {priority}\u001b[0m")
                add_to_chunks(f"\u001b[1;37mProposal:\u001b[0m\n{proposal}")
                continue

            if line.startswith("Acceptance: "):
                acceptance = line.replace("Acceptance: ", "").strip()
                add_to_chunks(f"\u001b[1;37mAcceptance:\u001b[0m\n{acceptance}")
                continue

            if line.startswith("Verification Prompt: "):
                verification = line.replace("Verification Prompt: ", "").strip()
                add_to_chunks(f"\u001b[1;37mVerification Prompt:\u001b[0m\n{verification}")
                continue

            if line.startswith("-" * 10):  # Separator line
                add_to_chunks("─" * 50)
                continue

        # Finalize last chunk
        current_chunk.append("```")
        formatted_parts.append("\n".join(current_chunk))
        
        return formatted_parts
        
    def format_pending_tasks(self, pending_proposals_df):
        """
        Convert pending_proposals_df to a more legible string format for Discord.
        
        Args:
            pending_proposals_df: DataFrame containing pending proposals
            
        Returns:
            Formatted string representation of the pending proposals
        """
        formatted_tasks = []
        for idx, row in pending_proposals_df.iterrows():
            task_str = f"Task ID: {idx}\n"
            task_str += f"Proposal: {row['proposal']}\n"
            task_str += "-" * 50  # Separator
            formatted_tasks.append(task_str)
        
        formatted_task_string =  "\n".join(formatted_tasks)
        output_string="NEW TASKS\n" + formatted_task_string
        return output_string

    def format_accepted_tasks(self, accepted_proposals_df):
        """
        Convert accepted_proposals_df to a legible string format for Discord.
        
        Args:
            accepted_proposals_df: DataFrame containing outstanding tasks
            
        Returns:
            Formatted string representation of the tasks
        """
        formatted_tasks = []
        for idx, row in accepted_proposals_df.iterrows():
            task_str = f"Task ID: {idx}\n"
            task_str += f"Proposal: {row['proposal']}\n"
            task_str += f"Acceptance: {row['acceptance']}\n"
            task_str += "-" * 50  # Separator
            formatted_tasks.append(task_str)
        
        formatted_task_string =  "\n".join(formatted_tasks)
        output_string="ACCEPTED TASKS\n" + formatted_task_string
        return output_string
    
    def format_verification_tasks(self, verification_proposals_df):
        """
        Format the verification_requirements dataframe into a string.

        Args:
            verification_proposals_df (pd.DataFrame): DataFrame containing tasks pending verification

        Returns:
        str: Formatted string of verification requirements
        """
        formatted_output = "TASKS PENDING VERIFICATION\n"
        for idx, row in verification_proposals_df.iterrows():
            formatted_output += f"Task ID: {idx}\n"
            formatted_output += f"Proposal: {row['proposal']}\n"
            formatted_output += f"Verification Prompt: {row['verification']}\n"
            formatted_output += "-" * 50 + "\n"
        return formatted_output
    
    def create_full_outstanding_pft_string(self, account_address):
        """ 
        This takes in an account address and outputs the current state of its outstanding tasks.
        Returns empty string for accounts with no PFT-related transactions.
        """ 
        memo_history = generic_pft_utilities.get_account_memo_history(account_address=account_address, pft_only=True)
        if memo_history.empty:
            return ""
        
        memo_history.sort_values('datetime', inplace=True)
        pending_proposals = post_fiat_task_generation_system.get_pending_proposals(account=memo_history)
        accepted_proposals = post_fiat_task_generation_system.get_accepted_proposals(account=memo_history)
        verification_proposals = post_fiat_task_generation_system.get_verification_proposals(account=memo_history)

        pending_string = self.format_pending_tasks(pending_proposals)
        accepted_string = self.format_accepted_tasks(accepted_proposals)
        verification_string = self.format_verification_tasks(verification_proposals)

        full_postfiat_outstanding_string=f"{pending_string}\n{accepted_string}\n{verification_string}"
        return full_postfiat_outstanding_string

    def _calculate_weekly_reward_totals(self, specific_rewards):
        """Calculate weekly reward totals with proper date handling.
        
        Returns DataFrame with weekly_total column indexed by date"""
        # Calculate daily totals
        daily_totals = specific_rewards[['directional_pft', 'simple_date']].groupby('simple_date').sum()

        if daily_totals.empty:
            logger.warning("No rewards data available to calculate weekly totals.")
            return pd.DataFrame(columns=['weekly_total'])

        # Extend date range to today
        today = pd.Timestamp.today().normalize()
        start_date = daily_totals.index.min()

        if pd.isna(start_date):
            logger.warning("Start date is NaT, cannot calculate weekly totals.")
            return pd.DataFrame(columns=['weekly_total'])
        
        date_range = pd.date_range(
            start=start_date,
            end=today,
            freq='D'
        )

        # Fill missing dates and calculate weekly totals
        extended_daily_totals = daily_totals.reindex(date_range, fill_value=0)
        extended_daily_totals = extended_daily_totals.resample('D').last().fillna(0)
        extended_daily_totals['weekly_total'] = extended_daily_totals.rolling(7).sum()

        # Return weekly totals
        weekly_totals = extended_daily_totals.resample('W').last()[['weekly_total']]
        weekly_totals.index.name = 'date'

        # if weekly totals are NaN, set them to 0
        weekly_totals = weekly_totals.fillna(0)

        return weekly_totals
    
    def _pair_rewards_with_tasks(self, specific_rewards, all_account_info):
        """Pair rewards with their original requests and proposals.
        
        Returns DataFrame with columns: memo_data, directional_pft, datetime, memo_type, request, proposal
        """
        # Get reward details
        reward_details = specific_rewards[
            ['memo_data', 'directional_pft', 'datetime', 'memo_type']
        ].sort_values('datetime')

        # Get original requests and proposals
        task_requests = all_account_info[
            all_account_info['memo_data'].apply(lambda x: constants.TaskType.REQUEST_POST_FIAT.value in x)
        ].groupby('memo_type').first()['memo_data']

        proposal_patterns = constants.TASK_PATTERNS[constants.TaskType.PROPOSAL]
        task_proposals = all_account_info[
            all_account_info['memo_data'].apply(lambda x: any(pattern in str(x) for pattern in proposal_patterns))
        ].groupby('memo_type').first()['memo_data']

        # Map requests and proposals to rewards
        reward_details['request'] = reward_details['memo_type'].map(task_requests).fillna('No Request String')
        reward_details['proposal'] = reward_details['memo_type'].map(task_proposals)

        return reward_details

    def get_reward_data(self, all_account_info):
        """Get reward time series and task completion history.
        
        Args:
            all_account_info: DataFrame containing account memo details
            
        Returns:
            dict with keys:
                - reward_ts: DataFrame of weekly reward totals
                - reward_summaries: DataFrame containing rewards paired with original requests/proposals
        """
        # Get basic reward data
        reward_responses = all_account_info[all_account_info['directional_pft'] > 0]
        specific_rewards = reward_responses[
            reward_responses.memo_data.apply(lambda x: "REWARD RESPONSE" in x)
        ]

        # Get weekly totals
        weekly_totals = self._calculate_weekly_reward_totals(specific_rewards)

        # Get reward summaries with context
        reward_summaries = self._pair_rewards_with_tasks(
            specific_rewards=specific_rewards,
            all_account_info=all_account_info
        )

        return {
            'reward_ts': weekly_totals,
            'reward_summaries': reward_summaries
        }

    @staticmethod
    def format_reward_summary(reward_summary_df):
        """
        Convert reward summary dataframe into a human-readable string.
        :param reward_summary_df: DataFrame containing reward summary information
        :return: Formatted string representation of the rewards
        """
        formatted_rewards = []
        for _, row in reward_summary_df.iterrows():
            reward_str = f"Date: {row['datetime']}\n"
            reward_str += f"Request: {row['request']}\n"
            reward_str += f"Proposal: {row['proposal']}\n"
            reward_str += f"Reward: {row['directional_pft']} PFT\n"
            reward_str += f"Response: {row['memo_data'].replace(constants.TaskType.REWARD.value, '')}\n"
            reward_str += "-" * 50  # Separator
            formatted_rewards.append(reward_str)
        
        output_string = "REWARD SUMMARY\n\n" + "\n".join(formatted_rewards)
        return output_string
    
    def _calculate_death_march_costs(self, settings: DeathMarchSettings, days: int = 1) -> tuple[int, int]:
        """Calculate death march check-ins and costs.
        
        Args:
            settings: User's death march settings
            days: Number of days for the death march
            
        Returns:
            tuple[int, int]: (checks_per_day, total_cost)
        """
        start_dt = datetime.combine(datetime.today(), settings.start_time)
        end_dt = datetime.combine(datetime.today(), settings.end_time)
        daily_duration = (end_dt - start_dt).total_seconds() / 60  # duration in minutes
        checks_per_day = int(daily_duration / settings.check_interval)
        total_cost = checks_per_day * days * 30  # 30 PFT per check-in
        
        return checks_per_day, total_cost

def init_bot():
    """Initialize and return the Discord bot with required intents"""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guild_messages = True
    return MyClient(intents=intents)

def configure_runtime():
    """Configure runtime settings based on user input"""

    # Network selection
    print(f"Network Configuration:\n1. Mainnet\n2. Testnet")
    network_choice = input("Select network (1/2) [default=2]: ").strip() or "2"
    RuntimeConfig.USE_TESTNET = network_choice == "2"
    network_config = get_network_config()

    # Local node configuration with network-specific context
    if network_config.local_rpc_url:
        print(f"\nLocal node configuration:")
        print(f"Local {network_config.name} node URL: {network_config.local_rpc_url}")
        use_local = input("Do you have a local node configured? (y/n) [default=n]: ").strip() or "n"
        RuntimeConfig.HAS_LOCAL_NODE = use_local == "y"
    else:
        print(f"\nNo local node configuration available for {network_config.name}")
        RuntimeConfig.HAS_LOCAL_NODE = False

    logger.debug(f"\nInitializing services for {network_config.name}...")
    logger.info(f"Using {'local' if RuntimeConfig.HAS_LOCAL_NODE else 'public'} endpoints...")
    

def init_services():
    """Initialize and return core services"""
    openai_request_tool = OpenAIRequestTool()
    post_fiat_task_generation_system = PostFiatTaskGenerationSystem()
    generic_pft_utilities = GenericPFTUtilities()
    generic_pft_utilities.run_transaction_history_updates()

    return (
        openai_request_tool,
        post_fiat_task_generation_system,
        generic_pft_utilities
    )

if __name__ == "__main__":
    # Initialize credential manager
    password = getpass.getpass("Enter your password: ")
    cred_manager = CredentialManager(password=password)

    # Initialize performance monitor
    monitor = PerformanceMonitor(time_window=60)
    monitor.start()

    # Configure logger
    configure_logger(
        log_to_file=True,
        output_directory=Path.cwd() / "nodetools",
        log_filename="nodetools.log",
        level="DEBUG"
    )

    configure_runtime()

    # Initialize services
    open_ai_request_tool, post_fiat_task_generation_system, generic_pft_utilities = init_services()

    logger.debug("---Services initialized successfully!---")

    # Initialize and run the bot
    client = init_bot()
    discord_credential_key = "discordbot_testnet_secret" if RuntimeConfig.USE_TESTNET else "discordbot_secret"
    client.run(cred_manager.get_credential(discord_credential_key))