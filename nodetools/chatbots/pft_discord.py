from xrpl.wallet import Wallet
import discord
from discord import Object, Interaction, SelectOption, app_commands
from discord.ui import Modal, TextInput, View, Select
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.utilities.credentials import CredentialManager
from nodetools.utilities.generic_pft_utilities import *
from nodetools.utilities.task_management import PostFiatTaskGenerationSystem
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.performance.monitor import PerformanceMonitor
from nodetools.prompts.chat_processor import ChatProcessor
import asyncio
from datetime import datetime, time
import pytz
import datetime
import nodetools.utilities.constants as constants
import getpass
    
class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get network configuration and set network-specific attributes
        self.network_config = constants.get_network_config()
        self.remembrancer = self.network_config.remembrancer_address

        # Initialize components
        self.open_ai_request_tool = OpenAIRequestTool()
        self.generic_pft_utilities = GenericPFTUtilities(node_name=self.network_config.node_name)
        self.post_fiat_task_generation_system = PostFiatTaskGenerationSystem()

        # Set network-specific attributes
        self.default_openai_model = constants.DEFAULT_OPEN_AI_MODEL
        self.conversations = {}
        self.user_seeds = {}
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        """Sets up the slash commands for the bot and initiates background tasks."""
        guild_id = self.network_config.discord_guild_id
        guild = Object(id=guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        print(f"MyClient.setup_hook: Slash commands synced to guild ID: {guild_id}")

        self.bg_task = self.loop.create_task(self.transaction_checker())

        @self.tree.command(name="pf_send", description="Open a transaction form")
        async def pf_send(interaction: Interaction):
            user_id = interaction.user.id

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /store_seed before initiating a transaction.", ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]

            class SimpleTransactionModal(discord.ui.Modal, title='Transaction Details'):
                address = discord.ui.TextInput(label='Recipient Address')
                amount = discord.ui.TextInput(label='Amount')
                message = discord.ui.TextInput(label='Message', style=discord.TextStyle.long, required=False)

                def __init__(self, seed, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.seed = seed  # Store the user's seed

                async def on_submit(self, interaction: discord.Interaction):
                    # Defer the interaction response to avoid timeout
                    await interaction.response.defer(ephemeral=True)

                    # Perform the transaction using the details provided in the modal
                    destination_address = self.address.value
                    amount = self.amount.value
                    message = self.message.value

                    # Trigger the transaction
                    proper_string = generic_pft_utilities.discord_send_pft_with_info_from_seed(
                        destination_address=destination_address, 
                        seed=self.seed,
                        user_name=interaction.user.name, 
                        message=message, 
                        amount=amount
                    )
                    await interaction.followup.send(
                        f'Transaction sent! {proper_string}',
                        ephemeral=True
                    )
            # Pass the user's seed to the modal
            await interaction.response.send_modal(SimpleTransactionModal(seed=seed))
            print("PF Send command executed!")
            #def send_xrp_with_info__seed_based(self,wallet_seed, amount, destination, memo):
        
            
        @self.tree.command(name="pf_accept", description="Accept tasks")
        async def pf_accept_menu(interaction: discord.Interaction):
            # Fetch the user's seed
            user_id = interaction.user.id
            if user_id not in self.user_seeds:
                await interaction.response.send_message("You must store a seed using /store_seed before accepting tasks.", ephemeral=True)
                return

            seed = self.user_seeds[user_id]

            # Fetch the tasks that are not yet accepted
            print(f"MyClient.pf_accept_menu: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            classic_address = wallet.classic_address
            memo_history = generic_pft_utilities.get_account_memo_history(account_address=classic_address, pft_only=False).copy()
            pf_df = generic_pft_utilities.get_proposal_acceptance_pairs(
                account_memo_detail_df=memo_history, 
                include_pending=True
            )

            # Filter out tasks that have already been accepted
            eligible_tasks = pf_df[pf_df['acceptance'] == ''].copy()

            map_of_eligible_tasks = eligible_tasks['proposal']

            # If there are no non-accepted tasks, notify the user
            if map_of_eligible_tasks.empty:
                await interaction.response.send_message("You have no tasks to accept.", ephemeral=True)
                return

            # Create dropdown options based on the non-accepted tasks
            options = [
                SelectOption(label=task_id, description=proposal[:100], value=task_id)
                for task_id, proposal in map_of_eligible_tasks.items()
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to accept", options=options)

            # Define the modal for inputting the acceptance string
            class AcceptanceModal(Modal):
                def __init__(self, task_id: str, task_text: str, seed: str, user_name: str):
                    super().__init__(title="Accept Task")
                    self.task_id = task_id
                    self.seed = seed
                    self.user_name = user_name
                    
                    # Add a label to display the full task description
                    self.task_description = discord.ui.TextInput(
                        label="Task Description (Do not modify)",
                        default=task_text,
                        style=discord.TextStyle.paragraph,
                        required=False
                    )
                    self.add_item(self.task_description)
                    
                    # Add the acceptance string input
                    self.acceptance_string = TextInput(
                        label="Acceptance String", 
                        placeholder="Type your acceptance string here",
                        style=discord.TextStyle.paragraph
                    )
                    self.add_item(self.acceptance_string)

                async def on_submit(self, interaction: discord.Interaction):
                    # Defer the response to avoid interaction timeout
                    await interaction.response.defer(ephemeral=True)
                    
                    acceptance_string = self.acceptance_string.value
                    
                    # Call the discord__task_acceptance function
                    output_string = post_fiat_task_generation_system.discord__task_acceptance(
                        seed_to_work=self.seed,
                        user_name=self.user_name,
                        task_id_to_accept=self.task_id,
                        acceptance_string=acceptance_string
                    )
                    
                    # Send a follow-up message with the result
                    await interaction.followup.send(output_string, ephemeral=True)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = map_of_eligible_tasks[selected_task_id]
                # Open the modal to get the acceptance string with the task text pre-populated
                await interaction.response.send_modal(AcceptanceModal(
                    task_id=selected_task_id,
                    task_text=task_text,
                    seed=seed,
                    user_name=interaction.user.name
                ))

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.response.send_message("Please choose a task to accept:", view=view, ephemeral=True)
        
        @self.tree.command(name="pf_refuse", description="Refuse tasks")
        async def pf_refuse_menu(interaction: discord.Interaction):
            # Fetch the user's seed
            user_id = interaction.user.id
            if user_id not in self.user_seeds:
                await interaction.response.send_message("You must store a seed using /store_seed before refusing tasks.", ephemeral=True)
                return

            seed = self.user_seeds[user_id]

            # Fetch the tasks that are not yet accepted
            print(f"MyClient.pf_refuse_menu: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            classic_address = wallet.classic_address
            memo_history = generic_pft_utilities.get_account_memo_history(account_address=classic_address, pft_only=False).copy()
            pf_df = generic_pft_utilities.get_proposal_refusal_pairs(
                account_memo_detail_df=memo_history, 
                exclude_refused=True
            )
            
            # TODO: Filter out tasks that have already been rewarded
            eligible_tasks = pf_df.copy()

            map_of_eligible_tasks = eligible_tasks['proposal']

            # If there are no non-accepted tasks, notify the user
            if map_of_eligible_tasks.empty:
                await interaction.response.send_message("You have no tasks to refuse.", ephemeral=True)
                return

            # Create dropdown options based on the non-accepted tasks
            options = [
                SelectOption(label=task_id, description=proposal[:100], value=task_id)
                for task_id, proposal in map_of_eligible_tasks.items()
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to refuse", options=options)

            # Define the modal for inputting the refusal string
            class RefusalModal(Modal):
                def __init__(self, task_id: str, task_text: str, seed: str, user_name: str):
                    super().__init__(title="Refuse Task")
                    self.task_id = task_id
                    self.seed = seed
                    self.user_name = user_name
                    
                    # Add a label to display the full task description
                    self.task_description = discord.ui.TextInput(
                        label="Task Description (Do not modify)",
                        default=task_text,
                        style=discord.TextStyle.paragraph,
                        required=False
                    )
                    self.add_item(self.task_description)
                    
                    # Add the refusal string input
                    self.refusal_string = TextInput(
                        label="Refusal Reason", 
                        placeholder="Type your reason for refusing this task",
                        style=discord.TextStyle.paragraph
                    )
                    self.add_item(self.refusal_string)

                async def on_submit(self, interaction: discord.Interaction):
                    # Defer the response to avoid interaction timeout
                    await interaction.response.defer(ephemeral=True)
                    
                    refusal_string = self.refusal_string.value
                    
                    # Call the discord__task_refusal function
                    output_string = post_fiat_task_generation_system.discord__task_refusal(
                        seed_to_work=self.seed,
                        user_name=self.user_name,
                        task_id_to_refuse=self.task_id,
                        refusal_string=refusal_string
                    )
                    
                    # Send a follow-up message with the result
                    await interaction.followup.send(output_string, ephemeral=True)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = map_of_eligible_tasks[selected_task_id]
                # Open the modal to get the refusal string with the task text pre-populated
                await interaction.response.send_modal(RefusalModal(
                    task_id=selected_task_id,
                    task_text=task_text,
                    seed=seed,
                    user_name=interaction.user.name
                ))

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.response.send_message("Please choose a task to refuse:", view=view, ephemeral=True)

        @self.tree.command(name="wallet_info", description="Get information about a wallet")
        async def wallet_info(interaction: discord.Interaction, wallet_address: str):
            try:
                account_info = generic_pft_utilities.generate_basic_balance_info_string_for_account_address(account_address=wallet_address)
                
                # Create an embed for better formatting
                embed = discord.Embed(title="Wallet Information", color=0x00ff00)
                embed.add_field(name="Wallet Address", value=wallet_address, inline=False)
                embed.add_field(name="Account Info", value=account_info, inline=False)
                
                await interaction.response.send_message(embed=embed, ephemeral=True)
            except Exception as e:
                await interaction.response.send_message(f"An error occurred: {str(e)}", ephemeral=True)

        @self.tree.command(name="pf_log", description="Send a long message to the remembrancer wallet")
        async def pf_remembrancer(interaction: discord.Interaction, message: str, encrypt: bool = False):
            user_id = interaction.user.id
            
            # Check if the user has a stored seed
            if user_id not in client.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed before using this command.",
                    ephemeral=True
                )
                return

            seed = client.user_seeds[user_id]
            user_name = interaction.user.name

            # Defer the response to avoid timeout for longer operations
            await interaction.response.defer(ephemeral=True)

            try:
                responses = generic_pft_utilities.send_memo(
                    wallet_seed=seed,
                    user_name=user_name,
                    destination=self.remembrancer,
                    memo=message,
                    compress=True,
                    encrypt=encrypt
                )

                transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object(response=responses[-1])
                clean_string = transaction_info['clean_string']

                mode = "Encrypted message" if encrypt else "Message"
                await interaction.followup.send(
                    f"{mode} sent to remembrancer successfully. Last chunk details:\n{clean_string}", 
                    ephemeral=True
                )

            except HandshakeRequiredException:
                try:
                    # Initiate handshake protocol
                    print(f"MyClient.pf_log_encrypted: Handshake required for {interaction.user.name}. Initiating handshake protocol.")
                    
                    handshake_response = generic_pft_utilities.send_handshake(
                        wallet_seed=seed,
                        user_name=user_name,
                        destination=self.remembrancer
                    )

                    await interaction.followup.send(
                        "Encryption handshake initiated. Please wait a few moments for the handshake to be processed.",
                        ephemeral=True
                    )

                    # Verification loop
                    handshake_verified = False
                    print(f"MyClient.pf_remembrancer: Attempting to verify handshake receipt from remembrancer ({self.remembrancer}) for {interaction.user.name}...")
                    for attempt in range(constants.TRANSACTION_VERIFICATION_ATTEMPTS):
                        print(f"MyClient.pf_remembrancer: Attempt {attempt+1} of {constants.TRANSACTION_VERIFICATION_ATTEMPTS}...")

                        # Check for remembrancer's handshake response
                        _, received_key = self.generic_pft_utilities.get_handshake_for_address(
                            wallet_address=self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed).address,
                            destination=self.remembrancer
                        )

                        if received_key:
                            handshake_verified = True
                            print(f"MyClient.pf_remembrancer: Handshake verified after {attempt+1} attempts, proceeding with message send...")
                            break
                        
                        await asyncio.sleep(constants.TRANSACTION_VERIFICATION_WAIT_TIME)

                    if handshake_verified:
                        await interaction.followup.send(
                            "Handshake verification timed out. Please try sending your message again.",
                            ephemeral=True
                        )
                        return

                    response = generic_pft_utilities.send_memo(
                        wallet_seed=seed,
                        user_name=user_name,
                        destination=self.remembrancer,
                        memo=message,
                        compress=True,
                        encrypt=encrypt
                    )

                    transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object(response=response)
                    clean_string = transaction_info['clean_string']

                    mode = "Encrypted message" if encrypt else "Message"
                    await interaction.followup.send(
                        f"{mode} sent to remembrancer successfully. Last chunk details:\n{clean_string}", 
                        ephemeral=True
                    )

                except Exception as e:
                    await interaction.followup.send(
                        f"An error occurred while initiating the encryption handshake protocol: {str(e)}", 
                        ephemeral=True
                    )

            except Exception as e:
                await interaction.followup.send(f"An error occurred while sending the message: {str(e)}", ephemeral=True)
        
        @self.tree.command(name="pf_chart", description="Generate a chart of your PFT rewards and metrics")
        async def pf_chart(interaction: discord.Interaction):
            user_id = interaction.user.id
            
            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed before generating a chart.", 
                    ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]
            print(f"MyClient.pf_chart: Spawning wallet to generate chart for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address

            # Defer the response since chart generation might take time
            await interaction.response.defer(ephemeral=True)

            try:
                # Call the charting function
                post_fiat_task_generation_system.output_pft_KPI_graph_for_address(user_wallet=wallet_address)
                
                # Create the file object from the saved image
                chart_file = discord.File(f'pft_rewards__{wallet_address}.png', filename='pft_chart.png')
                
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
                    ephemeral=True
                )
                
                # Clean up the file after sending
                import os
                os.remove(f'pft_rewards__{wallet_address}.png')

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
            if interaction.user.id != 402536023483088896:
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

        @self.tree.command(name="pf_outstanding", description="Show your outstanding tasks and verification tasks")
        async def pf_outstanding(interaction: discord.Interaction):
            user_id = interaction.user.id
            
            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed before viewing outstanding tasks.", 
                    ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]
            print(f"MyClient.pf_outstanding: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address

            # Defer the response to avoid timeout for longer operations
            await interaction.response.defer(ephemeral=True)

            try:
                # Get the unformatted output message
                output_message = generic_pft_utilities.create_full_outstanding_pft_string(account_address=wallet_address)
                
                # Format the message using the new formatting function
                formatted_chunks = generic_pft_utilities.format_tasks_for_discord(output_message)
                
                # Send the first chunk
                await interaction.followup.send(formatted_chunks[0], ephemeral=True)

                # Send the rest of the chunks
                for chunk in formatted_chunks[1:]:
                    await interaction.followup.send(chunk, ephemeral=True)

            except Exception as e:
                await interaction.followup.send(
                    f"An error occurred while fetching your outstanding tasks: {str(e)}", 
                    ephemeral=True
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

            class XRPTransactionModal(discord.ui.Modal, title='XRP Transaction Details'):
                def __init__(self, seed, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.seed = seed

                address = discord.ui.TextInput(label='Recipient Address')
                amount = discord.ui.TextInput(label='Amount (in XRP)')
                message = discord.ui.TextInput(
                    label='Message', 
                    style=discord.TextStyle.long, 
                    required=False,
                    placeholder='Insert an optional message'
                )
                destination_tag = discord.ui.TextInput(
                    label='Destination Tag',
                    required=False,
                    placeholder='Required for most exchanges'
                )

                async def on_submit(self, interaction: discord.Interaction):
                    await interaction.response.defer(ephemeral=True)

                    destination_address = self.address.value
                    amount = self.amount.value
                    message = self.message.value
                    destination_tag = self.destination_tag.value

                    # Create the memo
                    memo = generic_pft_utilities.construct_standardized_xrpl_memo(
                        memo_data=message,
                        memo_format=interaction.user.name,
                        memo_type="XRP_SEND"
                    )

                    try:
                        # Convert destination_tag to integer if it exists
                        dt = int(destination_tag) if destination_tag else None

                        # Call the send_xrp_with_info__seed_based function
                        response = generic_pft_utilities.send_xrp_with_info__seed_based(
                            wallet_seed=self.seed,
                            amount=amount,
                            destination=destination_address,
                            memo=memo,
                            destination_tag=dt
                        )

                        # Extract transaction information using the improved function
                        transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object__standard_xrp(response)
                        
                        # Create an embed for better formatting
                        embed = discord.Embed(title="XRP Transaction Sent", color=0x00ff00)
                        embed.add_field(name="Details", value=transaction_info['clean_string'], inline=False)
                        
                        # Add additional fields if available
                        if dt:
                            embed.add_field(name="Destination Tag", value=str(dt), inline=False)
                        if 'hash' in transaction_info:
                            embed.add_field(name="Transaction Hash", value=transaction_info['hash'], inline=False)
                        if 'xrpl_explorer_url' in transaction_info:
                            embed.add_field(name="Explorer Link", value=transaction_info['xrpl_explorer_url'], inline=False)
                        
                        await interaction.followup.send(embed=embed, ephemeral=True)
                    except Exception as e:
                        await interaction.followup.send(f"An error occurred: {str(e)}", ephemeral=True)

            # Create and send the modal
            modal = XRPTransactionModal(seed=seed)
            await interaction.response.send_modal(modal)

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
                    self.client.user_seeds[user_id] = self.seed.value  # Store the seed
                    await interaction.response.send_message(f'Seed stored successfully for user {interaction.user.name}.', ephemeral=True)

            # Pass the client instance to the modal
            await interaction.response.send_modal(SeedModal(client=self))
            print(f"Seed storage command executed by {interaction.user.name}")

        # Sync the commands to the guild
        await self.tree.sync(guild=guild)
        print(f"MyClient.setup_hook: Slash commands synced to guild ID: {guild_id}")

        # Sync the commands to the guild
        await self.tree.sync(guild=guild)
        print(f"MyClient.setup_hook: Slash commands synced to guild ID: {guild_id}")

        @self.tree.command(name="pf_initiate", description="Initiate your commitment")
        async def pf_initiate(interaction: discord.Interaction):
            user_id = interaction.user.id

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /store_seed before initiating.", ephemeral=True
                )
                return

            try:
                # Spawn the user's wallet
                print(f"MyClient.pf_initiate: Spawning wallet to initiate for {interaction.user.name}")
                seed = self.user_seeds[user_id]
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)

                # Check for existing initiation
                memo_history = generic_pft_utilities.get_account_memo_history(wallet.classic_address, pft_only=False)
                existing_initiations = memo_history[memo_history['memo_type'] == 'INITIATION_RITE']

                # Block re-initiation unless on testnet with ENABLE_REINITIATIONS = True
                if not existing_initiations.empty and not (constants.USE_TESTNET and constants.ENABLE_REINITIATIONS):
                    print(f"MyClient.pf_initiate: Blocking re-initiation for {interaction.user.name} ({wallet.classic_address})")
                    await interaction.response.send_message(
                        "You have already completed an initiation rite. Re-initiation is not allowed.", 
                        ephemeral=True
                    )
                    return

                # Define the modal to collect Google Doc Link and Commitment
                class InitiationModal(discord.ui.Modal, title='Initiation Commitment'):
                    google_doc_link = discord.ui.TextInput(
                        label='Google Doc Link', 
                        style=discord.TextStyle.short,
                        placeholder="Share the link to your Google Doc"
                    )
                    commitment_sentence = discord.ui.TextInput(
                        label='Commit to a Long-Term Objective',
                        style=discord.TextStyle.long,
                        placeholder="A 1-sentence commitment to a long-term objective"
                    )

                    async def on_submit(self, interaction: discord.Interaction):
                        await interaction.response.defer(ephemeral=True)
                        
                        try:
                            # Attempt the initiation rite
                            post_fiat_task_generation_system.discord__initiation_rite(
                                account_seed=seed, 
                                initiation_rite=self.commitment_sentence.value, 
                                google_doc_link=self.google_doc_link.value, 
                                username=interaction.user.name,
                                allow_reinitiation=constants.USE_TESTNET and constants.ENABLE_REINITIATIONS  # Allow re-initiation in test mode
                            )
                            
                            mode = "(TEST MODE)" if constants.USE_TESTNET else ""
                            await interaction.followup.send(
                                f"Initiation complete! {mode}\nCommitment: {self.commitment_sentence.value}\nGoogle Doc: {self.google_doc_link.value}",
                                ephemeral=True
                            )

                        except Exception as e:
                            await interaction.followup.send(
                                f"An error occurred during initiation: {str(e)}", 
                                ephemeral=True
                            )

                # Present the modal to the user
                await interaction.response.send_modal(InitiationModal())

            except Exception as e:
                await interaction.followup.send(f"An error occurred during initiation: {str(e)}", ephemeral=True)

        @self.tree.command(name="pf_request_task", description="Request a Post Fiat task")
        async def pf_task_slash(interaction: discord.Interaction, task_request: str):
            user_id = interaction.user.id
            
            # Check if the user has stored a seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message("You must store a seed using /pf_store_seed before generating a task.", ephemeral=True)
                return

            # Get the user's seed and other necessary information
            seed = self.user_seeds[user_id]
            user_name = interaction.user.name
            
            # Defer the response to avoid timeout
            await interaction.response.defer(ephemeral=True)
            
            try:
                # Send the Post Fiat request
                response = post_fiat_task_generation_system.discord__send_postfiat_request(
                    user_request=task_request,
                    user_name=user_name,
                    seed=seed
                )
                
                # Extract transaction information
                transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object(response=response)
                clean_string = transaction_info['clean_string']
                
                # Send the response
                await interaction.followup.send(f"Task Requested with Details: {clean_string}", ephemeral=True)
            except Exception as e:
                await interaction.followup.send(f"An error occurred while processing your request: {str(e)}", ephemeral=True)

        @self.tree.command(name="pf_initial_verification", description="Submit a task for verification")
        async def pf_submit_for_verification(interaction: discord.Interaction):
            user_id = interaction.user.id

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /store_seed before submitting a task for verification.", ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]

            # Fetch the tasks that are accepted but not completed
            print(f"MyClient.pf_initial_verification: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            wallet_address = wallet.classic_address
            memo_history = generic_pft_utilities.get_account_memo_history(wallet_address).copy()
            pf_df = generic_pft_utilities.get_proposal_acceptance_pairs(account_memo_detail_df=memo_history)
            # Filter for accepted tasks that have not been completed yet
            accepted_tasks = pf_df[
                (pf_df['acceptance'] != '') & 
                ~pf_df.index.isin(memo_history[
                    memo_history['memo_data'].str.contains(constants.TaskType.VERIFICATION_PROMPT.value, na=False)
                ]['memo_type'].unique())
            ].copy()
            
            # If there are no accepted tasks, notify the user
            if accepted_tasks.empty:
                await interaction.response.send_message("You have no accepted tasks to submit for verification.", ephemeral=True)
                return

            # Create dropdown options based on the accepted tasks
            options = [
                SelectOption(label=task_id, description=proposal[:100], value=task_id)
                for task_id, proposal in accepted_tasks['proposal'].items()
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to submit for verification", options=options)

            # Define the modal for inputting the completion justification
            class CompletionModal(Modal):
                def __init__(self, task_id: str, task_text: str, seed: str, user_name: str):
                    super().__init__(title="Submit Task for Verification")
                    self.task_id = task_id
                    self.seed = seed
                    self.user_name = user_name
                    
                    # Add a label to display the full task description
                    self.task_description = discord.ui.TextInput(
                        label="Task Description (Do not modify)",
                        default=task_text,
                        style=discord.TextStyle.paragraph,
                        required=False
                    )
                    self.add_item(self.task_description)
                    
                    # Add the completion justification input
                    self.completion_justification = TextInput(
                        label="Completion Justification", 
                        placeholder="Explain how you completed the task",
                        style=discord.TextStyle.paragraph
                    )
                    self.add_item(self.completion_justification)

                async def on_submit(self, interaction: discord.Interaction):
                    # Defer the response to avoid interaction timeout
                    await interaction.response.defer(ephemeral=True)
                    
                    completion_string = self.completion_justification.value
                    
                    # Call the discord__initial_submission function
                    output_string = post_fiat_task_generation_system.discord__initial_submission(
                        seed_to_work=self.seed,
                        user_name=self.user_name,
                        task_id_to_accept=self.task_id,
                        initial_completion_string=completion_string
                    )
                    
                    # Send a follow-up message with the result
                    await interaction.followup.send(output_string, ephemeral=True)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = accepted_tasks.loc[selected_task_id, 'proposal']
                # Open the modal to get the completion justification with the task text pre-populated
                await interaction.response.send_modal(CompletionModal(
                    task_id=selected_task_id,
                    task_text=task_text,
                    seed=seed,
                    user_name=interaction.user.name
                ))

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.response.send_message("Please choose a task to submit for verification:", view=view, ephemeral=True)

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
                    self.client = client

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

### Task Request
1. /pf_request_task: Request a new Post Fiat task.
2. /pf_accept: View and accept available tasks.
3. /pf_refuse: View and refuse available tasks.
4. /pf_initial_verification: Submit a completed task for verification.
5. /pf_final_verification: Submit final verification for a task to receive reward

### Transaction
1. /xrp_send: Send XRP to a destination address with a memo.
2. /pf_send: Open a transaction form to send PFT tokens with a memo.
3. /pf_log: take notes re your workflows
4. /pf_log_encrypted: take encrypted notes re your workflows

## Post Fiat operates on a Google Document.
1. Place your Funded Wallet Address at the top of the Google Document 
2. Set the Document to be shared (File/Share/Share With Others/Anyone With Link)
3. Note this is fully public as are transactions on the XRP ledger
4. The PF Initiate Function requires a document and a verbal committment
5. After your address place a section like
___x TASK VERIFICATION SECTION START x___ 
task verification details are here 
___x TASK VERIFICATION SECTION END x___

## Local Version
You can run a local version of the wallet. Please reference the Post Fiat Github
https://github.com/postfiatorg/pftpyclient

Note: XRP wallets need 15 XRP to transact.
"""
            
            await interaction.response.send_message(guide_text, ephemeral=True)

        @self.tree.command(name="pf_my_wallet", description="Show your wallet information")
        async def pf_my_wallet(interaction: discord.Interaction):
            user_id = interaction.user.id
            
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "No seed found for your account. Use /pf_store_seed to store a seed first.",
                    ephemeral=True
                )
                return

            try:
                seed = self.user_seeds[user_id]
                print(f"MyClient.pf_my_wallet: Spawning wallet to fetch info for {interaction.user.name}")
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
                wallet_address = wallet.classic_address

                # Get account info
                account_info = generic_pft_utilities.generate_basic_balance_info_string_for_account_address(account_address=wallet_address)
                
                # Get recent messages
                recent_messages = generic_pft_utilities.get_recent_messages_for_account_address(wallet_address)

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

                # Second embed for transaction history if needed
                if recent_messages['incoming_message'] or recent_messages['outgoing_message']:
                    embed2 = discord.Embed(title="Recent Transactions", color=0x00ff00)
                    
                    if recent_messages['incoming_message']:
                        incoming = truncate_field(recent_messages['incoming_message'])
                        embed2.add_field(name="Most Recent Incoming Transaction", value=incoming, inline=False)
                    
                    if recent_messages['outgoing_message']:
                        outgoing = truncate_field(recent_messages['outgoing_message'])
                        embed2.add_field(name="Most Recent Outgoing Transaction", value=outgoing, inline=False)
                    
                    embeds.append(embed2)

                # Send all embeds
                await interaction.response.send_message(embeds=embeds, ephemeral=True)
            
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}. Please try again later or contact support if the issue persists."
                await interaction.response.send_message(error_message, ephemeral=True)

        @self.tree.command(name="pf_rewards", description="Show your recent Post Fiat rewards")
        async def pf_rewards(interaction: discord.Interaction):
            user_id = interaction.user.id
            
            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /pf_store_seed before viewing rewards.",
                    ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]
            print(f"MyClient.pf_rewards: Spawning wallet to fetch rewards for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address

            # Defer the response to avoid timeout for longer operations
            await interaction.response.defer(ephemeral=True)

            try:
                memo_history = generic_pft_utilities.get_account_memo_history(wallet_address).copy().sort_values('datetime')
                reward_summary_map = generic_pft_utilities.get_reward_data(all_account_info=memo_history)
                recent_rewards = generic_pft_utilities.format_reward_summary(reward_summary_map['reward_summaries'].tail(10))

                # Split the message into chunks if it's too long
                chunks = []
                while len(recent_rewards) > 0:
                    if len(recent_rewards) > 1900:
                        chunk = recent_rewards[:1900]
                        last_newline = chunk.rfind('\n')
                        if last_newline != -1:
                            chunk = recent_rewards[:last_newline]
                        chunks.append(f"```\n{chunk}\n```")
                        recent_rewards = recent_rewards[len(chunk):]
                    else:
                        chunks.append(f"```\n{recent_rewards}\n```")
                        recent_rewards = ""

                # Send the first chunk
                await interaction.followup.send(chunks[0], ephemeral=True)

                # Send the rest of the chunks
                for chunk in chunks[1:]:
                    await interaction.followup.send(chunk, ephemeral=True)

            except Exception as e:
                await interaction.followup.send(f"An error occurred while fetching your rewards: {str(e)}", ephemeral=True)

        @self.tree.command(name="pf_final_verification", description="Submit final verification for a task")
        async def pf_final_verification(interaction: discord.Interaction):
            user_id = interaction.user.id

            # Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /store_seed before submitting final verification.", ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]

            # Fetch the tasks that are in the verification queue
            print(f"MyClient.pf_final_verification: Spawning wallet to fetch tasks for {interaction.user.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            wallet_address = wallet.classic_address
            memo_history = generic_pft_utilities.get_account_memo_history(wallet_address).copy()
            outstanding_verification = generic_pft_utilities.get_verification_df(account_memo_detail_df=memo_history)
            
            # If there are no tasks in the verification queue, notify the user
            if outstanding_verification.empty:
                await interaction.response.send_message("You have no tasks in the verification queue.", ephemeral=True)
                return

            # Create dropdown options based on the tasks in the verification queue
            options = [
                SelectOption(label=task_id, description=memo_data.replace(constants.TaskType.VERIFICATION_PROMPT.value,'')[:100], value=task_id)
                for task_id, memo_data in zip(outstanding_verification['memo_type'], outstanding_verification['memo_data'])
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task for final verification", options=options)

            # Define the modal for inputting the verification justification
            class VerificationModal(Modal):
                def __init__(self, task_id: str, task_text: str, seed: str, user_name: str):
                    super().__init__(title="Submit Final Verification")
                    self.task_id = task_id
                    self.seed = seed
                    self.user_name = user_name
                    
                    # Add a label to display the full task description
                    self.task_description = discord.ui.TextInput(
                        label="Task Description (Do not modify)",
                        default=task_text,
                        style=discord.TextStyle.paragraph,
                        required=False
                    )
                    self.add_item(self.task_description)
                    
                    # Add the verification justification input
                    self.verification_justification = TextInput(
                        label="Verification Justification", 
                        placeholder="Explain how you verified the task completion",
                        style=discord.TextStyle.paragraph
                    )
                    self.add_item(self.verification_justification)

                async def on_submit(self, interaction: discord.Interaction):
                    # Defer the response to avoid interaction timeout
                    await interaction.response.defer(ephemeral=True)
                    
                    justification_string = self.verification_justification.value
                    
                    # Call the discord__final_submission function
                    output_string = post_fiat_task_generation_system.discord__final_submission(
                        seed_to_work=self.seed,
                        user_name=self.user_name,
                        task_id_to_submit=self.task_id,
                        justification_string=justification_string
                    )
                    
                    # Send a follow-up message with the result
                    await interaction.followup.send(output_string, ephemeral=True)

            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = outstanding_verification[outstanding_verification['memo_type'] == selected_task_id]['memo_data'].values[0]
                # Open the modal to get the verification justification with the task text pre-populated
                await interaction.response.send_modal(VerificationModal(
                    task_id=selected_task_id,
                    task_text=task_text,
                    seed=seed,
                    user_name=interaction.user.name
                ))

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.response.send_message("Please choose a task for final verification:", view=view, ephemeral=True)

    async def on_ready(self):
        print(f'MyClient.on_ready: Logged in as {self.user} (ID: {self.user.id})')
        print('MyClient.on_ready: ------------------------------')
        print('MyClient.on_ready: Connected to the following guilds:')
        for guild in self.guilds:
            print(f'- {guild.name} (ID: {guild.id})')

        # Optionally, re-sync slash commands in all guilds
        await self.tree.sync()
        print('MyClient.on_ready: Slash commands synced across all guilds.')


    async def send_message_chunks(self, channel, message, user):
        max_chunk_size = 1900
        max_chunks = 5

        # Split the message into chunks
        chunks = []
        current_chunk = ""
        for line in message.split("\n"):
            if len(current_chunk) + len(line) + 1 <= max_chunk_size:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Send the message chunks
        for i, chunk in enumerate(chunks[:max_chunks], start=1):
            formatted_chunk = f"```\n{chunk}\n```"
            if i == 1:
                await channel.send(f"{user.mention}\n{formatted_chunk}")
            else:
                await channel.send(formatted_chunk)
            await asyncio.sleep(1)

        # Send a message if there are more chunks than the maximum allowed
        if len(chunks) > max_chunks:
            remaining_chunks = len(chunks) - max_chunks
            await channel.send(f"... ({remaining_chunks} more chunk(s) omitted)")

    async def send_long_message_to_channel(self, channel, long_message):
        while len(long_message) > 0:
            if len(long_message) > 1999:
                cutoff = long_message[:1999].rfind(' ')  # Find last space within limit
                if cutoff == -1:
                    cutoff = 1999  # No spaces found; cut off at limit
                to_send = long_message[:cutoff]
                long_message = long_message[cutoff:]
            else:
                to_send = long_message
                long_message = ''
            await channel.send(to_send)
        
    async def send_long_message(self, message, long_message):
        sent_messages = []
        while len(long_message) > 0:
            if len(long_message) > 1999:
                cutoff = long_message[:1999].rfind(' ')  # Find last space within limit
                if cutoff == -1:
                    cutoff = 1999  # No spaces found; cut off at limit
                to_send = long_message[:cutoff]
                long_message = long_message[cutoff:]
            else:
                to_send = long_message
                long_message = ''
            sent_message = await message.reply(to_send, mention_author=True)
            sent_messages.append(sent_message)
        return sent_messages

    async def send_long_message_then_delete(self, message, long_message, delete_after):
        sent_messages = await self.send_long_message(message, long_message)
        await asyncio.sleep(delete_after)
        for sent_message in sent_messages:
            try:
                await sent_message.delete()
            except discord.errors.NotFound:
                pass  # Message was already deleted
        try:
            await message.delete()
        except discord.errors.NotFound:
            pass  # Message was already deleted

    async def send_long_escaped_message(self, message, long_message):
        sent_messages = []
        chunk_size = 1950  # Reduced to account for added formatting

        # Split the message into chunks, preserving newlines
        chunks = []
        current_chunk = ""
        for line in long_message.split('\n'):
            if len(current_chunk) + len(line) + 1 > chunk_size:
                chunks.append(current_chunk)
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'
        if current_chunk:
            chunks.append(current_chunk)

        # Send chunks with proper formatting
        for chunk in chunks:
            to_send = f"```\n{chunk.rstrip()}\n```"

            # Ensure we don't exceed Discord's message limit
            if len(to_send) > 2000:
                to_send = to_send[:1997] + "```"

            sent_message = await message.reply(to_send, mention_author=True)
            sent_messages.append(sent_message)

        return sent_messages

    async def check_and_notify_new_transactions(self):
        CHANNEL_ID = self.network_config.discord_activity_channel_id
        channel = self.get_channel(CHANNEL_ID)
        
        if not channel:
            print(f"MyClient.check_and_notify_new_transactions: ERROR: Channel with ID {CHANNEL_ID} not found.")
            return

        # Call the function to get new messages and update the database
        messages_to_send = post_fiat_task_generation_system.sync_and_format_new_transactions()

        # DEBUGGING
        len_messages_to_send = len(messages_to_send)
        if len_messages_to_send > 0:
            print(f"MyClient.check_and_notify_new_transactions: Sending {len_messages_to_send} messages to the Discord channel") 

        # Send each new message to the Discord channel
        for message in messages_to_send:
            await channel.send(message)

        # except Exception as e:
        #     print(f"An error occurred while checking for new transactions: {str(e)}")

    async def transaction_checker(self):
        await self.wait_until_ready()
        while not self.is_closed():
            await self.check_and_notify_new_transactions()
            await asyncio.sleep(15)  # Check every 60 seconds

    async def death_march_reminder(self):
        await self.wait_until_ready()
        
        # Wait for 10 seconds after server start (for testing, change back to 30 minutes in production)
        await asyncio.sleep(30)
        
        target_user_id = 402536023483088896  # The specific user ID
        channel_id = 1229917290254827521  # The specific channel ID
        
        est_tz = pytz.timezone('US/Eastern')
        start_time = time(6, 30)  # 6:30 AM
        end_time = time(21, 0)  # 9:00 PM
        
        while not self.is_closed():
            try:
                now = datetime.now(est_tz).time()
                if start_time <= now <= end_time:
                    channel = self.get_channel(channel_id)
                    if channel:
                        if target_user_id in self.user_seeds:
                            seed = self.user_seeds[target_user_id]
                            print(f"MyClient.death_march_reminder: Spawning wallet to fetch info for {target_user_id}")
                            user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                            user_address = user_wallet.classic_address
                            tactical_string = self.post_fiat_task_generation_system.get_o1_coaching_string_for_account(user_address)
                            
                            # Send the message to the channel
                            await self.send_long_message_to_channel(channel, f"<@{target_user_id}> Death March Update:\n{tactical_string}")
                        else:
                            print(f"No seed found for user {target_user_id}")
                    else:
                        print(f"Channel with ID {channel_id} not found")
                else:
                    print("Outside of allowed time range. Skipping Death March reminder.")
            except Exception as e:
                print(f"An error occurred in death_march_reminder: {str(e)}")
            
            # Wait for 30 minutes before the next reminder (10 seconds for testing)
            await asyncio.sleep(30*60)  # Change to 1800 (30 minutes) for production
            
    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        
        #if message.author.id != 402536023483088896: 
            #print('IT IS ALEX')# Check if the user ID matches goodalexander's ID
        #    return

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
            op_df = self.open_ai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
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
                    generic_pft_utilities = GenericPFTUtilities(node_name=constants.NODE_NAME)
                    print(f"MyClient.tactics: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    memo_history = generic_pft_utilities.get_account_memo_history(user_wallet.classic_address)
                    full_user_context = generic_pft_utilities.get_full_user_context_string(user_wallet.classic_address, memo_history=memo_history)
                    
                    open_ai_request_tool = OpenAIRequestTool()
                    
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
                    
                    writable_df = open_ai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
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
                    print(f"MyClient.coach: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    wallet_address = user_wallet.classic_address

                    # Check PFT balance
                    pft_balance = self.generic_pft_utilities.get_account_pft_balance(wallet_address)
                    if pft_balance < 25000:
                        await message.reply(
                            f"You need at least 25,000 PFT to use the coach command. Your current balance is {pft_balance:,.2f} PFT.", 
                            mention_author=True
                        )
                        return

                    # Get user's full context
                    memo_history = self.generic_pft_utilities.get_account_memo_history(account_address=wallet_address)
                    full_context = self.generic_pft_utilities.get_full_user_context_string(account_address=wallet_address, memo_history=memo_history)
                    
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
                    response = self.open_ai_request_tool.o1_preview_simulated_request(
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
                    error_msg = f"An error occurred while processing your request: {str(e)}"
                    await message.reply(error_msg, mention_author=True)
            else:
                await message.reply("You must store a seed using !store_seed before using the coach.", mention_author=True)

        if message.content.startswith('!blackprint'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    print(f"MyClient.blackprint: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    #full_user_context = self.generic_pft_utilities.get_full_user_context_string(user_wallet.classic_address)
                    tactical_string = self.post_fiat_task_generation_system.generate_coaching_string_for_account(user_address)
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)

        if message.content.startswith('!deathmarch'):
            ## FOR REPEATED RESPONSE TARGET THIS AT USER ID 402536023483088896
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    print(f"MyClient.deathmarch: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    #full_user_context = self.generic_pft_utilities.get_full_user_context_string(user_wallet.classic_address)
                    tactical_string = self.post_fiat_task_generation_system.get_o1_coaching_string_for_account(user_address)
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)

        if message.content.startswith('!redpill'):
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                
                try:
                    print(f"MyClient.redpill: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    #full_user_context = self.generic_pft_utilities.get_full_user_context_string(user_wallet.classic_address)
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
                    print(f"MyClient.docrewrite: Spawning wallet to fetch info for {message.author.name}")
                    user_wallet = self.generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
                    user_address = user_wallet.classic_address
                    #full_user_context = self.generic_pft_utilities.get_full_user_context_string(user_wallet.classic_address)
                    tactical_string = self.post_fiat_task_generation_system.generate_document_rewrite_instructions(user_address)
                    
                    await self.send_long_message(message, tactical_string)
            
                except Exception as e:
                    error_message = f"An error occurred while generating tactical advice: {str(e)}"
                    await message.reply(error_message, mention_author=True)
            else:
                await message.reply("You must store a seed using /pf_store_seed before getting tactical advice.", mention_author=True)


        if message.content.startswith('!new_wallet'):
            wallet_maker =  generic_pft_utilities.create_xrp_wallet()
            await self.send_long_message_then_delete(message, wallet_maker, delete_after=60)

        if message.content.startswith('!wallet_info'):
            wallet_to_get = message.content.replace('!wallet_info','').strip()
            account_info = generic_pft_utilities.generate_basic_balance_info_string_for_account_address(account_address=wallet_to_get)
            await self.send_long_message(message, account_info)

        if message.content.startswith('!store_seed'):
            # Extract the seed from the message
            seed = message.content.replace('!store_seed', '').strip()
            # Store the seed for the user
            self.user_seeds[user_id] = seed
            await message.reply("Seed stored successfully.", mention_author=True)

        if message.content.startswith('!show_seed'):
            # Retrieve and show the stored seed for the user
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                await self.send_long_message_then_delete(message, f"Your stored seed is: {seed}", delete_after=30)
            else:
                await message.reply("No seed found for your account.", mention_author=True)

        if message.content.startswith('!pf_task'):
            if user_id in self.user_seeds:
                message_to_send = message.content.replace('!pf_task', '').strip()
                task_id = generic_pft_utilities.generate_custom_id()
                user_name = message.author.name
                memo_to_send = generic_pft_utilities.construct_standardized_xrpl_memo(memo_data=message_to_send, memo_format = user_name, memo_type=task_id)
                seed = self.user_seeds[user_id]
                response = post_fiat_task_generation_system.discord__send_postfiat_request(user_request= message_to_send, user_name=user_name, seed=seed)
                transaction_info = generic_pft_utilities.extract_transaction_info_from_response_object(response=response)
                clean_string = transaction_info['clean_string']
                await self.send_long_message(message, f"Task Requested with Details {clean_string}")
            else:
                await message.reply("You must store a seed before generating a task.", mention_author=True)

        if message.content.startswith('!my_wallet'):
            # Retrieve and show the stored seed for the user
            if user_id in self.user_seeds:
                seed = self.user_seeds[user_id]
                print(f"MyClient.my_wallet: Spawning wallet to fetch info for {message.author.name}")
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
                wallet_address = wallet.address
                account_info = generic_pft_utilities.generate_basic_balance_info_string_for_account_address(account_address=wallet_address)
                await self.send_long_message(message, f"Based on your seed your linked {account_info}")
            else:
                await message.reply("No seed found for your account.", mention_author=True)


        if message.content.startswith('!pf_initiate'):
            # check that xrp wallet is funded
            
            if user_id not in self.user_seeds:
                await message.reply("You must store a seed before initiating.", mention_author=True)
                return
            seed = self.user_seeds[user_id]
            print(f"MyClient.pf_initiate: Spawning wallet to initiate for {message.author.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address
            xrp_balance = generic_pft_utilities.get_xrp_balance(address=wallet_address)
            if xrp_balance < constants.MIN_XRP_BALANCE:
                await message.reply(f"You must fund your wallet with at least {constants.MIN_XRP_BALANCE} XRP before initiating.", mention_author=True)
                return

            memo_history = generic_pft_utilities.get_account_memo_history(account_address=wallet_address,pft_only=False)
            if len(memo_history[memo_history['memo_type']=="INITIATION_RITE"]) > 0:
                await message.reply("You have already performed an initiation rite with this wallet.", mention_author=True)
                return

        if message.content.startswith('!pf_outstanding'):
            seed = self.user_seeds[user_id]
            if user_id not in self.user_seeds:
                await message.reply("You must store a seed before getting outstanding tasks.", mention_author=True)
                return
            print(f"MyClient.pf_outstanding: Spawning wallet to fetch tasks for {message.author.name}")
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address
            output_message = generic_pft_utilities.create_full_outstanding_pft_string(account_address=wallet_address)
            await self.send_long_escaped_message(message, output_message)

def init_bot():
    """Initialize and return the Discord bot with required intents"""
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guild_messages = True
    return MyClient(intents=intents)

def init_services():
    """Initialize and return core services"""
    open_ai_request_tool = OpenAIRequestTool()
    post_fiat_task_generation_system = PostFiatTaskGenerationSystem()
    generic_pft_utilities = GenericPFTUtilities(node_name=constants.get_network_config().node_name)
    generic_pft_utilities.run_transaction_history_updates()

    return (
        open_ai_request_tool,
        post_fiat_task_generation_system,
        generic_pft_utilities
    )

if __name__ == "__main__":
    # Initialize credential manager
    password = getpass.getpass("Enter your password: ")
    cred_manager = CredentialManager(password)

    # Initialize performance monitor
    monitor = PerformanceMonitor(time_window=60)
    monitor.start()
    
    # Network selection with more context
    print(f"Network Configuration: ")
    for idx, network in enumerate(constants.Network):
        print(f"{idx+1}. {network.value.name}")
    network_choice = input("Select network (1/2) [default=2]: ").strip() or "2"
    selected_network = constants.Network.XRPL_MAINNET if network_choice == "1" else constants.Network.XRPL_TESTNET
    constants.USE_TESTNET = selected_network == constants.Network.XRPL_TESTNET

    # Local node configuration with network-specific context
    network_config = constants.get_network_config(network=selected_network)
    if network_config.local_rpc_url:
        print(f"\nLocal node configuration:")
        print(f"Local {network_config.name} node URL: {network_config.local_rpc_url}")
        use_local = input("Do you have a local node configured? (y/n) [default=n]: ").strip() or "n"
        constants.HAS_LOCAL_NODE = use_local == "y"
    else:
        print(f"\nNo local node configuration available for {network_config.name}")
        constants.HAS_LOCAL_NODE = False
    
    print(f"\nInitializing services for {network_config.name}...")
    print(f"Using {'local' if constants.HAS_LOCAL_NODE else 'public'} endpoints...")

    # Initialize services
    open_ai_request_tool, post_fiat_task_generation_system, generic_pft_utilities = init_services()

    print("---Services initialized successfully!---")

    # Initialize and run the bot
    client = init_bot()
    client.run(cred_manager.get_credential('discordbot_secret'))