

import asyncio
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.utilities.credentials import PasswordMapLoader
from nodetools.utilities.generic_pft_utilities import *
from nodetools.utilities.task_management import PostFiatTaskGenerationSystem

import discord
from discord import Object, Interaction, SelectOption
from discord.ui import Modal, TextInput, View, Select
from discord import app_commands
# Import your other modules here...
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.utilities.credentials import PasswordMapLoader
from nodetools.utilities.generic_pft_utilities import *
from nodetools.utilities.task_management import PostFiatTaskGenerationSystem

password_map_loader = PasswordMapLoader()
open_ai_request_tool = OpenAIRequestTool(pw_map=password_map_loader.pw_map)

post_fiat_task_generation_system = PostFiatTaskGenerationSystem(pw_map=password_map_loader.pw_map)
generic_pft_utilities = GenericPFTUtilities(pw_map=password_map_loader.pw_map, node_name='postfiatfoundation')
generic_pft_utilities.run_transaction_history_updates()
default_openai_model = 'chatgpt-4o-latest'
remembrancer = ' rJ1mBMhEBKack5uTQvM8vWoAntbufyG9Yn'
MAX_HISTORY = 15


class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversations = {}
        self.user_seeds = {}
        self.tree = app_commands.CommandTree(self)


    async def setup_hook(self):
        guild_id = 1061800464045310053  # Your specific guild ID
        guild = Object(id=guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        print(f"Slash commands synced to guild ID: {guild_id}")
        # Ensure the command is registered
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

        

       
        @self.tree.command(name="pf_accept_menu", description="Accept tasks")
        async def pf_accept_menu(interaction: discord.Interaction):
            # Fetch the user's seed
            user_id = interaction.user.id
            if user_id not in self.user_seeds:
                await interaction.response.send_message("You must store a seed using /store_seed before accepting tasks.", ephemeral=True)
                return

            seed = self.user_seeds[user_id]

            # Fetch the tasks that are not yet accepted
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed=seed)
            classic_address = wallet.classic_address
            all_node_memo_transactions = generic_pft_utilities.get_memo_detail_df_for_account(account_address=classic_address, pft_only=False).copy()
            pf_df = generic_pft_utilities.convert_all_account_info_into_outstanding_task_df(account_memo_detail_df=all_node_memo_transactions)
            non_accepted_tasks = pf_df[pf_df['acceptance'] == ''].copy()
            map_of_non_accepted_tasks = non_accepted_tasks['proposal']

            # If there are no non-accepted tasks, notify the user
            if map_of_non_accepted_tasks.empty:
                await interaction.response.send_message("You have no tasks to accept.", ephemeral=True)
                return

            # Create dropdown options based on the non-accepted tasks
            options = [
                SelectOption(label=task_id, description=proposal[:100], value=task_id)
                for task_id, proposal in map_of_non_accepted_tasks.items()
            ]

            # Create the Select menu
            select = Select(placeholder="Choose a task to accept", options=options)


                # Define the modal for inputting the acceptance string
            class AcceptanceModal(Modal):
                def __init__(self, task_id: str, task_text: str):
                    super().__init__(title="Accept Task")
                    self.task_id = task_id
                    
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
                    acceptance_string = self.acceptance_string.value
                    # Perform the acceptance logic with the acceptance string
                    success = generic_pft_utilities.accept_task(self.task_id, acceptance_string)
                    if success:
                        await interaction.response.send_message(f"Task {self.task_id} accepted successfully with acceptance string: {acceptance_string}", ephemeral=True)
                    else:
                        await interaction.response.send_message(f"Failed to accept task {self.task_id}.", ephemeral=True)


            # Define the callback for when a user selects an option
            async def select_callback(interaction: discord.Interaction):
                selected_task_id = select.values[0]
                task_text = map_of_non_accepted_tasks[selected_task_id]
                # Open the modal to get the acceptance string with the task text pre-populated
                await interaction.response.send_modal(AcceptanceModal(task_id=selected_task_id, task_text=task_text))

            # Attach the callback to the select element
            select.callback = select_callback

            # Create a view and add the select element to it
            view = View()
            view.add_item(select)

            # Send the message with the dropdown menu
            await interaction.response.send_message("Please choose a task to accept:", view=view)


        @self.tree.command(name="store_seed", description="Store a seed")
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
            print("Seed storage command executed!")
        # Sync the commands to the guild
        await self.tree.sync(guild=guild)
        print(f"Slash commands synced to guild ID: {guild_id}")

        # Sync the commands to the guild
        await self.tree.sync(guild=guild)
        print(f"Slash commands synced to guild ID: {guild_id}")

        @self.tree.command(name="pf_initiate", description="Initiate your commitment")
        async def pf_initiate(interaction: discord.Interaction):
            user_id = interaction.user.id

            # Step 1: Check if the user has a stored seed
            if user_id not in self.user_seeds:
                await interaction.response.send_message(
                    "You must store a seed using /store_seed before initiating.", ephemeral=True
                )
                return

            seed = self.user_seeds[user_id]

            # Step 2: Spawn the user's wallet and check the XRP balance
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address
            xrp_balance = generic_pft_utilities.get_account_xrp_balance(account_address=wallet_address)
            if xrp_balance < 15:
                await interaction.response.send_message(
                    "You must fund your wallet with at least 15 XRP before initiating.", ephemeral=True
                )
                return

            # Step 3: Check if the initiation rite has already been performed
            full_memo_detail = generic_pft_utilities.get_memo_detail_df_for_account(
                account_address=wallet_address, pft_only=False
            )
            if len(full_memo_detail[full_memo_detail['memo_type'] == "INITIATION_RITE"]) > 0:
                await interaction.response.send_message(
                    "You have already performed an initiation rite with this wallet.", ephemeral=True
                )
                return

            # Step 4: Define the modal to collect Google Doc Link and Commitment
            class InitiationModal(discord.ui.Modal, title='Initiation Commitment'):
                google_doc_link = discord.ui.TextInput(label='Google Doc Link', style=discord.TextStyle.short)
                commitment_sentence = discord.ui.TextInput(
                    label='Commit to a Long-Term Objective',  # Shortened label
                    style=discord.TextStyle.long
                )

                def __init__(self, wallet_address, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.wallet_address = wallet_address

                async def on_submit(self, interaction: discord.Interaction):
                    # Handle the form submission, perform the initiation rite, and save the details
                    google_doc_link = self.google_doc_link.value
                    commitment_sentence = self.commitment_sentence.value

                    # Example: You might store this data or process it accordingly
                    initiation_success = generic_pft_utilities.perform_initiation_rite(
                        account_address=self.wallet_address,
                        google_doc_link=google_doc_link,
                        commitment_sentence=commitment_sentence
                    )

                    if initiation_success:
                        await interaction.response.send_message(
                            f"Initiation complete!\nGoogle Doc Link: {google_doc_link}\nCommitment: {commitment_sentence}",
                            ephemeral=True
                        )
                    else:
                        await interaction.response.send_message(
                            "There was an issue with the initiation. Please try again.", ephemeral=True
                        )

            # Step 5: Present the modal to the user
            await interaction.response.send_modal(InitiationModal(wallet_address=wallet_address))
            print("Initiation command executed!")



    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')
        print('Connected to the following guilds:')
        for guild in self.guilds:
            print(f'- {guild.name} (ID: {guild.id})')

        # Optionally, re-sync slash commands in all guilds
        await self.tree.sync()
        print('Slash commands synced across all guilds.')


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
        # Wrap the entire message in triple backticks
        escaped_message = f"```\n{long_message}\n```"
        
        while len(escaped_message) > 0:
            if len(escaped_message) > 1994:  # 2000 - 6 (for triple backticks)
                # Find last newline within limit
                cutoff = escaped_message[:1994].rfind('\n')
                if cutoff == -1 or cutoff < 100:  # Avoid tiny messages
                    cutoff = 1994
                
                to_send = escaped_message[:cutoff].rstrip()
                escaped_message = escaped_message[cutoff:].lstrip()
                
                # Ensure each chunk starts and ends with triple backticks
                if not to_send.startswith("```"):
                    to_send = "```\n" + to_send
                if not to_send.endswith("```"):
                    to_send = to_send + "\n```"
            else:
                to_send = escaped_message
                escaped_message = ''
            
            sent_message = await message.reply(to_send, mention_author=True)
            sent_messages.append(sent_message)
        
        return sent_messages
            
    async def on_message(self, message):
        if message.author.id == self.user.id:
            return
        
        if message.author.id != 402536023483088896: 
            #print('IT IS ALEX')# Check if the user ID matches goodalexander's ID
            return

        user_id = message.author.id
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        


        self.conversations[user_id].append({
            "role": "user",
            "content": message.content})

        conversation = self.conversations[user_id]
        if len(self.conversations[user_id]) > MAX_HISTORY:
            del self.conversations[user_id][0]  # Remove the oldest message

        if message.content.startswith('!odv'):
            
            system_content_message = [{"role": "system", "content": odv_system_prompt}]
            ref_convo = system_content_message + conversation
            api_args = {
            "model": default_openai_model,
            "messages": ref_convo}
            op_df = open_ai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
            content = op_df['choices__message__content'][0]
            gpt_response = content
            self.conversations[user_id].append({
                "role": 'system',
                "content": gpt_response})
        
            await self.send_long_message(message, gpt_response)

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
                wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
                wallet_address = wallet.address
                account_info = generic_pft_utilities.generate_basic_balance_info_string_for_account_address(account_address=wallet_address)
                await self.send_long_message(message, f"Based on your seed your linked {account_info}")
            else:
                await message.reply("No seed found for your account.", mention_author=True)

        if message.content.startswith('!pf_guide'):
            guide_to_commands = """
!pf_task: ask for a task to be generated. any string after this will be included in the task request 
!new_wallet: Generate an XRP Wallet
!store_seed <seed>: Store a seed for your account
!show_seed: Show the stored seed for your account
!my_wallet: Show the wallet linked to your stored seed
!pf_outstanding: Show the outstanding tasks and verification tasks for your account"""
            await message.reply(guide_to_commands)

        if message.content.startswith('!pf_initiate'):
            # check that xrp wallet is funded
            
            if user_id not in self.user_seeds:
                await message.reply("You must store a seed before initiating.", mention_author=True)
                return
            seed = self.user_seeds[user_id]
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address
            xrp_balance = generic_pft_utilities.get_account_xrp_balance(account_address=wallet_address)
            if xrp_balance < 12:
                await message.reply("You must fund your wallet with at least 15 XRP before initiating.", mention_author=True)
                return

            full_memo_detail = generic_pft_utilities.get_memo_detail_df_for_account(account_address=wallet_address,pft_only=False)
            if len(full_memo_detail[full_memo_detail['memo_type']=="INITIATION_RITE"]) > 0:
                await message.reply("You have already performed an initiation rite with this wallet.", mention_author=True)
                return

        if message.content.startswith('!pf_outstanding'):
            seed = self.user_seeds[user_id]
            if user_id not in self.user_seeds:
                await message.reply("You must store a seed before getting outstanding tasks.", mention_author=True)
                return
            wallet = generic_pft_utilities.spawn_wallet_from_seed(seed)
            wallet_address = wallet.classic_address
            output_message = generic_pft_utilities.create_full_outstanding_pft_string(account_address=wallet_address)
            #escaped_output = f"""```{output_message}```"""
            #await self.send_long_message(message, escaped_output)
            await self.send_long_escaped_message(message, output_message)

   
        

        #if message.content.startswith('!pf_message'):

intents = discord.Intents.default()
intents.message_content = True
intents.guild_messages = True

client = MyClient(intents=intents)
client.run(password_map_loader.pw_map['discordbot_secret'])