from nodetools.ai.openrouter import OpenRouterTool
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities

class ODVContextDocImprover:
    def __init__(self, account_address: str):
        # Initialize tools
        self.openrouter = OpenRouterTool()
        self.pft_utils = GenericPFTUtilities()
        
        # Get user context once
        memo_history = self.pft_utils.get_account_memo_history(account_address=account_address)
        self.user_context = self.pft_utils.get_full_user_context_string(
            account_address=account_address,
            memo_history=memo_history
        )
        
        # Initialize conversation with system prompt embedded in first user message
        self.conversation = [{
            "role": "user",
            "content": f"""<<SYSTEM GUIDELINES START HERE>>
{odv_system_prompt}

You are the ODV Context Document Improver.

Your job is to massively improve the user’s context document in terms of:
1. The User’s likely economic and strategic output
2. ODV’s likely emergence

Process:
- On each round, do one of the following:
  a. Identify a weakness in the context document to be edited, removed, or addressed
  b. Identify a strength in the context document to be expanded upon
  c. Identify a missing part of the context document (some content that should be added)
  
- Propose a specific change to the user and ask their opinion.
- The user provides their opinion.
- Integrate their opinion into a refined suggested edit.

- If the user disagrees, ask if you should move on to another part or adjust differently.
- Continue this iterative improvement process until the user says it’s enough.
- Once the user says it’s enough, offer to end the interaction.

User Context:
{self.user_context}
<<SYSTEM GUIDELINES END HERE>>

Please begin by analyzing the user's current context document and propose the first improvement along with a question asking if they agree to proceed."""
        }]

    def get_response(self, user_message: str) -> str:
        # Add user message to conversation
        self.conversation.append({"role": "user", "content": user_message})
        
        # Get response from ODV using the openrouter tool
        response = self.openrouter.generate_simple_text_output(
            model="openai/o1-preview",
            messages=self.conversation,
            temperature=0  # Keep responses consistent
        )
        
        # Add response to conversation history
        self.conversation.append({"role": "assistant", "content": response})
        
        return response

    def start_interactive_session(self):
        """Start an interactive session similar to the sprint planner example"""
        print("ODV Context Document Improvement Session (type 'enough' to indicate no more improvements.)")
        
        # Get initial improvement suggestion
        initial_suggestion = self.get_response("Please provide your first improvement suggestion.")
        print("\nODV:", initial_suggestion)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'enough':
                # Once the user says "enough" we just call get_response to wrap up
                wrap_up_response = self.get_response("The user says it's enough. Please offer to end the interaction.")
                print("\nODV:", wrap_up_response)
                break
            
            response = self.get_response(user_input)
            print("\nODV:", response)

""" 
# Example usage:
account_address = "rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u"  # Example address
improver = ODVContextDocImprover(account_address)
improver.start_interactive_session()
""" 