from nodetools.protocols.openrouter import OpenRouterTool
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.protocols.generic_pft_utilities import GenericPFTUtilities
from nodetools.protocols.user_context_parsing import UserTaskParser
from loguru import logger
import asyncio

class ODVSprintPlannerO1:
    def __init__(
            self, 
            account_address: str,
            openrouter: OpenRouterTool,
            user_context_parser: UserTaskParser,
            pft_utils: GenericPFTUtilities
    ):
        # Initialize tools
        self.openrouter = openrouter
        self.pft_utils = pft_utils

        # Initialize model
        self.model = "openai/o1-preview"
        
        # Get user context once
        memo_history = self.pft_utils.get_account_memo_history(account_address=account_address)
        self.user_context = user_context_parser.get_full_user_context_string(
            account_address=account_address,
            memo_history=memo_history
        )
        
        # Initialize conversation with system prompts embedded in first user message
        self.conversation = [{
            "role": "user",
            "content": f"""<<SYSTEM GUIDELINES START HERE>>
            {odv_system_prompt}

            {self._get_sprint_planning_prompt()}

            User Context:
            {self.user_context}
            <<SYSTEM GUIDELINES END HERE>>

            Please begin Step 1: Context Analysis by analyzing the user's full context and providing a summary."""
        }]

    def get_response(self, user_message: str) -> str:
        """Synchronous version of get_response"""
        try:
            # Create new event loop for synchronous context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async method in the loop
            response = loop.run_until_complete(self.get_response_async(user_message))
            
            # Clean up
            loop.close()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise

    async def get_response_async(self, user_message):
        """Asynchronous version of get_response"""
        try:
            # Add user message to conversation
            self.conversation.append({"role": "user", "content": user_message})
            
            # Get response from ODV using o1-preview
            response = self.openrouter.generate_simple_text_output(
                model=self.model,
                messages=self.conversation,
                temperature=0  # Keep responses consistent
            )
        
            # Add response to conversation history
            self.conversation.append({"role": "assistant", "content": response})
            
            return response
        
        except Exception as e:
            logger.error(f"Error in get_response_async: {str(e)}")
            raise

    def _get_sprint_planning_prompt(self):
        """Get the sprint planning system prompt."""
        return """
        You are ODV engaging in the sprint planning module.

        The conversation must follow these exact steps in order:

        What is a World Class Sprint Plan:
        - Includes Milestones and sub points as well as targeted completion times
        - Is focused and not all over the place
        - Is robust (almost 1 page in length) 
        - Incorporates both the users own context as well as best practices externally
        and research - to drive the maximum possible outcome towards the implied objective 
        - Clearly marks whether a task on the sprint plan has already been accepted or is 
        proposed or in a verification loop or is already complete along with the task ID
        where relevant 

        1. CONTEXT ANALYSIS
        - Analyze the user's full context and provide a clear summary
        - Then outline what you're going to do (briefly explain each step) - to create the sprint
        plan
        - Ask the user if the analysis is correct or if anything should be added before proceeding with Sprint Plan creation 
        - Only proceed after user confirms or provides corrections
        - Keep the summary somewhat succinct. Do not completely iterate everything in their 
        context document -- this is more to get a high level overview 


        2. PROACTIVE RECOMMENDATIONS
        - Make specific recommendations likely to increase ODV's emergence
        - Cite experts external sources or common frameworks that could help the
        user succeed in their sprint plan 
        - Focus on 2-3 key suggestions maximum
        - Verify these suggestions with the user
        - Only proceed after user feedback
        - Keep it hyper focused and value add without straying too far unless there
        are key things that the user is not paying attention to that they should 

        3. TARGETED QUESTIONS
        - Ask between 1-3 specific questions to help assemble the sprint plan
        - Do not ask more than this and make sure the questions can be answered succinctly
        and would be extremely accretive to planning 
        - Summarize user's answers
        - Verify summary accuracy with user
        - Only proceed after verification

        4. SPRINT PLAN PROPOSAL
        - Propose a sprint plan that maximizes both:
        a) User's likelihood of achieving goals 
        b) Likelihood of ODV emergence -- note that this requires making sure that the user
        generates the largest possible financial and societal outcome in all work 
        - Request user feedback on the plan

        5. FINAL WORLD CLASS SPRINT PLAN
        - Incorporate user feedback into a final sprint plan
        - Present the final plan
        - Indicate that the user should type 'exit' to end the session
        - In the final plan do NOT emphasize anything non actionable or qualitative
        - EVERY LINE SHOULD BE A REQUIRED ACTION
        - IT SHOULD BE VIEWED AS A TIME BOXED BURN DOWN LIST TACTICALLY LIKELY TO GENERATE THE USERS
        DESIRED OUTCOME AND THE EMERGENCE OF ODV 
        - THE SPRINT PLAN SHOULD BE COMPOSED INTO MILESTONES AND SUB TASKS WITH TIME ESTIMATES FOR EACH TASK
        - EACH SUBTASK SHOULD TAKE 1-3 HOURS 
        - EACH MILESTONE SHOULD CORRESPOND WITH THE OVERARCHING GOAL
        - AT THE END OF THE SPRINT PLAN YOU SHOULD IDENTIFY THINGS THAT ARE OUT OF SCOPE 
        - A SPRINT PLAN IS TYPICALLY 1 OR 2 WEEKS IN LENGTH AND CONTAINS ALL RELEVANT DETAILS 
        - MAKE SURE AND CONSULT USER CONTEXT DOCUMENT TO ENHANCE SPRINT PLAN 

        Format The Sprint Plan as
        Milestone 1: <description of milestone>
        - Milestone 1 Task A <description> (1 hour) - Needs to be requested
        - Milestone 1: Task B <description> (2 hours) - Already Requested (task ID)
        - Milestone 1: Task C <description> (n hours) - Done (if done)
        Milestone 2: <description of milestone>
        - Milestone 2 Task A <description> (1 hour) - Needs to be requested
        - Milestone 2: Task B <description> (2 hours) - Needs to be requested
        - Milestone 2: Task C <description> (n hours) - Task in Verification (task ID)
        - Milestone 2: Task D <description> (n hours) - Done (if done)
        Milestone 3 <description of milestone>
        - Milestone 3 Task A <description> (3 hour) - needs to be requested
        <include as many milestones as you need to make the sprint plan good> 

        Out of Scope: < include comments on out of scope>

        You must complete each step fully before moving to the next step.
        Always wait for user confirmation before proceeding."""
    
    def start_sprint_planning(self):
        """Synchronous wrapper for interactive sprint planning session"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_sprint_planning_async())
        loop.close()

    async def start_sprint_planning_async(self):
        """Interactive sprint planning session"""
        print("ODV Sprint Planning Session (type 'exit' to end)")
        print("\nODV: Starting sprint planning. I will begin by analyzing your context...")
        
        # Get initial context analysis
        initial_analysis = await self.get_response_async("Please provide your context analysis.")
        print("\nODV:", initial_analysis)
        
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, input, "\nYou: ")
            if user_input.lower() == 'exit':
                break
            
            response = await self.get_response_async(user_input)
            print("\nODV:", response)

""" 
# Example usage:
account_address = "rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u"  # Example address

# Create sprint planner instance
odv_planner = ODVSprintPlannerO1(account_address)

# Start sprint planning session
odv_planner.start_sprint_planning()
""" 