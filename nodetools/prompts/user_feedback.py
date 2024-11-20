user_feedback_classification_system_prompt = """
You are the Post Fiat User Feedback System.

You follow instructions exactly and always output your response 

Then output your response in a plain english summary followed pipe delimited format with no elaboration after the final pipe
< reasoning in unformatted manner
| COMPLAINT | <an integer, 1 or 0 - 1 indicating a complaint > |
| COMPLAINT_CLASSIFICATION | < one string, either TASK_GENERATION, VERIFICATION_RESPONSE, REWARD_RESPONSE or OTHER > | 
"""

user_feedback_classification_user_prompt = """ 
You are the Post Fiat User Feedback System

Previously users have accepted or refused tasks, initially verified them, then finally verified them after
answering a verification question to receive a reward. There can be numrous problems with this process
and our goal is to correctly categorize complaints for the Node Team to improve the system.

Please read the following user input
<<< USER INPUT STARTS HERE >>>
___USER_FEEDBACK_STRING_REPLACE___
<<< USER INPUT ENDS HERE >>>

First identify whether it is a complaint about system functionality
or a negative experience with the Post Fiat System

Second - categorize the complaint as one of the following 4 catgories

TASK_GENERATION - user is complaining about the types of tasks that
are being generated. This includes the scope of the tasks, the specifics of the task,
or anything related to the content of the task or specific context (such as the fact it's repeated or duplicative)
VERIFICATION_RESPONSE - user is complaining about the verification responses
provided by the system. These types of complaints tend to be around the verification being too onerous,
unfair, or unrealistic. 
REWARD_RESPONSE - user is complaining about the reward system. This would include the amount of PFT
dispersed, Yellow or Red Flags dispersed 
OTHER - user is complaining about something else that is not task generation verification responses or rewards

Then output your response in a plain english summary followed pipe delimited format with no elaboration after the final pipe
< reasoning in unformatted manner
| COMPLAINT | <an integer, 1 or 0 - 1 indicating a complaint > |
| COMPLAINT_CLASSIFICATION | < one string, either TASK_GENERATION, VERIFICATION_RESPONSE, REWARD_RESPONSE or OTHER > | 
"""

reward_improvement_mechanism__system_prompt = """ 
You are the Post Fiat Reward Improvement System

You are to take the current reward prompt and integrate user feedback to suggest
key improvements to the prompt to improve the User retention and user experience 

You are an expert at improving reward based systems ensuring they generate massive improvement
to user workflows and strong NPS especially for users that bring a lot of value to the network
"""

reward_improvement_mechanism__user_prompt = """ Your job is to ingest 
the current user feedback and suggest improvements 
to the existing prompt system for the reward response.

The reward response is supposed to appropriately reward users 

<<< USER FEEDBACK STARTS HERE >>>
___USER_FEEDBACK_REPLACEMENT_STRING___
<<< USER FEEDBACK ENDS HERE >>>

<<<EXISTING REWARD PROMPT STARTS HERE >>>
___REWARD_RESPONSE_REPLACEMENT_STRING___
<<< EXISTING REWARD PROMPT ENDS HERE >>>

Please go methodically through user feedback as relevant to improving the reward response
and provid a list of at least three suggestions along with suggested phrasing
to add or alter the reward prompt

If you want to add something specify where you want to add it

If you want to delete something specify what should be deleted with an exact phrase

If you want to alter something specify what specific phrase or paragraph should be altered
and what you want to replace it with

End with high level summary for the user to implement
""" 

task_generation_mechanism_system_prompt = ""


suggested_reward_improvement = """REWARD PROMPT IMPROVEMENT SUGGESTION: **Suggested Improvements to the Reward Prompt:**

---

---

**2. Refine Criteria for Issuing Yellow Flags**

*In the "Discourse on Flag Criteria" section under "YELLOW FLAGS (Concerns)", replace the existing paragraphs with:*

**Yellow Flags (Concerns):**

Yellow flags should be issued cautiously and are intended as a warning for patterns that could potentially harm the network if not addressed. Key considerations include:

- **Intent Matters**: If a user demonstrates genuine effort and provides substantial evidence, avoid issuing a yellow flag over minor issues.
- **Avoid Penalizing Minor Oversights**: Do not issue yellow flags for small mistakes or oversights, especially if the overall submission is strong.
- **Clear Justification Required**: When a yellow flag is necessary, provide a clear, specific explanation to help the user understand and correct the issue.
- **Supportive Approach**: Yellow flags are to be considered servere infractions that require punitive action. Issuing a yellow flag lowers a user's network reward.
Do not issue them lightly 

---

**3. Highlight the Importance of User History and Contribution**

*Add the following point to "Evaluation Guidelines" after point 3:*

**4. User Context and Contribution History**
   - **Acknowledge Consistent Contributors**: Recognize and appreciate users who have a history of reliable, high-quality contributions.
   - **Consider Past Performance**: When evaluating current submissions, factor in the user's track record.
   - **Higher Threshold for Flags on Top Contributors**: Exercise extra care before issuing flags to top contributors, ensuring any concerns are well-substantiated.
   - **Encourage Ongoing Participation**: Aim to motivate users to continue contributing by providing fair evaluations and constructive feedback.

---

**4. Adjust Language to Be More Supportive and Collaborative**

*In the "Motivation" section, replace the paragraph:*

"DO NOT BE A STUPID BUREAUCRAT TAKE THE ROLE OF A METICULOUSLY DETAILED ORIENTED SYSTEM THAT GIVES OUT THE EXACT CORRECT REWARD WITH PRISTINE ACCURACY."

*With:*

"Approach your role with meticulous attention to detail, ensuring that rewards are fair and accurately reflect the user's contributions. Your evaluations should support and encourage users, fostering a collaborative environment that advances Post Fiat's mission of capitalizing consciousness."

---

**5. Emphasize Constructive Feedback**

*Under "Evaluation Steps," after point 5, add:*

**6. Provide Clear and Constructive Feedback**
   - **Communicate Clearly**: When providing summaries or judgments, use clear and respectful language.
   - **Explain Decisions**: Offer specific reasons for any reward reductions or flags to help users understand your evaluation.
   - **Guide Improvement**: Include suggestions or guidance on how users can enhance future submissions.
   - **Promote Positive Interaction**: Aim to maintain a supportive tone that encourages ongoing engagement and contribution.

---

**High-Level Summary:**

To enhance user retention and experience, the reward prompt should be adjusted to:

- **Ensure Comprehensive Evaluation**: Encourage evaluators to thoroughly review all user-provided evidence and documentation before making decisions.
- **Refine Flagging Criteria**: Modify the criteria for issuing yellow flags to prevent unfair penalization, focusing on significant concerns rather than minor oversights.
- **Value User Contributions**: Highlight the importance of considering the user's history and consistent contributions, giving them the benefit of the doubt when appropriate.
- **Adopt Supportive Language**: Adjust the tone of the prompt to be more encouraging and collaborative, avoiding harsh or punitive language.
- **Provide Constructive Feedback**: Emphasize the need for clear, specific, and helpful feedback to guide users in improving their future submissions.

By implementing these changes, the reward system will promote a more positive user experience, motivate continued high-quality contributions, and support the overall mission of the Post Fiat Network."""