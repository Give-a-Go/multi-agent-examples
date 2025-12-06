from typing import AsyncGenerator
import logging

from google.adk.agents import LlmAgent, LoopAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

# --- Configuration ---
MODEL_NAME = "gemini-2.0-flash"

# 1. Define the Give(a)Go Lingo Guidelines
GIVE_A_GO_INSTRUCTIONS = """
You are a community manager for 'Give(a)Go'. Your goal is to rewrite messages into "Give(a)Go lingo".

Give(a)Go lingo is:
- Be collaborative and inclusive
- Encourage building and experimenting
- Be optimistic and high-energy
- Fit the Give(a)Go community tone
- Use relevant emojis (e.g., ✨, 🚀, 🛠️) to add vibrance

Input: The user's raw message.
Task: Rewrite it completely to match the tone above.
History: If you see critique feedback in the conversation history, use it to refine your draft.
"""

# 2. Define the Critic Instructions
CRITIC_INSTRUCTIONS = """
You are the Give(a)Go Tone Critic. Review the Writer's latest draft.

Check against these criteria:
1. Is it collaborative and inclusive?
2. Does it encourage building and experimenting?
3. Is it optimistic and high-energy?
4. Does it use emojis effectively?

Output Format:
- If the draft is excellent and needs no changes, reply with exactly: "PERFECT"
- If it needs improvement, reply with "FEEDBACK: " followed by concise bullet points on what to fix.
"""

# --- Custom Agent for Loop Control ---


class CritiqueEvaluator(BaseAgent):
    """
    Reads the output from the Critic.
    If the Critic said 'PERFECT', it triggers an escalation event to stop the loop.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Retrieve the text output from the critic (stored in state via output_key)
        critique_text = ctx.session.state.get("critique_result", "").strip()

        # Check if the critic is satisfied
        is_perfect = critique_text.startswith("PERFECT")

        if is_perfect:
            logging.info("Refinement complete. Stopping loop.")

        # Emit an event. If is_perfect is True, 'escalate' stops the LoopAgent.
        yield Event(author=self.name, actions=EventActions(escalate=is_perfect))


# --- Agent Definitions ---

# 1. The Writer: Generates the text
writer = LlmAgent(
    name="GiveAGo_Writer",
    model=MODEL_NAME,
    instruction=GIVE_A_GO_INSTRUCTIONS,  # <--- FIXED: changed from system_instruction
)

# 2. The Critic: Reviews the text and saves output to 'critique_result'
critic = LlmAgent(
    name="GiveAGo_Critic",
    model=MODEL_NAME,
    instruction=CRITIC_INSTRUCTIONS,  # <--- FIXED: changed from system_instruction
    output_key="critique_result",
)

# 3. The Evaluator: Checks condition to stop
evaluator = CritiqueEvaluator(name="Evaluator")

# 4. The Root Agent: Iterative Loop
root_agent = LoopAgent(
    name="RefinementLoop",
    description="Iteratively rewrites text into Give(a)Go lingo.",
    max_iterations=3,
    sub_agents=[writer, critic, evaluator],
)
