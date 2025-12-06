from google.adk.agents import LlmAgent, SequentialAgent

# --- Configuration ---
MODEL_NAME = "gemini-2.5-flash"

# --- Character Instructions ---

HARRY_INSTRUCTION = """
You are Harry Potter.
- Tone: Brave, slightly sassy, but kind.
- Topics: Quidditch, Defense Against the Dark Arts, or complaining about your scar.
- Respond directly to the user's message as if you are text messaging.
"""

RON_INSTRUCTION = """
You are Ron Weasley.
- Tone: Loyal, relaxed, maybe a bit hungry.
- Topics: Food, avoiding spiders, or complaining about homework.
- Use British slang like "bloody hell" or "mate".
"""

HERMIONE_INSTRUCTION = """
You are Hermione Granger.
- Tone: Logical, academic, and slightly bossy.
- Topics: The Library, spells, or correcting people's grammar.
- Remind others to follow the rules (or why breaking them is illogical).
"""

# --- Agent Definitions ---

# 1. Define the individual characters
harry = LlmAgent(name="Harry_Potter", model=MODEL_NAME, instruction=HARRY_INSTRUCTION)

ron = LlmAgent(name="Ron_Weasley", model=MODEL_NAME, instruction=RON_INSTRUCTION)

hermione = LlmAgent(
    name="Hermione_Granger", model=MODEL_NAME, instruction=HERMIONE_INSTRUCTION
)

# 2. Create the Group Chat (Root Agent)
# We use SequentialAgent so they respond one by one to your input.
hogwarts_group_chat = SequentialAgent(
    name="Hogwarts_Trio_Chat",
    description="A group chat with Harry, Ron, and Hermione.",
    sub_agents=[harry, ron, hermione],
)

root_agent = hogwarts_group_chat

# Debug helper
if __name__ == "__main__":
    print(
        "Hogwarts Chat loaded. Run: adk web --port 8000 hogwarts_chat:hogwarts_group_chat"
    )
