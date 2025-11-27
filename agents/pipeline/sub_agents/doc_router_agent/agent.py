# agents/doc_router_agent/agent.py

from google.adk import Agent
from .classifier import classify_domain
from .prompt import ROUTER_PROMPT


class DocRouterAgent(Agent):
    def __init__(self):
        super().__init__(name="doc_router_agent")

    async def run(self, ctx):
        query = ctx.input_text

        # Rule-based
        rule = classify_domain(query)

        # LLM-based
        llm_result = await ctx.llm.query(
            ROUTER_PROMPT.format(query=query),
            response_format="json"
        )

        # merge
        final = self.merge(rule, llm_result)
        return final

    def merge(self, rule, llm):
        if rule["confidence"] >= llm["confidence"]:
            return rule
        return llm


root_agent = DocRouterAgent()
