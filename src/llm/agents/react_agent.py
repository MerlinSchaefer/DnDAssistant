import re
from typing import Union

from langchain import agents as lc_agents
from langchain import memory as lc_memory
from langchain import prompts as lc_prompts
from langchain import tools as lc_tools
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain_core import retrievers as lcc_retrievers
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import chat_models

from src.llm.agents._invokable_agent import InvokableAgent
from src.llm.prompts.assistant import react_prompt
from src.llm.tools import get_document_retrieval_tool

FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = "Invalid Format: Missing 'Action:' after 'Thought:"
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = "Invalid Format: Missing 'Action Input:' after 'Action:'"
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class CustomReActSingleInputOutputParser(ReActSingleInputOutputParser):
    """Custom parser that removes backticks from the input before parsing."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            return self._parse_logic(text)
        except OutputParserException as e:
            # Append FINAL_ANSWER_ACTION to the text and try parsing again
            text_with_final_answer = f"{FINAL_ANSWER_ACTION} {text}"
            try:
                return self._parse_logic(text_with_final_answer)
            except OutputParserException:
                # Raise the original exception if parsing still fails
                raise e

    def _parse_logic(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}")
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish({"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text)

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")


class ConversationalRetrievalAgentWithMemory(InvokableAgent):
    def __init__(
        self,
        memory,  #: lc_memory.FileChatMessageHistory,
        retriever: lcc_retrievers.BaseRetriever,
        chat_model: chat_models.BaseChatModel,
    ):
        """Creates a retrieval agent executor with memory.

        See https://python.langchain.com/docs/modules/memory/agent_with_memory.

        Args:
            memory: The memory object to be used by the agent.
            retriever: The retriever that the agent can use to answer queries.
            chat_model: The chat model that the agent can use to answer queries.
            system_prompt: The (user) system prompt used to fill the prompt template.
                        Will be empty outside of chat_mode.USER.
            chat_mode: One of three ChatModes (open, strict, user) to determine
                             which prompt template to load.

        Returns:
            agent_executor: The agent executor that can use a retrieval tool to answer queries and has memory.
                It is to be invoked using `agent_executor.invoke({"input": "[User query]"})`.
        """

        tools = [
            get_document_retrieval_tool(retriever=retriever),
        ]
        self._agent_executor = create_react_agent_with_memory(
            prompt_template=react_prompt,
            memory=memory,
            tools=tools,
            chat_model=chat_model,
        )

    def _invoke(self, query: str) -> str:
        result = self._agent_executor.invoke({"input": query})
        return result["output"]


def create_react_agent_with_memory(
    prompt_template: lc_prompts.BasePromptTemplate,
    memory: lc_memory.ConversationBufferMemory,
    tools: list[lc_tools.Tool],
    chat_model: chat_models.BaseChatModel,
) -> lc_agents.AgentExecutor:
    """Auxiliary function for creating a general agent executor with memory that can use tools.

    Args:
        prompt_template: The prompt template for the agent. Must contain instructions how to use the tools.
        memory: The memory object to be used by the agent.
        tools: The tools that the agent can use.
        chat_model: The chat model that the agent can use to answer queries.

    Returns:
        agent_executor: The agent executor that can use the given tools and has memory.
            It is to be invoked using `agent_executor.invoke({"input": "[User query]"})`.
    """
    agent = lc_agents.create_react_agent(
        llm=chat_model,
        tools=tools,
        prompt=prompt_template,
        output_parser=CustomReActSingleInputOutputParser(),
    )
    agent_executor = lc_agents.AgentExecutor(
        agent=agent,  # type: ignore
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,  # type: ignore
    )
    return agent_executor
