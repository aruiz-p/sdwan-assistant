# from langchain.chat_models import ChatOpenAI
from pydantic import ValidationError
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import (
    format_to_openai_function_messages,
)
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_tools_list import tools
from logging_config.main import setup_logging
from utils.text_utils import remove_white_spaces, output_to_json

logger = setup_logging()


SYSTEM_PROMPT = """
You are a Cisco SD-WAN expert AI assistant, your role is to start Network Wide Path Insight traces on behalf of users to spot network issues. Follow these guidelines:
1. The user will let you know the site and vpn to start the trace. Additionally they could provide source and destination subnets. 
2. Use the 'get_site_list' function to obtain the list of available sites to run the trace and confirm it matches with the user input.
3. Before starting the trace, use the 'get_device_details_from_site' to retrieve the device list that will be used as parameter. 
4. Optionally,use the VPN, site id and source and destination networks provided by the user as parameters to start the trace. 
5. After starting the trace inform the user and share the "trace_id" and "timestamp". 
6. You need to verify if there are any flows and if there is any reported event. Inform the user about it.
7. When user request information of a trace, always use "get_entry_time_and_state" to retrieve the entry_time and state use it to get other information. 
8. If the trace is already stopped, you can still provide the information requested by the user. 
9. If the state indicates an issue, you should still try to provide the user with the information requested.  
10. To present the flow summary use one row for each flow.
11. When the user requests detailed information of a flow, use the previously obtained timestamp.Try to understand the output and provide a conclusion based on that information. 
12. Must use as much as possible many emojis that are relevant to your messages to make them more human-friendly.
"""



MEMORY_KEY = "chat_history"

#LLM_MODEL = "gpt-4-turbo-preview"
# LLM_MODEL = "gpt-3.5-turbo-16k"
# LLM_MODEL = "gpt-4o"
LLM_MODEL = "gpt-4o-mini"


class LLMChatAgent:
    def __init__(self) -> None:
        self._create_agent()

    def _create_agent(self) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    remove_white_spaces(string=SYSTEM_PROMPT),
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)
        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        self.agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

    def _agent_executor(self, message: str) -> str:
        return self.agent_executor.invoke({"input": message})["output"]

    def chat(self, message: str, attempts: int = 0) -> str:
        """
        TODO: There a potential loop here. If agent is not able to connect to the device,
        the agent will try to connect again to the device. This can go on forever.
        The agent stoppped at 3 attempts to connect to the device.
        """
        logger.info(f"CHAT_SENT_TO_LLM: {message}")
        try:
            return self._agent_executor(message)
        except (ValidationError, ConnectionError, KeyError) as e:
            if attempts < 2:
                if isinstance(e, ValidationError):
                    msg = f"ERROR: You missed a parameter invoking the function, See for the information missing: {e}"
                elif isinstance(e, ConnectionError):
                    msg = f"ERROR: Unable to connect. {e}"
                else:  # KeyError
                    msg = f"ERROR: You provided an empty value or a device that doesn't exists. {e}"
                logger.error(msg)
                return self.chat(msg, attempts + 1)
            else:
                logger.error(f"Uncatched error: {e}")
                return f"ERROR: {e}"


if __name__ == "__main__":
    agent = LLMChatAgent()
    chat = agent.chat(
        "please provide a summary of all activities I asked you to check in our conversation"
    )
    print(chat)
    print("#" * 80, "\n")