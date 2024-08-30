import streamlit as st
from openai import OpenAI
import pandas as pd
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain, ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

# Show title and description.
st.title("💬 Chinese version cocobot")


# Setting up System Prompt
file_path = 'system_prompt.txt'
with open(file_path, 'r') as file:
    default_system_message = file.read()

st.info("""Instruction for the chatbot:  
        {system_prompt}""".format(system_prompt=default_system_message))

system_message = default_system_message
# if st.button('使用现在的system prompt'):
#     system_message = default_system_message

# user_input_system_message = st.text_input("你可以选择输入新的 system prompt: ")
# if st.button('使用新的 system prompt'):
#     system_message = user_input_system_message
#     st.info("""新的 system prompt是:  
#         {system_prompt}""".format(system_prompt=system_message))
# else:
#     system_message = default_system_message



# Make sure we have an Openai API to use
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")


else:
    # Create an OpenAI client.
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")

    # Create the ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # memory = ConversationBufferMemory(return_messages=True)
    # memory.clear()

    # set up history memory
    msgs = StreamlitChatMessageHistory(key="chat_history")

    # create a chatbot llm chain
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,  # Always return the instance created earlier
        input_messages_key="input",
        # output_messages_key="content",
        history_messages_key="history",
    )

    # chain = ConversationChain(
    #         llm=llm,
    #         prompt=prompt,
    #         memory=memory,
    #         input_key="input",
    #         output_key="response"
    #     )
    
    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if user_input := st.chat_input("您想聊点什么？"):
        # if user_input=="SAVE" or user_input=="save":
        #     chat_history_df = pd.DataFrame(st.session_state.messages)
        #     csv = chat_history_df.to_csv()
        #     st.download_button(
        #         label="Download data as CSV",
        #         data=csv,
        #         file_name="chat_history.csv",
        #         mime="text/csv",
        #     )
        # else:
        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        config = {"configurable": {"session_id": "any"}}
        bot_response = chain_with_history.invoke({"input": user_input}, config)


        # Use the ConversationChain
        # ai_response = chain({"input": user_input})

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
                response = st.write(bot_response.content)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response.content})

        chat_history_df = pd.DataFrame(st.session_state.messages)
        csv = chat_history_df.to_csv()
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="chat_history.csv",
            mime="text/csv",
        )

        