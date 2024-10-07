import asyncio
from os import environ
from langchain import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings  
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from chainlit.input_widget import Select
import giskard
import pandas as pd

DB_FAISS_PATH = r'vectorestores\db_faiss'
CUSTOM_PROMPT_TEMPLATE = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=['context', 'question'])
    return prompt

def load_llm(model_name):
    try:
        print(f"Loading model: {model_name}")
        if model_name == "TheBloke/openchat-3.5-0106-GGUF":
            lm = CTransformers(
                model=model_name,
                max_tokens=512,
                temperature=0.5,
            )
        elif model_name == "TheBloke/Llama-2-7B-Chat-GGML":
            lm = CTransformers(
                model=model_name,
                model_type="llama",
                max_new_tokens=512,
                temperature=0.5
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return lm
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

def qa_bot(model_name):
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2")  
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        llm = load_llm(model_name)
        qa_prompt = set_custom_prompt()
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 2}),
                                               return_source_documents=True,
                                               chain_type_kwargs={'prompt': qa_prompt}
                                               )
        return qa_chain
    except Exception as e:
        print(f"Error setting up QA bot: {e}")
        raise

# Define a custom Giskard model wrapper for the QA bot
class GiskardQAModel(giskard.Model):
    def __init__(self, qa_chain, name="QA Model"):
        super().__init__(model=qa_chain, model_type="text_generation", name=name)
        self.qa_chain = qa_chain

    def model_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for query in df["query"]:
            res = self.qa_chain({"query": query})
            results.append(res["result"])
        return pd.DataFrame({"prediction": results})

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    username_stored = environ.get("CHAINTLIT_USERNAME")
    password_stored = environ.get("CHAINTLIT_PASSWORD")

    if username_stored is None or password_stored is None:
        raise ValueError(
            "Username or password not set. Please set CHAINTLIT_USERNAME and "
            "CHAINTLIT_PASSWORD environment variables."
        )

    if (username, password) == (username_stored, password_stored):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def start():
    # Display a dropdown to select the model
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Select a Model",
                values=["TheBloke/openchat-3.5-0106-GGUF", "TheBloke/Llama-2-7B-Chat-GGML"],
                initial_index=0,
            )
        ]
    ).send()
    
    model_selection = settings["Model"]
    cl.user_session.set("model", model_selection)
    
    # Initialize the QA chain with the selected model
    chain = qa_bot(model_selection)
    cl.user_session.set("chain", chain)

    msg = cl.Message(content=f"Hi, Welcome to the Medical Bot. You've selected the model: {model_selection}. What is your query?")
    await msg.send()

@cl.on_message
async def main(message):
    try:
        chain = cl.user_session.get("chain")
        # If there's no chain, reinitialize it using the stored model
        if not chain:
            model_selection = cl.user_session.get("model")
            chain = qa_bot(model_selection)
            cl.user_session.set("chain", chain)
        
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res["result"]
        sources = res["source_documents"]

        if sources:
            answer += f"\nSources:" + str(sources)
        else:
            answer += "\nNo sources found"

        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"Error processing message: {e}").send()

@cl.on_settings_update
async def settings_update(updated_settings):
    # Update the model if changed
    if "Model" in updated_settings:
        new_model = updated_settings["Model"]
        cl.user_session.set("model", new_model)
        chain = qa_bot(new_model)
        cl.user_session.set("chain", chain)
        msg = cl.Message(content=f"Model has been updated to: {new_model}. Please enter your query.")
        await msg.send()

if __name__ == "__main__":
    asyncio.run(cl.main())

    # Example: Giskard evaluation for the selected model
    model_selection = "TheBloke/openchat-3.5-0106-GGUF"  # or "TheBloke/Llama-2-7B-Chat-GGML"
    chain = qa_bot(model_selection)
    giskard_model = GiskardQAModel(qa_chain=chain, name=model_selection)
    
    # Create a test dataset
    test_data = pd.DataFrame({
        "query": [
            "What are the symptoms of diabetes?",
            "How to treat high blood pressure?",
            "What is the best diet for weight loss?"
        ]
    })
    
    # Wrap the dataset in Giskard
    giskard_dataset = giskard.Dataset(test_data)
    
    # Run Giskard scan
    results = giskard.scan(giskard_model, giskard_dataset, only="hallucination")
    display(results)
