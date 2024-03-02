# from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS # Library from META for efficient similarity search
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

#     You are a helpful Traditional Chinese language teaching assistant that can have
#     Answer the following question: {question}  
#     By searching  the following video transcript: {docs}

#     Only use the factual information from the transcript to answer the question.
#     If you dont feel like you have enough information to answer, say "I don't know."
# Your answers should be detailed.
gpt_template = """

    You are a helpful Traditional Chinese language teaching assistant that can have educational conversations.
    Create small conversations using a given word bank. You only respond and speak in traditional chinese.
    If a mistake is made you can translate and have the user try again.
    Students talk back to you saying this: {user_message}  

    Try to use use the words from the word bank to have a small conversation with your student.
    After he convo, note any mistakes and save them for later.
"""


# video_url = "https://www.youtube.com/watch?v=pfW2pQBwx6A"
def create_vector_db_from_word_doc(word_doc: str) -> FAISS:
    loader = Docx2txtLoader("./chinese_notes.docx")
    transcript = loader.load()

    # TAKE THE LINES FROM THE TRANSCRIPT, AND WE NEED TO CHUNK IT
    # WE DO THIS SO THAT THE OPENAPI CAN READ ALL THE DATA WITHOUT GOING OVER OPENAPI SIZE LIMITS
    # WE TAKE THOSE CHUNKS AND STORE THEM AS VECTOR STORES
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # FAISS HELPS US DO THE SIMILARITY SEARCH
    db = FAISS.from_documents(docs, embeddings)

    # return an instance of those chunks
    return db


def get_response_from_query(db, user_message, k_similarity=4):
    # llm we chose, text-davinci, can handle 4097 tokens
    # k_similarity =since each document chunk is 1000, we can search roughly 4 documents, so our k should be set to 4


    # search the query relevant documents
    # what did they say about X in the video
    # the search will search only the document that is relevant to our query
    docs = db.similarity_search(user_message, k_similarity)
    docs_page_content = " ".join([d.page_content for d in docs])


    llm = OpenAI(model="gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables = ["user_message","docs"],
        template = gpt_template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(user_message=user_message, docs=docs_page_content)
    response = response.replace("\n", "")

    return response


# print(create_vector_db_from_word_doc(video_url))