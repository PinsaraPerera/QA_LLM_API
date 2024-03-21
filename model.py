from langchain_community.llms import CTransformers
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA


def load_llm():
    # load the locally downloaded model
    llm = CTransformers(
        model = "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
        model_type = "mistral",
        max_new_tokens = 1048,
        temperature = 0.3,   
    )

    return llm

def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen  = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_answer_gen = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    document_answer_gen = splitter_answer_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)



    llm_ques_gen_pipeline = load_llm()

    prompt_template = """
    You are an expert at creating questions based on materials and documentation.
    Your goal is to prepare a student for their exam and tests.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the students for their tests.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an expert at creating practice questions based on material and documentation.
    Your goal is to help a students to prepare for a MCQ test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables = ["existing_answer", "text"],
        template = refine_template
    )

    ques_gen_chain = load_summarize_chain(
        llm= llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt = PROMPT_QUESTIONS,
        refine_prompt= REFINE_PROMPT_QUESTIONS,
    )

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_document(document_answer_gen, embeddings)

    llm_answer_gen = load_llm()

    ques_list = ques.split('\n')

    filtered_ques_list = [element for element in ques_list if element.endswith("?") or element.endswith(".")]

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm = llm_answer_gen,
        chain_type="stuff",
        retriever = vector_store.as_retriever(),
    )

    return answer_generation_chain, filtered_ques_list

def get_csv(file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)

    output_file = base_folder + "QA.csv"

    with open(output_file, 'w', newline= '', encoding= "utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([ques, answer])

    return output_file

