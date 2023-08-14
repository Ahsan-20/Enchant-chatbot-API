from fastapi import Response
import os
import json
import openai
from fastapi import APIRouter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from fastapi import UploadFile, File, Form
import shutil
import tiktoken
from dotenv import load_dotenv
from elevenlabs import generate
import os
from fastapi import UploadFile, File, Form
from elevenlabs import generate
load_dotenv()


router = APIRouter()




def append_chat_to_json(chat_file, user_id, new_chat_sentences):
    # Check if the chat file exists, if not create an empty JSON structure
    if not os.path.exists(chat_file):
        with open(chat_file, 'w') as f:
            json.dump({}, f)

    # Load existing chat data from the JSON file
    with open(chat_file, 'r') as f:
        chat_data = json.load(f)

    # Check if user_id already exists in the chat data
    if user_id not in chat_data:
        chat_data[user_id] = []

    # Append the new chat sentences to the user's chat history
    for sentence in new_chat_sentences:
        chat_data[user_id].append(sentence)

    # Save the updated chat data back to the JSON file
    with open(chat_file, 'w') as f:
        json.dump(chat_data, f, indent=2)


def load_chat_by_user_id(chat_file, user_id):
    # Check if the chat file exists, if not return an empty list
    if not os.path.exists(chat_file):
        return []

    # Load existing chat data from the JSON file
    with open(chat_file, 'r') as f:
        chat_data = json.load(f)

    # Get the chat history for the specified user_id
    if user_id in chat_data:
        return chat_data[user_id]
    else:
        return []

def get_messages(my_list,no=10):
    if len(my_list) >= no:
        return my_list[-no:]
    else:
        return my_list
      
def tokens_count(string: str) -> int:
      encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
      return len(encoding.encode(string))

@router.post("/transcribe")
async def transcribe(customer: UploadFile = File(...), user_id: str = Form(...), company_id: str = Form(...)):
    
    name = 'Bella'
    file_extension = os.path.splitext(customer.filename)[1].lower()
    
    if file_extension not in [".mp3", ".m4a", ".wav"]:
        return {"error": "Unsupported file format."}

    # Save the uploaded file
    file_path = f"temp_{user_id}/uploaded_audio{file_extension}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create parent directory if it doesn't exist
    with open(file_path, "wb") as f:
        f.write(await customer.read())

    # Transcribe the audio
    with open(file_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)

    os.remove(file_path)

    try:
        shutil.rmtree(os.path.dirname(file_path))
    except OSError as e:
        pass  # The directory could not be removed.
    
    chat_file = 'chat_data.json'
    user_chat_history = load_chat_by_user_id(chat_file, user_id)

    query=transcript.text
    print(query)
    embeddings = OpenAIEmbeddings()
    persist_directory = f'trained_db/{company_id}/{company_id}_all_embeddings'

    prompt_template = f"""You are a Sales agent of my company named as Saamaan.Pk you are on call with the client and trying to sell the products we offer. your end goal is to convince client
    so that he agree to buy our product. offer them below products and answer to any query they have for the product. Your name is Bella. try to start the conversation by telling about our company and the producta we offer.
    Try to find the interest if the client and offer them products accordingly.
    The company details and product details given below as. 
    context:

    {{context}}

    following is the chat history between you and the user:
    {str(get_messages(user_chat_history))}

    Question: {{question}}

    Answer :"""
        
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo',max_tokens=500)
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)

    Vectordb = FAISS.load_local(persist_directory, embeddings)
    retriever = Vectordb.as_retriever(search_type="mmr")
    docs = retriever.get_relevant_documents(query)
    ans = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    result = ans['output_text']
    new_chat_sentences = [
    "USER: "+query,
    "AI: "+result
    ]
    print(docs)
    append_chat_to_json(chat_file, user_id, new_chat_sentences)

    print(result)
    audio = generate(
        text=result,
        voice=name,
        api_key=os.getenv("ELEVENLAB_API_KEY"),
        model="eleven_multilingual_v1"
    )

    headers = {
        "Content-Type": "audio/mpeg",
        "Content-Disposition": "attachment; filename=audio_file.mp3"  # Change the filename if needed
    }

    # Return the audio as a response with the appropriate headers
    return Response(content=audio, headers=headers)
