from flask import Flask, render_template, request, redirect, url_for, Response, stream_with_context
from flask import send_from_directory
from flask import jsonify

from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaDownloadProgress

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import ConversationChain

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

from pdf2image import convert_from_path
from PIL import Image

import fitz # PyMuPDF
from rapidfuzz import process, fuzz

from urllib.parse import unquote
from threading import Thread
import subprocess
import threading
import traceback
import platform
import tempfile
import datetime
import requests
import logging
import sqlite3
import signal
import PyPDF2
import base64
import queue
import uuid
import json
import time
import nltk
import zlib
import ast
import sys
import os
import io
import re

from logging.handlers import RotatingFileHandler
from nltk.corpus import stopwords



app = Flask(__name__)

# Route for the home page, rendering the initial model selection form (legacy)
@app.route('/')
def index():
    return render_template('chat.html')

# model_selection.html triggers window.location.href to '/chat', which triggers this route, which loads the chat.html template at the end!
@app.route('/chat')
def chat():
    return render_template('chat.html')

# Route to display the file loading form
@app.route('/load_file')
def load_file():
    return render_template('model_selection.html', show_file_form=True)

@app.route('/download/<filename>')
def download_file(filename):
    # return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=False, mimetype='application/pdf')

@app.route('/pdf/<filename>')
def pdf_viewer(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename)



#########################------------------GLOBALS!----------------------###############################
LLAMA_CPP_PROCESS = None
LLM = None
CHAT_ID = None
SEQUENCE_ID = None
LOADED_UP = False
LLM_LOADED_UP = False
VECTORDB_LOADED_UP = False
LLM_CHANGE_RELOAD_TRIGGER_SET = False
VECTORDB_CHANGE_RELOAD_TRIGGER_SET = False
VECTOR_STORE = None
HF_BGE_EMBEDDINGS = None
AZURE_OPENAI_EMBEDDINGS = None
HISTORY_MEMORY_WITH_BUFFER = None   #Init in load_model_and_vectordb(); reset in load_chat_history() when old chats loaded, and in load_model_and_vectordb() when 'New Chat' selected; used for non-RAG convChain init in stream, and for saving context in stream for RAG chains and lastly, for setting HISTORY_SUMMARY in stream() via load_memory_variables({})
HISTORY_SUMMARY = {}    #Set in stream() via HISTORY_MEMORY_WITH_BUFFER.load_memory_variables({}), and in load_chat_history() from chat_history DB; cleared in load_model_and_vectordb() when 'New Chat' selected; used to init prompt templates in stream() and lastly, for storage to chat_history DB in stream() and get_references()

# Dict for user queries:  queries[session_id] = user_input
QUERIES = {}

# If modifying these scopes, delete the file token.json.
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly", "https://www.googleapis.com/auth/drive.readonly"]
GDRIVE_CREDS = None
#########################------------------------------------------------###############################



#########################------------Setup & Handle Logging-------------###############################
try:
    # 1 - Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.ERROR)

    # 2 - Create a RotatingFileHandler
    # maxBytes: max file size of log file after which a new file is created; set to 1024 * 1024 * 5 for 5MB: 1024x1024 is 1MB, then a multiplyer for the number of MB
    # backupCount: number of backup files to keep specifying how many old log files to keep
    handler = RotatingFileHandler('server_log.log', maxBytes=1024*1024*5, backupCount=2)
    handler.setLevel(logging.ERROR)

    # 3 - Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)

    # 4 - Add the handler to the logger
    logger.addHandler(handler)
    # Logger ready! Usage: logger.error(f"This is an error message with error {e}")
except Exception as e:
    print(f"\n\nCould not establish logger, encountered error: {e}")


def handle_api_error(message, exception=None):
    error_message = f"{message} {str(exception) if exception else '; No exception info.'}".strip()
    #traceback_details = traceback.format_exc()
    #full_message = f"\n\n{error_message}\n\nTraceback: {traceback_details}\n\n"
    full_message = f"\n\n{error_message}\n\n"

    if logger:
        logger.error(full_message)
        print(full_message)
    else:
        print(full_message)
    return jsonify(success=False, error=error_message), 500 #internal server error


def handle_local_error(message, exception=None):
    error_message = f"{message} {str(exception) if exception else '; No exception info.'}".strip()
    #traceback_details = traceback.format_exc()
    full_message = f"\n\n{error_message}\n\n"
    if logger:
        logger.error(full_message)
        print(full_message)
    else:
        print(full_message)
    raise Exception(exception)


def handle_error_no_return(message, exception=None):
    error_message = f"{message} {str(exception) if exception else '; No exception info.'}".strip()
    #traceback_details = traceback.format_exc()
    full_message = f"\n\n{error_message}\n\n"
    if logger:
        logger.error(full_message)
        print(full_message)
    else:
        print(full_message)

#########################-------------------------------------###############################



if not os.path.exists('config.json'):
    try:
        with open('config.json', 'w') as file:
            json.dump({}, file)
    except Exception as e:
        handle_error_no_return("Could not init config.json. Multiple app restarts may be required to get the app to init correctly. Printing error and proceeding: ", e)



# Method to write to config.json | input- dict of key:values to be written to config.json
def write_config(config_updates, filename='config.json'):

    # Open config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except Exception as e:
        config = {}     #init emply config dict
        handle_error_no_return("Could not read config.json when attempting to write, encountered error: ", e)
        
    restart_required = False
    if LLM_LOADED_UP:
        llm_trigger_keys_for_app_restart = ['use_local_llm', 'use_azure_open_ai', 'use_gpu', 'model_choice', 'local_llm_chat_template_format', 'local_llm_context_length', 'local_llm_max_new_tokens', 'local_llm_gpu_layers', 'base_template']
                
        for key in llm_trigger_keys_for_app_restart:
            if key in config_updates and config_updates[key] != config.get(key):
                global LLM_CHANGE_RELOAD_TRIGGER_SET
                LLM_CHANGE_RELOAD_TRIGGER_SET = True
                restart_required = True
                break
    
    if VECTORDB_LOADED_UP:
        vectordb_trigger_keys_for_app_restart = ['embedding_model_choice']

        for key in vectordb_trigger_keys_for_app_restart:
            if key in config_updates and config_updates[key] != config.get(key):
                global VECTORDB_CHANGE_RELOAD_TRIGGER_SET
                VECTORDB_CHANGE_RELOAD_TRIGGER_SET = True
                restart_required = True
                break

    config.update(config_updates)

    # Write updated config.json:
    try:
        with open(filename, 'w') as file:
            json.dump(config, file, indent=4)
    except Exception as e:
        handle_local_error("Could not update config.json, encountered error: ", e)
     
    return {'success': True, 'restart_required':restart_required}
            

# Method to read from config.json | input- list of keys to be read from config.json; output- dict of key:value pairs; MANAGE DEFAULTS HERE!
def read_config(keys, default_value=None, filename='config.json'):
    
    # Open config file to read-in all current params:
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
    except Exception as e:
        handle_error_no_return("Could not read config.json, encountered error: ", e)
        return {key: default_value for key in keys}     #because a read scenario wherein config.json does not exist shouldn't occur!
    
    return_dict = {}
    update_config_dict = {}
    base_directory = config.get('base_directory', '/app/storage')   # specifying default if not found

    for key in keys:
        if key in config:
            return_dict[key] = config[key]
        else:
            default_value = {
                'windows_base_directory':'C:/web_app_storage',
                'unix_and_docker_base_directory':'/app/storage',
                'mac_base_directory':'app',
                'upload_folder':base_directory + '/uploaded_pdfs',
                'vectordb_sbert_folder':base_directory + '/chroma_db_sbert_embeddings',
                'vectordb_openai_folder':base_directory + '/chroma_db_openai_embeddings',
                'vectordb_bge_large_folder':base_directory + '/chroma_db_bge_large_embeddings',
                'vectordb_bge_base_folder':base_directory + '/chroma_db_bge_base_embeddings',
                'sqlite_images_db':base_directory + '/images_database_main.db',
                'sqlite_history_db':base_directory + '/chat_history.db',
                'sqlite_docs_loaded_db':base_directory + '/docs_loaded.db',
                'model_dir':base_directory + '/models',
                'highlighted_docs':base_directory + '/highlighted_pdfs',
                'ocr_pdfs':base_directory + '/ocr_pdfs',
                'pdfs_to_txts':base_directory + '/pdfs_to_txts',
                'local_llm_server':'hf-waitress',
                'model_choice':'Meta-Llama-3-8B-Instruct.f16.gguf',
                'do_rag':True,
                'force_enable_rag':False,
                'force_disable_rag':False,
                'use_local_llm':True,
                'use_gpu':True,
                'use_gpu_for_embeddings':False,
                'azure_cv_free_tier':True,
                'use_azure_open_ai':False,
                'use_openai_embeddings':False,
                'azure_openai_api_type':'azure',
                'azure_openai_api_version':'2023-05-15',
                'azure_openai_max_tokens':4096,
                'azure_openai_temperature':0.7,
                'use_bge_large_embeddings':False,
                'use_bge_base_embeddings':False,
                'use_sbert_embeddings':True,
                'embedding_model_choice':'sbert_mpnet_base_v2',
                'use_ocr':False,
                'ocr_service_choice':'None',
                'local_llm_model_type':'llama',
                'local_llm_chat_template_format':'llama3',
                'local_llm_context_length':8192,
                'local_llm_max_new_tokens':2048,
                'local_llm_gpu_layers':47,
                'local_llm_temperature':0.8,
                'local_llm_top_k':40,
                'local_llm_top_p':0.95,
                'local_llm_min_p':0.05,
                'local_llm_n_keep':0,
                'server_timeout_seconds':10,
                'server_retry_attempts':3,
                'base_template':"Answer the user's question in as much detail as possible. Be as accurate as possible. Do not make up answers or fabricate false information! Whenever additional context is provided, mention the document names and page numbers the user should reference as per your best judgement.",
            }.get(key, 'undefined')

            if default_value == 'undefined':
                raise KeyError(f"Key \'{key}\' not found in config.json and no default value has been defined either.\n")
            
            return_dict[key] = default_value
            update_config_dict[key] = default_value

    if update_config_dict:
        # Write Defaults
        try:
            write_config(update_config_dict)
        except Exception as e:
            handle_error_no_return("Could not write defaults to config.json. Encountered error: ", e)
    
    ##print(f"return_dict: {return_dict}")

    return return_dict


# Method for API route to read from config.json
# Deviates from typical RESTful principals to use a POST call to fetch values but practical & justifyable because we:
# 1. Do not want to make the URL huge with a ever-growing list of query-params 2. Do not wish to expose values via query-params
@app.route('/config_reader_api', methods=['POST'])
def config_reader_api():
    # keys = request.args.getlist('keys') # Assuming keys are passed as query parameters
    
    try:
        keys = request.json.get('keys', []) # Could also do keys = request.json['keys'] but this way we can provide a default list should 'keys' be missing!
    except Exception as e:
        return handle_api_error("Server-side error - could not read keys for config_reader_api request. Encountered error:", e)

    try:
        values = read_config(keys)  # send list of keys, get dict of key:values
    except Exception as e:
        return handle_api_error("Server-side error - could not read keys from config.json. Encountered error: ", e)
    
    return jsonify(success=True, values=values)


# Method for API route to write to config.json
@app.route('/config_writer_api', methods=['POST'])
def config_writer_api():

    try:
        config_updates = request.json['config_updates']
        print(f"config_updates for config_writer_api: {config_updates}")
    except Exception as e:
        return handle_api_error("Server-side error - could not read values for config_writer_api request. Encountered error: ", e)
    
    try:
        write_return = write_config(config_updates)
    except Exception as e:
        return handle_api_error("Server-side error - could not write keys to config.json. Encountered error: ", e)
    
    return jsonify({"success": write_return['success'], "restart_required": write_return['restart_required']})



#########################------------Setup Directories-------------###############################
BASE_DIRECTORY = ""

if platform.system() == 'Windows':
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    import azure.ai.vision as sdk
    
    #BASE_DIRECTORY = 'C:/temp_web_app_storage'
    try:
        read_return = read_config(['windows_base_directory'])   #passing list of values to read
        BASE_DIRECTORY = str(read_return['windows_base_directory']) #received dict of key:values
    except Exception as e:
        handle_local_error("Could not read windows_base_directory on boot, encountered error: ", e)

elif platform.system() == 'Linux':
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    import azure.ai.vision as sdk
    
    #BASE_DIRECTORY = '/app/storage'
    try:
        read_return = read_config(['unix_and_docker_base_directory'])
        BASE_DIRECTORY = str(read_return['unix_and_docker_base_directory'])
    except Exception as e:
        handle_local_error("Could not read unix_and_docker_base_directory on boot, encountered error: ", e)

else:   #Likely 'Darwin' and hence MacOS
    #BASE_DIRECTORY = 'app'
    try:
        read_return = read_config(['mac_base_directory'])
        BASE_DIRECTORY = str(read_return['mac_base_directory'])
    except Exception as e:
        handle_local_error("Could not read mac_base_directory on boot, encountered error: ", e)

try:
    write_config({'base_directory':BASE_DIRECTORY})
except Exception as e:
    handle_local_error("Could not write OS BASE_DIRECTORY on boot, encountered error: ", e)


###---Notes on the above workflow:---###
# 1. Everytime the app runs, the OS platform is detected
# 2. Following which the apporpriate base directory is requested as above
# 3. If this is the very first run:
#   a. read_config does not find the directory data in config.json
#   b. the else clause is triggered and defaults set for both, write_config and return
# 4. If this isn't the very first run:
#   a. read_config simply returns the OS specific directory - this allows the user to update the directory via config.json!
# 4. On return, BASE_DIRECTORY is set and write_config has os specific directories set (windows_base_directory, unix_and_docker_base_directory, and mac_base_directory)
# 5. write_config is invoked for BASE_DIRECTORY
# 6. write_config detects a write-attempt for BASE_DIRECTORY and updates all related app directories too, which can be subsequently read as required
# 7. This ensures that directories are set correctly at each run while also allowing the user to set their preferred directory via config.json


# Having set the values for the directories above, proceed to actually create them on disk IF they don't alread exist!
if not os.path.exists(BASE_DIRECTORY):

    # Create a directory for app storage 
    try:
        os.mkdir(BASE_DIRECTORY)
    except Exception as e:
        handle_local_error("Failed to create Base App Directory, encountered error: ", e)
        
try:
    read_return = read_config(['model_dir', 'highlighted_docs', 'upload_folder', 'ocr_pdfs', 'pdfs_to_txts', 'index_dir'])
    model_dir = read_return['model_dir']
    highlighted_docs = read_return['highlighted_docs']
    upload_folder = read_return['upload_folder']
    ocr_pdfs = read_return['ocr_pdfs']
    pdfs_to_txts = read_return['pdfs_to_txts']
    index_dir = read_return['index_dir']
except Exception as e:
    handle_local_error("Could not read paths for app directories (model_dir, highlighted_docs, upload_folder) from config.json on boot, encountered error: ", e)


# If the base directory does not currently exist...
if not os.path.exists(model_dir):

    # Create a directory for app storage
    try:
        os.mkdir(model_dir)
    except Exception as e:
        handle_local_error("Failed to create Model Directory (model_dir), encountered error: ", e)


# If the highlighted_docs directory does not currently exist...
if not os.path.exists(highlighted_docs):

    # Create a directory for app storage
    try:
        os.mkdir(highlighted_docs)
    except Exception as e:
        handle_local_error("Failed to create Highlighted Docs Directory (highlighted_docs), encountered error: ", e)


# If the upload_folder directory does not currently exist...
if not os.path.exists(upload_folder):

    # Create a directory for app storage
    try:
        os.mkdir(upload_folder)
    except Exception as e:
        handle_local_error("Failed to create Uploaded Docs Directory (upload_folder), encountered error: ", e)
        

# If the ocr_pdfs directory does not currently exist...
if not os.path.exists(ocr_pdfs):

    # Create a directory for app storage
    try:
        os.mkdir(ocr_pdfs)
    except Exception as e:
        handle_local_error("Failed to create OCR'ed Docs Directory (ocr_pdfs), encountered error: ", e)


# If the pdfs_to_txts directory does not currently exist...
if not os.path.exists(pdfs_to_txts):

    # Create a directory for app storage
    try:
        os.mkdir(pdfs_to_txts)
    except Exception as e:
        handle_local_error("Failed to create txt-docs Directory (pdfs_to_txts), encountered error: ", e)


app.config['UPLOAD_FOLDER'] = upload_folder
app.config['DOWNLOAD_FOLDER'] = highlighted_docs


def clean_text_string(text_to_be_cleaned):
    
    # Clean text
    # text_to_be_cleaned = text_to_be_cleaned.replace("►", "").replace("■", "").replace("▼", "")
    # text_to_be_cleaned = text_to_be_cleaned.replace("Confidential Copy \n            for \n         DKPPU", "")
    #clean_text = re.sub(r'\n(?=[a-z.])', ' ', text)     # replaces newline chars immediately followed by a small-letter or dot with a space as they're likely to be the same sentence split-up across lines.
    clean_text = re.sub(r'\n+', '\n', text_to_be_cleaned)

    # This regex substitutes anything that is not a word character or whitespace with an empty string.
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)

    # This regex substitutes any sequence of whitespace characters with a single space.
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text


def PDFtoAzureDocAiTXT(input_filepath):

    print("\n\nProcessing Document - PDF to Azure DocAI TXT\n\n")
    
    try:
        read_return = read_config(['azure_doc_ai_endpoint', 'azure_doc_ai_subscription_key', 'ocr_pdfs'])
        azure_doc_ai_endpoint = read_return['azure_doc_ai_endpoint']
        azure_doc_ai_subscription_key = read_return['azure_doc_ai_subscription_key']
        ocr_pdfs = read_return['ocr_pdfs']
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for PDFtoAzureDocAiTXT, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path):
        print("Azure-OCR'ed doc already exists! Returning existing file.")
        return output_text_file_path

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        docai_client = DocumentAnalysisClient(azure_doc_ai_endpoint, AzureKeyCredential(azure_doc_ai_subscription_key))
    except Exception as e:
        handle_local_error("Could not create ComputerVisionClient for Azure DocAI, encountered error: ", e)

    try:
        with open(input_filepath, "rb") as pdf_file:
            # 1 - Get page count:
            try:
                pypdf_reader = PyPDF2.PdfReader(pdf_file)
                page_count = len(pypdf_reader.pages)
                page_range = f"1-{page_count}" if page_count > 1 else "1"
                print(f"page_range: {page_range}")
            except Exception as e:
                handle_local_error("Could not get page count for call to Azure DocAI, encountered error: ", e)

            # 2 - Reset file-read stream's internal pointer, which has now been set to the end of the file due to the above read operation!
            pdf_file.seek(0)

            # 3 - Call Azure DocAI:
            try:
                poller = docai_client.begin_analyze_document("prebuilt-layout", pdf_file, pages=page_range)
                result = poller.result()
            except Exception as e:
                handle_local_error("Could not get results for begin_analyze_document for Azure DocAI, encountered error: ", e)

        # print(f"result: \n{result}")

        used_regions = set()   # set will avoid duplicates

        if hasattr(result, 'tables'):
            for table in result.tables:
                #print("Found table")
                if table.cells:     # Check if there are cells in the table 
                    for cell in table.cells:
                        #print(f"Row {cell.row_index}, Column {cell.column_index}, Text: {cell.content}")
                        cell_text = f'Row {cell.row_index}, Column {cell.column_index}: {cell.content}'

                        # Get page number
                        page_number = ""
                        if cell.bounding_regions:   # Check if there are bounding regions
                            for region in cell.bounding_regions:
                                page_number = region.page_number
                                cell_polygon = region.polygon
                                cell_polygon_tuple = tuple((point.x, point.y) for point in cell_polygon)    # lists aren't hashable to cast to a tuple
                                used_regions.add(cell_polygon_tuple)

                        try:
                            output_text_file.write(f"[PAGE:{page_number}]\n{cell_text}\n")
                        except Exception as e:
                            handle_local_error("could not write to output text file, encountered error: ", e)

        # Get paragraphs
        if hasattr(result, 'paragraphs'):
            for paragraph in result.paragraphs:
                para_page_number = paragraph.bounding_regions[0].page_number
                para_polygon = paragraph.bounding_regions[0].polygon
                para_polygon_tuple = tuple((point.x, point.y) for point in para_polygon)
                
                if para_polygon_tuple in used_regions:
                    continue
                
                para_content = paragraph.content
                #print(f"\n---Processing Page: {para_page_number}---\n")
                #print(f"paragraph: {para_content}")

                # write the extracted text to the file:
                try:
                    output_text_file.write(f"[PAGE:{para_page_number}]\n{para_content}\n")
                    used_regions.add(para_polygon_tuple)
                except Exception as e:
                    handle_local_error("could not write to output text file, encountered error: ", e)

    except Exception as e:
        handle_local_error("Error processing document with azure DocAI: ", e)

    # Close all files
    output_text_file.close()

    return output_text_file_path


def PDFtoAzureOCRTXT(input_filepath):
    
    print("\n\nProcessing Document - PDF to Azure OCR TXT\n\n")
    
    try:
        read_return = read_config(['azure_ocr_endpoint', 'azure_ocr_subscription_key', 'ocr_pdfs', 'azure_cv_free_tier'])
        azure_ocr_endpoint = read_return['azure_ocr_endpoint']
        azure_ocr_subscription_key = read_return['azure_ocr_subscription_key']
        ocr_pdfs = read_return['ocr_pdfs']
        azure_cv_free_tier = read_return['azure_cv_free_tier']
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for PDFtoAzureOCRTXT, please provide required API config. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")
    output_text_file_path = os.path.join(ocr_pdfs, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path):
        print("OCR'ed doc already exists! Returning existing file.")
        return output_text_file_path

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    try:
        computervision_client = ComputerVisionClient(azure_ocr_endpoint, CognitiveServicesCredentials(azure_ocr_subscription_key))
    except Exception as e:
        handle_local_error("Could not create ComputerVisionClient for Azure OCR, encountered error: ", e)
    
    calls_made = 0

    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to Text OCR\n\n")
    for page_number, image in enumerate(pages, start = 1):
        
        # Convert to bytes and create a stream
        try:
            img_stream = io.BytesIO()
            image.save(img_stream, format='PNG')
            img_stream.seek(0)  # Reset the stream position to the beginning
        except Exception as e:
            handle_local_error("Could not convert image to Byte Stream for Azure OCR, encountered error: ", e)
            continue

        # Send to Azure OCR
        try:
            if azure_cv_free_tier:
                if calls_made < 20:
                    print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                    result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                    #analyze_result = computervision_client.begin_analyze_document("prebuilt-layout", img_stream).result()
                    calls_made += 1
                else:
                    print("Sleeping for 60secs due to AzureOCR free-tier restrictions!")
                    time.sleep(63)  #free tier restrictions!
                    print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                    result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                    #analyze_result = computervision_client.begin_analyze_document("prebuilt-layout", img_stream).result()
                    calls_made = 1  #reset counter
            else:
                print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
        except HttpResponseError as e:
            print(f"\n\nHttpResponseError e: {e}\n\n")
            if e.status_code == 429:
                print("Exceeded free-tier usage limits, waiting for one-minute and retrying")
                time.sleep(63)  #free tier restrictions!
                img_stream.seek(0)
                print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                calls_made = 1  #reset counter
        except Exception as e:
            if "(429)" in str(e):
                print("Exceeded free-tier usage limits, waiting for one-minute and retrying")
                time.sleep(63)  #free tier restrictions!
                img_stream.seek(0)
                print(f"Submitting page {page_number} to AzureComputerVision for OCR")
                result = computervision_client.recognize_printed_text_in_stream(image=img_stream)
                calls_made = 1  #reset counter\
            else:
                handle_local_error("Could not convert image to Byte Stream for Azure OCR, encountered error: ", e)

        for region in result.regions:
            for line in region.lines:
                #print(" ".join([word.text for word in line.words]))

                try:
                    clean_text = str(" ".join([word.text for word in line.words]))
                except Exception as e:
                    handle_error_no_return("Could not obtain line from Azure OCR result, encountered error: ", e)
                    continue

                # Write the extracted text to the file:
                try:
                    output_text_file.write(f"[PAGE:{page_number}]\n{clean_text}\n")
                except Exception as e:
                    handle_local_error("Could not write to output text file, encountered error: ", e)

    # Close all files
    output_text_file.close()

    return output_text_file_path


def PDFtoTXT(input_file):

    print("\n\nProcessing Document - PDF to TXT\n\n")

    try:
        read_return = read_config(['pdfs_to_txts'])
        pdfs_to_txts = read_return['pdfs_to_txts']
    except Exception as e:
        handle_local_error("Missing pdfs_to_txts directory for PDFtoTXT in config.json, encountered error: ", e)
    
    # Initialize PDF file reader
    try:
        pdf_file = open(input_file, 'rb')
    except Exception as e:
        handle_local_error("Could not open PDF file, encountered error: ", e)

    try:
        source_filename = os.path.basename(input_file)
    except Exception as e:
        handle_local_error("Could not open PDF file, encountered error: ", e)

    # Initialize PDF reader
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
    except Exception as e:
        handle_local_error("Could not initialize PDF reader, encountered error: ", e)

    # Set output path
    output_text_file_name = source_filename.replace(".pdf",".txt")
    output_text_file_path = os.path.join(pdfs_to_txts, output_text_file_name).replace("\\","/")

    if os.path.exists(output_text_file_path):
        print("PyPDF2-extracted .txt already exists! Returning existing file.")
        return output_text_file_path

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)

    # Loop through all the pages and extract text
    for page_num in range(len(pdf_reader.pages)):
        
        try:
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
        except Exception as e:
            handle_error_no_return("Could not extract text from page, encountered error: ", e)

        #clean_text = text
        # Clean text
        clean_text = clean_text_string(text)
        page_number = int(page_num) + 1
        
        # Optionally, you can include page numbers in the text file
        # output_text_file.write(f'\n\n--- Page {page_num + 1} ---\n\n')
        
        # Write the extracted text to the file
        try:
            output_text_file.write(f"[PAGE:{page_number}]\n{clean_text}\n")
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)

    # Close all files
    pdf_file.close()
    output_text_file.close()

    return output_text_file_path


def extract_images_from_pdf(pdf_path):
    
    print("Extracting Images from PDF")

    try:
        source_filename = os.path.basename(pdf_path)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)
    
    with open(pdf_path, 'rb') as file:
        
        try:
            pdf_reader = PyPDF2.PdfReader(file)
        except Exception as e:
            handle_local_error("Could not read PDF, encountered error: ", e)

        images = []

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            text = page.extract_text().strip()
            if not text:
                print("Scanned page, skipping")
                continue

            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':

                        # Log details about the image object:
                        try:
                            image_obj = xObject[obj]
                            obj_details = {
                                'Object Reference': obj,
                                'Width': image_obj.get('/Width', 'Unknown'),
                                'Height': image_obj.get('/Height', 'Unknown'),
                                'Color Space': image_obj.get('/ColorSpace', 'Unkown'),
                                'Filter': image_obj.get('/Filter', 'Unknown'),
                                'Bits Per Component': image_obj.get('/BitsPerComponent', 'Unknown')
                            }
                            #print(f"\n\nImage Object Details: {obj_details}\n\n")   # Filter is indicative of format: '/DCTDecode': 'JPEG', '/FlateDecode': 'PNG or others','/JPXDecode': 'JPEG 2000', etc.

                            # data  = image_obj._data

                            if obj_details['Filter'] == '/FlateDecode':
                                #print("\n\nDecoding PNG!\n\n")
                                try:
                                    data  = image_obj._data
                                    decompressed_data = zlib.decompress(data)
                                except Exception as e:
                                    error_message = f"\n\nPNG decompression exception: {e}\n\n"
                                    if logger:
                                        logger.error(error_message)
                                        print(error_message)
                                    else:
                                        print(error_message)
                            else:
                                decompressed_data  = image_obj._data

                            text = page.extract_text()  
                            # clean_text = text

                            # Clean text
                            clean_text = clean_text_string(text)

                            try:
                                if obj_details['Filter'] == '/FlateDecode':
                                    # Determine Color Space:
                                    color_space = image_obj.get('/ColorSpace')

                                    if color_space == '/DeviceRGB':
                                        mode = 'RGB'
                                    elif color_space == '/DeviceCMYK':
                                        mode = 'CMYK'
                                    elif color_space == '/DeviceGray':
                                        mode = 'L'
                                    else:
                                        mode = 'L'  # Default to grayscale if unsure

                                    # Create image from bytes
                                    image = Image.frombytes(mode, ((obj_details['Width']), (obj_details['Height'])), decompressed_data) # 'L' for 8-bit pixels, black and white
                                    with io.BytesIO() as output:
                                        image.save(output, format='JPEG')
                                        binary_data = output.getvalue()
                                        format = "JPEG"
                                        images.append((binary_data,clean_text,format))

                                else:
                                    # Load image from bytes
                                    image = Image.open(io.BytesIO(decompressed_data))

                                    # Determine format (JPEG)
                                    format = image.format
                                    
                                    #print(f"\n\nImage format: {format}\n\n")  # This will print the format

                                    # If image loads, append image to images DB
                                    images.append((decompressed_data,clean_text,format))

                            except Exception as e:
                                error_message = f"\n\nEncountered unrecognized or invalid image data for object detailed below. Exception: {e}\n\n"
                                if logger:
                                    logger.error(error_message)
                                    logger.error(obj_details)
                                    print(error_message)
                                    print(f"\n\nImage Object Details: {obj_details}\n\n")   # Filter is indicative of format: '/DCTDecode': 'JPEG', '/FlateDecode': 'PNG or others','/JPXDecode': 'JPEG 2000', etc.

                                else:
                                    print(error_message)
                                    print(f"\n\nImage Object Details: {obj_details}\n\n")   # Filter is indicative of format: '/DCTDecode': 'JPEG', '/FlateDecode': 'PNG or others','/JPXDecode': 'JPEG 2000', etc.


                        except Exception as e:
                            handle_error_no_return("Could not process image object. Exception: ", e)

        # print("Images array:")
        # print(images)
        return images


def store_images_to_db(images):

    print("\n\nStoring Images to Database\n\n")

    try:
        read_return = read_config(['sqlite_images_db'])
        sqlite_images_db = read_return['sqlite_images_db']
    except Exception as e:
        handle_local_error("Missing sqlite_images_db in config.json for method store_images_to_db. Error: ", e)

    try:
        conn = sqlite3.connect(sqlite_images_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to Images DB, encountered error: ", e)
    
    # If the database does not currently exist...
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    image_data BLOB NOT NULL,
                    surrounding_text TEXT,
                    metadata TEXT,
                    format TEXT
            )
        ''')

        conn.commit()
    except Exception as e:
        handle_local_error("Could not create Images DB, encountered error: ", e)
    
    try:
        for image_data, surrounding_text, format in images:
            #print("surrounding_text: ", surrounding_text)
            # Check if the image_data already exists in the database:
            cursor.execute("SELECT COUNT(*) FROM images WHERE image_data = ?", (image_data,))
            if cursor.fetchone()[0] == 0:
                print("\nInserting new image into images DB\n")
                cursor.execute("INSERT INTO images (image_data, surrounding_text, format) VALUES (?, ?, ?)", (image_data, surrounding_text, format))
        conn.commit()
    except Exception as e:
        handle_local_error("Could not store images to Images DB, encountered error: ", e)
    finally:
        conn.close()


def record_doc_loaded_to_db(document_name, embedding_model, vectordb_used, chunk_size, chunk_overlap):

    print("\n\nRecording document loading to records DB\n\n")

    try:
        read_return = read_config(['sqlite_docs_loaded_db'])
        sqlite_docs_loaded_db = read_return['sqlite_docs_loaded_db']
    except Exception as e:
        handle_local_error("Missing sqlite_docs_loaded_db in config.json for method store_images_to_db. Error: ", e)

    try:
        conn = sqlite3.connect(sqlite_docs_loaded_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to document_records DB, encountered error: ", e)
    
    # If the database does not currently exist...
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_records (
                    id INTEGER PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vectordb_used TEXT,
                    chunk_size INTEGER,
                    chunk_overlap INTEGER
            )
        ''')

        conn.commit()
    except Exception as e:
        handle_local_error("Could not create document_records DB, encountered error: ", e)
    
    try:
        cursor.execute("INSERT INTO document_records (document_name, embedding_model, vectordb_used, chunk_size, chunk_overlap) VALUES (?, ?, ?, ?, ?)", (document_name, embedding_model, vectordb_used, chunk_size, chunk_overlap))
        conn.commit()
        conn.close()
    except Exception as e:
        handle_local_error("Could not update document_records DB, encountered error: ", e)



# List-splitter function for a large number of embeddings!
def split_embeddings_list(all_splits, max_emmbeddings_list_size):
    for i in range(0, len(all_splits), max_emmbeddings_list_size):  # Step through the large list in steps of max size
        yield all_splits[i:i + max_emmbeddings_list_size]   # Yield a slice of all_splits from index i upto but NOT including i+max_size 


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self): #to provide string-representation of an object
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"



def chunk_docs_with_page_numbers(input_file, chunk_size=250):
    documents = []
    current_chunk = ""
    current_page = 1

    def add_chunk(chunk, page):
        if chunk:
            documents.append(Document(
                page_content=chunk.strip(),
                metadata={'source': input_file, 'page_number': page}
            ))

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('[PAGE:'):
                    new_page = int(line.strip()[6:-1])
                    if new_page != current_page:
                        add_chunk(current_chunk, current_page)
                        current_chunk = ""
                        current_page = new_page
                    continue

                if len(current_chunk) + len(line) > chunk_size:
                    add_chunk(current_chunk, current_page)
                    current_chunk = line
                else:
                    current_chunk += line

                if len(current_chunk) >= chunk_size:
                    add_chunk(current_chunk, current_page)
                    current_chunk = ""
        
        # Add any remaining content
        add_chunk(current_chunk, current_page)

    except Exception as e:
        handle_local_error("Could not chunk document, encountered error: ", e)

    return documents


# Document vectorization and chunking
def LoadNewDocument(input_file):

    global VECTOR_STORE
    
    ### L1 - Load Data from Source ###
    print("\nLoading Document")

    try:
        read_return = read_config(['use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder'])
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
    except Exception as e:
        handle_local_error("Missing values in config.json, could not LoadNewDocument. Error: ", e)

    chunk_sz = 250
    chunk_olp = 0

    ### L2 - Chunk Source Data ###
    print("Chunking Doc")
    try:
        numbered_splits = chunk_docs_with_page_numbers(input_file, chunk_sz)

        print(f"\n\nnumbered_splits sample: {numbered_splits[:3]}\n\n")
        print(f"\n\nnumbered_splits type: {type(numbered_splits[3])}\n\n")
        # print(f"\n\nall_splits sample: {all_splits[:3]}\n\n")
        # print(f"\n\nall_splits type: {type(all_splits[3])}\n\n")
        
    except Exception as e:
        handle_local_error("Failed to chunk document for storage to VectorDB, encountered error: ", e)

    ### L3 - Store Chunks in VectorDB ###
    print("Storing to VectorDB: ChromaDB")
    try:
        # Return VectorStore initialized from documents and embeddings.
        if use_sbert_embeddings:
            # Ideally should use MAX_BATCH_SIZE obtained elsewhere 
            if len(numbered_splits) > 5000:
                split_docs = split_embeddings_list(numbered_splits, 5000)
                for split_docs_list in split_docs:
                    VECTOR_STORE = Chroma.from_documents(documents=split_docs_list, embedding=HuggingFaceEmbeddings(), persist_directory=vectordb_sbert_folder)
            else:
                VECTOR_STORE = Chroma.from_documents(documents=numbered_splits, embedding=HuggingFaceEmbeddings(), persist_directory=vectordb_sbert_folder)
        
        elif use_openai_embeddings:
            print("Using OpenAI Text Ada Model via Azure OpenAI")

            list_position = 0
            token_count = 0

            for i in range(list_position, len(numbered_splits)):

                token_count += len(str(numbered_splits[i]))
                if token_count >= 108000:
                    VECTOR_STORE = Chroma.from_documents(documents=numbered_splits[list_position:i+1], embedding=AZURE_OPENAI_EMBEDDINGS, persist_directory=vectordb_openai_folder)  #AZURE_OPENAI_EMBEDDINGS defined on line 407
                    list_position = i+1
                    token_count = 0
                    print("Loaded batch, sleeping for one minute to stay within rate-limit")
                    time.sleep(63)
                    continue

            # post-loop, if any splits are left to be processed but were missed due to token_count not reaching the limit:
            if list_position < len(numbered_splits):
                VECTOR_STORE = Chroma.from_documents(documents=numbered_splits[list_position:], embedding=AZURE_OPENAI_EMBEDDINGS, persist_directory=vectordb_openai_folder) #AZURE_OPENAI_EMBEDDINGS defined on line 407

        elif use_bge_base_embeddings or use_bge_large_embeddings:
            persist_directory = ""
            if use_bge_base_embeddings:
                persist_directory = vectordb_bge_base_folder
            elif use_bge_large_embeddings:
                persist_directory = vectordb_bge_large_folder
            VECTOR_STORE = Chroma.from_documents(documents=numbered_splits, embedding=HF_BGE_EMBEDDINGS, persist_directory=persist_directory)    #HF_BGE_EMBEDDINGS defined in process_model() line 2133

    except Exception as e:
        handle_local_error("Could not store to VectorDB, encountered error: ", e)

    return chunk_sz, chunk_olp


def find_images_in_db(reference_pages):

    print("\nSearching for relevant Images\n")

    try:
        read_return = read_config(['sqlite_images_db'])
        sqlite_images_db = read_return['sqlite_images_db']
    except Exception as e:
        handle_local_error("Missing sqlite_images_db in config.json for method find_images_in_db. Error: ", e)

    matched_images = []
    matched_images_found = False

    try:
        conn = sqlite3.connect(sqlite_images_db)
        conn.row_factory = sqlite3.Row
        print("Database connected for image search")
    except Exception as e:
        handle_local_error("Could not connect to images DB for image search, encountered error: ", e)

    for doc in reference_pages:
                
        #source_filename = os.path.basename(doc)

        for search_string in reference_pages[doc]:

            # Only search for non-empty search strings
            if search_string:
                try:
                    images = conn.execute('SELECT DISTINCT id, image_data FROM images WHERE surrounding_text LIKE ?', ('%' + search_string + '%',)).fetchall()
                except Exception as e:
                    handle_error_no_return("Could not select images from Images DB, encountered error: ", e)
                for row in images:
                    print("Matching image found!")
                    matched_images_found = True
                    image_id = row['id']
                    image_data = row['image_data']
                    matched_images.append((image_id, image_data))

    conn.close()
    matched_images = set(matched_images)
    return matched_images_found, matched_images


def highlight_text_on_page(highlight_list, stream_session_id):

    print("\nHighlighting Document\n")
    threshold = 80

    try:
        read_return = read_config(['upload_folder', 'highlighted_docs'])
        upload_folder = read_return['upload_folder']
        highlighted_pdfs_path = read_return['highlighted_docs']
    except Exception as e:
        handle_local_error("Missing upload_folder in config.json for method highlight_text_on_page. Error: ", e)
    
    for index, doc in enumerate(highlight_list, start=1):

        try:
            pdf_path = os.path.join(upload_folder, doc).replace("\\","/")
            output_file_extension = "_" + stream_session_id + '.pdf'
            output_file_name = doc.replace(".pdf",output_file_extension) 
            output_pdf_path = os.path.join(highlighted_pdfs_path, output_file_name).replace("\\","/")
            highlight_doc = fitz.open(pdf_path)
        except Exception as e:
            handle_error_no_return("Could not open doc for highlighting, encountered error: ", e)
            continue
        
        for target in highlight_list[doc]:
            try:
                text_to_highlight = str(target[1])
                text_to_highlight = re.sub(r'Row \d+, Column \d+: ', '', text_to_highlight)
                page_number = int(target[0])
                page = highlight_doc.load_page(page_number-1)
                page_text = page.get_text("text")

                # Split the page text into overlapping phrases
                words = page_text.split()
                phrases = [' '.join(words[i:i+len(text_to_highlight.split())]) for i in range(len(words))]

                # Find fuzzy matches
                good_matches = []
                for phrase in phrases:
                    score = fuzz.partial_ratio(text_to_highlight.lower(), phrase.lower())
                    if score >= threshold:
                        good_matches.append(phrase)

                for match in good_matches:
                    if len(str(match)) > 3:
                        text_instances = page.search_for(match)
                        for inst in text_instances:
                            try:
                                #print(f"HIGHLIGHTING inst {inst} in document {doc}")
                                page.add_highlight_annot(inst)
                            except Exception as e:
                                handle_error_no_return("Could not highlight text instance, encountered error: ", e)
                                continue

            except Exception as e:
                handle_error_no_return("Error loading page or searching for text to highlight, encountered error: ", e)
                continue
            
        try:
            highlight_doc.save(output_pdf_path, garbage=0, deflate=False, clean=False)
        except Exception as e:
            handle_error_no_return("Could not save highlighted doc, encountered error: ", e)
            continue

    return True


def highlighter_interface(reference_pages, stream_session_id):

    user_should_refer_pages_in_doc = {}
    highlight_list = {}
    docs_have_relevant_info = False

    print(f"\n\nreference_pages: {reference_pages}\n\n")

    for file_path, content in reference_pages.items():
        source_filename = os.path.basename(file_path)
        print(f"\nsource_filename basename: {source_filename}\n")
        output_file_extension = "_" + stream_session_id + '.pdf'
        output_file_name = source_filename.replace(".pdf",output_file_extension) 
        page_numbers = set()
        highlight_strings = set()

        for item in content:
            # Each item in the list has two elements
            page_text, page_number = item
            page_numbers.add(int(page_number))
            highlight_strings.add((int(page_number), str(page_text[:50])))

        if page_numbers:
            user_should_refer_pages_in_doc[output_file_name] = page_numbers
            docs_have_relevant_info = True

        if highlight_strings:
            highlight_list[source_filename] = list(highlight_strings)

    if docs_have_relevant_info:
        try:
            highlight_text_on_page(highlight_list, str(stream_session_id))
        except Exception as e:
            handle_error_no_return("Could not highlight text, encountered error: ", e)

    return docs_have_relevant_info, user_should_refer_pages_in_doc


def determine_sequence_id_for_chat(chat_id):

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method store_chat_history_to_db. Error: ", e)

    # Connect to or create the DB
    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to DB for chat history storage, encountered error: ", e)

    try:
        # Determine sequence_id
        cursor.execute("SELECT COALESCE(MAX(sequence_id), 0) FROM chat_history WHERE chat_id = ?", (int(chat_id),))
        # "The COALESCE function accepts two or more arguments and returns the first non-null argument."
        # This accounts for a new chat!
        # Note that trailing comma! Without it, the simple select query will produce an error: "parameters are of unsupported type" !!
        # This is because the SQLite3 module can have trouble recognizing single-item tuples as tuples, so a trailing comma helps alleviate this! 

        result = cursor.fetchone()
        current_sequence_id = result[0]     # 'result' will be a list, so extract the first value
        
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)

    return int(current_sequence_id)


def store_llama_cpp_chat_history_to_db(chat_id, sequence_id, user_query_for_history_db, model_response_for_history_db, current_prompt_template):

    global SEQUENCE_ID

    print(f"\n\nStoring chat history for chat with CHAT_ID: {chat_id}")

    try:
        read_return = read_config(['sqlite_history_db', 'model_choice', 'base_template'])
        sqlite_history_db = read_return['sqlite_history_db']
        model_choice = read_return['model_choice']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method store_chat_history_to_db. Error: ", e)

    # Connect to or create the DB
    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to DB for chat history storage, encountered error: ", e)

    try:
        prev_sequence_id = determine_sequence_id_for_chat(chat_id)
        #print("prev_sequence_id: ", prev_sequence_id)
        SEQUENCE_ID = prev_sequence_id + 1
        #print("current_sequence_id: ", SEQUENCE_ID)
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)
       
    # print(type(CHAT_ID))
    # print(type(current_sequence_id))
    # print(type(user_query_for_history_db))
    # print(type(model_response_for_history_db))

    try:
        # Store conversation history into DB
        cursor.execute("INSERT INTO chat_history (chat_id, sequence_id, user_query, llm_response, llm_model, prompt_template) VALUES (?, ?, ?, ?, ?, ?)", (int(chat_id), int(sequence_id), user_query_for_history_db, model_response_for_history_db, model_choice, str(current_prompt_template)))
        conn.commit()
    except Exception as e:
        handle_local_error("Could not insert chat history into DB, encountered error: ", e)



def store_chat_history_to_db(user_query_for_history_db, model_response_for_history_db, current_historical_summary):

    global SEQUENCE_ID

    print(f"\n\nStoring chat history for chat with CHAT_ID: {CHAT_ID}")

    try:
        read_return = read_config(['sqlite_history_db', 'model_choice', 'base_template'])
        sqlite_history_db = read_return['sqlite_history_db']
        model_choice = read_return['model_choice']
        base_template = read_return['base_template']
    except Exception as e:
        handle_local_error("Missing keys in config.json for method store_chat_history_to_db. Error: ", e)

    # Connect to or create the DB
    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        handle_local_error("Could not establish connection to DB for chat history storage, encountered error: ", e)

    try:
        prev_sequence_id = determine_sequence_id_for_chat(CHAT_ID)
        #print("prev_sequence_id: ", prev_sequence_id)
        SEQUENCE_ID = prev_sequence_id + 1
        #print("current_sequence_id: ", SEQUENCE_ID)
    except Exception as e:
        handle_local_error("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)
       
    # print(type(CHAT_ID))
    # print(type(current_sequence_id))
    # print(type(user_query_for_history_db))
    # print(type(model_response_for_history_db))

    try:
        # Store conversation history into DB
        cursor.execute("INSERT INTO chat_history (chat_id, sequence_id, user_query, llm_response, llm_model, prompt_template, history_summary) VALUES (?, ?, ?, ?, ?, ?, ?)", (int(CHAT_ID), int(SEQUENCE_ID), user_query_for_history_db, model_response_for_history_db, model_choice, str(base_template), str(current_historical_summary)))
        conn.commit()
    except Exception as e:
        handle_local_error("Could not insert chat history into DB, encountered error: ", e)

    conn.close()


@app.route('/login_to_google_drive')
def login_to_google_drive():
    global GDRIVE_CREDS
    if os.path.exists("gdrive_token.json"):
        GDRIVE_CREDS = Credentials.from_authorized_user_file("gdrive_token.json", GDRIVE_SCOPES)
    if not GDRIVE_CREDS or not GDRIVE_CREDS.valid:
        if GDRIVE_CREDS and GDRIVE_CREDS.expired and GDRIVE_CREDS.refresh_token:
            GDRIVE_CREDS.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "gdrive_credentials.json", GDRIVE_SCOPES
            )
            GDRIVE_CREDS = flow.run_local_server(port=0)
            with open("gdrive_token.json", "w") as token:
                token.write(GDRIVE_CREDS.to_json())
    return jsonify(success=True)


def categorize_mimetype(mimetype):
    mimetype = mimetype.lower()

    word_mimetypes = {
        # Microsoft Word (modern formats)
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-word.document.macroEnabled.12",
        "application/vnd.ms-word.template.macroEnabled.12",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",

        # Microsoft Word (legacy format)
        "application/msword",

        # Google Docs
        "application/vnd.google-apps.document",

        # OpenDocument Presentation
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.text-template",
        "application/vnd.oasis.opendocument.text-web"
    }

    excel_mimetypes = {
        # Microsoft Excel (modern formats)
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroEnabled.12",
        "application/vnd.ms-excel.template.macroEnabled.12",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.template",

        # Microsoft Excel (legacy format)
        "application/vnd.ms-excel",

        # Google Sheets
        "application/vnd.google-apps.spreadsheet",

        # OpenDocument Presentation
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.spreadsheet-template",
        "text/csv",
        "text/tab-separated-values"
    }

    presentation_mimetypes = {
        # Microsoft PowerPoint (modern formats)
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
        "application/vnd.ms-powerpoint.presentation.macroEnabled.12",  # .pptm
        "application/vnd.openxmlformats-officedocument.presentationml.template",  # .potx
        "application/vnd.ms-powerpoint.template.macroEnabled.12",  # .potm
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",  # .ppsx
        "application/vnd.ms-powerpoint.slideshow.macroEnabled.12",  # .ppsm

        # Microsoft PowerPoint (legacy format)
        "application/vnd.ms-powerpoint",  # .ppt, .pot, .pps

        # Google Slides
        "application/vnd.google-apps.presentation",

        # OpenDocument Presentation
        "application/vnd.oasis.opendocument.presentation",  # .odp
        "application/vnd.oasis.opendocument.presentation-template"  # .otp
    }

    pdf_mimetypes = {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "application/vnd.pdf",
        "text/pdf",
        "text/x-pdf"
    }

    text_mimetypes = {
        "application/rtf",
        "text/rtf",
        "text/plain"
    }

    image_mimetypes = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/svg+xml",
        "image/tiff",
        "image/x-icon",
        "image/vnd.microsoft.icon",
        "image/heic",
        "image/heif"
    }

    video_mimetypes = {
        "video/mp4",
        "video/mpeg",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-ms-wmv",
        "video/x-flv",
        "video/webm",
        "video/3gpp",
        "video/3gpp2",
        "video/x-matroska"
    }

    audio_mimetypes = {
        "audio/mpeg",
        "audio/x-wav",
        "audio/wav",
        "audio/x-m4a",
        "audio/aac",
        "audio/ogg",
        "audio/webm",
        "audio/flac",
        "audio/x-ms-wma",
        "audio/x-aiff"
    }

    folder_mimetypes = {
        "application/vnd.google-apps.folder",  # Google Drive folder
        "application/x-directory",             # Generic directory mime type
        "inode/directory",                     # Often used in Unix-like systems
        "folder",                              # Some systems might use this
    }

    # Word file
    if mimetype in word_mimetypes:
        return "word"
    # Excel files
    elif mimetype in excel_mimetypes:
        return "excel"
    # Presentation files
    elif mimetype in presentation_mimetypes:
        return "presentation"
    # Text files
    elif mimetype in text_mimetypes:
        return "text"
    # PDFs
    elif mimetype in pdf_mimetypes:
        return "pdf"
    # Images
    elif mimetype in image_mimetypes:
        return "image"
    # Videos
    elif mimetype in video_mimetypes:
        return "video"
    # Audio files
    elif mimetype in audio_mimetypes:
        return "audio"
    # Folders
    elif mimetype in folder_mimetypes:
        return "folder"
    # Other
    else:
        return "other"

@app.route('/fetch_file_list_from_google_drive')
def fetch_file_list_from_google_drive():
    gdrive_files = []
    try:
        service = build("drive", "v3", credentials=GDRIVE_CREDS)

        about_result = service.about().get(fields="storageQuota,user").execute()
        print(f"about_result: {about_result}")

        results = (
            service.files().list(
                q="trashed=false",
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, version)"
            ).execute()
        )
        
        items = results.get("files", [])
        print(f"len(items): {len(items)}")

        if not items:
            print("No files found.")
            return jsonify(success=True, gdrive_files=gdrive_files)
        else:
            # print("\n\nFiles:\n\n")
            # print("Name       ID      mimeType        fileExtension       Category      version")
            for item in items:
                category = categorize_mimetype(item['mimeType'])
                #print(f"{item['name']}      ({item['id']})      {item['mimeType']}      {category}        {item['version']}")
                gdrive_files.append({
                    'name': item['name'],
                    'id': item['id'],
                    'mimeType': item['mimeType'],
                    'version': item['version'],
                    'type': category
                })

    except Exception as e:
        return handle_api_error("Could not fetch GDrive files, encountered error: ", e)
    
    return jsonify({'success': True, 'gdrive_files': gdrive_files})


def download_folder(service, folder_id, path, indent=''):

    print(f"\n\nDownloading GoogleDrive Folder with id: {folder_id}\n\n")

    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        return handle_api_error("Server-side error - could not create nested directory in the download_folder() method: ", e)

    query = f"'{folder_id}' in parents"
    fields = "files(id, name, mimeType)"
    
    gdrive_folder_contents = service.files().list(q=query, fields=fields).execute()
    items = gdrive_folder_contents.get('files', [])

    for item in items:
        file_id = item['id']
        filename = item['name']
        mime_type = item['mimeType']
        print(f"folder item mime_type f{mime_type}")

        if "folder" in str(mime_type):
            sub_folder_path = os.path.join(path, filename)    # in this case, filename will be the folder name

            try:
                download_folder(service, file_id, sub_folder_path)  # in this case, file_or_folder_id will be the folder id
            except Exception as e:
                return handle_api_error("Could not download_folder in the download_folder() method, encountered error: ", e)
        else:
            try:
                filename_with_extension, file_content = download_gdrive_file(service, file_id, filename, mime_type)
            except Exception as e:
                return handle_api_error("Server-side error - could not get_file_content in the download_folder() method: ", e)

            try:
                # filepath = os.path.join(path, secure_filename(filename_with_extension))
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename_with_extension))
                print(f"Saving {filename_with_extension} to {filepath}")
                with open(filepath, 'wb') as f:
                    f.write(file_content)

                try:
                    vector_embed_filepath(filename_with_extension, filepath)
                except Exception as e:
                    return handle_api_error("Could not vector_embed_filepath() in the download_folder() method, encountered error: ", e)

            except Exception as e:
                return handle_api_error("Server-side error - could not save Google Drive file in the download_folder() method: ", e)

    return True


def download_gdrive_file(service, file_id, filename, mime_type):

    print(f"\n\nDownloading GoogleDrive File with mime_type: {mime_type}\n\n")

    file_mime_category = categorize_mimetype(mime_type)

    filename_with_extension = filename

    try:
        if file_mime_category in ["word", "excel", "presentation"]:
            # Handle Google Docs Editors files
            if 'google-apps' in mime_type:
                if file_mime_category == "word":
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    file_extension = '.docx'
                elif file_mime_category == "excel":
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    file_extension = '.xlsx'
                elif file_mime_category == "presentation":
                    export_mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    file_extension = '.pptx'
                
                print(f"Downloading google-apps file with mimeType {file_mime_category}")
                gdrive_request = service.files().export_media(fileId=file_id, mimeType=export_mime_type)
                if not filename.endswith(file_extension):
                    filename_with_extension += file_extension
            
            else:
                # For non-Google formats, use get_media
                print(f"Downloading office file with non google-apps mimeType {file_mime_category}")
                gdrive_request = service.files().get_media(fileId=file_id)
        else:
            print(f"Downloading non-office file with mimeType {file_mime_category}")
            gdrive_request = service.files().get_media(fileId=file_id)
        
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, gdrive_request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
    
    except Exception as e:
        return handle_api_error("Error downloading file from Google Drive in the download_gdrive_file() method: ", e)
    
    try:
        file_content = file.getvalue()
    except Exception as e:
        return handle_api_error("Server-side error - could not getValue() for downloaded Google Drive file  in the download_gdrive_file() method: ", e)

    return filename_with_extension, file_content


def gdrive_downloader(service, file_or_folder_id, filename, mime_type, path=app.config['UPLOAD_FOLDER']):
    file_mime_category = categorize_mimetype(mime_type)

    if file_mime_category == "folder":
        download_path = os.path.join(path, secure_filename(filename))    # in this case, filename will be the folder name
        download_folder(service, file_or_folder_id, download_path)
        return filename, None    # Return None for file_content as it's a folder
    else:
        filename_with_extension, file_content = download_gdrive_file(service, file_or_folder_id, filename, mime_type)
        return filename_with_extension, file_content


@app.route('/google_drive_loader', methods=['POST'])
def google_drive_loader():

    try:
        gdrive_file_id = str(request.form['file_id'])
        gdrive_file_mimeType = str(request.form['file_mimeType'])
    except Exception as e:
        return handle_api_error("Server-side error reading Google Drive file details for download in the google_drive_loader() method: ", e)
    
    try:
        service = build("drive", "v3", credentials=GDRIVE_CREDS)
    except Exception as e:
        return handle_api_error("Could not create Google service handler in the google_drive_loader() method, check credentials and re-try: ", e)

    try:
        file_metadata = service.files().get(fileId=gdrive_file_id, fields='name, mimeType').execute()
        original_filename = file_metadata.get('name', 'untitled')
        mime_type = file_metadata.get('mimeType', gdrive_file_mimeType)
    except Exception as e:
        return handle_api_error("Could not read GoogleDrive file metadata in the google_drive_loader() method, encountered error: ", e)
    
    try:
        filename_with_extension, file_content = gdrive_downloader(service, gdrive_file_id, original_filename, mime_type)
    except Exception as e:
        return handle_api_error("Server-side error - could not getValue() for downloaded Google Drive file in the google_drive_loader() method: ", e)

    if file_content is not None:
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename_with_extension))

            print(f"Saving {filename_with_extension} to {filepath}")
            with open(filepath, 'wb') as f:
                f.write(file_content)

            try:
                vector_embed_filepath(filename_with_extension, filepath)
            except Exception as e:
                return handle_api_error("Could not vector_embed_filepath() in the google_drive_loader() method, encountered error: ", e)

        except Exception as e:
            return handle_api_error("Server-side error - could not save file downloaded from Google Drive in the google_drive_loader() method: ", e)
    
    return jsonify({'success': True})


# Route for loading all models from model dir
@app.route('/load_local_models')
def load_local_models():

    try:
        read_return = read_config(['model_dir'])
        model_dir = read_return['model_dir']
    except Exception as e:
        return handle_api_error("Missing model_dir in config.json for method load_local_models. Error: ", e)
    
    try:
        models = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    except Exception as e:
        return handle_api_error("Could not load list of local models, encountered error: ", e)
        
    #print(f"locally available models: {models}")
    return jsonify({'success': True, 'models': models})


@app.route('/upload_new_llm', methods=['POST'])
def upload_new_llm():

    try:
        read_return = read_config(['model_dir'])
        model_dir = read_return['model_dir']
    except Exception as e:
        return handle_api_error("Could not determine model_dir in upload_new_llm. Error: ", e)

    try:
        input_file = request.files['file']
    except Exception as e:
        return handle_api_error("Server-side error recieving LLM file: ", e)

    # Ensure the filename is secure
    filename = secure_filename(input_file.filename)

    try:
        filepath = os.path.join(model_dir, filename)

        print("Loading new LLM - filename: ", filename)
        print("Loading new LLM - filepath: ", filepath)

        # Save the uploaded file to the specified path
        input_file.save(filepath)
    except Exception as e:
        return handle_api_error("Failed to save LLM to model_dir, encountered error: ", e)

    return jsonify(success=True)


# Route to handle the submission of the first form (LLM & embeddings model and GPU selection)
@app.route('/process_model', methods=['POST'])
def process_model():
    
    global HF_BGE_EMBEDDINGS

    ###---New config.json---###

    config_update_dict = {}

    use_azure_open_ai = 'use_azure' in request.form
    use_openai_embeddings = 'use_openai_embeddings' in request.form
    use_sbert_embeddings = 'use_sbert_embeddings' in request.form
    use_bge_large_embeddings = 'use_bge_large_embeddings' in request.form
    use_bge_base_embeddings = 'use_bge_base_embeddings' in request.form
    use_gpu_for_embeddings = request.form.get('use_gpu_for_embeds', False)    # default no
    model_choice = str(request.form['model_choice'])
    use_gpu = request.form.get('use_gpu', False)

    config_update_dict.update({'use_azure_open_ai':use_azure_open_ai, 'use_openai_embeddings':use_openai_embeddings, 'use_sbert_embeddings':use_sbert_embeddings, 'use_bge_large_embeddings':use_bge_large_embeddings, 'use_bge_base_embeddings':use_bge_base_embeddings, 'use_gpu_for_embeddings':use_gpu_for_embeddings, 'model_choice':model_choice, 'use_gpu':use_gpu})

    try:
        if use_bge_base_embeddings or use_bge_large_embeddings:
            model_name = ""
            if use_bge_base_embeddings:
                model_name = "BAAI/bge-base-en"
            elif use_bge_large_embeddings:
                model_name = "BAAI/bge-large-en"
            model_kwargs = {}
            if use_gpu_for_embeddings:
                model_kwargs.update({"device": "cuda"})
            else:
                model_kwargs.update({"device": "cpu"})
            encode_kwargs = {"normalize_embeddings": True}
            HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
    except Exception as e:
        return handle_api_error("Could not load BGE embeddings in process_model, encountered error: ", e)
    
    try:
        write_config(config_update_dict)
    except Exception as e:
        handle_local_error("Could not write updates to config.json, encountered error: ", e)

    # Redirect to the next step
    return redirect(url_for('load_file'))


def convert_to_pdf_with_unoconv(input_file_path, output_file_path):
    print("\n\nConverting non-PDF document to PDF format\n\n")
    if platform.system() == 'Windows':
        subprocess.run(['python', 'unoconv.py', '-f', 'pdf', '-o', output_file_path, input_file_path], check=True)
    else:
        subprocess.run(['unoconv', '-f', 'pdf', '-o', output_file_path, input_file_path], check=True)


def reload_vector_store():
    global VECTOR_STORE
    print("\nRe-Loading VectorDB: ChromaDB")

    vectordb_used = ""

    try:
        read_return = read_config(['use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder', 'embedding_model_choice'])
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
        embedding_model_choice = read_return['embedding_model_choice']
    except Exception as e:
        handle_local_error("Missing values in config.json when reloading VectorDB, could not fully complete process_new_file. Please try restarting the application. Error: ", e)

    try:
        if use_sbert_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_sbert_folder, embedding_function=HuggingFaceEmbeddings())
            vectordb_used = vectordb_sbert_folder
        elif use_openai_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_openai_folder, embedding_function=AZURE_OPENAI_EMBEDDINGS)
            vectordb_used = vectordb_openai_folder
        elif use_bge_base_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
            vectordb_used = vectordb_bge_base_folder
        elif use_bge_large_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
            vectordb_used = vectordb_bge_large_folder
    except Exception as e:
        handle_local_error("Could not reload VectorDB when trying to process_new_file. Please try restarting the application. Error: ", e)
    
    return embedding_model_choice, vectordb_used


def convert_non_pdf_to_pdf_with_unoconv(filename, filepath):
    print("Converting to PDF file")

    try:
        conv_filename = os.path.splitext(filename)[0] + '.pdf'
        conv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], conv_filename)

        convert_to_pdf_with_unoconv(filepath, conv_filepath)

        return conv_filename, conv_filepath
    except subprocess.CalledProcessError as e:
        return handle_api_error("Could not convert file to PDF, encountered error: ", e)
    except Exception as e:
        return handle_api_error("Unexpected error when converting file to PDF, encountered error: ", e)


def vector_embed_filepath(filename, filepath):
    print("Vector Embedding Document")

    if not filename.lower().endswith('.pdf'):
        _, filepath = convert_non_pdf_to_pdf_with_unoconv(filename, filepath)

    use_ocr = False
    try:
        read_return = read_config(['use_ocr', 'ocr_service_choice'])
        use_ocr = read_return['use_ocr']
        ocr_service_choice = read_return['ocr_service_choice']
    except Exception as e:
        handle_local_error("Could not determine use_ocr in config.json for process_new_file. Disabling OCR and proceeding. Error: ", e)
    
    print("Processing PDF file")
    
    if use_ocr:
        try:
            if ocr_service_choice == 'AzureVision':
                input_file = PDFtoAzureOCRTXT(filepath)
            elif ocr_service_choice == 'AzureDocAi':
                input_file = PDFtoAzureDocAiTXT(filepath)
        except Exception as e:
            handle_error_no_return("Failed to OCR text from PDF. Will now attempt to extract text via PyPDF2. Encountered error: ", e)
            try:
                input_file = PDFtoTXT(filepath)
            except Exception as e:
                handle_local_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
    else:
        try:
            input_file = PDFtoTXT(filepath)
        except Exception as e:
            handle_local_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
    
    # try:
    #     images = extract_images_from_pdf(filepath)
    # except Exception as e:
    #     handle_error_no_return("Failed to extract images from the PDF document, encountered error: ", e)

    # try:
    #     store_images_to_db(images)
    # except Exception as e:
    #     handle_error_no_return("Failed to save images to database, encountered error: ", e)
    
    try:
        chunk_size, chunk_overlap = LoadNewDocument(input_file)
    except Exception as e:
        handle_local_error("Failed to extract text from PDF: ", e)
    
    try:
        embedding_model_choice, vectordb_used = reload_vector_store()
    except Exception as e:
        handle_local_error("Could not reload vector store when attempting to vector_embed_filepath(), encountered error: ", e)

    try:
        record_doc_loaded_to_db(filename, embedding_model_choice, vectordb_used, chunk_size, chunk_overlap)
    except Exception as e:
        handle_error_no_return("Unable to record document loading to records DB, encountered error: ", e)

    return True


# Route to handle the submission of the second form (file loading)
@app.route('/process_new_file', methods=['POST'])
def process_new_file():

    try:
        input_file = request.files['file']
    except Exception as e:
        return handle_api_error("Server-side error recieving file: ", e)

    # Ensure the filename is secure
    filename = secure_filename(input_file.filename)
    if "PDF" in filename:
        filename = filename.replace("PDF", "pdf")

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print("Loading new file - filename: ", filename)
        print("Loading new file - filepath: ", filepath)

        # Save the uploaded file to the specified path
        input_file.save(filepath)
    except Exception as e:
        return handle_api_error("Failed to save document to app folder, encountered error: ", e)

    try:
        vector_embed_filepath(filename, filepath)
    except Exception as e:
        return handle_api_error("Could not vector_embed_filepath() in the process_new_file() method, encountered error: ", e)

    return jsonify(success=True)


# Route to store user rating: 
# ATTN: comment out print() statements, as users may elect to leave a rating as a response is being generated, which is when the stdout is redirected to the event stream! 
@app.route('/store_user_rating', methods=['POST'])
def store_user_rating():
    
    # print("Stroing user rating")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json for method store_user_rating. Error: ", e)
    
    try:
        user_rating = request.form['rating']
        chat_id_for_rating = request.form['chat_id']
        sequence_id_for_rating = request.form['sequence_id']
    except Exception as e:
        return handle_api_error("Server-side error, could not read user rating or failed to obtain chat/sequence ID, encountered error: ", e)

    # print("user_rating: ", user_rating)
    # print("chat_id_for_rating: ", chat_id_for_rating)
    # print("sequence_id_for_rating: ", sequence_id_for_rating)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        cursor = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history DB for storage of user-rating, encountered error: ", e)

    try:
        cursor.execute(
            '''
            UPDATE chat_history
            SET user_rating = ?
            WHERE chat_id = ? AND sequence_id = ?
            ''',
            (user_rating, chat_id_for_rating, sequence_id_for_rating)
        )
        conn.commit()
    except Exception as e:
        return handle_api_error("Could not store user-rating to chat history db, encountered error: ", e)

    conn.close()

    return jsonify(success=True)



def is_llama_cpp_local_server_online():
    try:
        response = requests.get('http://localhost:8080/health')
        
        if response.status_code == 200:
            data = response.json()  # parse the JSON response to determine the server status
            if data['status'] == 'ok':
                print(f"Server ready: {data['slots_idle']} idle slots, {data['slots_processing']} processing slots.")
                return {"server_available":True, "loading_model":False, "status_code":200}
            elif data['status'] == 'no slot available':
                print("No slots available. Server is running but cannot handle more requests.")
                return {"server_available":False, "loading_model":False, "status_code":200}
            
        elif response.status_code == 503:   # model still loading or no slots
            data = response.json()
            if data['status'] == 'loading model':
                print("Server is loading the selected LLM, please wait")
                return {"server_available":False, "loading_model":True, "status_code":503}
            else:
                print("No slots available. Server is running but cannot handle more requests.")
                return {"server_available":False, "loading_model":False, "status_code":503}
        
        elif response.status_code == 500:
            print("Server error: Failed to load LLM.")
            logger.error("llama.cpp - 500 event")
            return {"server_available":False, "loading_model":False, "status_code":500}
        
        else:
            return {"server_available":False, "loading_model":False, "status_code":500}
    
    except requests.exceptions.ConnectionError as e:
        error_message = "\n\nECONNREFUSED event\n\n"
        if logger:
            logger.error(error_message)
            print(error_message)
        else:
            print(error_message)
        return {"server_available":False, "loading_model":True, "status_code":500}
    except Exception as e:
        error_message = f"\n\nCould not check llama.cpp local-server health, encountered error: {e}\n\n"
        if logger:
            logger.error(error_message)
            print(error_message)
        else:
            print(error_message)
        return {"server_available":False, "loading_model":False, "status_code":500}
    

def send_ctrl_c_to_process(process):
    if process.poll() is None:  # check if process is still running via poll(), which returns None if a process is still running 
        if platform.system() == 'Windows':
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        try:
            # Wait a bit for the process to terminate gracefully:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("Process did not terminate within timeout, will be force-killed.")
            process.kill()  # Sends 'SIGKILL' on Unix-like to force-kill immediately / 'TerminateProcess' on Windows which still allows for graceful termination
            process.wait()
            if process.poll() is not None:
                print("Process has been killed successfully.")
            else:
                print("Process still running after force kill attempt.")


def terminate_llama_cpp_process(process):
    try:
        # process.terminate() sends 'SIGTERM' on Unix-like systems / 'TerminateProcess' on Windows, allows for graceful termination
        # process.wait()
        send_ctrl_c_to_process(process)
        if process.poll() is not None:  # process has indeed terminated
            print("Process terminated gracefully.")
    except Exception as e:
        handle_local_error("Failed to terminate llama.cpp process, encountered error: ", e)


@app.route('/llama_cpp_server_starter')
def llama_cpp_server_starter():

    global LLM_CHANGE_RELOAD_TRIGGER_SET
    global LLAMA_CPP_PROCESS
    global LLM_LOADED_UP

    if LLM_LOADED_UP and not LLM_CHANGE_RELOAD_TRIGGER_SET:
        model_choice = 'undefined'
        try:
            read_return = read_config(['model_choice'])
            model_choice = read_return['model_choice']
        except Exception as e:
            handle_error_no_return("Missing model_choice in config.json when attempting to return without re-loading. Printing error and proceeding: ", e)
        print(f'\n\nAlready loaded! Simply returning model choice: {model_choice}\n\n')
        return jsonify({'success': True, 'llm_model': model_choice})
    elif LLM_CHANGE_RELOAD_TRIGGER_SET:
        print('\n\nProceeding to reload the LLM & resetting the LLM_CHANGE_RELOAD_TRIGGER_SET flag.\n\n')
        LLM_CHANGE_RELOAD_TRIGGER_SET = False

    try:
        if is_llama_cpp_local_server_online()['server_available']:
            print("Server online. Terminating and reloading from config.json")
            try:
                terminate_llama_cpp_process(LLAMA_CPP_PROCESS)
                LLAMA_CPP_PROCESS = None
            except Exception as e:
                LLM_LOADED_UP = True
                handle_error_no_return("Failed to terminate running llama.cpp process, server was likely launched by a previous session. Retruning with the currently loaded LLM. To change, shutdown the previously launched server manually and reload this page. Technical error-details follow: ", e)
                try:
                    read_return = read_config(['model_choice'])
                    model_choice = read_return['model_choice']
                    return jsonify({'success': True, 'llm_model': model_choice})
                except Exception as e:
                    handle_error_no_return("Missing values in config.json when preparing to launch llama.cpp server, encountered error: ", e)
                return jsonify({'success': True, 'llm_model': 'undefined'})
                
                
    except Exception as e:
        handle_error_no_return("Could not pre-check if llama.cpp server is running, it may be offline. Printing error and proceeding: ", e)


    try:
        read_return = read_config(['model_dir', 'model_choice', 'local_llm_context_length', 'local_llm_max_new_tokens', 'local_llm_gpu_layers', 'server_timeout_seconds', 'server_retry_attempts', 'use_gpu'])
        model_dir = read_return['model_dir']
        model_choice = read_return['model_choice']
        local_llm_context_length = read_return['local_llm_context_length']
        local_llm_max_new_tokens = read_return['local_llm_max_new_tokens']
        local_llm_gpu_layers = read_return['local_llm_gpu_layers']
        server_timeout_seconds = read_return['server_timeout_seconds']
        server_retry_attempts = read_return['server_retry_attempts']
        use_gpu = read_return['use_gpu']
    except Exception as e:
        return handle_api_error("Missing values in config.json when preparing to launch llama.cpp server, encountered error: ", e)


    try:
        cpp_model = os.path.join(model_dir, model_choice)
    except Exception as e:
        return handle_api_error("Could not os.join path to model file to launch llama.cpp server, encountered error: ", e)

    if not use_gpu:
        local_llm_gpu_layers = 0

    try:
        cpp_app = ['llama-server', '-m', cpp_model, '-ngl', str(local_llm_gpu_layers), '-c', str(local_llm_context_length), '-n', str(local_llm_max_new_tokens), '--host', '0.0.0.0']

        if platform.system() == 'Windows':
            LLAMA_CPP_PROCESS = subprocess.Popen(cpp_app, creationflags=subprocess.CREATE_NEW_CONSOLE)  # Windows only! Comment when containerizing or deploying to Linux/MacOS!
        else:           
            # Platform & container agnostic:
            with open('llama_cpp_server_output_log.txt', 'w') as f:
                LLAMA_CPP_PROCESS = subprocess.Popen(cpp_app, stdout=f, stderr=subprocess.STDOUT, text=True)    #stdout has already been redirected to the file, so simply direct stderr to stdout!

    except Exception as e:
        return handle_api_error("Could not launch llama.cpp process, encountered error: ", e)


    timeout = server_timeout_seconds   
    attempts = server_retry_attempts

    try:
        for _ in range(attempts):
            if is_llama_cpp_local_server_online()['server_available']:
                print("llama.cpp server launched succesfully! Returning.")
                LLM_LOADED_UP = True
                return jsonify({'success': True, 'llm_model': model_choice})
            time.sleep(timeout)
    except Exception as e:
        handle_error_no_return("Could not check server status after launch attempt, printing error and retrying: ", e)

    return handle_api_error("Failed to start llama.cpp local-server")



@app.route('/load_vectordb')
def load_vectordb():

    global VECTOR_STORE
    global HF_BGE_EMBEDDINGS
    global AZURE_OPENAI_EMBEDDINGS
    global VECTORDB_CHANGE_RELOAD_TRIGGER_SET
    global VECTORDB_LOADED_UP

    if VECTORDB_LOADED_UP and not VECTORDB_CHANGE_RELOAD_TRIGGER_SET:
        print(f'\n\nVectorDB already loaded! Simply returning.\n\n')
        return jsonify({'success': True})
    elif VECTORDB_CHANGE_RELOAD_TRIGGER_SET:
        print('\n\nProceeding to reload VectorDB & resetting the VECTORDB_CHANGE_RELOAD_TRIGGER_SET flag.\n\n')
        VECTORDB_CHANGE_RELOAD_TRIGGER_SET = False

    try:
        read_return = read_config(['use_gpu_for_embeddings', 'use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder'])
        use_gpu_for_embeddings = read_return['use_gpu_for_embeddings']
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to load_vectordb. Error: ", e)
    
    
    ### 1 - Load VectorDB from disk
    print("\n\nLoading VectorDB: ChromaDB\n\n")
    try:
        if use_sbert_embeddings:
            VECTOR_STORE = Chroma(persist_directory=vectordb_sbert_folder, embedding_function=HuggingFaceEmbeddings())
            # try:
            #     # chroma_client = VECTOR_STORE.PersistentClient
            #     # max_batch_size = chroma_client._producer.max_batch_size
            #     max_batch_size = VECTOR_STORE.max_batch_size
            #     print(f"max_batch_size: {max_batch_size}")
            # except Exception as e:
            #     print(f"Could not get max_batch_size. Error: {e}")
        
        elif use_openai_embeddings:

            try:
                read_return = read_config(['azure_openai_text_ada_api_url', 'azure_openai_text_ada_api_key', 'azure_openai_api_type', 'azure_openai_api_version', 'azure_openai_text_ada_deployment_name'])
                azure_openai_text_ada_api_url = read_return['azure_openai_text_ada_api_url']
                azure_openai_text_ada_api_key = read_return['azure_openai_text_ada_api_key']
                azure_openai_api_type = read_return['azure_openai_api_type']
                azure_openai_api_version = read_return['azure_openai_api_version']
                azure_openai_text_ada_deployment_name = read_return['azure_openai_text_ada_deployment_name']
            except Exception as e:
                return handle_api_error("Missing values for Azure OpenAI Embeddings in method load_model_and_vectordb in config.json. Error: ", e)
            
            try:
                os.environ["OPENAI_API_BASE"] = azure_openai_text_ada_api_url
                os.environ["OPENAI_API_KEY"] = azure_openai_text_ada_api_key
                os.environ["OPENAI_API_TYPE"] = azure_openai_api_type
                os.environ["OPENAI_API_VERSION"] = azure_openai_api_version
            except Exception as e:
                return handle_api_error("Could not set OS environment variables for Azure OpenAI Embeddings in load_model_and_vectordb, encountered error: ", e)

            
            AZURE_OPENAI_EMBEDDINGS = OpenAIEmbeddings(deployment=azure_openai_text_ada_deployment_name)
            VECTOR_STORE = Chroma(persist_directory=vectordb_openai_folder, embedding_function=AZURE_OPENAI_EMBEDDINGS)
        
        elif use_bge_base_embeddings:
            model_name = "BAAI/bge-base-en"
            model_kwargs = {}
            if use_gpu_for_embeddings:
                model_kwargs.update({"device": "cuda"})
            else:
                model_kwargs.update({"device": "cpu"})
            encode_kwargs = {"normalize_embeddings": True}
            HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
                
        
        elif use_bge_large_embeddings:
            model_name = "BAAI/bge-large-en"
            model_kwargs = {}
            if use_gpu_for_embeddings:
                model_kwargs.update({"device": "cuda"})
            else:
                model_kwargs.update({"device": "cpu"})
            encode_kwargs = {"normalize_embeddings": True}
            HF_BGE_EMBEDDINGS = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
            VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
        
        #VECTOR_STORE = Chroma(persist_directory=VECTORDB_SBERT_FOLDER, embedding_function=HuggingFaceEmbeddings())
    except Exception as e:
        return handle_api_error("Could not load VectorDB, encountered error: ", e)
    
    VECTORDB_LOADED_UP = True
    return jsonify(success=True)


@app.route('/set_prompt_template', methods=['POST'])
def set_prompt_template():

    base_template = ""

    try:
        base_template = request.form['prompt_template']
    except Exception as e:
        return handle_api_error("Server-side error, could not read prompt_template from the POST request in method set_prompt_template, encountered error: ", e)
    
    try:
        write_config({'base_template':base_template})
    except Exception as e:
        return handle_api_error("Could not update base_template in method set_prompt_template, encountered error: ", e)

    return jsonify({'success':True})


@app.route('/fetch_file_list_for_vector_db', methods=['POST'])
def fetch_file_list_for_vector_db():

    print("Loading file list for selected VectorDB")

    try:
        selected_embedding_model_choice = request.form['embedding_model_choice']
    except Exception as e:
        return handle_api_error("Server-side error, could not read embedding_model_choice from the POST request in method fetch_file_list_for_vector_db, encountered error: ", e)

    # For the VectorDB presently picked by the user in the dropdown, obtain the associated VectorDB folder for the select query:
    vdb_for_select = ""
    try:
        if selected_embedding_model_choice == 'bge_large':
            read_return = read_config(['vectordb_bge_large_folder'])
            vdb_for_select = read_return['vectordb_bge_large_folder']
            
        elif selected_embedding_model_choice == 'bge_base':
            read_return = read_config(['vectordb_bge_base_folder'])
            vdb_for_select = read_return['vectordb_bge_base_folder']

        elif selected_embedding_model_choice == 'sbert_mpnet_base_v2':
            read_return = read_config(['vectordb_sbert_folder'])
            vdb_for_select = read_return['vectordb_sbert_folder']

        elif selected_embedding_model_choice == 'openai_text_ada':
            read_return = read_config(['vectordb_openai_folder'])
            vdb_for_select = read_return['vectordb_openai_folder']

    except Exception as e:
        return handle_api_error("Could not create new VectorDB in reset_vector_db_on_disk, encountered error: ", e)

    try:
        read_return = read_config(['sqlite_docs_loaded_db'])
        sqlite_docs_loaded_db = read_return['sqlite_docs_loaded_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_docs_loaded_db in config.json in method fetch_file_list_for_vector_db. Error: ", e)

    file_row_list = []
    
    try:
        conn = sqlite3.connect(sqlite_docs_loaded_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to sqlite_docs_loaded_db database to load file list, encountered error: ", e)

    try:
        c.execute("SELECT document_name, vectordb_used, chunk_size, chunk_overlap FROM document_records where vectordb_used = ?", (vdb_for_select,))
    except Exception as e:
        return handle_api_error("Could not get document list from document_records db, encountered error: ", e)
    
    try:
        result = c.fetchall()

        for list_item in result:
            file_row_list.append(list(list_item))
    except Exception as e:
        return handle_api_error("Could not parse document list from document_records db, encountered error: ", e)

    #print(f'returning docs loaded list: {file_row_list}')

    return jsonify({'success': True, 'file_row_list': file_row_list})


@app.route('/reset_vector_db_on_disk', methods=['POST'])
def reset_vector_db_on_disk():

    print("Resetting selected VectorDB")

    try:
        selected_embedding_model_choice = request.form['embedding_model_choice']
    except Exception as e:
        return handle_api_error("Server-side error, could not read embedding_model_choice from the POST request in method reset_vector_db_on_disk, encountered error: ", e)

    try:
        read_return = read_config(['base_directory'])
        base_directory = read_return['base_directory']
    except Exception as e:
        handle_local_error("Could not read base_directory from config.json for reset_vector_db_on_disk. Error: ", e)

    try:
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d-%Hhr-%Mmin-%Ssec')
    except Exception as e:
        return handle_api_error("Could not obtain timestamp in reset_vector_db_on_disk, encountered error: ", e)

    # Now that we have all pre-requisite data to create a new VectorDB, proceed to do so by checking the model the user had currently picked from the dropdown: 
    try:
        if selected_embedding_model_choice == 'bge_large':
            vectordb_bge_large_folder = base_directory + '/chroma_db_bge_large_embeddings' + '-' + formatted_datetime
            write_config({'vectordb_bge_large_folder':vectordb_bge_large_folder})
            
        elif selected_embedding_model_choice == 'bge_base':
            vectordb_bge_base_folder = base_directory + '/chroma_db_bge_base_embeddings' + '-' + formatted_datetime
            write_config({'vectordb_bge_base_folder':vectordb_bge_base_folder})

        elif selected_embedding_model_choice == 'sbert_mpnet_base_v2':
            vectordb_sbert_folder = base_directory + '/chroma_db_sbert_embeddings' + '-' + formatted_datetime
            write_config({'vectordb_sbert_folder':vectordb_sbert_folder})

        elif selected_embedding_model_choice == 'openai_text_ada':
            vectordb_openai_folder = base_directory + '/chroma_db_openai_embeddings' + '-' + formatted_datetime
            write_config({'vectordb_openai_folder':vectordb_openai_folder})

    except Exception as e:
        return handle_api_error("Could not create new VectorDB in reset_vector_db_on_disk, encountered error: ", e)

    restart_required = True
    global VECTORDB_CHANGE_RELOAD_TRIGGER_SET
    VECTORDB_CHANGE_RELOAD_TRIGGER_SET = True
    try:
        read_return = read_config(['embedding_model_choice'])
        set_embedding_model_choice = read_return['embedding_model_choice']
        if set_embedding_model_choice != selected_embedding_model_choice:
            restart_required = False
            VECTORDB_CHANGE_RELOAD_TRIGGER_SET = False
    except Exception as e:
        handle_error_no_return("Could not compare selected and set embedding models when determining if restart_required in reset_vector_db_on_disk(), encountered error: ", e)

    #print(f'returning docs loaded list: {file_row_list}')

    return jsonify({'success': True, "restart_required": restart_required})


@app.route('/load_chat_history_list')
def load_chat_history_list():

    print("loading chat history list for sidebar")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method load_chat_history_list. Error: ", e)

    history_id_list = []
    
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to sqlite_history_db database to load chat history list, encountered error: ", e)

    try:
        c.execute("SELECT DISTINCT chat_id FROM chat_history")
    except Exception as e:
        return handle_api_error("Could not get list from chat history db, encountered error: ", e)
    
    try:
        result = c.fetchall()

        for list_item in result:
            history_id_list.append(list_item)
    except Exception as e:
        return handle_api_error("Could not parse chat history list from db, encountered error: ", e)

    #print(f'returning chat hsitory list: {history_id_list}')

    return jsonify({'success': True, 'history_list': history_id_list})


@app.route('/load_chat_history', methods=['POST'])
def load_chat_history():

    global CHAT_ID
    global SEQUENCE_ID
    global HISTORY_SUMMARY
    global HISTORY_MEMORY_WITH_BUFFER

    print("loading chat history")

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method load_chat_history. Error: ", e)

    # Clear chat history of current chat, prep for loading historical chat summary:
    # try:
    #     HISTORY_MEMORY_WITH_BUFFER.chat_memory.clear()
    #     HISTORY_MEMORY_WITH_BUFFER = ConversationSummaryBufferMemory(llm=LLM, max_token_limit=300, return_messages=False)
    #     HISTORY_SUMMARY = {}
    # except Exception as e:
    #     handle_error_no_return("Could not clear memory when loading chat history, encountered error: ", e)

    try:
        chat_id_for_history_search = request.form['chat_id']
        CHAT_ID = request.form['chat_id']
    except Exception as e:
        return handle_api_error("Could not retrieve Chat ID from request form, encountered error: ", e)

    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history database, encountered error: ", e)

    sequence_id_for_history_search = 1
    retrieve_history = True
    chat_history = []
    old_chat_model = ""

    while(retrieve_history):

        try:
            c.execute("SELECT user_query FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
            
            user_message = str(result[0])

            user_message = user_message.strip('\n')
            regex_to_swap_multiple_spaces_with_newline = r' {2,}'
            user_message = re.sub(regex_to_swap_multiple_spaces_with_newline, '<br>', user_message)

            user_message = '<div class="user-message">' + user_message + '</div>'

            chat_history.append(user_message)

            c.execute("SELECT llm_response FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()

            result = str(result[0])
            result_parts = result.split("pdf_pane_data=",1)
            # llm_response = '<div class="llm-wrapper"> <div class="llm-response">' + str(result[0]) + '</div>'
            llm_response = '<div class="response-and-viewer-container"><div class="llm-wrapper"> <div class="llm-response">' + result_parts[0]

        except Exception as e:
            return handle_api_error("Could not retrieve chat history, encountered error: ", e)
        
        llm_response = llm_response.strip('\n')
        llm_response = llm_response.replace('\n\n', '<br><br>')
        llm_response = llm_response.replace('\n', '<br>')
        
        try:
            c.execute("SELECT user_rating FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            result = c.fetchone()
        except Exception as e:
            handle_error_no_return("Could not fetch user rating, encountered error: ", e)

        response_rated = False
        user_rating_for_history_chat = None

        if result[0]:
            response_rated = True
            try:
                user_rating_for_history_chat = int(result[0])
                #print(f'rating exists: {user_rating_for_history_chat}')
            except Exception as e:
                handle_error_no_return("Could not retrieve integer value of user rating, encountered error: ", e)


        llm_rating = f'''<div class="star-rating" data-rated={response_rated} rating-chat-id={chat_id_for_history_search} rating-sequence-id={sequence_id_for_history_search}>
        <i class="far fa-star" data-rate="1"></i>
        <i class="far fa-star" data-rate="2"></i>
        <i class="far fa-star" data-rate="3"></i>
        <i class="far fa-star" data-rate="4"></i>
        <i class="far fa-star" data-rate="5"></i>
        </div>
        </div>
        </div>'''


        if user_rating_for_history_chat:
            rating_parts = llm_rating.split("far", user_rating_for_history_chat)
            if len(rating_parts) <= user_rating_for_history_chat:
                llm_rating = "fas".join(rating_parts)
            else:
                llm_rating = "fas".join(rating_parts[:-1]) + "fas" + "far".join(rating_parts[-1:])

        llm_response += llm_rating

        if len(result_parts) > 1:
            llm_response += result_parts[1]
            llm_response += "</div>"
            llm_response = llm_response.strip('\n')
            llm_response = llm_response.replace('\n\n', '<br><br>')
            llm_response = llm_response.replace('\n', '<br>')

        chat_history.append(llm_response)

        # Increment sequence ID for next iteration:
        sequence_id_for_history_search += 1

        # But first, check to see if next sequence exists!
        try:
            c.execute("SELECT EXISTS(SELECT 1 FROM chat_history WHERE chat_id = ? AND sequence_id = ?)", (int(chat_id_for_history_search), int(sequence_id_for_history_search)))
            exists = c.fetchone()[0]
        except Exception as e:
            return handle_api_error("Could not determine if next sequence exists in chat history DB, encountered error: ", e)
            
        if not exists:
            SEQUENCE_ID = sequence_id_for_history_search - 1
            retrieve_history = False
            try:
                c.execute("SELECT llm_model FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (CHAT_ID, SEQUENCE_ID))
                result = c.fetchone()
                old_chat_model = str(result[0])
            except Exception as e:
                handle_error_no_return("Could not determine previously used LLM in chat, encountered error: ", e)
            try:
                c.execute("SELECT history_summary FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (CHAT_ID, SEQUENCE_ID))
                result = c.fetchone()
                history_summary_dict = str(result[0])
            except Exception as e:
                handle_error_no_return("Could not fetch history summary of last chat, encountered error: ", e)
            c.close()

    # Convert History Summary and add a new key indicating it was recently cleared!
    if history_summary_dict is not None and history_summary_dict != "" and history_summary_dict != 'None':
        print(f"\n\history_summary_dict string from old chat: {history_summary_dict}\n\n")
        try:
            HISTORY_SUMMARY = ast.literal_eval(history_summary_dict)    #cast as dictionary
            HISTORY_SUMMARY["has_been_reset"] = True
        except Exception as e:
            handle_error_no_return("Could not cast history summary string from DB to dict and/or set has_been_reset boolean, encountered error: ", e)

    # Temp prints:
    print(f"\n\nHISTORY_SUMMARY: {HISTORY_SUMMARY}\n\n")
    # print(f"\n\history_summary_dict: {history_summary_dict}\n\n")
    # print(f"\n\nHISTORY_MEMORY_WITH_BUFFER.summary: {HISTORY_MEMORY_WITH_BUFFER.summary}\n\n")
    # print(f"\n\nHISTORY_MEMORY_WITH_BUFFER.chat_memory.messages: {HISTORY_MEMORY_WITH_BUFFER.chat_memory.messages}\n\n")
    print(f'\n\nChat history loaded for chat with model: {old_chat_model}\n\n')

    return jsonify({'success': True, 'chat_history': chat_history, 'old_chat_model': old_chat_model})


@app.route('/init_chat_history_db')
def init_chat_history_db():

    global CHAT_ID

    try:
        read_return = read_config(['sqlite_history_db'])
        sqlite_history_db = read_return['sqlite_history_db']
    except Exception as e:
        return handle_api_error("Missing sqlite_history_db in config.json in method init_chat_history_db. Error: ", e)

    # Connect to chat_history.db to determine appropriate chat_id
    try:
        conn = sqlite3.connect(sqlite_history_db)
        c = conn.cursor()
    except Exception as e:
        return handle_api_error("Could not connect to chat history database, encountered error: ", e)

    # If the database does not currently exist...
    try:
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY,
                        chat_id INTEGER,
                        sequence_id INTEGER,
                        user_query TEXT,
                        llm_response TEXT,
                        user_rating INTEGER,
                        llm_model TEXT, 
                        prompt_template TEXT,
                        history_summary TEXT
            )
        ''')
        conn.commit()
    except Exception as e:
        return handle_api_error("Could not create new chat history db, encountered error: ", e)

    try:
        c.execute("SELECT COALESCE(MAX(chat_id), 0) FROM chat_history")
        # "The COALESCE function accepts two or more arguments and returns the first non-null argument."
        # This accounts for an empty DB!

        result = c.fetchone()

        # 'result' will be a tuple, so extract the first element
        max_chat_id = result[0]

        new_chat_id = max_chat_id + 1
        CHAT_ID = new_chat_id

        print(f"Chat history DB initialised with CHAT_ID: {CHAT_ID}")
    except Exception as e:
        return handle_api_error("Could not set CHAT_ID, encountered error: ", e)

    conn.close()

    return jsonify({'success': True, 'chat_id': CHAT_ID})


def fetch_image_from_db(image_id):

    try:
        read_return = read_config(['sqlite_images_db'])
        sqlite_images_db = read_return['sqlite_images_db']
    except Exception as e:
        handle_local_error("Missing sqlite_history_db in config.json in method fetch_image_from_db. Error: ", e)
    
    # 1 - Connect to DB
    try:
        conn = sqlite3.connect(sqlite_images_db)
    except Exception as e:
        handle_local_error("Could not connect to images database, encountered error: ", e)
    
    # 2 - Get Images
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT image_data FROM images WHERE id = ?', (image_id,)).fetchone()
        images_bytes = row['image_data'] if row else None
    except Exception as e:
        handle_local_error("Could not fetch image from DB, encountered error: ", e)
    
    conn.close()
    
    return images_bytes


@app.route('/image_display/<int:image_id>')
def image_display(image_id):
    print(f"\n\nprepping image for display: {image_id}\n\n")

    try: 
        image_bytes = fetch_image_from_db(image_id)
    except Exception as e:
        handle_local_error("Could not fetch image for display, encountered error: ", e)
    
    try:
        encoded = base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        handle_local_error("Could not encode image for display URI, encountered error: ", e)

    # Return an HTML response with the embedded image:
    data_uri = f"data:image/jpeg;base64,{encoded}"
    html_content = f'<img src="{data_uri}" alt="Image">'

    return html_content


def extract_significant_phrases(query):
    print("Extracting significant phrases")

    try:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        handle_error_no_return("Failed to download & set stopwords, encountered error: ", e)
    
    try:
        tokens = [token for token in query.lower().split() if token not in stop_words]
    except Exception as e:
        handle_local_error("Could not extract significant tokens, encountered error: ", e)

    print(f"\nReturning tokens: {tokens}\n")
    return tokens


def calculate_relevance_score(phrases, document_content):
    #print("calculating relevance score")
    
    try:
        content_lower = document_content.lower()
    except Exception as e:
        handle_local_error("Could not read document_content in calculate_relevance_score(), encountered error: ", e)
    
    #print(f"document content: {content_lower}")
    
    #score = sum(1 for phrase in phrases if phrase in content_lower)
    
    score = 0
    try:
        for phrase in phrases:
            if phrase in content_lower:
                print(f"Match found to enable RAG: {phrase}")
                score += 1
    except Exception as e:
        handle_local_error("Could not compare phrases in calculate_relevance_score(), encountered error: ", e)
    
    return score


def filter_relevant_documents(query, search_results, threshold=1):

    print("Checking relevant docs to determin if RAG is required")

    do_rag = False
    page_contents = []

    try:
        significant_phrases = extract_significant_phrases(query)
    except Exception as e:
        handle_local_error("Could not extract significant phrases, encountered error: ", e)
    
    print(f"significant tokens: {significant_phrases}")
    #relevant_documents = []

    try:
        for document in search_results:
            # check for non-empty source field
            if document.page_content:
                page_contents.append(document.page_content)

            if not do_rag:  # if do_rag has already been set to true, why look?
                if document.metadata.get('source'):
                    score = calculate_relevance_score(significant_phrases, document.page_content)
                    if score >= threshold:
                        #relevant_documents.append(document)
                        print("Must do RAG!")
                        do_rag = True
    except Exception as e:
        handle_local_error("Could not read calculate relevance score, encountered error: ", e)

    #return relevant_documents
    return page_contents, do_rag



@app.route('/setup_for_llama_cpp_response', methods=['POST'])
def setup_for_llama_cpp_response():

    global QUERIES

    do_rag = True
    similarity_threshold = 0.8

    stream_session_id = ""
    key_for_vector_results = ""
    # Generate a unique session ID using universally Unique Identifier via the uuid4() method, wherein the randomness of the result is dependent on the randomness of the underlying operating system's random number generator
    # UUI is a standard used for creating unique strings that have a very high likelihood of being unique across all time and space, for ex: f47ac10b-58cc-4372-a567-0e02b2c3d479
    try:
        stream_session_id = str(uuid.uuid4())
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
    except Exception as e:
        return handle_api_error("Error creating unique stream_session_id when attempting to setup_for_streaming_response. Error: ", e)
    

    # Determine do_rag
    try:
        read_return = read_config(['use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'force_enable_rag', 'force_disable_rag', 'local_llm_chat_template_format', 'base_template'])
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        force_enable_rag = read_return['force_enable_rag']
        force_disable_rag = read_return['force_disable_rag']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
        base_template = read_return['base_template']

    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to setup_for_streaming_response. Error: ", e)

    try:
        # Attempt to get query data
        user_query = request.json['user_query']
        chat_id = request.json['chat_id']

        # Store the query associated with the ID
        QUERIES[stream_session_id] = user_query
    except KeyError:
        return handle_api_error("Could not obtain and/or store user_query in setup_for_streaming_response, encountered error: ", e)

    print("chat_id: ", chat_id)

    # Perform similarity search on the vector DB
    print("\n\nPerforming similarity search to determine if RAG necessary\n\n")
    embedding_function = None
    try:
        if use_sbert_embeddings:
            embedding_function=HuggingFaceEmbeddings()
        elif use_openai_embeddings:
            embedding_function=AZURE_OPENAI_EMBEDDINGS
        elif use_bge_base_embeddings:
            embedding_function=HF_BGE_EMBEDDINGS
        elif use_bge_large_embeddings:
            embedding_function=HF_BGE_EMBEDDINGS
    except Exception as e:
        handle_error_no_return("Could not set embedding_function for similarity_search when attempting to setup_for_streaming_response, encountered error: ", e)
    
    try:
        # docs = VECTOR_STORE.similarity_search(user_query, embedding_fn=embedding_function)
        # docs_with_relevance_score = VECTOR_STORE.similarity_search_with_relevance_scores(user_query, 10, embedding_fn=embedding_function)
        docs_list_with_cosine_distance = VECTOR_STORE.similarity_search_with_score(user_query, 7, embedding_fn=embedding_function)
        # print(f'\n\nsimple similarity search results: \n {docs}\n\n')
        # print(f'\n\nRelevance Score similarity search results (range 0 to 1): \n {docs_with_relevance_score}\n\n')
        # print(f'\n\nDocs list most similar to query based on cosine distance: \n {docs_list_with_cosine_distance}\n\n')
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determine do_rag when attempting to setup_for_streaming_response, encountered error: ", e)

    # docs = [
    #     doc for doc, score in docs_list_with_cosine_distance
    #     if score >= similarity_threshold
    # ]

    docs = [doc for doc, score in docs_list_with_cosine_distance]

    print("\n\nDetermining do_rag \n\n")
    # We do not modify the force_enable_rag or force_disable_rag flags in this method, we simply respond to them here. UI updates should handle those flags.
    if force_enable_rag:
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        try:
            do_rag = True
        except Exception as e:
            do_rag = False
            handle_error_no_return("Error force-enabling RAG, disabling RAG and continuing: could not filter_relevant_documents during setup_for_streaming_response, encountered error: ", e)
    elif force_disable_rag:
        print("\n\nFORCE_DISABLE_RAG True, force disabling RAG and returning\n\n")
        do_rag = False
    else:
        try:
            _, do_rag = filter_relevant_documents(user_query, docs)
        except Exception as e:
            do_rag = False
            handle_error_no_return("RAG Error, disabling RAG and continuing: could not filter_relevant_documents during setup_for_streaming_response, encountered error: ", e)
    
    print(f'Do RAG? {do_rag}')

    try:
        write_config({'do_rag':do_rag})
    except Exception as e:
        handle_error_no_return("Could not write do_rag to config during setup_for_streaming_response, encountered error: ", e)

    
    # Having determined do_rag, time to build the prompt template!
    
    if do_rag:  # add similarity search results for RAG!
        try:
            QUERIES[key_for_vector_results] = docs
            user_query += f"\n\nThe following context might be helpful in answering the user query above:\n{docs}"
            print(f"RAG formatted user_query: \n{user_query}\n")
        except Exception as e:
            try:
                write_config({'do_rag':False})
            except Exception as e:
                handle_error_no_return("Could not write do_rag to config during setup_for_streaming_response, encountered error: ", e)
            handle_error_no_return("RAG Error: Could not update QUERIES dict and user_query during setup_for_streaming_response, proceeding without RAG. Encountered error: ", e)

    current_sequence_id = determine_sequence_id_for_chat(chat_id)
    formatted_prompt = ""
    print("current_sequence_id: ", current_sequence_id)
    if current_sequence_id > 0:    # get the last prompt so we can continue the completions

        try:
            read_return = read_config(['sqlite_history_db'])
            sqlite_history_db = read_return['sqlite_history_db']
        except Exception as e:
            handle_error_no_return("Missing keys in config.json for method store_chat_history_to_db. Error: ", e)

        # Connect to or create the DB
        try:
            conn = sqlite3.connect(sqlite_history_db)
            cursor = conn.cursor()
        except Exception as e:
            handle_error_no_return("Could not establish connection to DB for chat history storage, encountered error: ", e)

        try:
            # Determine sequence_id
            cursor.execute("SELECT prompt_template FROM chat_history WHERE chat_id = ? AND sequence_id = ?", (int(chat_id), int(current_sequence_id)))
            # "The COALESCE function accepts two or more arguments and returns the first non-null argument."
            # This accounts for a new chat!
            # Note that trailing comma! Without it, the simple select query will produce an error: "parameters are of unsupported type" !!
            # This is because the SQLite3 module can have trouble recognizing single-item tuples as tuples, so a trailing comma helps alleviate this! 

            result = cursor.fetchone()
            formatted_prompt = str(result[0])
            
        except Exception as e:
            handle_error_no_return("Could not determine sequence ID for storage to chat history DB, encountered error: ", e)
    
    if formatted_prompt == "":  # could not be updated above
        current_sequence_id = 0 # reset chat sequence id

    if local_llm_chat_template_format == 'llama3':

        if current_sequence_id > 0:
            formatted_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            formatted_prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_template}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif local_llm_chat_template_format == 'llama2':

        if current_sequence_id > 0:
            formatted_prompt += f"<s>[INST] {user_query} [/INST]"
        else:
            formatted_prompt += f"<s>[INST] <<SYS>>\n {base_template} \n<</SYS>>\n\n {user_query}  [/INST]"

    elif local_llm_chat_template_format == 'chatml':
        
        if current_sequence_id > 0:
            formatted_prompt += f"\n<|im_start|>user\n{user_query}<|im_end|>\n"
        else:
            formatted_prompt += f"<|im_start|>system\n{base_template}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    elif local_llm_chat_template_format == 'phi3':

        if current_sequence_id > 0:
            formatted_prompt += f"<|user|>\n{user_query}<|end|>\n<|assistant|>\n"
        else:
            formatted_prompt += f"<|system|>\n{base_template}<|end|>\n<|user|>\n{user_query}<|end|>\n<|assistant|>\n"

    elif local_llm_chat_template_format == 'command-r':

        if current_sequence_id > 0:
            formatted_prompt += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_query}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        else:
            formatted_prompt += f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{base_template}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_query}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

    elif local_llm_chat_template_format == 'deepseek':
        
        if current_sequence_id > 0:
            formatted_prompt += f"### Instruction:\n{user_query}\n### Response:\n"
        else:
            formatted_prompt += f"{base_template}### Instruction:\n{user_query}\n### Response:\n"

    elif local_llm_chat_template_format == 'deepseek-coder-v2':
        
        if current_sequence_id > 0:
            formatted_prompt += f"User: {user_query}\nAssistant: "
        else:
            formatted_prompt += f"<|begin_of_sentence|>{base_template}\nUser: {user_query}\nAssistant: "

    elif local_llm_chat_template_format == 'vicuna':

        if current_sequence_id > 0:
            formatted_prompt += f"USER: {user_query}\nASSISTANT: "
        else:
            formatted_prompt += f"{base_template}\n\nUSER: {user_query}\nASSISTANT: "

    elif local_llm_chat_template_format == 'openchat':

        if current_sequence_id > 0:
            formatted_prompt += f"GPT4 Correct User: {user_query}<|end_of_turn|>GPT4 Correct Assistant: "
        else:
            formatted_prompt += f"<s>GPT4 Correct System: {base_template}<|end_of_turn|>GPT4 Correct User: {user_query}<|end_of_turn|>GPT4 Correct Assistant: "

    elif local_llm_chat_template_format == 'gemma2':

        if current_sequence_id > 0:
            formatted_prompt += f"<start_of_turn>user\n{user_query}<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt += f"<start_of_turn>user\n{base_template}\n{user_query}<end_of_turn>\n<start_of_turn>model\n"

    # Return a bunch of stuff
    new_sequence_id = int(current_sequence_id) + 1
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag, "formatted_user_prompt": formatted_prompt, "sequence_id":new_sequence_id})



@app.route('/get_references', methods=['POST'])
def get_references():

    print("\n\nGetting References\n\n")

    try:
        read_return = read_config(['do_rag', 'upload_folder', 'local_llm_chat_template_format'])
        do_rag = read_return['do_rag']
        upload_folder = read_return['upload_folder']
        local_llm_chat_template_format = read_return['local_llm_chat_template_format']
    except Exception as e:
        return handle_api_error("Missing values in config.json when attempting to get_references. Error: ", e)

    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['user_query']
        llm_response = request.json['llm_response']
        formatted_user_prompt = request.json['formatted_user_prompt']
        chat_id = request.json['chat_id']
        sequence_id = request.json['sequence_id']
    except Exception as e:
        return handle_api_error("Could not read request content in method get_references, encountered error: ", e)

    if local_llm_chat_template_format == 'llama3':
        formatted_user_prompt += f"{llm_response}<|eot_id|>"
    elif local_llm_chat_template_format == 'llama2':
        formatted_user_prompt += f"{llm_response}</s>"
    elif local_llm_chat_template_format == 'chatml':
        formatted_user_prompt += f"{llm_response}<|im_end|>\n"
    elif local_llm_chat_template_format == 'phi3':
        formatted_user_prompt += f"{llm_response}<|end|>\n"
    elif local_llm_chat_template_format == 'command-r':
        formatted_user_prompt += f"{llm_response}<|END_OF_TURN_TOKEN|>"
    elif local_llm_chat_template_format == 'deepseek':
        formatted_user_prompt += f"{llm_response}\n<|EOT|>\n"
    elif local_llm_chat_template_format == 'deepseek-coder-v2':
        formatted_user_prompt += f"{llm_response}<|end_of_sentence|>"
    elif local_llm_chat_template_format == 'vicuna':
        formatted_user_prompt += f"{llm_response} </s>\n"
    elif local_llm_chat_template_format == 'openchat':
        formatted_user_prompt += f"{llm_response}<|end_of_turn|>"
    elif local_llm_chat_template_format == 'gemma2':
        formatted_user_prompt += f"{llm_response}<end_of_turn>\n"

    if not do_rag:
        print("\n\nSkipping RAG, storing chat history and returning\n\n")
        try:
            store_llama_cpp_chat_history_to_db(chat_id, sequence_id, user_query, llm_response, formatted_user_prompt)
        except Exception as e:
            handle_error_no_return("Could not store_llama_cpp_chat_history_to_db in get_references(), encountered error: ", e)
        return jsonify({'success': True})
        
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        docs = QUERIES[key_for_vector_results]
    except Exception as e:
        return handle_api_error("Could not obtain relevant data from QUERIES dict, encountered error: ", e)

    # Having obtained the relevant info, clear the QUERIES{} dict so as to not bloat it!
    try:
        del QUERIES[key_for_vector_results]
    except Exception as e:
        handle_error_no_return("Error clearing queries dict in method get_references: ", e)

    reference_response = ""

    all_sources = {}
    reference_pages = {}

    for doc in docs:
        
        try:
            relevant_page_text = str(doc.page_content)
            relevant_page_number = str(doc.metadata.get('page_number'))
            source_filepath = str(doc.metadata.get('source'))
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
        
        relevant_page_text = relevant_page_text.replace('\n', ' ')

        # relevant_page_text = relevant_page_text.split('\n', 1)[0]
        # relevant_page_text = relevant_page_text.strip()
        # relevant_page_text = re.sub(r'[\W_]+Page \d+[\W_]+', '', relevant_page_text)
        
        try:
            source_filename = os.path.basename(source_filepath)
            _, file_extension = os.path.splitext(source_filepath)
        except Exception as e:
            handle_error_no_return("Could not parse path with OS lib, encountered error: ", e)
            continue

        # The source_filepath will likely always reference a TXT file because of how we're loading the VectorDB!
        # Check if the PDF version of the source doc exists
        if file_extension == '.txt':

            #print("\n\ntxt file\n\n")

            # Construct the path to the potential PDF version
            pdf_version_path = os.path.join(upload_folder, os.path.basename(source_filepath).replace('.txt', '.pdf'))   # not catching an error here as os.path.basename(source_filepath) has already been caught just above!

            # Check if PDF version of the source TXT exists!
            if os.path.exists(pdf_version_path):

                #print("\n\pdf exists\n\n")

                source_filename = source_filename.replace('.txt', '.pdf')
                
                if pdf_version_path in reference_pages:
                    reference_pages[pdf_version_path].extend([[relevant_page_text,relevant_page_number]])
                    #reference_pages[pdf_version_path].extend({'page_content': relevant_page_text, 'page_number': relevant_page_number})
                else:
                    reference_pages[pdf_version_path] = [[relevant_page_text,relevant_page_number]]
                    #reference_pages[pdf_version_path] = {'page_content': relevant_page_text, 'page_number': relevant_page_number}

                # Add this file to our sources dictionary if it's not already present
                if source_filename not in all_sources:
                    source_filepath = pdf_version_path
                    all_sources.update({source_filename: source_filepath})

            # Else PDF does not exist, TXT is the source
            else:
                print("\n\nNo PDF source doc found in the 'uploaded_pdfs' dir, RAG ACTIVE BUT REFERENCING WILL NOT DISPLAY!\n\n")
                # Check if the TXT is already in the sources dict
                if source_filename not in all_sources:
                    try:
                        source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                        all_sources.update({source_filename: source_filepath})
                    except Exception as e:
                        handle_error_no_return("Could not construct filepath for TXT file, encountered error: ", e)


        # If file is not a TXT file
        else:
            # Check if the TXT is already in the sources dict
            if source_filename not in all_sources:
                try:
                    source_filepath = os.path.join(upload_folder, source_filename) # reconstructed path using the OS module just to be safe
                    all_sources.update({source_filename: source_filepath})
                except Exception as e:
                    handle_error_no_return("Could not construct filepath for non-TXT file, encountered error: ", e)


    try:
        docs_have_relevant_info, user_should_refer_pages_in_doc = highlighter_interface(reference_pages, stream_session_id)
    except Exception as e:
        handle_error_no_return("Could not complete highlighter_interface, encountered error: ", e)

    # try:
    #     matched_images_found, matched_images_in_bytes = find_images_in_db(reference_pages)
    # except Exception as e:
    #     handle_error_no_return("Could not search for images, encountered error: ", e)

    refer_pages_string = ""
    download_link_html = ""
    images_iframe_html = ""
    matched_images_found = False

    if docs_have_relevant_info:

        refer_pages_string = "<br><br><h6>Refer to the following pages in the mentioned docs:</h6>"
        
        for index, doc in enumerate(user_should_refer_pages_in_doc, start=1):
            pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
            tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
            frame_doc_path = f"/pdf/{doc}"
            try:
                stream_id_string_to_remove = f"_{stream_session_id}"
                doc_name_without_stream_id = str(doc).replace(stream_id_string_to_remove, "")
                refer_pages_string += f"<br><h6>{doc_name_without_stream_id}: "
                for page in user_should_refer_pages_in_doc[doc]:
                    frame_doc_path += f"#page={str(page)}" 
                    refer_pages_string += f'<a href="javascript:void(0)" onclick="goToPageAndSwitchTab(\'{pdf_iframe_id}\', \'{frame_doc_path}\', \'tab{tab_name_string}\', \'{stream_session_id}\')">Page {page}</a>, '
                    frame_doc_path = f"/pdf/{doc}"
                refer_pages_string = refer_pages_string.strip(', ') + "</h6>"
            except Exception as e:
                handle_error_no_return("Could not construct refer_pages_string, encountered error: ", e)

        pdf_right_pane_id = f"stream{stream_session_id}PdfPane"
        download_link_html = f'<div class="pdf-viewer-container" id="{pdf_right_pane_id}">'

        # Add tab buttons
        download_link_html += '<div class="tab-buttons">'
        for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
            tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
            stream_id_string_to_remove = f"_{stream_session_id}"
            doc_name_without_stream_id = str(source).replace(stream_id_string_to_remove, "")
            default_open = ' defaultTabs' if index == 1 else ''
            download_link_html += f'<button class="tab-button{default_open}" stream-session-id="{stream_session_id}" onclick="openTab(event, \'tab{tab_name_string}\', \'{stream_session_id}\')">{doc_name_without_stream_id}</button>'
        download_link_html += '</div>'

        # Add tab content
        for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
            try:
                download_link_url = url_for('download_file', filename=source)
                pdf_iframe_id = f"stream{stream_session_id}PdfViewer{str(index)}"
                tab_name_string = f"stream{stream_session_id}tabName{str(index)}"
                download_link_html += f'<div id="tab{tab_name_string}" class="tab-content" stream-session-id="{stream_session_id}">'
                download_link_html += f'<iframe id="{pdf_iframe_id}" src="{download_link_url}" width="100%" height="600"></iframe>'
                download_link_html += "</div>"
            except Exception as e:
                handle_error_no_return("Could not construct download_link_html, encountered error: ", e)

        download_link_html += "</div>"

    if matched_images_found:
        image_gallery_id = f"image_gallery_for_stream_{stream_session_id}"
        images_iframe_html = f'''
        <h6>Browse a gallery of relevant images by clicking on the thumbnail below:</h6>
        <i class="fas fa-images thumbnail-icon" onclick="openImageGalleryModal('{image_gallery_id}')"></i>
        <div id="{image_gallery_id}" class="image-gallery-modal">
        <span class="image-gallery-close" onclick="closeImageGalleryModal('{image_gallery_id}')">&times;</span>
        <div class="image-gallery-content">
        '''
        for image_id, image_bytes_data in matched_images_in_bytes:
            #print(f"\n\nmatched image id: {image_id}")
            try:
                image_link_url = url_for('image_display', image_id=image_id)
                images_iframe_html += f'<iframe src="{image_link_url}" frameborder="0" class="gallery-thumbnail"></iframe>'
            except Exception as e:
                handle_error_no_return("Could not construct images_iframe_html, encountered error: ", e)
        
        images_iframe_html += f'</div></div>'

    
    # reference_response = refer_pages_string + download_link_html + images_iframe_html
    reference_response = refer_pages_string + images_iframe_html

    try:
        # model_response_for_history_db = str(llm_response) + refer_pages_string
        model_response_for_history_db = str(llm_response)
        model_response_for_history_db += f"\n\n{reference_response}"
        model_response_for_history_db += f"\n\npdf_pane_data={download_link_html}"
        model_response_for_history_db = model_response_for_history_db.strip('\n')

        formatted_user_query = str(user_query).strip('\n')

        user_query_for_history_db = formatted_user_query
    except Exception as e:
        handle_error_no_return("Could not prep data to store_chat_history_to_db in get_references(), encountered error: ", e)

    try:
        store_llama_cpp_chat_history_to_db(chat_id, sequence_id, user_query_for_history_db, model_response_for_history_db, formatted_user_prompt)
    except Exception as e:
        handle_error_no_return("Could not store_chat_history_to_db in get_references(), encountered error: ", e)

    return jsonify({'success': True, 'response': reference_response, 'pdf_frame':download_link_html})


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)