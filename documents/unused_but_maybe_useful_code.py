# JS: // let streamed_content = dataObj.replace(/(?<![A-Z]:|\/|[0-9]|[ivxlcdm])([.?!])(?=\s|$|[0-9])(?!\s*\/)/g, '$1<br><br>');

def preprocess_string(s):
    """
    This function removes all non-alphanumeric characters from the string, 
    converts it to lowercase, and trims whitespace.
    It's not used in the current implementation of LARS.
    """
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()




def PDFtoMSTrOCR(input_filepath):
    
    print("\n\nProcessing Document - PDF to MS TrOCR TXT\n\n")

    try:
        read_return = read_config(['model_dir'])
        model_directory = read_return['model_dir']
    except Exception as e:
        handle_local_error("Missing model_dir in config.json for PDFtoMSTrOCR. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Set output path
    output_text_file_path = input_filepath.replace(".pdf","_ms_tr_ocr_cleaned.txt") 
    raw_output_text_file_path = input_filepath.replace(".pdf","_ms_tr_ocr_raw.txt") 

    # Init list for Whoosh indexing
    pdf_data = []

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
        raw_output_text_file = open(raw_output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Setting up Cleaner LLM:
    llm_name = 'openhermes-2.5-mistral-7b.Q8_0.gguf'
    llm_dir = model_directory + '/' + llm_name
    config = {'context_length': 8192, 'max_new_tokens': 8192, 'gpu_layers':50}
    cleaner_llm = CTransformers(model=llm_dir, model_type="llama", config=config)
    # cleanup_template = PromptTemplate(template="Correct the following text for any gramatical and formatting errors, otherwise leaving it unchanged: {input}", input_variables=["input"])
    conv_chain = ConversationChain(llm = cleaner_llm)

    #Load OCR TrOCR model:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    
    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to MS TrOCR\n\n")
    for page_number, page_image in enumerate(pages, start = 1):

        rgb_image = page_image.convert("RGB")
        width, height = rgb_image.size

        page_text = ""

        block_no = 0

        # original_stdout = sys.stdout

        # Process the page in 240x71 blocks:
        # for y in range(0, height, 71):
        #     for x in range(0, width, 240):
                
        #         print(f"Processing block {block_no}")

        #         # long-term: discard output
        #         # f = open(os.devnull, 'w')
        #         # sys.stdout = f

        #         # Crop block:
        #         block = rgb_image.crop((x, y, x + 240, y + 71))

        #         # Process block with TrOCR:
        #         pixel_values = processor(images=block, return_tensors="pt").pixel_values

        #         generated_ids = model.generate(pixel_values)
        #         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        #         print(f"generated block text: {generated_text}")

        #         if str(generated_text) != "0.00":
        #             page_text += generated_text

        #         # Reset stdout to its original value
        #         # sys.stdout = original_stdout

        #         block_no += 1

        for y in range(0, height, 50):
                
            print(f"Processing block {block_no}")

            # long-term: discard output
            # f = open(os.devnull, 'w')
            # sys.stdout = f

            # Crop block:
            block = rgb_image.crop((0, y, width, y + 50))

            # Process block with TrOCR:
            pixel_values = processor(images=block, return_tensors="pt").pixel_values

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            print(f"generated block text: {generated_text}")

            if str(generated_text) != "0.00":
                page_text += generated_text

            # Reset stdout to its original value
            # sys.stdout = original_stdout

            block_no += 1

        # Save raw output of the above process for analysis:
        try:
            raw_output_text_file.write(page_text + '\n')
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)
        
        # Clean Text:
        print(f"\n\nCleaning Page Text with {llm_name}\n\n")
        llm_input = f"Correct the following text for any gramatical and formatting errors, otherwise leaving it unchanged: {page_text}"
        try:
            clean_text = conv_chain.predict(input=llm_input)
        except Exception as e:
            handle_local_error("Could not clean text with LLM, encountered error: ", e)

        # Write the cleaned up text to the file
        try:
            output_text_file.write(clean_text + '\n')
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)

        # Whoosh prep
        #whoosh_clean_text = preprocess_string(clean_text)
        whoosh_page_dict_entry = {"title": source_filename, "content": clean_text, "pagenumber":page_number+1}
        pdf_data.append(whoosh_page_dict_entry)

    # Close all files
    raw_output_text_file.close()
    output_text_file.close()

     # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(pdf_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path



#Local OCR using PyTesseract - Not used in LARS
def PDFtoOCRTXT(input_filepath):
    
    print("\n\nProcessing Document - PDF to OCR TXT\n\n")

    try:
        read_return = read_config(['base_directory'])
        app_base_directory = read_return['base_directory']
    except Exception as e:
        handle_local_error("Missing base_directory in config.json for PDFtoOCRTXT. Error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Set output path
    output_text_file_path = input_filepath.replace(".pdf","_ocr_300.txt") 

    # Init list for Whoosh indexing
    pdf_data = []

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to Text OCR\n\n")
    for page_number, page_image in enumerate(pages, start=1):

        try:
            custom_config = r'--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
            text = pytesseract.image_to_string(page_image, config='--psm 3')    # Page Segmentation Mode (PSM) 3 - Default; Fully Automatic Page Segmentation & OCR, but no Orientation and Script Detection (OSD). PSM 3,4 & 6 are common for docs. For a full list of PSMs, ask ChatGPT "Can you give me a walkthrough of all the different Page Segmentation Modes in Python's PyTesseract?" 
            #text = pytesseract.image_to_string(page_image, config=custom_config)
        except Exception as e:
            handle_error_no_return("Could not OCR text from page, encountered error: ", e)
            
        # Optionally save image for review
        try:
            ocr_img_directory = app_base_directory + '/OCR_IMAGES'
            if not os.path.exists(ocr_img_directory):
                os.makedirs(ocr_img_directory)
            ocr_img_filename = f'{ocr_img_directory}/{source_filename}_page_{page_number}.jpg'
            page_image.save(ocr_img_filename)
        except Exception as e:
            error_message = f"Could not save OCR image for {source_filename}_page_{page_number}, encountered error: "
            handle_error_no_return(error_message, e)

        # clean_text = text
        # Clean text
        clean_text = clean_text_string(text)
        
        # Optionally, you can include page numbers in the text file
        # output_text_file.write(f'\n\n--- Page {page_num + 1} ---\n\n')
        
        # Write the extracted text to the file
        try:
            output_text_file.write(clean_text + '\n')
        except Exception as e:
            handle_local_error("Could not write to output text file, encountered error: ", e)

        # Whoosh prep
        #whoosh_clean_text = preprocess_string(clean_text)
        whoosh_page_dict_entry = {"title": source_filename, "content": clean_text, "pagenumber":page_number+1}
        pdf_data.append(whoosh_page_dict_entry)

    # Close all files
    output_text_file.close()

    # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(pdf_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path




def PDFtoAzureOCRTXT_url(input_filepath):
    
    print("\n\nProcessing Document - PDF to Azure OCR TXT\n\n")

    try:
        read_return = read_config(['azure_ocr_endpoint', 'azure_ocr_subscription_key'])
        azure_ocr_endpoint = read_return['azure_ocr_endpoint']
        azure_ocr_subscription_key = read_return['azure_ocr_subscription_key']
    except Exception as e:
        handle_local_error("Missing Azure OCR Endpoint URL & Subscription Key for PDFtoAzureOCRTXT_url, please provide required API config. Error: ", e)
    
    try:
        os.environ["azure_ocr_endpoint"] = azure_ocr_endpoint
        os.environ["azure_ocr_subscription_key"] = azure_ocr_subscription_key
    except Exception as e:
        handle_local_error("Could not set OS environment variables for Azure OCR, encountered error: ", e)

    try:
        source_filename = os.path.basename(input_filepath)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Convert PDF to  a list of images
    try:
        print("\n\nConverting PDF to a list of Images\n\n")
        pages = convert_from_path(input_filepath, 300) # 300dpi - good balance between quality and performance
    except Exception as e:
        handle_local_error("Could not image PDF file, encountered error: ", e)
    
    # Set output path
    output_text_file_path = input_filepath.replace(".pdf","_azure_ocr_300.txt") 

    # Init list for Whoosh indexing
    pdf_data = []

    # Initialize text output
    try:
        output_text_file = open(output_text_file_path, 'w', encoding='utf-8')
    except Exception as e:
        handle_local_error("Could not initialize/access output text file, encountered error: ", e)
    
    # Init Azure VisionServiceOptions
    service_options = sdk.VisionServiceOptions(os.environ["azure_ocr_endpoint"], os.environ["azure_ocr_subscription_key"])
    
    # Iterate over each page and apply OCR:
    print("\n\nBeginning image to Text OCR\n\n")
    for page_number, image in enumerate(pages, start = 1):
    #for image in pages:
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Save the image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
            image.save(temp_image_file, format='PNG')
            temp_image_path = temp_image_file.name

            # Setup cision source with byte array
            vision_source = sdk.VisionSource(url=temp_image_path)

            # Set analysis options:
            analysis_options = sdk.ImageAnalysisOptions()
            analysis_options.features = sdk.ImageAnalysisFeature.TEXT


            # Send to Azure OCR & analyze the image
            image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)
            result = image_analyzer.analyze()

            if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
                if result.text is not None:
                    # print("Text:")
                    for line in result.text.lines:
                        # print(f"Line: {line.content}")
                        clean_text = line.content

                        # Write the extracted text to the file:
                        try:
                            output_text_file.write(clean_text + '\n')
                        except Exception as e:
                            handle_local_error("Could not write to output text file, encountered error: ", e)

                        # Whoosh prep
                        #whoosh_clean_text = preprocess_string(clean_text)
                        whoosh_page_dict_entry = {"title": source_filename, "content": clean_text, "pagenumber":page_number+1}
                        pdf_data.append(whoosh_page_dict_entry)

            else:
                # Handle errors:
                error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
                print(" Analysis failed.")
                print("   Error reason: {}".format(error_details.reason))
                print("   Error code: {}".format(error_details.error_code))
                print("   Error message: {}".format(error_details.message))

    # Close all files
    output_text_file.close()

    # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(pdf_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path



def TxtCleaner(input_file):
    """
    Processes a text file by cleaning and subsequent indexing.

    :param input_file: Path to the input text file within the app's folder.
    :return: path to the cleaned output text file.
    """

    print("\nProcessing Text File")

    # Ensure the file is .txt:
    if not input_file.lower().endswith('.txt'):
        raise ValueError("File must be a .txt file")
    
    # Get filename
    try:
        source_filename = os.path.basename(input_file)
    except Exception as e:
        handle_local_error("Could not extract filename, encountered error: ", e)

    # Set output path:
    output_text_file_path = input_file.replace(".txt","_cleaned.txt")

    # Init list for Whoosh indexing
    text_data = []

    try:
        # Read and process the file, \ is a continuation char in Python used to split long lines of code for readibility!
        with open(input_file, 'r', encoding='utf-8') as input_file ,\
                open(output_text_file_path, 'w', encoding='utf-8') as output_text_file:
            
            # enumerate returns a tuple containing the count and value of an iterable such as a file or list. It starts at 0 but here we specify 1 as the start index:
            for line_num, line in enumerate(input_file, 1):
                
                # Clean text:
                clean_line = line.replace("►", "").replace("■", "").replace("▼", "")
                clean_line = clean_line.replace("Confidential Copy \n            for \n         DKPPU", "")
                clean_line = re.sub(r'\n(?=[a-z.])', '', clean_line)
                clean_line = re.sub(r'\n+', '\n', clean_line)
                clean_line = re.sub(r'[^\w\s]', '', clean_line)     # This regex substitutes anything that is not a word character or whitespace with an empty string.
                clean_line = re.sub(r'\s+', ' ', clean_line).strip()    # This regex substitutes any sequence of whitespace characters with a single space.

                # Write the cleaned text to the output file 
                output_text_file.write(clean_line + '\n')

                # Whoosh prep
                whoosh_page_dict_entry = {"title": source_filename, "content": clean_line, "pagenumber":line_num}
                text_data.append(whoosh_page_dict_entry)
    except Exception as e:
        handle_local_error("Could not create text file, encountered error: ", e)

    # Create Whoosh Index; if error, log exception and proceed to returning output_text_file_path
    try:
        whoosh_indexer(text_data)
    except Exception as e:
        handle_error_no_return("Could not index file, encountered error: ", e)

    return output_text_file_path



def find_text_in_pdf_dpr(pdf_path, target_text):
    print("pdf_path, target_text: ", pdf_path, ", ", target_text)

    page_numbers = []

    try:
        with open(pdf_path, 'rb') as file:

            reader = PyPDF2.PdfReader(file)

            for page_num in range(len(reader.pages)):

                page = reader.pages[page_num]

                content = page.extract_text()

                if target_text in content:
                    print("found match!")
                    page_numbers.append(page_num + 1)
    except Exception as e:
        handle_local_error("Could not find page numbers from PDF, encountered error: ", e)

    print("page numbers before returning: ")
    print(page_numbers)
    return page_numbers



def find_text_in_pdf(reference_pages):

    user_should_refer_pages_in_doc = {}
    docs_have_relevant_info = False

    for doc_path in reference_pages:

        source_filename = os.path.basename(doc_path)

        try:
            text = extract_text(doc_path)
            pages = text.split("\f")
            page_numbers = []

            for page_num, content in enumerate(pages):
                for target_text in reference_pages[doc_path]:
                    target_text = preprocess_string(target_text)
                    content = preprocess_string(content)
                    if target_text in content:
                        page_numbers.append(page_num + 1)
                        docs_have_relevant_info = True
            
            page_numbers = set(page_numbers)

            user_should_refer_pages_in_doc[source_filename] = page_numbers
        except Exception as e:
            handle_local_error("Could not find page numbers from PDF, encountered error: ", e)

    return docs_have_relevant_info, user_should_refer_pages_in_doc




def whoosh_text_in_pdf(reference_pages):

    print("Searching Index")

    try:
        read_return = read_config(['index_dir'])
        index_dir = read_return['index_dir']
    except Exception as e:
        handle_local_error("Missing index_dir in config.json for method whoosh_text_in_pdf. Error: ", e)

    user_should_refer_pages_in_doc = {}
    docs_have_relevant_info = False

    try:
        # Open the index
        ix = open_dir(index_dir)

        # Create a 'searcher' object
        with ix.searcher() as searcher:
            query_parser = QueryParser("content", ix.schema)

            for doc in reference_pages:
                
                source_filename = os.path.basename(doc)
                page_numbers = []
                
                for search_string in reference_pages[doc]:

                    # Only search for non-empty search strings
                    if search_string:

                        query = query_parser.parse(search_string)

                        results = searcher.search(query)

                        for hit in results:
                            print(f"Found in {hit['title']} on page {hit['pagenumber']}")
                            page_numbers.append(int(hit['pagenumber']))
                            docs_have_relevant_info = True

                page_numbers = set(page_numbers)
                user_should_refer_pages_in_doc[source_filename] = page_numbers

    except Exception as e:
        handle_error_no_return("Could not search Whoosh Index, encountered error: ", e)

    return docs_have_relevant_info, user_should_refer_pages_in_doc



# Route to handle the submission of the second form (file loading)
@app.route('/process_file', methods=['POST'])
def process_file():

    use_ocr = False
    try:
        read_return = read_config(['use_ocr', 'ocr_service_choice'])
        use_ocr = read_return['use_ocr']
        ocr_service_choice = read_return['ocr_service_choice']
    except Exception as e:
        handle_api_error("Could not determine use_ocr in config.json for process_new_file. Disabling OCR and proceeding. Error: ", e)

    try:
        load_new_file = request.form.get('load_new_file', 'n').lower()
    except Exception as e:
        handle_api_error("Server-side error - could not interpret user selection. Encountered error: ", e)

    if load_new_file ==  'y':

        try:
            input_file = request.files['input_file']
        except Exception as e:
            handle_api_error("Server-side error recieving file: ", e)

        # Ensure the filename is secure
        filename = secure_filename(input_file.filename)
        if "PDF" in filename:
            filename = filename.replace("PDF", "pdf")

        pdf_file = False
        txt_file = False

        if filename.endswith('.pdf'):
            pdf_file = True
        elif filename.endswith('.txt'):
            txt_file = True
        else:
            return jsonify(success=False, error="Invalid file format, expected a PDF or TXT file"), 400 #HTTP Bad Request


        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print("Loading new file - filename: ", filename)
            print("Loading new file - filepath: ", filepath)

            # Save the uploaded file to the specified path
            input_file.save(filepath)
        except Exception as e:
            handle_api_error("Failed to save document to app folder, encountered error: ", e)
        
        # print("input_file: ", input_file)
        
        if pdf_file:
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
                        handle_api_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)
            else:
                try:
                    input_file = PDFtoTXT(filepath)
                except Exception as e:
                    handle_api_error("Failed to extract text from the PDF document, even via fallback PyPDF2, encountered error: ", e)

            try:
                images = extract_images_from_pdf(filepath)
            except Exception as e:
                handle_error_no_return("Failed to extract images from the PDF document, encountered error: ", e)

            try:
                store_images_to_db(images)
            except Exception as e:
                handle_error_no_return("Failed to save images to database, encountered error: ", e)

        if txt_file:
            print("Processing Text file")

            try:
                # Need to set to filepath as input_file just contains the file itself from the POST!
                input_file = TxtCleaner(filepath)
            except Exception as e:
                handle_api_error("Failed to extract text from PDF: ", e)
        
        try:
            LoadNewDocument(input_file)         
        except Exception as e:
            handle_api_error("Failed to extract text from PDF: ", e)


    # Don't get confused about not loading the VectorDB here! You'll notice that we're doing an additional VectorDB load step in the route method below, 'process_new_file()', but not here!
    # This is because this current route is triggered BEFORE the initital model and VectorDB loading occurs, so after this '/load_model_and_vectordb' triggers and loads the DB anyway!
    # However, when '/process_new_file' is invoked mid-chat, the VectorDB must be RE-LOADED! Hence the extra step in the route below. 
    
    #return "File processed (or not) and ready for chat!"
    #return redirect(url_for('load_model_and_vectordb'))
    return jsonify(success=True)


@app.route('/load_model_and_vectordb')
def load_model_and_vectordb():
    
    global LLM
    global VECTOR_STORE
    global LOADED_UP
    global LLM_CHANGE_RELOAD_TRIGGER_SET
    global HISTORY_SUMMARY
    global HISTORY_MEMORY_WITH_BUFFER
    global HF_BGE_EMBEDDINGS
    global AZURE_OPENAI_EMBEDDINGS

    try:
        read_return = read_config(['model_choice', 'use_gpu_for_embeddings', 'use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'vectordb_sbert_folder', 'vectordb_openai_folder', 'vectordb_bge_base_folder', 'vectordb_bge_large_folder', 'use_azure_open_ai'])
        model_choice = read_return['model_choice']
        use_gpu_for_embeddings = read_return['use_gpu_for_embeddings']
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        vectordb_sbert_folder = read_return['vectordb_sbert_folder']
        vectordb_openai_folder = read_return['vectordb_openai_folder']
        vectordb_bge_base_folder = read_return['vectordb_bge_base_folder']
        vectordb_bge_large_folder = read_return['vectordb_bge_large_folder']
        use_azure_open_ai = read_return['use_azure_open_ai']
    except Exception as e:
        handle_api_error("Missing values in config.json when attempting to load_model_and_vectordb. Error: ", e)


    # global CONVERSATION_RAG_CHAIN_WITH_SUMMARY_BUFFER

    if LOADED_UP and not LLM_CHANGE_RELOAD_TRIGGER_SET:
        print(f'\n\nAlready loaded! Clearing chat history and returning model choice: {model_choice}\n\n')
        HISTORY_MEMORY_WITH_BUFFER.chat_memory.clear()
        HISTORY_MEMORY_WITH_BUFFER = ConversationSummaryBufferMemory(llm=LLM, max_token_limit=300, return_messages=False)
        HISTORY_SUMMARY = {}
        return jsonify({'success': True, 'llm_model': model_choice})
    elif LLM_CHANGE_RELOAD_TRIGGER_SET:
        print('\n\nForce restarting app! Preserving chat history and proceeding to reload the VectorDB & LLM. Resetting reset flag too.\n\n')
        LLM_CHANGE_RELOAD_TRIGGER_SET = False
        

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
                read_return = read_config(['azure_openai_base_url', 'azure_openai_api_key', 'azure_openai_api_type', 'c'])
                azure_openai_base_url = read_return['azure_openai_base_url']
                azure_openai_api_key = read_return['azure_openai_api_key']
                azure_openai_api_type = read_return['azure_openai_api_type']
                azure_openai_api_version = read_return['azure_openai_api_version']
            except Exception as e:
                handle_error_no_return("Missing values for Azure OpenAI Embeddings in method load_model_and_vectordb in config.json. Error: ", e)
            
            try:
                os.environ["OPENAI_API_BASE"] = azure_openai_base_url
                os.environ["OPENAI_API_KEY"] = azure_openai_api_key
                os.environ["OPENAI_API_TYPE"] = azure_openai_api_type
                os.environ["OPENAI_API_VERSION"] = azure_openai_api_version
            except Exception as e:
                handle_error_no_return("Could not set OS environment variables for Azure OpenAI Embeddings in load_model_and_vectordb, encountered error: ", e)
            
            AZURE_OPENAI_EMBEDDINGS = OpenAIEmbeddings(deployment="openai-ada-embedding")
            VECTOR_STORE = Chroma(persist_directory=vectordb_openai_folder, embedding_function=AZURE_OPENAI_EMBEDDINGS)
        
        elif use_bge_base_embeddings:
            if HF_BGE_EMBEDDINGS is not None:
                VECTOR_STORE = Chroma(persist_directory=vectordb_bge_base_folder, embedding_function=HF_BGE_EMBEDDINGS)
            else:
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
            if HF_BGE_EMBEDDINGS is not None:
                VECTOR_STORE = Chroma(persist_directory=vectordb_bge_large_folder, embedding_function=HF_BGE_EMBEDDINGS)
            else:
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
        handle_api_error("Could not load VectorDB, encountered error: ", e)


    ### 2 - Load LLM Model from config.json ###
    print("\n\nLoading LLM from config.json\n\n")
    try:

        if not use_azure_open_ai:

            try:
                read_return = read_config(['use_gpu', 'model_dir', 'local_llm_context_length', 'local_llm_max_new_tokens', 'local_llm_gpu_layers', 'local_llm_model_type', 'local_llm_temperature'])
                use_gpu = read_return['use_gpu']
                local_llm_context_length = read_return['local_llm_context_length']
                local_llm_max_new_tokens = read_return['local_llm_max_new_tokens']
                local_llm_gpu_layers = read_return['local_llm_gpu_layers']
                local_llm_model_type = read_return['local_llm_model_type']
                local_llm_temperature = read_return['local_llm_temperature']
                model_dir = read_return['model_dir']
            except Exception as e:
                handle_api_error("Missing values in config.json for setting-up local-LLM in method load_model_and_vectordb. Error: ", e)

            llm_model = model_dir + '/' + model_choice

            config = {'context_length': local_llm_context_length, 'max_new_tokens': local_llm_max_new_tokens, 'temperature': local_llm_temperature}
            
            if use_gpu:
                config.update({'gpu_layers':local_llm_gpu_layers})
            
            LLM = CTransformers(model=llm_model, model_type=local_llm_model_type, config=config, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

        else:

            try:
                read_return = read_config(['azure_openai_base_url', 'azure_openai_api_key', 'azure_openai_api_type', 'azure_openai_deployment_name', 'azure_openai_api_version', 'azure_openai_max_tokens', 'azure_openai_temperature'])
                azure_openai_base_url = read_return['azure_openai_base_url']
                azure_openai_api_key = read_return['azure_openai_api_key']
                azure_openai_api_type = read_return['azure_openai_api_type']
                azure_openai_deployment_name = read_return['azure_openai_deployment_name']
                azure_openai_api_version = read_return['azure_openai_api_version']
                azure_openai_max_tokens = read_return['azure_openai_max_tokens']
                azure_openai_temperature = read_return['azure_openai_temperature']
            except Exception as e:
                handle_api_error("Missing values in config.json for setting-up Azure-OpenAI-LLM in method load_model_and_vectordb. Error: ", e)
            
            LLM = AzureChatOpenAI(
                openai_api_base=azure_openai_base_url,
                openai_api_version=azure_openai_api_version,
                deployment_name=azure_openai_deployment_name,
                openai_api_key=azure_openai_api_key,
                openai_api_type=azure_openai_api_type,
                max_tokens=azure_openai_max_tokens, 
                temperature=azure_openai_temperature,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()]
            )

    except Exception as e:
        handle_api_error("Could not load LLM, encountered error: ", e)

    print("\n\n")


    ### 3 - Define History memory w/ buffer:
    try:
        HISTORY_MEMORY_WITH_BUFFER = ConversationSummaryBufferMemory(llm=LLM, max_token_limit=300, return_messages=False)
    except Exception as e:
        handle_api_error("Could not setup memory buffer for LLM, encountered error: ", e)
    
    LOADED_UP = True
    print(f'\n\nDone loading! Returning model choice: {model_choice}\n\n')
    return jsonify({'success': True, 'llm_model': model_choice})


 # Do not delete as vectorDB folder remains on disk
    # Once new VectorDB is created, proceed to update records DB:
    # try:
    #     read_return = read_config(['sqlite_docs_loaded_db'])
    #     sqlite_docs_loaded_db = read_return['sqlite_docs_loaded_db']
    # except Exception as e:
    #     handle_api_error("Missing sqlite_docs_loaded_db in config.json in method reset_vector_db_on_disk. Error: ", e)
    
    # try:
    #     conn = sqlite3.connect(sqlite_docs_loaded_db)
    #     c = conn.cursor()
    # except Exception as e:
    #     handle_api_error("Could not connect to sqlite_docs_loaded_db database to delete file list, encountered error: ", e)

    # try:
    #     c.execute("DELETE FROM document_records where embedding_model = ?", (selected_embedding_model_choice,))
    #     conn.commit()
    #     print(f"Deleted all records where embedding_model = {selected_embedding_model_choice}")
    # except Exception as e:
    #     handle_api_error("Could not delete document list from document_records db, encountered error: ", e)


    

@app.route('/setup_for_streaming_response', methods=['POST'])
def setup_for_streaming_response():

    print("\n\nSetting up to stream response\n\n")

    global QUERIES
    do_rag = True   # We will only return an internal server error in the events that do_rag cannot be written, the user_query cannot be read or if a unique stream_session_id cannot be established

    stream_session_id = ""
    # Generate a unique session ID using universally Unique Identifier via the uuid4() method, wherein the randomness of the result is dependent on the randomness of the underlying operating system's random number generator
    # UUI is a standard used for creating unique strings that have a very high likelihood of being unique across all time and space, for ex: f47ac10b-58cc-4372-a567-0e02b2c3d479
    try:
        stream_session_id = str(uuid.uuid4())
    except Exception as e:
        handle_api_error("Error creating unique stream_session_id when attempting to setup_for_streaming_response. Error: ", e)


    try:
        read_return = read_config(['use_sbert_embeddings', 'use_openai_embeddings', 'use_bge_base_embeddings', 'use_bge_large_embeddings', 'force_enable_rag', 'force_disable_rag'])
        use_sbert_embeddings = read_return['use_sbert_embeddings']
        use_openai_embeddings = read_return['use_openai_embeddings']
        use_bge_base_embeddings = read_return['use_bge_base_embeddings']
        use_bge_large_embeddings = read_return['use_bge_large_embeddings']
        force_enable_rag = read_return['force_enable_rag']
        force_disable_rag = read_return['force_disable_rag']
    except Exception as e:
        handle_api_error("Missing values in config.json when attempting to setup_for_streaming_response. Error: ", e)


    # We do not modify the force_enable_rag or force_disable_rag flags in this method, we simply respond to them here. UI updates should handle those flags.
    if force_enable_rag:
        
        print("\n\nFORCE_ENABLE_RAG True, force enabling RAG and returning\n\n")
        
        do_rag = True
        
        try:
            write_config({'do_rag':do_rag})
        except Exception as e:
            handle_api_error("Could not force_enable_rag when attempting to setup_for_streaming_response, encountered error: ", e)
        
        return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag})
    
    if force_disable_rag:

        print("\n\nFORCE_DISABLE_RAG True, force disabling RAG and returning\n\n")

        do_rag = False

        try:
            write_config({'do_rag':do_rag})
        except Exception as e:
            handle_api_error("Could not force_disable_rag when attempting to setup_for_streaming_response, encountered error: ", e)

        return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag})

    try:
        # Attempt to get the user's query
        user_query = request.json['message']
        # Store the query associated with the ID
        QUERIES[stream_session_id] = user_query
    except KeyError:
        handle_api_error("Could not obtain and/or store user_query in setup_for_streaming_response, encountered error: ", e)


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
        docs = VECTOR_STORE.similarity_search(user_query, embedding_fn=embedding_function)
        # docs_with_relevance_score = VECTOR_STORE.similarity_search_with_relevance_scores(user_query, 10, embedding_fn=embedding_function)
        # docs_list_with_cosine_distance = VECTOR_STORE.similarity_search_with_score(user_query, 10, embedding_fn=embedding_function)
        # print(f'\n\nsimple similarity search results: \n {docs}\n\n')
        # print(f'\n\nRelevance Score similarity search results (range 0 to 1): \n {docs_with_relevance_score}\n\n')
        # print(f'\n\nDocs list most similar to query based on cosine distance: \n {docs_list_with_cosine_distance}\n\n')
    except Exception as e:
        handle_error_no_return("Could not perform similarity_search to determin do_rag when attempting to setup_for_streaming_response, encountered error: ", e)


    print("\n\nDetermining do_rag \n\n")
    try:
        page_contents, do_rag = filter_relevant_documents(user_query, docs)
    except Exception as e:
        handle_error_no_return("Force enabling RAG and returning: could not determine do_rag during setup_for_streaming_response, encountered error: ", e)
    
    print(f'Do RAG? {do_rag}')

    try:
        write_config({'do_rag':do_rag})
    except Exception as e:
        handle_api_error("Could not write do_rag during setup_for_streaming_response, encountered error: ", e)


    # Return the stream_session_id
    return jsonify({"success": True, "stream_session_id": stream_session_id, "do_rag": do_rag})
    
    
    # if matched_images_found:
    #     images_iframe_html = "<br><h6>Refer to the images below:</h6>"
    #     for image_id, image_bytes_data in matched_images_in_bytes:
    #         #print(f"\n\nmatched image id: {image_id}")
    #         try:
    #             image_link_url = url_for('image_display', image_id=image_id)
    #             images_iframe_html += f'<br><iframe width="750" height="400" src="{image_link_url}" frameborder="0"></iframe><br>'
    #         except Exception as e:
    #             handle_error_no_return("Could not construct images_iframe_html, encountered error: ", e)


#############################################################################
##############---NOTES ON THE BELOW CUSTOM CLASS APPROACH---#################
#############################################################################

# class CustomStream(io.StringIO)
#   defines a new class 'CustomStream' that inherits the StringIO class from the io module.
#   'StringIO' is an in-memory, file-like object that can be used as a string buffer, essentially a file in-memory rather than on disk

# def __init__(self, callback=None)
#   initialization method for instances of 'CustomStream' accepting one optional argument that defaults to None if not provided

# super().__init__()
#   calls the 'init' method of the parent class 'StringIO', which here is also the super & base class!
#   necessary to ensure that parent/base/super class 'StringIO' is properly initialized for instances of 'CustomStream'
# 
# self.callback = callback
#   The passed 'callback' attribute is stored as an instance attrib, meaning each instance of 'CustomStream' will have its own 'callback' attrib
 
# def write(self, data)
#   Overwrites the 'write' method of parent class 'StringIO'; this method is called whenever data is written to our 'CustomStream'
 
# PRIMARY MOTIVATION FOR THIS CUSTOM CLASS!! If we have a callback, call it:
# if self.callback:
#   self.callback(data)
#
#   The method checks if a 'callback' function has been set for the instance, i.e. 'self.callback' is not 'None'
#   If there is a 'callback', it calls that function with the provided data, 
#   which allows us to "hook" into the write process & execute additional logic whenever data is written to the 'CustomStream'   

# return super().write(data)
#   Finally, this calls the 'write' method of the parent class 'StringIO' using the 'super()' function
#   This ensures that the actual writing of the data to the in-memory buffer, which is the primary function of 'StringIO' still happens!
#   The provided data is passed for this to the base method
 
# In summary, this 'CustomStream' class provides a custom implementation of 'StringIO' that supports a callback mechanism:
# Everytime data is written to this custom stream, the 'callback', if provided, is executed thus allowing for additional functionality during the write

# In this application, this mechanism is used to queue data for the streaming response! 
# It extends the 'StringIO' class by adding a new feature: the ability to trigger a 'callback' function whenever a 'write()' occurs!

# So to use this:

# 1. We define a queue: 
#       data_queue = queue.Queue()

# 2. We define a callback function that puts data in this queue:
#        def callback(data):
#           data_queue.put(data)

# 3. We create an instance of our custom stream passing this callback function:
#        custom_stream = CustomStream(callback=callback)

# 4. We redirect stdout to our custom stream temporarily
#       original_stdout = sys.stdout
#        sys.stdout = custom_stream

# 5. We start the llm_task() thread & the LLM() function now outputs here, finally resetting stdout and putting None into the queue: data_queue.put(None)

# 6. While the thread runs, we start a while loop that keeps yielding from the queue and stopping when None is read: yield f"data: {line}\n\n"

# 7. A final yield signals the end of the stream, to be handled at the client-side:  yield "event: END\ndata: null\n\n"

#############################################################################
#################---XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX---####################
#############################################################################


class CustomStream(io.StringIO):
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback

    def write(self, data):
        # If we have a callback, call it
        if self.callback:
            self.callback(data)

        return super().write(data)


@app.route('/stream/<stream_session_id>')
def stream(stream_session_id):

    print("stream route triggered")

    global QUERIES
    global HISTORY_MEMORY_WITH_BUFFER

    try:
        read_return = read_config(['do_rag', 'base_template'])
        do_rag = read_return['do_rag']
        base_template = read_return['base_template']
    except Exception as e:
        error_message = f"\n\Missing values in config.json in main method stream!. Error: {e}\n\n"
        if logger:
            logger.error(error_message)
            print(error_message)
        else:
            print(error_message)
        return jsonify(success=False, error=error_message), 500 # internal server error

    key_for_llm_result = "LlmResponseforQueryID_" + stream_session_id
    key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id

    user_query = request.args.get('input')
    #print(f"do_rag: {do_rag}")

    print(f'\n\nuser query passed to the LLM: {user_query}\n\n')

    if do_rag:
        ### 0 - If memory has been reset due to an old chat loading up, delete the additional key that added for a non-RAG resuming scenario:
        if 'has_been_reset' in HISTORY_SUMMARY:
            del HISTORY_SUMMARY['has_been_reset']

        ### 1 - Define Template:
        rag_prompt_template_variables = """

        Use the following context to answer the user's question:
        
        Context:{context}
        Question:{question}
        """
        history_summary_for_rag = re.sub(r"\{|\}", "", str(HISTORY_SUMMARY))    #search through the string str(HISTORY_SUMMARY) for all instances of { and } and replace them with an empty string "", effectively removing these characters from the string

        rag_history_prompt = "The conversation so far: " + history_summary_for_rag

        rag_prompt_template = rag_history_prompt + "\n" + base_template + "\n" + rag_prompt_template_variables

        print(f"\n\nrag_prompt_template: {rag_prompt_template}\n\n")

        rag_qa_chain_prompt = PromptTemplate.from_template(rag_prompt_template)

        ### 2 - Setup Chain
        qa_chain = RetrievalQA.from_chain_type(LLM, retriever=VECTOR_STORE.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt":rag_qa_chain_prompt})

    else:
        ### 0 - If memory has been reset due to an old chat loading up, delete the additional key that added for a non-RAG resuming scenario:
        if 'has_been_reset' in HISTORY_SUMMARY:
            del HISTORY_SUMMARY['has_been_reset']

        ### 1 - Define Template keping in mind if memory has been reset due to an an old chat loading up:
        non_rag_prompt_template_variables = "\nCurrent conversation:\n{history}\nHuman: {input}\nAI Assistant:"

        history_summary_for_non_rag = re.sub(r"\{|\}", "", str(HISTORY_SUMMARY))    #search through the string str(HISTORY_SUMMARY) for all instances of { and } and replace them with an empty string "", effectively removing these characters from the string
        
        non_rag_history_prompt = "The conversation so far: " + history_summary_for_non_rag
        
        non_rag_prompt_template = non_rag_history_prompt + "\n" + base_template + "\n" + non_rag_prompt_template_variables

        #print(f"\n\nnon_rag_prompt_template: {non_rag_prompt_template}\n\n")
        non_rag_qa_chain_prompt = PromptTemplate(template=non_rag_prompt_template, input_variables=["history","input"])
        print(f"\n\non_rag_qa_chain_prompt: {non_rag_qa_chain_prompt}\n\n")

        ### 2 - Setup Chain
        conversation_chain_with_summary_buffer = ConversationChain(
            llm = LLM,
            prompt=non_rag_qa_chain_prompt
        )

    if not user_query:
        return "Session not found", 404
    
    stop_thread = threading.Event()
    # Will be set() in llm_task() methods 'finally' block to stop the thread: Once the inferencing is complete in the 'try' block, the 'finally' block adds a 'None' object to the queue 
    # and sets the threading event below. The 'None' causes the yielding while loop to invoke join() on the llm_task() thread for synchronization, as it causes the invoking thread, 
    # in this case main thread, to wait until the ll_task thread object completes execution before resuming. Meanwhile setting the stop_thread causes the llm_task threads while() to complete!

    def generate():

        data_queue = queue.Queue()

        def callback(data):
            data_queue.put(data)

        custom_stream = CustomStream(callback=callback)

        # Redirect stdout to our custom stream temporarily
        original_stdout = sys.stdout
        sys.stdout = custom_stream

        def llm_task():
            global QUERIES
            global HISTORY_SUMMARY
            result = ""
            while not stop_thread.is_set():
                # Call LLM
                try:
                    
                    if do_rag:
                        result = qa_chain({"query": user_query})

                        # Experimental RAG chains here:

                    else:
                        result = conversation_chain_with_summary_buffer.predict(input=user_query)
                        
                        # Experimental chains here:
                finally:
                    # Reset stdout to its original value
                    sys.stdout = original_stdout

                    # experimental outputs here:
                    #print(f"history_buffer_result: {history_buffer_result}")

                    # Save the LLM's formatted response for reference searching 
                    formatted_llm_output = ""
                    
                    if do_rag:
                        print("\n\nStoring RAG-Context history:\n")

                        formatted_llm_output = str(result['result'])
                        QUERIES[key_for_llm_result] = formatted_llm_output
                        QUERIES[key_for_vector_results] = result

                        HISTORY_MEMORY_WITH_BUFFER.save_context({"input":user_query}, {"output":formatted_llm_output})
                    else:
                        HISTORY_MEMORY_WITH_BUFFER.save_context({"input":user_query}, {"output":result})

                    
                    HISTORY_SUMMARY = HISTORY_MEMORY_WITH_BUFFER.load_memory_variables({})
                    print(f"\n\nHISTORY_SUMMARY:{HISTORY_SUMMARY}\n\n")
                    print(f"\n\nHISTORY_MEMORY_WITH_BUFFER.chat_memory.messages: {HISTORY_MEMORY_WITH_BUFFER.chat_memory.messages}\n\n")


                    if not do_rag:
                        formatted_user_query = str(user_query).strip('\n')
                        formatted_llm_output = str(result)
                        formatted_llm_output = formatted_llm_output.strip('\n')
                        
                        # If RAG is done, get_references() method stores history. If not, we store history right here
                        print(f"\n\nStoring chat history with non-RAG LLM output: {formatted_llm_output}\n\n")
                        
                        # Storing to history DB as get_references() will not be invoked in non-RAG chains!
                        store_chat_history_to_db(formatted_user_query, formatted_llm_output, HISTORY_SUMMARY)

                    # Stop thread
                    data_queue.put(None)
                    stop_thread.set()

        # Start the LLM task in a separate thread
        thread = threading.Thread(target=llm_task)
        thread.start()

        i = 0
        # Continuously yield data as it becomes available
        while True:
            line = data_queue.get()
            if line is None:
                print("None read, breaking & stopping thread")
                thread.join()
                break
            if i == 0:
                line = line.strip('\n')
                i += 1
            line = line.replace('\n\n', '</br></br>')
            line = line.replace('\n', '</br>')
            #line = re.sub(r'\s{2,}', lambda match: '&nbsp;' * len(match.group()), line)
            yield f"data: {line}\n\n"

        # This part ensures that after LLM finishes, the stream is closed
        yield "event: END\ndata: null\n\n"

        print("LLM stream done")

    print("\n\nStarting inferencing!\n\n")
    return Response(generate(), content_type='text/event-stream')


@app.route('/lc_get_references', methods=['POST'])
def lc_get_references():

    print("\n\nGetting References\n\n")

    try:
        read_return = read_config(['do_rag', 'upload_folder'])
        do_rag = read_return['do_rag']
        upload_folder = read_return['upload_folder']
    except Exception as e:
        handle_api_error("Missing values in config.json when attempting to get_references. Error: ", e)

    if not do_rag:
        print("\n\nSkipping RAG and returning\n\n")
        return jsonify({'success': True, 'chat_id': CHAT_ID, 'sequence_id': SEQUENCE_ID})

    try:
        stream_session_id = request.json['stream_session_id']
        user_query = request.json['message']
    except Exception as e:
        handle_api_error("Could not read request content in method get_references, encountered error: ", e)
        
    try:
        key_for_vector_results = "VectorDocsforQueryID_" + stream_session_id
        key_for_llm_result = "LlmResponseforQueryID_" + stream_session_id

        docs = QUERIES[key_for_vector_results]
        llm_response = QUERIES[key_for_llm_result]
    except Exception as e:
        handle_api_error("Could not obtain relevant data from QUERIES dict, encountered error: ", e)

    # Having obtained the relevant info, clear the QUERIES{} dict so as to not bloat it!
    try:
        del QUERIES[key_for_vector_results]
        del QUERIES[key_for_llm_result]
    except Exception as e:
        handle_error_no_return("Error clearing queries dict in method get_references: ", e)

    reference_response = ""

    all_sources = {}
    reference_pages = {}

    try:
        print(f"\n\ndocs['source_documents']: {docs['source_documents']}\n\n")
        print(f"\n\ndocs['result']: {docs['result']}\n\n")
    except Exception as e:
        handle_api_error("Could not parse vector DB search results during get_references() ops, encountered error: ", e)
    

    relevant_pages = "<br><br>Relevant Pages & Topics:<br><br>"

    for doc in docs['source_documents']:
        try:
            relevant_pages += str(doc.page_content)
            relevant_pages += "<br>In Source Document:<br>"
            relevant_pages += str(doc.metadata)
            relevant_pages += "<br><br>"

            relevant_page_text = str(doc.page_content)

            source_filepath = str(doc.metadata["source"])
        except Exception as e:
            handle_error_no_return("Could not access doc.page_content and/or doc.metadata, encountered error: ", e)
            continue
    
        relevant_page_text = relevant_page_text.split('\n', 1)[0]
        relevant_page_text = relevant_page_text.strip()
        relevant_page_text = re.sub(r'[\W_]+Page \d+[\W_]+', '', relevant_page_text)

        source_filepath = source_filepath.replace('\\', '/')
        
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

                source_filename = source_filename.replace('.txt', '.pdf')
                
                if pdf_version_path in reference_pages:
                    reference_pages[pdf_version_path].extend([relevant_page_text])
                else:
                    reference_pages[pdf_version_path] = [relevant_page_text]

                # Add this file to our sources dictionary if it's not already present
                if source_filename not in all_sources:
                    source_filepath = pdf_version_path
                    all_sources.update({source_filename: source_filepath})

            # Else PDF does not exist, TXT is the source
            else:
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

    # print(f"\n\nreference_pages: {reference_pages}\n\n")

    try:
        docs_have_relevant_info, user_should_refer_pages_in_doc = whoosh_text_in_pdf_and_highlight(reference_pages, stream_session_id)
        # docs_have_relevant_info, user_should_refer_pages_in_doc = whoosh_text_in_pdf(reference_pages)
    except Exception as e:
        handle_error_no_return("Could not search Whoosh Index, encountered error: ", e)

    try:
        matched_images_found, matched_images_in_bytes = find_images_in_db(reference_pages)
    except Exception as e:
        handle_error_no_return("Could not search for images, encountered error: ", e)

    refer_pages_string = ""
    download_link_html = ""
    images_iframe_html = ""

    if docs_have_relevant_info:
        
        # refer_pages_string = "<br><br>Refer to the following pages in the mentioned docs:<br>"
        # for doc in user_should_refer_pages_in_doc:
        #     try:
        #         # Remove duplicates from reference_pages dict
        #         refer_pages_string += "<br>" + str(doc) + ": " + str(user_should_refer_pages_in_doc[doc]).replace("{", "").replace("}", "") + "<br>"
        #     except Exception as e:
        #         error_message = f"\n\nCould not construct refer_pages_string, encountered error: {e}\n\n"
        #         if logger:
        #             logger.error(error_message)
        #             print(error_message)
        #         else:
        #             print(error_message)


        refer_pages_string = "<br><br><h6>Refer to the following pages in the mentioned docs:</h6><br>"
        
        # for doc in user_should_refer_pages_in_doc:
        for index, doc in enumerate(user_should_refer_pages_in_doc, start=1):
            # pdf_iframe_id = str(doc) + "PdfViewer"
            pdf_iframe_id = "stream" + stream_session_id + "PdfViewer" + str(index)
            frame_doc_path = f"/pdf/{doc}"
            # frame_doc_path = upload_folder + f"/{doc}" 
            try:
                refer_pages_string += f"<br><h6>{doc}: "
                for page in user_should_refer_pages_in_doc[doc]:
                    frame_doc_path += "#page=" + str(page) 
                    refer_pages_string += f'<a href="javascript:void(0)" onclick="goToPage(\'{pdf_iframe_id}\', \'{frame_doc_path}\')">Page {page}</a>, '
                    frame_doc_path = f"/pdf/{doc}"
                refer_pages_string = refer_pages_string.strip(', ') + "</h6><br>"
            except Exception as e:
                handle_error_no_return("Could not construct refer_pages_string, encountered error: ", e)

        # download_link_html = "<br><h6>Refer to the source documents below:</h6>"
        pdf_right_pane_id = "stream" + stream_session_id + "PdfPane"
        download_link_html = f'<div class="pdf-viewer" id={pdf_right_pane_id}>'

        for index, source in enumerate(user_should_refer_pages_in_doc, start=1):
            try:
                # print("\n\nlooping sources\n\n")
                download_link_url = url_for('download_file', filename=source)
                pdf_iframe_id = "stream" + stream_session_id + "PdfViewer" + str(index)
                download_link_html += f'<br><h6><a href="{download_link_url}" target="_blank"><iframe id="{pdf_iframe_id}" src="{download_link_url}" width="100%" height="600"></iframe></a></h6><br>'
            except Exception as e:
                handle_error_no_return("Could not construct download_link_html, encountered error: ", e)

        download_link_html += "</div>"
        
        # print(f"\n\nall_sources: {all_sources}\n\n")
        # for source in all_sources:
        #     try:
        #         # print("\n\nlooping sources\n\n")
        #         download_link_url = url_for('download_file', filename=source)
        #         pdf_iframe_id = str(source) + "PdfViewer"
        #         download_link_html += f'<br><a href="{download_link_url}" target="_blank"><iframe id="{pdf_iframe_id}" src="{download_link_url}" width="600" height="400"></iframe></a><br>'
        #     except Exception as e:
        #         error_message = f"\n\nCould not construct download_link_html, encountered error: {e}\n\n"
        #         if logger:
        #             logger.error(error_message)
        #             print(error_message)
        #         else:
        #             print(error_message)
    
    if matched_images_found:
        images_iframe_html = "<br><h6>Refer to the images below:</h6>"
        for image_id, image_bytes_data in matched_images_in_bytes:
            #print(f"\n\nmatched image id: {image_id}")
            try:
                image_link_url = url_for('image_display', image_id=image_id)
                images_iframe_html += f'<br><iframe width="750" height="400" src="{image_link_url}" frameborder="0"></iframe><br>'
            except Exception as e:
                handle_error_no_return("Could not construct images_iframe_html, encountered error: ", e)

    
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
        store_chat_history_to_db(user_query_for_history_db, model_response_for_history_db, HISTORY_SUMMARY)
    except Exception as e:
        handle_error_no_return("Could not store_chat_history_to_db in get_references(), encountered error: ", e)

    return jsonify({'success': True, 'response': reference_response, 'pdf_frame':download_link_html, 'chat_id': CHAT_ID, 'sequence_id': SEQUENCE_ID})



//Make a GET request to the server to load the LLM & vectorDB
                // fetch('/load_model_and_vectordb')
                // .then(response => {
                //     if (!response.ok) {
                //         return response.json().then(err => { throw new Error(err.error)});
                //     }
                //     return response
                // })
                // .then(response => response.json())
                // .then(data => {
                //     if (data.success) {

                //         llm_model = data.llm_model
                //         LLM_MODEL = String(data.llm_model)
                        
                //         // If LLM & VectorDB loaded successfully, init the chat history DB 
                //         fetch('/init_chat_history_db')
                //             .then(response => {
                //                 if (!response.ok) {
                //                     return response.json().then(err => { throw new Error(err.error)});
                //                 }
                //                 return response
                //             })
                //             .then(response => response.json())
                //             .then(data => {
                //                 if (data.success) {
                //                     // If LLM, VectorDB and chat history DB initialized successfully, continue

                //                     curr_chat_id = data.chat_id

                //                     curr_chat_id = " Chat ".concat(String(curr_chat_id))

                //                     display_chatid_and_model = String(curr_chat_id).concat(": ", String(llm_model))

                //                     document.getElementById('model_header').innerHTML = display_chatid_and_model;
                //                     document.getElementById('model_header').style.display = 'block';

                //                     // Load menu items for the chat history menu
                //                     loadChatHistoryMenu();
                //                     document.getElementById('ModelAndDBLoading').style.display = 'none';
                //                     document.getElementById('ReadyToChat').style.display = 'block';
                                    
                //                     var timeoutDelayInMilliseconds = 1500; //1.5 seconds
                //                     setTimeout(function() {
                //                         document.getElementById('ReadyToChat').style.display = 'none';
                //                     }, timeoutDelayInMilliseconds);
                                    
                //                 } else {
                //                     throw new Error('Error when initializing the chat history DB');
                //                 }
                //             })
                //             .catch(error => {
                //                 let full_error_message = "There was an error in initializing the chat history DB: " + String(error.message);
                //                 console.error(full_error_message);
                //                 alert(full_error_message);
                //             });
                //     } else {
                //         throw new Error('Data error when loading the model or vectorDB');
                //     }
                // })
                // .catch(error => {
                //     let full_error_message = "There was an error in loading the model or vectorDB: " + String(error.message);
                //     console.error(full_error_message);
                //     alert(full_error_message);
                // });
                

                // TEMPLATE: Make a GET request to the server
            //    fetch('/init_chat_history_db')
            //         .then(response => {
            //             if (!response.ok) {
            //                 throw new Error(`Server-side HTTP error! Status: ${response.status}`);
            //             }
            //             return response
            //         })
            //         .then(response => response.json())
            //         .then(data => {
            //             if (data.success) {

                            
            //             } else {
            //                 throw new Error('Data error when fetching history-menu list');
            //             }
            //         })
            //         .catch(error => {
            //             console.error("There was an error in fetching the history-menu list: ", error.message);
            //         });


            function sendMessage() {
    
                document.getElementById('processingQ').style.display = 'block';
    
                let userInput = document.getElementById('user-input').value;
    
                // Append user input to the chat area
                document.getElementById('chat-area').innerHTML += '<div class="user-message">' + userInput + '</div>';
    
                // Make AJAX call to the app.py server to get the models response
                fetch('/get_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({'message': userInput})
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server-side error! Status: ${response.status}`);
                    }
                    return response
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        CHAT_ID = data.chat_id;
                        SEQUENCE_ID = data.sequence_id;

                        document.getElementById('processingQ').style.display = 'none';
                        const responseAndRating = `
                        <div class="llm-wrapper">
                            <div class="llm-response">
                                ${data.response}
                            </div>
                            <div class="star-rating" data-rated="False" rating-chat-id=${data.chat_id} rating-sequence-id=${data.sequence_id}>
                                <i class="far fa-star" data-rate="1"></i>
                                <i class="far fa-star" data-rate="2"></i>
                                <i class="far fa-star" data-rate="3"></i>
                                <i class="far fa-star" data-rate="4"></i>
                                <i class="far fa-star" data-rate="5"></i>
                            </div>
                        </div>
                        `;
                        document.getElementById('chat-area').innerHTML += responseAndRating;
                        //document.getElementById('chat-area').innerHTML += '<div class="llm-response">' + data.response + '</div>';
                    } else {
                        throw new Error('Internal Server Error: Check server-log and server command-line for more details.');
                    }
                })
                .catch(error => {
                    errorHandler("fetching response", "/get_response", String(error.message))
                });
    
                // Clear the input field
                document.getElementById('user-input').value = '';
            }