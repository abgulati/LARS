# LARS - The LLM & Advanced Referencing Solution

<p align="center">
<img src="https://github.com/abgulati/LocalLLM/blob/main/web_app/static/images/LARS_Logo_3.png"  align="center">
</p>

LARS is an application that enables you to run LLM's (Large Language Models) locally on your device, upload your own documents and engage in conversations wherein the LLM grounds its responses with your uploaded content. This grounding helps increase accuracy and reduce the common issue of AI-generated inaccuracies or "hallucinations." This technique is commonly known as "Retrieval Augmented Generation", or RAG.

There are many desktop applications for running LLMs locally, and LARS aims to be the ultimate open-source RAG-centric LLM application. Towards this end, LARS takes the concept of RAG much further by adding detailed citations to every response, supplying you with specific document names, page numbers, text-highlighting, and images relevant to your question, and even presenting a document reader right within the response window. While all the citations are not always present for every response, the idea is to have at least some combination of citations brought up for every RAG response and that’s generally found to be the case. 

### Here's a list detailing LARS's feature-set as it stands today:

1. Advanced Citations: The main showcase feature of LARS - LLM-generated responses are appended with detailed citations comprising document names, page numbers, text highlighting and image extraction for any RAG centric responses, with a document reader presented for the user to scroll through the document right within the response window and download highlighted PDFs
2. Vast number of supported file-formats:
    - PDFs
    - Word files: doc, docx, odt, rtf, txt
    - Excel files: xls, xlsx, ods, csv
    - PowerPoint presentations: ppt, pptx, odp
    - Image files: bmp, gif, jpg, png, svg, tiff
    - Rich Text Format (RTF)
    - HTML files
3. Conversion memory: Users can ask follow-up questions, including for prior conversations
4. Full chat-history: Users can go back and resume prior conversations
5. Users can force enable or disable RAG at any time via Settings
6. Users can change the system prompt at any time via Settings
7. Drag-and-drop in new LLMs - change LLM's via Settings at any time
8. Built-in prompt-templates for the most popular LLMs and then some: Llama3, Llama2, ChatML, Phi3, Command-R, Deepseek Coder, Vicuna and OpenChat-3.5
9. Pure llama.cpp backend - No frameworks, no Python-bindings, no abstractions - just pure llama.cpp! Upgrade to newer versions of llama.cpp independent of LARS
10. GPU-accelerated inferencing: Nvidia CUDA-accelerated inferencing supported
11. Tweak advanced LLM settings - Change LLM temperature, top-k, top-p, min-p, n-keep, set the number of model layers to be offloaded to the GPU, and enable or disable the use of GPUs, all via Settings at any time 
12. Four embedding models - sentence-transformers/all-mpnet-base-v2, BGE-Base, BGE-Large, OpenAI Text-Ada
13. Sources UI - A table is displayed for the selected embedding model detailing the documents that have been uploaded to LARS, including vectorization details such as chunk_size and chunk_overlap 
14. A reset button is provided to empty and reset the vectorDB
15. Three text extraction methods: a purely local text-extraction option and two OCR options via Azure for better accuracy and scanned document support - Azure ComputerVision OCR has an always free-tier 
16. A custom parser for the Azure AI Document-Intelligence OCR service for enhanced table-data extraction while preventing double-text by accounting for the spatial coordinates of the extracted text

### A demonstration video showcasing these features can be viewed at the link below:

[LARS Feature-Demonstration Video](https://www.youtube.com/watch?v=Mam1i86n8sU&t=1s&ab_channel=AbheekGulati)

<a href="https://www.youtube.com/watch?v=Mam1i86n8sU&t=1s&ab_channel=AbheekGulati" target="_blank" ><img src="https://github.com/abgulati/LocalLLM/blob/main/web_app/static/images/LARS_Logo_3.png" alt="LARS Feature-Demonstration Video" style="max-width:50%;"></a>


## Dependencies

0. Build Tools:

    - On Windows:

        - Download Microsoft Visual Studio Build Tools 2022 from the [Official Site - "Tools for Visual Studio"](https://visualstudio.microsoft.com/downloads/)

    - On Linux (Ubuntu and Debian-based), install the following packages:

        - build-essential includes GCC, G++, and make
        - libffi-dev for Foreign Function Interface (FFI)
        - libssl-dev for SSL support   

        ```
        apt-get update
        apt-get install -y software-properties-common build-essential libffi-dev libssl-dev
        ```


1. Nvidia CUDA (if supported Nvidia GPU present):
    - Install Nvidia [GPU Drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

    - Install Nvidia [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) - LARS built and tested with v12.2.2

    - Verify Installation via the terminal:
        ```
        nvcc -V
        nvidia-smi
        ```

    - CMAKE-CUDA Fix (Very Important!):

        Copy all the four files from the following directory:   
        ```C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\visual_studio_integration\MSBuildExtensions```
        
        and Paste them to the following directory:   
        ```C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations```


2. llama.cpp:
    - Download from the [Official Repo](https://github.com/ggerganov/llama.cpp):
    
        ```
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp
        ```
    
    - Install CMAKE on Windows from the [Official Site](https://cmake.org/download/)

        - add to PATH:   
        ```C:\Program Files\CMake\bin```
    
    - Build llama.cpp with CMAKE:
    
        - Build with CUDA:   
        
        ```
        cmake -B build -DLLAMA_CUDA=ON
        cmake --build build  --config Release
        ```
    
        - Build without CUDA:   
        
        ```
        cmake -B build
        cmake --build build  --config Release
        ```
    
    - Add to PATH:   
        ```path_to_cloned_repo\llama.cpp\build\bin\Release```
    
    - Verify Installation via the terminal:
    
        ```
        server
        ```


3. Python:
    - Built and tested with Python v3.11.x

    - Windows:

        - Download v3.11.9 from the [Official Site](https://www.python.org/downloads/windows/)

        - During installation, ensure you check "Add Python 3.11 to PATH" or manually add it later, either via:

            - Advanced System Settings -> Environment Variables -> System Variables -> EDIT PATH Variable -> Add the below (change as per your installation location):    
                ```
                C:\Users\user_name\AppData\Local\Programs\Python\Python311\
                ```
            
            - Or via PowerShell:   
                ```
                Set PATH=%PATH%;C:\Users\user_name\AppData\Local\Programs\Python\Python311
                ```

    - Linux (Ubuntu and Debian-based):
    
        - via deadsnakes PPA:   

        ```
        add-apt-repository ppa:deadsnakes/ppa -y
        apt-get update
        apt-get install -y python3.11 python3.11-venv python3.11-dev
        python3.11 -m ensurepip
        ```

    - Verify Installation via the terminal:

        ```
        python --version
        ```


4. LibreOffice:
    - This is an optional, but highly recommended dependency - Only PDFs are supported if this setup is not completed

    - Windows:
        - Download from the [Official Site](https://www.libreoffice.org/download/download-libreoffice/)

        - Add to PATH, either via:

            - Advanced System Settings -> Environment Variables -> System Variables -> EDIT PATH Variable -> Add the below (change as per your installation location):    
                ```
                C:\Program Files\LibreOffice\program
                ```
            
            - Or via PowerShell:   
                ```
                Set PATH=%PATH%;C:\Program Files\LibreOffice\program
                ```

    - Ubuntu & Debian-based Linux - Download from the [Official Site](https://www.libreoffice.org/download/download-libreoffice/) or install via terminal:

        ```
        sudo apt update
        sudo apt install libreoffice
        ```

    - Fedora and other RPM-based distros - Download from the [Official Site](https://www.libreoffice.org/download/download-libreoffice/) or install via terminal:

        ```
        sudo dnf update
        sudo dnf install libreoffice
        ```

    - MacOS - Download from the [Official Site](https://www.libreoffice.org/download/download-libreoffice/) or install via Homebrew:

        ```
        brew install --cask libreoffice
        ```

    - Verify Installation:
        - On Windows and MacOS: Run the LibreOffice application
        
        - On Linux via the terminal: 
            ```
            libreoffice
            ```

5. Poppler:
    - LARS utilizes the pdf2image Python library to convert each page of a document into an image as required for OCR. This library is essentially a wrapper around the Poppler utility which handles the conversion process.

    - Windows:
    
        - Download from the [Official Repo](https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.02.0-0)

        - Add to PATH, either via:

            - Advanced System Settings -> Environment Variables -> System Variables -> EDIT PATH Variable -> Add the below (change as per your installation location):    
                ```
                path_to_installation\poppler_version\Library\bin
                ```
            
            - Or via PowerShell:   
                ```
                Set PATH=%PATH%;path_to_installation\poppler_version\Library\bin
                ```   

    - Linux:

        ```
        apt-get update
        apt-get install -y poppler-utils wget
        ```

6. PyTesseract:
    - This is an optional dependency - Tesseract-OCR is not actively used in LARS but methods to use it are present in the source code

    - Windows:

        - Download Tesseract-OCR for Windows via [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

        - Add to PATH, either via:

            - Advanced System Settings -> Environment Variables -> System Variables -> EDIT PATH Variable -> Add the below (change as per your installation location):    
                ```
                C:\Program Files\Tesseract-OCR
                ```
            
            - Or via PowerShell:   
                ```
                Set PATH=%PATH%;C:\Program Files\Tesseract-OCR
                ```


## Installing LARS

1. Clone the repository:
    ```
    git clone https://github.com/abgulati/LocalLLM
    cd LocalLLM
    ```

    - If prompted for GitHub authentication, use a [Personal Access Token](https://github.com/settings/tokens) as passwords are deprecated. Also accessible via:      
        ```GitHub Settings -> Developer settings (located on the bottom left!) -> Personal access tokens```

2. Install Python dependencies:
    - Windows via PIP:
        ```
        pip install -r .\requirements.txt
        ```
    
    - Linux via PIP:
        ```
        pip3 install -r ./requirements.txt
        ```

    - Note on Azure: Some required Azure libraries are NOT available on the MacOS platform! A separate requirements file is therefore included for MacOS excluding these libraries:

    - MacOS:
        ```
        pip3 install -r ./requirements_mac.txt
        ```


### Troubleshooting Installation Issues

- If you encounter errors with ```pip install```, try the following:

1. Remove version numbers:

    - If a specific package version causes an error, edit the corresponding requirements.txt file to remove the version constraint, that is the `==version.number` segment, for example:    
    ```urllib3==2.0.4```    
    becomes simply:    
    ```urllib3```

2. Create and use a Python virtual environment:

    - It's advisable to use a virtual environment to avoid conflicts with other Python projects

    - Windows:
    
        - Create a Python virtual environment (venv):
            ```
            python -m venv larsenv
            ```

        - Activate, and subsequently use, the venv:
            ```
            .\larsenv\Scripts\activate
            ```

        - Deactivate venv when done:
            ```
            deactivate
            ```

    - Linux and MacOS:

        - Create a Python virtual environment (venv):
            ```
            python3 -m venv larsenv
            ```

        - Activate, and subsequently use, the venv:
            ```
            source larsenv/bin/activate
            ```

        - Deactivate venv when done:
            ```
            deactivate
            ```

3. If problems persist, consider opening an issue on the [LARS GitHub repository](https://github.com/abgulati/LocalLLM/issues) for support.


## First Run - Important Steps for First-Time Setup

- After installing, run LARS using:
    ```
    cd web_app
    python app.py   # Use 'python3' on Linux/macOS
    ```

- Navigate to ```http://localhost:5000/``` in your browser

- All application directories required by LARS will now be created on disk

- Eventually (after approximately 60 seconds) you'll see an alert on the page indicating an error:
    ```
    Failed to start llama.cpp local-server
    ```

- This indicates that first-run has completed, all app directories have been created, but no LLMs are present in the ```models``` directory and may now be moved to it

- Move your LLMs (any file format supported by llama.cpp, preferably GGUF) to the newly created ```models``` dir, located by default in the following locations:   
    - Windows: ```C:/web_app_storage/models```
    - Linux: ```/app/storage/models```
    - MacOS: ```/app/models```

- Once you've placed your LLMs in the appropriate ```models``` dir above, refresh ```http://localhost:5000/```

- You'll once again receive an error alert stating ```Failed to start llama.cpp local-server``` after approximately 60 seconds

- This is because your LLM now needs to be selected in the LARS ```Settings``` menu

- Accept the alert and click on the ```Settings``` gear icon in the top-right

- In the ```LLM Selection``` tab, select your LLM and the appropriate Prompt-Template format from the appropriate dropdowns

- Modify Advanced Settings to correctly set ```GPU``` options, the ```Context-Length```, and optionally, the token generation limit (```Maximum tokens to predict```) for your selected LLM

- Hit ```Save``` and if an automatic refresh is not triggered, manually refresh the page

- If all steps have been executed correctly, first-time setup is now complete, and LARS is ready for use

- LARS will also remember your LLM settings for subsequent use


## General User Guide - Post First-Run Steps

1. Document Formats Supported:

    - If LibreOffice is installed and added to PATH as detailed in Step 4 of the [Dependencies](https://github.com/abgulati/LocalLLM?tab=readme-ov-file#dependencies) section, the following formats are supported:

        - PDFs
        - Word files: doc, docx, odt, rtf, txt
        - Excel files: xls, xlsx, ods, csv
        - PowerPoint presentations: ppt, pptx, odp
        - Image files: bmp, gif, jpg, png, svg, tiff
        - Rich Text Format (RTF)
        - HTML files

    - If LibreOffice is not setup, only PDFs are supported

2. OCR Options for Text Extraction:

    - LARS provides three methods for extracting text from documents, accommodating various document types and quality:

        - Local Text Extraction: Uses PyPDF2 for efficient text extraction from non-scanned PDFs. Ideal for quick processing when high accuracy is not critical, or entirely local processing is a necessity.

        - Azure ComputerVision OCR - Enhances text extraction accuracy and supports scanned documents. Useful for handling standard document layouts. [Offers a free tier suitable for initial trials and low-volume use, capped at 5000 transactions/month at 20 transactions/minute.](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/computer-vision/)

        - Azure AI Document Intelligence OCR - Best for documents with complex structures like tables. A custom parser in LARS optimizes the extraction process.

        - NOTES: 

            - Azure OCR options incur API-costs in most cases and are not bundled with LARS.
            
            - A limited free-tier for ComputerVision OCR is available as linked above. This service is cheaper overall but slower and may not work for non-standard document layouts (other than A4 etc).

            - Consider the document types and your accuracy needs when selecting an OCR option.

3. LLMs: 

    - Only local-LLMs are presently supported
    
    - The ```Settings``` menu provides many options for the power-user to configure and change the LLM via the ```LLM Selection``` tab

    - Very-Important: Select the appropriate prompt-template format for the LLM you're running

    - LLMs trained for the following prompt-template formats are presently supported:

        - Meta Llama-3
        - Meta Llama-2
        - Mistral & Mixtral MoE LLMs
        - Microsoft Phi-3
        - OpenHermes-2.5-Mistral 
        - Nous-Capybara 
        - OpenChat-3.5
        - Cohere Command-R and Command-R+
        - DeepSeek Coder
    
    - Tweak Core-configuration settings via ```Advanced Settings``` (triggers LLM-reload and page-refresh):
        
        - Number of layers offloaded to the GPU
        - Context-size of the LLM
        - Maximum number of tokens to be generated per response

    - Tweak settings to change response behavior at any time:

        - Temperature – randomness of the response 
        - Top-p – Limit to a subset of tokens with a cumulative probability above <top-p>
        - Min-p – Minimum probability for considering a token, relative to most likely <min_p>
        - Top-k – Limit to K most probable tokens
        - N-keep – Prompt-tokens retained when context-size exceeded <n_keep> (-1 to retain all)

4. Embedding models and Vector Database:

    - Four embedding models are provided in LARS:

        - sentence-transformers/all-mpnet-base-v2 (default)
        - bge-base-en-v1.5
        - bge-large-en-v1.5 (highest [MTEB](https://huggingface.co/spaces/mteb/leaderboard) ranked model available in LARS)
        - Azure-OpenAI Text-Ada (incurs API cost, not bundled with LARS)

    - With the exception of the Azure-OpenAI embeddings, all other models run entirely locally and for free. On first run, these models will be downloaded from the HuggingFace Hub. This is a one-time download and they'll subsequently be present locally.

    - The user may switch between these embedding models at any time via the ```VectorDB & Embedding Models``` tab in the ```Settings``` menu

    - Docs-Loaded Table: In the ```Settings``` menu, a table is displayed for the selected embedding model displaying the list of documents embedded to the associated vector-database. If a document is loaded multiple times, it’ll have multiple entries in this table, which could be useful for debugging any issues.

    - Clearing the VectorDB: Use the ```Reset``` button and provide confirmation to clear the selected vector database. This creates a new vectorDB on-disk for the selected embedding model. The old vectorDB is still preserved and may be reverted to by manually modifying the config.json file.

5. Edit System-Prompt:

    - The System-Prompt serves as an instruction to the LLM for the entire conversation
    
    - LARS provides the user with the ability to edit the System-Prompt via the ```Settings``` menu by selecting the ```Custom``` option from the dropdown in the ```System Prompt``` tab
    
    - Changes to the System-Prompt will start a new chat

6. Force Enable/Disable RAG:

    - Via the ```Settings``` menu, the user may force enable or disable RAG (Retrieval Augmented Generation – the use of content from your documents to improve LLM-generated responses) whenever required 
    
    - This is often useful for the purposes of evaluating LLM responses in both scenarios
    
    - Force disabling will also turn off attribution features
    
    - The default setting, which uses NLP to determine when RAG should and shouldn’t be performed, is the recommended option
    
    - This setting can be changed at any time

7. Chat History: 

    - Use the chat history menu on the top-left to browse and resume prior conversations

    - Very-Important: Be mindful of prompt-template mismatches when resuming prior conversations! Use the ```Information``` icon on the top-right to ensure the LLM used in the prior-conversation, and the LLM presently in use, are both based on the same prompt-template formats!

8. User rating: 

    - Each response may be rated on a 5-point scale by the user at any time
    
    - Ratings data is stored in the ```chat-history.db``` SQLite3 database located in the app directory:
        - Windows: ```C:/web_app_storage```
        - Linux: ```/app/storage```
        - MacOS: ```/app```
    
    - Ratings data is very valuable for evaluation and refinement of the tool for your workflows

9. Dos and Don’ts:

    - Do NOT tweak any settings or submit additional queries while a response to a query is already being generated! Wait for any ongoing response generation to complete.

## Troubleshooting

- If a chat goes awry, or any odd responses are generated, simply try starting a ```New Chat``` via the menu on the top-left

- Alternatively, start a new chat by simply refreshing the page

- If issues are faced with citations or RAG performance, try resetting the vectorDB as described in Step 4 of the [General User Guide](https://github.com/abgulati/LocalLLM?tab=readme-ov-file#general-user-guide---post-first-run-steps) above

- If any application issues crop up and are not resolved simply by starting a new chat or restarting LARS, try deleting the config.json file by following the steps below:

    - Shut-down the LARS app server by terminating the Python program with ```CTRL+C```
    - Backup and delete the ```config.json``` file located in ```LocalLLM/web_app``` (same directory as ```app.py```)

- For any severe data and citation issues that are not resolved even by resetting the VectorDB as described in Step 4 of the [General User Guide](https://github.com/abgulati/LocalLLM?tab=readme-ov-file#general-user-guide---post-first-run-steps) above, perform the following steps:

    - Shut-down the LARS app server by terminating the Python program with ```CTRL+C```
    - Backup and delete the entire app directory:
        - Windows: ```C:/web_app_storage```
        - Linux: ```/app/storage```
        - MacOS: ```/app```

- If problems persist, consider opening an issue on the [LARS GitHub repository](https://github.com/abgulati/LocalLLM/issues) for support.

## Current Development Roadmap

| Category                                      | Tasks In-Progress                                                                                              |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Minor tasks, bug-fixes & enhancements:        | Azure CV-OCR free-tier UI toggle                                                                               |
|                                               | 0B text-file cleanup                                                                                           |
| Major features / detailed research:           | Local OCR via Vision LLMs                                                                                      |
|                                               | "Summarized" RAG via recursive summary generation: [RAPTOR](https://arxiv.org/html/2401.18059v1) investigation |
|                                               | Nvidia TensorRT-LLM inferencing backend for compatible GPUs                                                    |

# Support and Donations
I hope that LARS has been valuable in your work, and I invite you to support its ongoing development! If you appreciate the tool and would like to contribute to its future enhancements, consider making a donation. Your support helps me to continue improving LARS and adding new features.

How to Donate
To make a donation, please use the following link to my PayPal:

[Donate via PayPal](https://www.paypal.com/donate/?business=35EP992TTK5J6&no_recurring=0&item_name=If+you+appreciate+my+work+and+would+like+to+contribute+to+its+ongoing+development+%26+enhancements%2C+consider+making+a+donation%21&currency_code=CAD)

Your contributions are greatly appreciated and will be used to fund further development efforts.