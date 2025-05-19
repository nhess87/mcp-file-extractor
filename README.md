## Getting Started

### Tool Overview

This tool is designed to extract content from files such as PDFs, Word documents, images, and plain text files. It supports intelligent OCR extraction and multimodal interpretation using the following services:

Azure Document Intelligence – for high-accuracy OCR and form recognition

Gemini Multimodal (Google) – for advanced image and document understanding

OpenAI Multimodal (GPT-4V) – for interpreting and extracting structured and unstructured information from visual and textual inputs

Whether you're processing scanned documents, structured forms, or images with embedded text, this tool provides a unified interface for extracting readable and usable content using cutting-edge AI models.


### Follow the steps below to set up and run the MCP SSE server.

### 1. Navigate to the Project Directory

```bash
cd mcp-file-extractor
```
### 2. Create and Activate Virtual Environment
```
uv venv 

.venv\Scripts\activate

```

### 3. Install Dependencies
uv pip install -r requirements.txt

### 4. Configure Environment Variables
Create a copy of .env_template file and name it ".env" in the same directory and modify/insert the necessary values such as Database string and etc.

### 5.5. Run the Server
Go to the root directory : cd packages
then run:
python -m parser_tools.src.server_ssep
