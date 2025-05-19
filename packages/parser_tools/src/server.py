from typing import Any, Optional, Annotated, Union
import httpx
from mcp.server.fastmcp import FastMCP
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv 
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.applications import Starlette
from mcp import mcp_tools, _mcp_server as mcp_server
from mcp.transport.starlette import SseServerTransport
from mcp.server import Server
import uvicorn
import os
import argparse
import datetime
from parser_tools.tests.healthcheck import check_database,check_external_apis, check_environment_variables, check_external_api, get_uptime


def setup_logger(name:str) -> logging.Logger:
    #Ensure the logs directory exists
    log_dir= Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure file handler for logging
    log_file= log_dir / "server.log"
    
    #Get the logger and configure handlers
    logger= logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        fh= logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s -%(levelname)s -%(message)s"))
        logger.addHandler(fh)
        
    return logger

logger = setup_logger(__name__)


base_path= Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(base_path))

dotenv_path= Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path= dotenv_path)


# Initialize FastMCP server
mcp= FastMCP("file_tool")

mcp_tools= []

async def health_check_endpoint(request:Optional[Request] =None) -> JSONResponse:
    external_apis_to_check = []

    # GET public APIs or Apis without Authentication (if needed)
    for key, value in os.environ.items():
        if key.startswith("API_URL_PUBLIC_") and value:
            print(f"this is key: {key}")
            external_apis_to_check.append({"name": key, "url": value, "headers": {}})

    # GET APIs with Authentication (if needed)
    for key, value in os.environ.items():
        if key.startswith("API_URL_PRIVATE_") and value:
            name = key.replace("API_URL_PRIVATE_", "")

            headers = {
            "Accept": "application/json"
            }
            api_key = os.getenv(f"API_KEY_{name}")
            logger.info(f"this is api key: {api_key}")
            if name=="BRAVE":
                headers["X-Subscription-Token"]= api_key 
            else:
                headers["Authorization"] = f"Bearer {api_key}"

            # Custom Headers (if required by the API)
            # headers["Custom-Header-Name"] = "Custom-Header-Value" 

            external_apis_to_check.append({
                "name": key,
                "url": value,
                "headers": headers
            })
            logger.info(f"this is external apis to check: {external_apis_to_check}")

    # Databases
    database_list = [
        os.getenv("DATABASE_URL"),
        os.getenv("VECTOR_DATABASE_URL")
    ]

    env_check = await check_environment_variables()
    db_check = await check_database(database_list)
    api_check = await check_external_apis(external_apis_to_check)

    db_status_ok = all(db.get("status") == "ok" for db in db_check.values())
    api_status_ok = all(api.get("status") == "ok" for api in api_check.values())
    env_status_ok = env_check.get("status") == "ok" and env_check.get("missing") in (None, [], {})

    overall_status = "ok" if all([db_status_ok, api_status_ok, env_status_ok]) else "unhealthy"
    logger.info(f"Overall health check status: {overall_status}")

    result = {
        "status": overall_status,
        "timestamp": datetime.datetime.now().isoformat(),
        "uptime": get_uptime(),
        "checks": {
            "database": db_check,
            "external_api": api_check,
            "env_vars": env_check,
        }
    }

    status_code = 200 if overall_status == "ok" else 500
    if overall_status != "ok":
        logger.warning("Health check failed", extra=result)

    logger.debug(f"Full health check result: {result}")
    return JSONResponse(result, status_code=status_code)

async def preflight_health_check():
    response = await health_check_endpoint(None)
    if response.status_code != 200:
        logger.error("Preflight health check failed. Exiting.")
    exit(1)

def tool_wrapper(*args, **kwargs):
    def decorator(func):
        mcp_tools.append({
        "name": kwargs.get("name", func.**name**),
        "description": kwargs.get("description", func.**doc**),
        "signature": str(func.**annotations**),
        })
        return mcp.tool(**kwargs)(func)
    if args and callable(args[0]):
    # Used as @tool_wrapper with no arguments
        return decorator(args[0])
    else:
        # Used as @tool_wrapper(name=..., description=...)
        return decorator
    
#---------------------------Text Extractor Tool---------------------------------------
@tool_wrapper(
    name="extract_text_from_file",
    description="Extract text from a user file stored in an Azure Blob Storage container with name based on task_id.",
)
async def extract_text_from_file(
    task_id: Annotated[]
) -> str:
    from parser_tools.src.extractor_utils.fntools import makeStorageConnString, get_latest_blob
    from parser_tools.src.extractor_utils.parsers import extract_text_function
    
    try:
        logger.info(f"task_id is {task_id}")
        postgres_host= os.getenv("DB_HOST")
        postgres_database= os.getenv("DB_NAME")
        postgres_user= os.getenv("DB_USER")
        postgres_password= os.getenv("DB_PASSWORD")
        postgres_port= os.getenv("DB_PORT")
        
        
        db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}"
        
        STORAGE_ACCOUNT_NAME = os.getenv("STORAGE_ACCOUNT_NAME")
        STORAGE_ACCOUNT_KEY = os.getenv("STORAGE_ACCOUNT_KEY")

        storage_conn_string = makeStorageConnString(STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY)
        text_extractor= text_extractor
        db_url= db_url
        container_name = f"conv-task-{task_id}"
        #table_name = f"task_{task_id}"

        latest_blob_name, uploaded_file = get_latest_blob(container_name, storage_conn_string)
        logger.info(f"Latest blob in container {container_name}: {latest_blob_name}")

        extracted_text = extract_text_function(
            extractorOCR=text_extractor,
            StorageConnString=storage_conn_string,
            blob_name=latest_blob_name, #Process only the latest file
            uploaded_file=uploaded_file,
        )
        return extracted_text

    except Exception as e:
        logger.error(f"Error in the process: {e}")
        return ""


async def list_tools_endpoint(request: Request):
    return JSONResponse({"tools": mcp_tools})


def create_starlette_app(mcp_server, *, debug: bool = False) -> Starlette:
    
    ##Create a Starlette application to serve the provided MCP server with SSE.
    
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
            Route("/tools/list", endpoint=list_tools_endpoint),
            Route("/health", endpoint=health_check_endpoint),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default=os.getenv('SERVER_HOST', '127.0.0.1'), help='Host to bind to')
    parser.add_argument('--port', type=int, default=int(os.getenv('SERVER_PORT', 8080)), help='Port to listen on')
    args = parser.parse_args()

    logger.info(f"Starting server with arguments: {args}")

    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=args.host, port=args.port, log_level="info", access_log=True)
