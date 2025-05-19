import datetime
import os
import time
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

start_time= time.time()
# checking all the required environment variables
async def check_environment_variables() -> dict:
    required_env_vars = [
        "DATABASE_URL", "SERVER_HOST", "SERVER_PORT", 
        "VECTOR_DATABASE_URL", "OPENAI_API_KEY"
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    result= {
        "status": "ok" if not missing_vars else "missing",
        "missing": None if not missing_vars else missing_vars
    }
    logger.info(f"Environment variable check: {result}")
    return result


async def check_database(db_list: list[str]) -> dict[str, dict[str, str]]:
    from urllib.parse import urlparse
    import asyncpg
    logger.info("here's the db_list: ", db_list)
    results = {}
    for db_url in db_list:
        try:
            # Parse DB connection URL
            parsed_url = urlparse(db_url)
            db_name = parsed_url.path.lstrip("/") or "unknown"
            logger.info(f"Checking database: {db_name}")

            DB_CONFIG = {
                "user": parsed_url.username,
                "password": parsed_url.password,
                "database": db_name,
                "host": parsed_url.hostname,
                "port": parsed_url.port,
            }

            # Attempt to connect
            conn = await asyncpg.connect(**DB_CONFIG)
            await conn.close()

            results[db_name] = {"status": "ok"}
            logger.info(f"Database {db_name} check: ok")

        except Exception as e:
            error_msg = f"{e}"
            logger.error(f"Database {db_name} check failed: {error_msg}")
            results[db_name] = {"status": "error", "error": error_msg}

    return results

async def check_external_apis(api_list: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    import httpx
    results = {}

    async with httpx.AsyncClient() as client:
        for api in api_list:
            api_url = api.get("url")
            headers = api.get("headers", {})
            
            if api.get("name")=="API_URL_PRIVATE_BRAVE":  
                try:
                    ###for calling this url, params are required
                    params = { 
                    "q": "test",
                    "count": 1
                    }
                    logger.info(f"Checking the header: {headers} for BRAVE API")
                    res = await client.get(api_url, params=params, headers=headers, timeout=10.0)
                    results[api_url] = {"status": "ok"}
                    logger.info(f"External API check for {api_url}: Status OK")

                except httpx.HTTPError as e:
                    error_msg = f"HTTP error checking {api_url}: {e}"
                    results[api_url] = {"status": "error", "error": error_msg}
                    logger.warning(error_msg)
                except Exception as e:
                    error_msg = f"Error checking {api_url}: {e}"
                    results[api_url] = {"status": "error", "error": error_msg}
                    logger.error(error_msg, exc_info=True)
            ###########
            ###create /define your own external api logic call here with new elif statement
            ####For each API call you implement (within each elif block), make sure to initialize the results dictionary like this results[api_url] = {"status": "ok"}
            #############
            else:
                try:
                    res = await client.get(api_url, headers=headers, timeout=10.0)
                    res.raise_for_status()
                    results[api_url] = {"status": "ok"}
                    logger.info(f"External API check for {api_url}: Status OK")
                except httpx.HTTPError as e:
                    error_msg = f"HTTP error checking {api_url}: {e}"
                    results[api_url] = {"status": "error", "error": error_msg}
                    logger.warning(error_msg)
                except Exception as e:
                    error_msg = f"Error checking {api_url}: {e}"
                    results[api_url] = {"status": "error", "error": error_msg}
                    logger.error(error_msg, exc_info=True)

    logger.info(f"External API checks: {results}")
    return results

def get_uptime() -> str:
    delta = time.time() - start_time
    return str(datetime.timedelta(seconds=int(delta)))