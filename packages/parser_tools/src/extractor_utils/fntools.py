from io import BytesIO
import logging
from azure.storage.blob import BlobServiceClient


logger = logging.getLogger("tasks")

def makeStorageConnString(STORAGE_ACCOUNT_NAME, STORAGE_ACCOUNT_KEY):
    return f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"



def get_latest_blob(container_name: str, storage_conn_string: str) -> str:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(storage_conn_string)
        container_client = blob_service_client.get_container_client(container_name)

        blobs = list(container_client.list_blobs())

        if not blobs:
            logger.info("No files found in the container.")
            return None, None

        latest_blob = max(blobs, key=lambda b: b.last_modified)
        blob_name = latest_blob.name
        logger.info(f"Latest blob in container {container_name}: {blob_name}")

        # Download the actual blob content
        blob_client = container_client.get_blob_client(blob_name)
        stream = blob_client.download_blob()
        content = stream.readall()
        uploaded_file = BytesIO(content)

        return blob_name, uploaded_file
    except Exception as e:
        logger.info(f"Error fetching latest blob: {e}")
        return None, None