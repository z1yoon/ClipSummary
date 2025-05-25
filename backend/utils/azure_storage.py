import os
import asyncio
import tempfile
from typing import Optional, BinaryIO
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient
from fastapi import UploadFile
import logging

logger = logging.getLogger(__name__)

class AzureBlobStorage:
    """Simple and clean Azure Blob Storage interface with signed URL support"""
    
    def __init__(self):
        self.connection_string = os.getenv("BLOB_CONNECTION_STRING")
        self.container_name = os.getenv("BLOB_CONTAINER_NAME", "uploads")
        
        if not self.connection_string:
            raise ValueError("BLOB_CONNECTION_STRING not found in environment variables")
        
        # Initialize sync client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        # Extract account name and key for SAS generation
        conn_parts = dict(item.split('=', 1) for item in self.connection_string.split(';') if '=' in item)
        self.account_name = conn_parts.get('AccountName')
        self.account_key = conn_parts.get('AccountKey')
        
        if not self.account_name or not self.account_key:
            raise ValueError("Could not extract account name and key from connection string")
        
        # Create container if it doesn't exist
        try:
            self.blob_service_client.create_container(self.container_name)
            logger.info(f"Created container: {self.container_name}")
        except Exception:
            logger.info(f"Container {self.container_name} already exists")
    
    def generate_upload_url(self, upload_id: str, filename: str, expiry_hours: int = 1) -> dict:
        """Generate a signed URL for direct upload to Azure Blob Storage"""
        blob_name = f"{upload_id}/{filename}"
        
        try:
            # Calculate expiry time
            expiry_time = datetime.utcnow() + timedelta(hours=expiry_hours)
            
            # Generate SAS token with upload permissions
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=BlobSasPermissions(write=True, create=True),
                expiry=expiry_time
            )
            
            # Construct the full upload URL
            upload_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            
            logger.info(f"Generated signed upload URL for {blob_name}, expires at {expiry_time}")
            
            return {
                "upload_url": upload_url,
                "blob_name": blob_name,
                "expires_at": expiry_time.isoformat(),
                "upload_method": "PUT",
                "headers": {
                    "x-ms-blob-type": "BlockBlob",
                    "Content-Type": "application/octet-stream"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL for {blob_name}: {str(e)}")
            raise
    
    def verify_blob_exists(self, upload_id: str, filename: str) -> bool:
        """Verify that a blob was successfully uploaded"""
        blob_name = f"{upload_id}/{filename}"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            # Check if blob exists and get its properties
            properties = blob_client.get_blob_properties()
            
            # Verify it has content
            if properties.size > 0:
                logger.info(f"Verified blob {blob_name} exists with size {properties.size} bytes")
                return True
            else:
                logger.warning(f"Blob {blob_name} exists but is empty")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify blob {blob_name}: {str(e)}")
            return False
    
    def get_blob_size(self, upload_id: str, filename: str) -> Optional[int]:
        """Get the size of an uploaded blob"""
        blob_name = f"{upload_id}/{filename}"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            properties = blob_client.get_blob_properties()
            return properties.size
            
        except Exception as e:
            logger.error(f"Failed to get blob size for {blob_name}: {str(e)}")
            return None

    async def download_video(self, upload_id: str, filename: str, local_path: str) -> str:
        """Download video from Azure Blob Storage to local path for processing"""
        blob_name = f"{upload_id}/{filename}"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            logger.info(f"Downloading blob {blob_name} to {local_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download blob to local file using streaming for large files
            with open(local_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                
                # Stream download in chunks to handle large files efficiently
                chunk_size = 64 * 1024 * 1024  # 64MB chunks
                for chunk in download_stream.chunks():
                    download_file.write(chunk)
            
            file_size = os.path.getsize(local_path)
            logger.info(f"Successfully downloaded {blob_name} ({file_size} bytes) to {local_path}")
            
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {str(e)}")
            raise
    
    def get_blob_url(self, upload_id: str, filename: str) -> str:
        """Get the URL for a blob (for direct streaming if needed)"""
        blob_name = f"{upload_id}/{filename}"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, 
            blob=blob_name
        )
        return blob_client.url
    
    def delete_blob(self, upload_id: str, filename: str) -> bool:
        """Delete a blob from storage"""
        blob_name = f"{upload_id}/{filename}"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            blob_client.delete_blob()
            logger.info(f"Deleted blob: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete blob {blob_name}: {str(e)}")
            return False

# Global instance
azure_storage = AzureBlobStorage()