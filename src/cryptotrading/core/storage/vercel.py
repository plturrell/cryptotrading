"""
Vercel storage implementations (Blob and KV) conforming to unified interface
"""

import os
import json
import base64
import asyncio
from typing import Dict, List, Optional, Union
import aiohttp
import requests

from .base import StorageInterface, SyncStorageInterface


class VercelBlobStorage(StorageInterface):
    """Async Vercel Blob storage implementation"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('BLOB_READ_WRITE_TOKEN')
        if not self.token:
            raise ValueError("Vercel Blob token required. Set BLOB_READ_WRITE_TOKEN env var.")
        
        self.base_url = "https://blob.vercel-storage.com"
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    async def read(self, key: str) -> Optional[bytes]:
        """Read data from Vercel Blob"""
        async with aiohttp.ClientSession() as session:
            # First, get the blob URL
            list_url = f"{self.base_url}/list"
            params = {'prefix': key, 'limit': 1}
            
            async with session.get(list_url, headers=self.headers, params=params) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                blobs = data.get('blobs', [])
                
                if not blobs or blobs[0]['pathname'] != key:
                    return None
                
                # Download the blob
                blob_url = blobs[0]['url']
                async with session.get(blob_url) as blob_resp:
                    if blob_resp.status == 200:
                        return await blob_resp.read()
        
        return None
    
    async def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to Vercel Blob"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Vercel Blob expects base64 encoded data
        encoded_data = base64.b64encode(data).decode('utf-8')
        
        async with aiohttp.ClientSession() as session:
            put_url = f"{self.base_url}/upload"
            
            payload = {
                'pathname': key,
                'data': encoded_data,
                'access': 'public'
            }
            
            async with session.post(put_url, headers=self.headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise Exception(f"Failed to write to Vercel Blob: {text}")
    
    async def delete(self, key: str) -> None:
        """Delete data from Vercel Blob"""
        async with aiohttp.ClientSession() as session:
            delete_url = f"{self.base_url}/delete"
            payload = {'urls': [key]}
            
            async with session.post(delete_url, headers=self.headers, json=payload) as resp:
                if resp.status not in (200, 204):
                    text = await resp.text()
                    raise Exception(f"Failed to delete from Vercel Blob: {text}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Vercel Blob"""
        data = await self.read(key)
        return data is not None
    
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys with optional prefix filter"""
        async with aiohttp.ClientSession() as session:
            list_url = f"{self.base_url}/list"
            params = {'limit': 1000}
            if prefix:
                params['prefix'] = prefix
            
            async with session.get(list_url, headers=self.headers, params=params) as resp:
                if resp.status != 200:
                    return []
                
                data = await resp.json()
                blobs = data.get('blobs', [])
                return [blob['pathname'] for blob in blobs]


class VercelBlobStorageSync(SyncStorageInterface):
    """Synchronous Vercel Blob storage implementation"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('BLOB_READ_WRITE_TOKEN')
        if not self.token:
            raise ValueError("Vercel Blob token required. Set BLOB_READ_WRITE_TOKEN env var.")
        
        self.base_url = "https://blob.vercel-storage.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        })
    
    def read(self, key: str) -> Optional[bytes]:
        """Read data from Vercel Blob"""
        # First, get the blob URL
        list_url = f"{self.base_url}/list"
        params = {'prefix': key, 'limit': 1}
        
        resp = self.session.get(list_url, params=params)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        blobs = data.get('blobs', [])
        
        if not blobs or blobs[0]['pathname'] != key:
            return None
        
        # Download the blob
        blob_url = blobs[0]['url']
        blob_resp = requests.get(blob_url)
        
        if blob_resp.status_code == 200:
            return blob_resp.content
        
        return None
    
    def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to Vercel Blob"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Vercel Blob expects base64 encoded data
        encoded_data = base64.b64encode(data).decode('utf-8')
        
        put_url = f"{self.base_url}/upload"
        payload = {
            'pathname': key,
            'data': encoded_data,
            'access': 'public'
        }
        
        resp = self.session.post(put_url, json=payload)
        if resp.status_code not in (200, 201):
            raise Exception(f"Failed to write to Vercel Blob: {resp.text}")
    
    def delete(self, key: str) -> None:
        """Delete data from Vercel Blob"""
        delete_url = f"{self.base_url}/delete"
        payload = {'urls': [key]}
        
        resp = self.session.post(delete_url, json=payload)
        if resp.status_code not in (200, 204):
            raise Exception(f"Failed to delete from Vercel Blob: {resp.text}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Vercel Blob"""
        data = self.read(key)
        return data is not None
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys with optional prefix filter"""
        list_url = f"{self.base_url}/list"
        params = {'limit': 1000}
        if prefix:
            params['prefix'] = prefix
        
        resp = self.session.get(list_url, params=params)
        if resp.status_code != 200:
            return []
        
        data = resp.json()
        blobs = data.get('blobs', [])
        return [blob['pathname'] for blob in blobs]


class VercelKVStorage(StorageInterface):
    """Async Vercel KV (Redis) storage implementation"""
    
    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self.url = url or os.getenv('KV_REST_API_URL')
        self.token = token or os.getenv('KV_REST_API_TOKEN')
        
        if not self.url or not self.token:
            raise ValueError("Vercel KV credentials required. Set KV_REST_API_URL and KV_REST_API_TOKEN.")
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    async def read(self, key: str) -> Optional[bytes]:
        """Read data from Vercel KV"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/get/{key}"
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get('result')
                    if result:
                        # KV stores strings, convert back to bytes
                        if isinstance(result, str):
                            return result.encode('utf-8')
                        return result
        return None
    
    async def write(self, key: str, data: Union[bytes, str]) -> None:
        """Write data to Vercel KV"""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/set/{key}"
            payload = {'value': data}
            
            async with session.post(url, headers=self.headers, json=payload) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    raise Exception(f"Failed to write to Vercel KV: {text}")
    
    async def delete(self, key: str) -> None:
        """Delete data from Vercel KV"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/del/{key}"
            async with session.post(url, headers=self.headers) as resp:
                if resp.status not in (200, 204):
                    text = await resp.text()
                    raise Exception(f"Failed to delete from Vercel KV: {text}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Vercel KV"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/exists/{key}"
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return bool(data.get('result', 0))
        return False
    
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys with optional prefix filter"""
        pattern = f"{prefix}*" if prefix else "*"
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.url}/keys/{pattern}"
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('result', [])
        return []