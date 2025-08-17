"""
Test unified storage abstraction with different backends
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from cryptotrading.core.storage import (
    StorageFactory,
    get_storage,
    get_sync_storage,
    LocalFileStorage,
    LocalFileStorageSync
)


class TestStorageFactory:
    """Test storage factory environment detection"""
    
    def test_local_storage_default(self, monkeypatch):
        """Test that local storage is default when not on Vercel"""
        monkeypatch.delenv('VERCEL', raising=False)
        monkeypatch.delenv('BLOB_READ_WRITE_TOKEN', raising=False)
        monkeypatch.delenv('KV_REST_API_URL', raising=False)
        
        storage = StorageFactory.get_sync_storage()
        assert isinstance(storage, LocalFileStorageSync)
    
    def test_vercel_detection(self, monkeypatch):
        """Test Vercel environment detection"""
        monkeypatch.setenv('VERCEL', '1')
        monkeypatch.setenv('BLOB_READ_WRITE_TOKEN', 'test-token')
        
        # Should attempt to create Vercel storage but fall back to local if token invalid
        storage = StorageFactory.get_sync_storage()
        # Note: Will be LocalFileStorageSync because token is fake
        assert storage is not None
    
    def test_explicit_storage_type(self):
        """Test explicit storage type selection"""
        storage = StorageFactory.get_sync_storage('local')
        assert isinstance(storage, LocalFileStorageSync)
    
    def test_async_storage(self):
        """Test async storage creation"""
        storage = StorageFactory.get_async_storage('local')
        assert isinstance(storage, LocalFileStorage)


class TestLocalStorage:
    """Test local file storage implementation"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sync_storage(self, temp_dir):
        """Create sync storage instance"""
        return LocalFileStorageSync(base_path=temp_dir)
    
    @pytest.fixture
    def async_storage(self, temp_dir):
        """Create async storage instance"""
        return LocalFileStorage(base_path=temp_dir)
    
    def test_sync_write_read(self, sync_storage):
        """Test synchronous write and read"""
        # Write text
        sync_storage.write_text('test.txt', 'Hello, World!')
        assert sync_storage.read_text('test.txt') == 'Hello, World!'
        
        # Write JSON
        data = {'key': 'value', 'number': 42}
        sync_storage.write_json('data.json', data)
        assert sync_storage.read_json('data.json') == data
        
        # Write bytes
        sync_storage.write('binary.bin', b'\x00\x01\x02\x03')
        assert sync_storage.read('binary.bin') == b'\x00\x01\x02\x03'
    
    def test_sync_exists_delete(self, sync_storage):
        """Test existence check and deletion"""
        sync_storage.write_text('exists.txt', 'content')
        
        assert sync_storage.exists('exists.txt')
        assert not sync_storage.exists('not-exists.txt')
        
        sync_storage.delete('exists.txt')
        assert not sync_storage.exists('exists.txt')
    
    def test_sync_list_keys(self, sync_storage):
        """Test listing keys"""
        # Create files in different directories
        sync_storage.write_text('file1.txt', 'content1')
        sync_storage.write_text('dir/file2.txt', 'content2')
        sync_storage.write_text('dir/subdir/file3.txt', 'content3')
        sync_storage.write_text('other/file4.txt', 'content4')
        
        # List all keys
        all_keys = sync_storage.list_keys()
        assert len(all_keys) == 4
        assert 'file1.txt' in all_keys
        assert 'dir/file2.txt' in all_keys
        assert 'dir/subdir/file3.txt' in all_keys
        assert 'other/file4.txt' in all_keys
        
        # List with prefix
        dir_keys = sync_storage.list_keys('dir/')
        assert len(dir_keys) == 2
        assert 'dir/file2.txt' in dir_keys
        assert 'dir/subdir/file3.txt' in dir_keys
    
    @pytest.mark.asyncio
    async def test_async_write_read(self, async_storage):
        """Test asynchronous write and read"""
        # Write text
        await async_storage.write_text('test.txt', 'Hello, Async!')
        assert await async_storage.read_text('test.txt') == 'Hello, Async!'
        
        # Write JSON
        data = {'async': True, 'value': 123}
        await async_storage.write_json('async_data.json', data)
        assert await async_storage.read_json('async_data.json') == data
    
    @pytest.mark.asyncio
    async def test_async_concurrent_operations(self, async_storage):
        """Test concurrent async operations"""
        # Write multiple files concurrently
        tasks = []
        for i in range(10):
            task = async_storage.write_text(f'concurrent_{i}.txt', f'Content {i}')
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Read them back concurrently
        read_tasks = []
        for i in range(10):
            task = async_storage.read_text(f'concurrent_{i}.txt')
            read_tasks.append(task)
        
        results = await asyncio.gather(*read_tasks)
        
        for i, content in enumerate(results):
            assert content == f'Content {i}'
    
    def test_path_safety(self, sync_storage):
        """Test that paths are sanitized"""
        # Try to escape base directory
        sync_storage.write_text('../escape.txt', 'content')
        
        # File should be created inside base directory
        base_path = Path(sync_storage.base_path)
        assert not (base_path.parent / 'escape.txt').exists()
        assert sync_storage.exists('escape.txt')  # Stripped the ../
    
    def test_nested_directories(self, sync_storage):
        """Test creating nested directory structures"""
        sync_storage.write_text('a/b/c/d/deep.txt', 'Deep content')
        assert sync_storage.read_text('a/b/c/d/deep.txt') == 'Deep content'
        
        # Check that intermediate directories were created
        base_path = Path(sync_storage.base_path)
        assert (base_path / 'a' / 'b' / 'c' / 'd').is_dir()


class TestStorageInterface:
    """Test storage interface contracts"""
    
    def test_sync_interface_contract(self):
        """Test that sync storage implements all required methods"""
        storage = LocalFileStorageSync()
        
        # Check all required methods exist
        assert hasattr(storage, 'read')
        assert hasattr(storage, 'write')
        assert hasattr(storage, 'delete')
        assert hasattr(storage, 'exists')
        assert hasattr(storage, 'list_keys')
        assert hasattr(storage, 'read_json')
        assert hasattr(storage, 'write_json')
        assert hasattr(storage, 'read_text')
        assert hasattr(storage, 'write_text')
    
    @pytest.mark.asyncio
    async def test_async_interface_contract(self):
        """Test that async storage implements all required methods"""
        storage = LocalFileStorage()
        
        # Check all required methods exist and are coroutines
        assert asyncio.iscoroutinefunction(storage.read)
        assert asyncio.iscoroutinefunction(storage.write)
        assert asyncio.iscoroutinefunction(storage.delete)
        assert asyncio.iscoroutinefunction(storage.exists)
        assert asyncio.iscoroutinefunction(storage.list_keys)
        assert asyncio.iscoroutinefunction(storage.read_json)
        assert asyncio.iscoroutinefunction(storage.write_json)
        assert asyncio.iscoroutinefunction(storage.read_text)
        assert asyncio.iscoroutinefunction(storage.write_text)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])