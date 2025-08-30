"""
AWS Data Exchange Service
Provides access to premium financial and economic datasets through AWS Data Exchange
"""

import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataExchangeDataset:
    """AWS Data Exchange dataset information"""

    dataset_id: str
    name: str
    description: str
    provider: str
    categories: List[str]
    last_updated: datetime
    asset_count: int


@dataclass
class DataExchangeAsset:
    """AWS Data Exchange asset information"""

    asset_id: str
    name: str
    dataset_id: str
    file_format: str
    size_bytes: int
    created_at: datetime


class AWSDataExchangeService:
    """AWS Data Exchange client for financial data"""

    def __init__(self, region_name: str = "us-east-1"):
        """Initialize AWS Data Exchange service"""
        try:
            # Initialize AWS clients
            self.dataexchange = boto3.client("dataexchange", region_name=region_name)
            self.s3 = boto3.client("s3", region_name=region_name)

            # Default bucket for temporary data processing
            self.temp_bucket = os.getenv(
                "AWS_DATA_EXCHANGE_BUCKET", "cryptotrading-dataexchange-temp"
            )

            logger.info("AWS Data Exchange service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize AWS Data Exchange service: {e}")
            raise

    def discover_financial_datasets(self) -> List[DataExchangeDataset]:
        """Discover available financial datasets"""
        try:
            datasets = []
            paginator = self.dataexchange.get_paginator("list_data_sets")

            # Filter for financial/economic datasets
            financial_keywords = [
                "financial",
                "market",
                "economic",
                "trading",
                "price",
                "crypto",
                "stock",
                "forex",
                "commodity",
                "derivatives",
            ]

            for page in paginator.paginate():
                for dataset in page["DataSets"]:
                    # Check if dataset is financial-related
                    name = dataset.get("Name", "").lower()
                    description = dataset.get("Description", "").lower()

                    is_financial = any(
                        keyword in name or keyword in description for keyword in financial_keywords
                    )

                    if is_financial:
                        datasets.append(
                            DataExchangeDataset(
                                dataset_id=dataset["Id"],
                                name=dataset["Name"],
                                description=dataset.get("Description", ""),
                                provider=dataset["Origin"],
                                categories=[],  # Will be populated from tags if available
                                last_updated=datetime.fromisoformat(
                                    dataset["UpdatedAt"].replace("Z", "+00:00")
                                ),
                                asset_count=0,  # Will be populated separately
                            )
                        )

            logger.info(f"Found {len(datasets)} financial datasets")
            return datasets

        except Exception as e:
            logger.error(f"Error discovering datasets: {e}")
            return []

    def get_dataset_assets(self, dataset_id: str) -> List[DataExchangeAsset]:
        """Get assets (files) within a dataset"""
        try:
            assets = []
            paginator = self.dataexchange.get_paginator("list_data_set_revisions")

            # Get latest revision
            revisions = list(paginator.paginate(DataSetId=dataset_id))
            if not revisions or not revisions[0]["Revisions"]:
                return assets

            latest_revision = revisions[0]["Revisions"][0]
            revision_id = latest_revision["Id"]

            # Get assets in the latest revision
            asset_paginator = self.dataexchange.get_paginator("list_revision_assets")
            for page in asset_paginator.paginate(DataSetId=dataset_id, RevisionId=revision_id):
                for asset in page["Assets"]:
                    assets.append(
                        DataExchangeAsset(
                            asset_id=asset["Id"],
                            name=asset["Name"],
                            dataset_id=dataset_id,
                            file_format=asset.get("AssetType", "unknown"),
                            size_bytes=asset.get("AssetDetails", {})
                            .get("S3SnapshotAsset", {})
                            .get("Size", 0),
                            created_at=datetime.fromisoformat(
                                asset["CreatedAt"].replace("Z", "+00:00")
                            ),
                        )
                    )

            return assets

        except Exception as e:
            logger.error(f"Error getting dataset assets: {e}")
            return []

    def create_data_job(self, asset_id: str, dataset_id: str, revision_id: str) -> str:
        """Create a job to export data to S3"""
        try:
            # Create export job
            response = self.dataexchange.create_job(
                Type="EXPORT_ASSETS_TO_S3",
                Details={
                    "ExportAssetsToS3": {
                        "AssetDestinations": [
                            {
                                "AssetId": asset_id,
                                "Bucket": self.temp_bucket,
                                "Key": f"data-exchange/{dataset_id}/{asset_id}/{datetime.now().isoformat()}",
                            }
                        ],
                        "DataSetId": dataset_id,
                        "RevisionId": revision_id,
                    }
                },
            )

            job_id = response["Id"]
            logger.info(f"Created data export job: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Error creating data job: {e}")
            raise

    def start_data_job(self, job_id: str) -> bool:
        """Start the data export job"""
        try:
            self.dataexchange.start_job(JobId=job_id)
            logger.info(f"Started data job: {job_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting data job: {e}")
            return False

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of data export job"""
        try:
            response = self.dataexchange.get_job(JobId=job_id)
            return {
                "job_id": job_id,
                "state": response["State"],
                "type": response["Type"],
                "created_at": response["CreatedAt"],
                "updated_at": response.get("UpdatedAt"),
                "errors": response.get("Errors", []),
            }

        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {"job_id": job_id, "state": "ERROR", "errors": [str(e)]}

    def wait_for_job_completion(self, job_id: str, timeout_minutes: int = 30) -> bool:
        """Wait for job to complete"""
        import time

        timeout = datetime.now() + timedelta(minutes=timeout_minutes)

        while datetime.now() < timeout:
            status = self.get_job_status(job_id)
            state = status.get("state")

            if state == "COMPLETED":
                logger.info(f"Job {job_id} completed successfully")
                return True
            elif state in ["ERROR", "CANCELLED"]:
                logger.error(f"Job {job_id} failed with state: {state}")
                return False
            elif state in ["IN_PROGRESS", "WAITING"]:
                logger.info(f"Job {job_id} still running, state: {state}")
                time.sleep(30)  # Check every 30 seconds
            else:
                logger.warning(f"Unknown job state: {state}")
                time.sleep(30)

        logger.error(f"Job {job_id} timed out after {timeout_minutes} minutes")
        return False

    def download_and_process_data(self, dataset_id: str, asset_id: str) -> pd.DataFrame:
        """Download and process data from completed export job"""
        try:
            # Get the latest revision
            revisions = self.dataexchange.list_data_set_revisions(DataSetId=dataset_id)
            if not revisions["Revisions"]:
                raise ValueError("No revisions found for dataset")

            revision_id = revisions["Revisions"][0]["Id"]

            # Create and start export job
            job_id = self.create_data_job(asset_id, dataset_id, revision_id)
            if not self.start_data_job(job_id):
                raise RuntimeError("Failed to start data job")

            # Wait for completion
            if not self.wait_for_job_completion(job_id):
                raise RuntimeError("Data job failed or timed out")

            # Download from S3
            s3_key = f"data-exchange/{dataset_id}/{asset_id}/{datetime.now().isoformat()}"

            with tempfile.NamedTemporaryFile() as temp_file:
                self.s3.download_file(self.temp_bucket, s3_key, temp_file.name)

                # Try to read as CSV first, then other formats
                try:
                    df = pd.read_csv(temp_file.name)
                except:
                    try:
                        df = pd.read_json(temp_file.name)
                    except:
                        try:
                            df = pd.read_excel(temp_file.name)
                        except:
                            logger.error("Unable to parse downloaded data")
                            raise

            logger.info(f"Successfully downloaded and processed data: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error downloading/processing data: {e}")
            raise

    def get_available_crypto_datasets(self) -> List[Dict[str, Any]]:
        """Get specifically crypto-related datasets"""
        datasets = self.discover_financial_datasets()
        crypto_datasets = []

        crypto_keywords = ["crypto", "bitcoin", "ethereum", "blockchain", "digital asset"]

        for dataset in datasets:
            name_desc = (dataset.name + " " + dataset.description).lower()
            if any(keyword in name_desc for keyword in crypto_keywords):
                crypto_datasets.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "name": dataset.name,
                        "description": dataset.description,
                        "provider": dataset.provider,
                        "last_updated": dataset.last_updated.isoformat(),
                    }
                )

        return crypto_datasets

    def get_available_economic_datasets(self) -> List[Dict[str, Any]]:
        """Get specifically economic-related datasets"""
        datasets = self.discover_financial_datasets()
        econ_datasets = []

        econ_keywords = [
            "economic",
            "gdp",
            "inflation",
            "employment",
            "interest rate",
            "fed",
            "monetary",
        ]

        for dataset in datasets:
            name_desc = (dataset.name + " " + dataset.description).lower()
            if any(keyword in name_desc for keyword in econ_keywords):
                econ_datasets.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "name": dataset.name,
                        "description": dataset.description,
                        "provider": dataset.provider,
                        "last_updated": dataset.last_updated.isoformat(),
                    }
                )

        return econ_datasets

    def load_dataset_to_database(
        self, dataset_id: str, asset_id: str, table_name: str
    ) -> Dict[str, Any]:
        """Load AWS Data Exchange data directly to database"""
        try:
            # Download and process data
            df = self.download_and_process_data(dataset_id, asset_id)

            # Import database connection
            from api.data_loader import Session, engine

            # Store to database
            df.to_sql(table_name, engine, if_exists="append", index=False)

            # Get summary stats
            session = Session()
            try:
                result = session.execute(f"SELECT COUNT(*) FROM {table_name}")
                record_count = result.fetchone()[0]
            finally:
                session.close()

            return {
                "status": "success",
                "dataset_id": dataset_id,
                "asset_id": asset_id,
                "table_name": table_name,
                "records_loaded": record_count,
                "data_shape": df.shape,
                "columns": list(df.columns),
            }

        except Exception as e:
            logger.error(f"Error loading dataset to database: {e}")
            return {"status": "error", "error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Test the service
    service = AWSDataExchangeService()

    print("Discovering financial datasets...")
    datasets = service.discover_financial_datasets()
    print(f"Found {len(datasets)} financial datasets")

    for dataset in datasets[:5]:  # Show first 5
        print(f"- {dataset.name} ({dataset.provider})")
        print(f"  {dataset.description[:100]}...")

    print("\nCrypto-specific datasets:")
    crypto_datasets = service.get_available_crypto_datasets()
    for dataset in crypto_datasets[:3]:
        print(f"- {dataset['name']}")

    print("\nEconomic datasets:")
    econ_datasets = service.get_available_economic_datasets()
    for dataset in econ_datasets[:3]:
        print(f"- {dataset['name']}")
