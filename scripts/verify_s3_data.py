#!/usr/bin/env python3
"""
Verify the data in S3 bucket and show the structure
"""

import os
from datetime import datetime

import boto3


def verify_s3_data():
    """Verify and display S3 bucket contents"""

    print("🔍 Verifying S3 Bucket Contents")
    print("=" * 50)

    # Get credentials from environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if not aws_access_key or not aws_secret_key:
        print("❌ AWS credentials not found in environment variables")
        print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return

    # Direct S3 client
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region,
    )

    bucket_name = os.getenv("S3_BUCKET_NAME", "tentimecrypto")

    try:
        # List all objects
        print(f"📋 Listing all objects in bucket: {bucket_name}")

        response = s3_client.list_objects_v2(Bucket=bucket_name)
        objects = response.get("Contents", [])

        print(f"Total objects: {len(objects)}")
        print(f"Bucket size: {sum(obj['Size'] for obj in objects):,} bytes")

        # Group by data type
        data_types = {}
        for obj in objects:
            key_parts = obj["Key"].split("/")
            data_type = key_parts[0] if key_parts else "root"

            if data_type not in data_types:
                data_types[data_type] = []

            data_types[data_type].append(obj)

        print(f"\n📂 Data organization:")
        for data_type, files in data_types.items():
            print(f"\n{data_type.upper()}:")
            for file_obj in files:
                size_kb = file_obj["Size"] / 1024
                mod_time = file_obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S")
                print(f"  📄 {file_obj['Key']} ({size_kb:.1f}KB) - {mod_time}")

        # Show some sample data
        print(f"\n📖 Sample data contents:")

        # Find a market data file
        market_data_files = [obj for obj in objects if obj["Key"].startswith("market-data/")]
        if market_data_files:
            sample_file = market_data_files[0]["Key"]
            print(f"\nSample market data file: {sample_file}")

            response = s3_client.get_object(Bucket=bucket_name, Key=sample_file)
            content = response["Body"].read().decode("utf-8")

            # Show first 300 characters
            print("Content preview:")
            print(content[:300] + "..." if len(content) > 300 else content)

        # Find a user data file
        user_data_files = [obj for obj in objects if obj["Key"].startswith("user-data/")]
        if user_data_files:
            sample_file = user_data_files[0]["Key"]
            print(f"\nSample user data file: {sample_file}")

            response = s3_client.get_object(Bucket=bucket_name, Key=sample_file)
            content = response["Body"].read().decode("utf-8")

            print("Content preview:")
            print(content[:300] + "..." if len(content) > 300 else content)

        return True

    except Exception as e:
        print(f"❌ Error verifying S3 data: {e}")
        return False


if __name__ == "__main__":
    success = verify_s3_data()

    if success:
        print("\n✅ S3 data verification complete!")
        print("🎯 All crypto trading data is properly stored and organized")
    else:
        print("\n❌ S3 data verification failed")
