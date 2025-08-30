#!/bin/bash

# AWS Data Exchange Setup Script
# Sets up AWS Data Exchange service for cryptotrading platform

echo "=================================================="
echo "  AWS Data Exchange Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Step 1: Check Python and pip
echo -e "${YELLOW}Step 1: Checking Python environment...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo -e "${RED}✗ Python3 not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

if command -v pip3 &> /dev/null; then
    echo "✓ pip3 is available"
else
    echo -e "${RED}✗ pip3 not found. Please install pip.${NC}"
    exit 1
fi

# Step 2: Install AWS dependencies
echo ""
echo -e "${YELLOW}Step 2: Installing AWS dependencies...${NC}"
if pip3 install -r requirements_aws_deps.txt; then
    echo "✓ AWS dependencies installed"
else
    echo -e "${RED}✗ Failed to install AWS dependencies${NC}"
    exit 1
fi

# Step 3: Check AWS CLI (optional but recommended)
echo ""
echo -e "${YELLOW}Step 3: Checking AWS CLI...${NC}"
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version)
    echo "✓ Found: $AWS_VERSION"
else
    echo -e "${YELLOW}⚠ AWS CLI not found (optional but recommended)${NC}"
    echo "  Install with: pip3 install awscli"
fi

# Step 4: Environment setup
echo ""
echo -e "${YELLOW}Step 4: Environment setup...${NC}"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Check current environment variables
AWS_KEY_SET=false
AWS_SECRET_SET=false
AWS_REGION_SET=false
AWS_BUCKET_SET=false

if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
    AWS_KEY_SET=true
    echo "✓ AWS_ACCESS_KEY_ID is set"
else
    echo "✗ AWS_ACCESS_KEY_ID not set"
fi

if [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
    AWS_SECRET_SET=true
    echo "✓ AWS_SECRET_ACCESS_KEY is set"
else
    echo "✗ AWS_SECRET_ACCESS_KEY not set"
fi

if [ ! -z "$AWS_DEFAULT_REGION" ]; then
    AWS_REGION_SET=true
    echo "✓ AWS_DEFAULT_REGION is set to: $AWS_DEFAULT_REGION"
else
    echo "✗ AWS_DEFAULT_REGION not set"
fi

if [ ! -z "$AWS_DATA_EXCHANGE_BUCKET" ]; then
    AWS_BUCKET_SET=true
    echo "✓ AWS_DATA_EXCHANGE_BUCKET is set to: $AWS_DATA_EXCHANGE_BUCKET"
else
    echo "✗ AWS_DATA_EXCHANGE_BUCKET not set"
fi

# Interactive setup if needed
if [ "$AWS_KEY_SET" = false ] || [ "$AWS_SECRET_SET" = false ] || [ "$AWS_REGION_SET" = false ] || [ "$AWS_BUCKET_SET" = false ]; then
    echo ""
    echo -e "${YELLOW}Interactive AWS Setup${NC}"
    echo "Enter your AWS credentials and configuration:"
    
    if [ "$AWS_KEY_SET" = false ]; then
        echo -n "AWS Access Key ID: "
        read AWS_ACCESS_KEY_ID
        echo "export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> .env
    fi
    
    if [ "$AWS_SECRET_SET" = false ]; then
        echo -n "AWS Secret Access Key: "
        read -s AWS_SECRET_ACCESS_KEY
        echo ""
        echo "export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> .env
    fi
    
    if [ "$AWS_REGION_SET" = false ]; then
        echo -n "AWS Region (default: us-east-1): "
        read AWS_REGION
        AWS_REGION=${AWS_REGION:-us-east-1}
        echo "export AWS_DEFAULT_REGION=$AWS_REGION" >> .env
    fi
    
    if [ "$AWS_BUCKET_SET" = false ]; then
        echo -n "S3 Bucket for Data Exchange (will be created if not exists): "
        read AWS_BUCKET
        echo "export AWS_DATA_EXCHANGE_BUCKET=$AWS_BUCKET" >> .env
    fi
    
    echo ""
    echo "✓ Environment variables added to .env file"
    echo "  Run: source .env  (to load in current session)"
fi

# Step 5: Test AWS connection
echo ""
echo -e "${YELLOW}Step 5: Testing AWS connection...${NC}"
echo "Running AWS Data Exchange CLI check..."

if python3 cli_aws_data_exchange.py --check-setup; then
    echo -e "${GREEN}✓ AWS Data Exchange setup completed successfully!${NC}"
else
    echo -e "${RED}✗ AWS connection test failed${NC}"
    echo "Please check your AWS credentials and try again"
    exit 1
fi

# Step 6: Create S3 bucket if needed
echo ""
echo -e "${YELLOW}Step 6: S3 bucket setup...${NC}"
if [ ! -z "$AWS_DATA_EXCHANGE_BUCKET" ] || [ ! -z "$AWS_BUCKET" ]; then
    BUCKET_NAME=${AWS_DATA_EXCHANGE_BUCKET:-$AWS_BUCKET}
    echo "Checking S3 bucket: $BUCKET_NAME"
    
    if aws s3 ls "s3://$BUCKET_NAME" 2>/dev/null; then
        echo "✓ S3 bucket exists: $BUCKET_NAME"
    else
        echo "Creating S3 bucket: $BUCKET_NAME"
        if aws s3 mb "s3://$BUCKET_NAME"; then
            echo "✓ S3 bucket created: $BUCKET_NAME"
        else
            echo -e "${YELLOW}⚠ Could not create S3 bucket (may need different name or permissions)${NC}"
        fi
    fi
fi

# Final instructions
echo ""
echo -e "${GREEN}=================================================="
echo "  Setup Complete!"
echo "==================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Load environment: source .env"
echo "2. Test the CLI: python3 cli_aws_data_exchange.py --interactive"
echo "3. Access web UI: http://localhost:5001 → AWS Data Exchange"
echo ""
echo "CLI Usage Examples:"
echo "  python3 cli_aws_data_exchange.py --check-setup"
echo "  python3 cli_aws_data_exchange.py --discover crypto"
echo "  python3 cli_aws_data_exchange.py --interactive"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "- Ensure your AWS user has AWSDataExchangeFullAccess policy"
echo "- Data Exchange may incur AWS costs for premium datasets"
echo "- Free tier includes some sample datasets"