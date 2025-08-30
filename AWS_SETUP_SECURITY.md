# 🔐 AWS Security Setup Guide

## ⚠️ CRITICAL SECURITY ISSUE RESOLVED

**FIXED**: Removed hardcoded AWS credentials from all scripts
**IMPLEMENTED**: AWS Secrets Manager integration for secure credential storage

## 🔧 Secure AWS Configuration with Secrets Manager

### 1. AWS Secrets Manager Setup (Recommended)

**Why Secrets Manager?**
- ✅ No credentials in source code or environment variables
- ✅ Centralized credential management
- ✅ Automatic credential rotation
- ✅ Fine-grained access control
- ✅ Audit logging and monitoring

**Setup Steps:**

1. **Configure AWS CLI** (one-time setup):
```bash
aws configure
# OR use IAM roles for EC2/ECS/Lambda
```

2. **Run the setup script**:
```bash
python3 scripts/setup_s3_credentials.py
```

3. **Follow the interactive prompts** to securely store your S3 credentials in Secrets Manager

4. **Test the integration**:
```bash
python3 scripts/setup_real_s3.py
```

### 2. Alternative: Environment Variables (Not Recommended)

Only use environment variables for development/testing:

```bash
# Temporary setup for development only
export AWS_ACCESS_KEY_ID='your_access_key'
export AWS_SECRET_ACCESS_KEY='your_secret_key'
export S3_BUCKET_NAME='tentimecrypto'
export AWS_DEFAULT_REGION='us-east-1'
```

### 2. AWS IAM Best Practices

Create an IAM user with minimal required permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::tentimecrypto",
                "arn:aws:s3:::tentimecrypto/*",
                "arn:aws:s3:::cryptotrading-dataexchange-temp",
                "arn:aws:s3:::cryptotrading-dataexchange-temp/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:CreateSecret",
                "secretsmanager:UpdateSecret"
            ],
            "Resource": "arn:aws:secretsmanager:*:*:secret:cryptotrading/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "dataexchange:ListDataSets",
                "dataexchange:ListRevisions",
                "dataexchange:ListAssets",
                "dataexchange:CreateJob",
                "dataexchange:StartJob",
                "dataexchange:GetJob"
            ],
            "Resource": "*"
        }
    ]
}
```

### 3. Alternative: AWS IAM Roles (Recommended)

For production deployments, use IAM roles instead of access keys:

1. **EC2 Instance**: Attach IAM role to EC2 instance
2. **ECS/Fargate**: Use task execution roles
3. **Lambda**: Built-in execution roles

### 4. Testing Your Setup

Test the Secrets Manager integration:

```bash
# Test Secrets Manager setup
python3 scripts/setup_s3_credentials.py

# Test S3 integration
python3 scripts/setup_real_s3.py

# Verify all credentials are secure
python3 scripts/verify_security_fix.py
```

### 5. Security Checklist

- [x] ✅ Hardcoded credentials removed
- [x] ✅ AWS Secrets Manager integration implemented
- [ ] ✅ IAM permissions minimized
- [x] ✅ `.env` added to `.gitignore`
- [ ] ✅ Secrets Manager secret created
- [ ] ✅ S3 integration tested
- [ ] ✅ AWS credentials rotated regularly
- [ ] ✅ CloudTrail logging enabled
- [ ] ✅ S3 bucket policies configured

## 🚨 Never Do This

❌ **NEVER** commit files containing:
- AWS access keys
- AWS secret keys  
- Database passwords
- API tokens
- Private keys

## ✅ Production Recommendations

1. **✅ Use AWS Secrets Manager** (implemented)
2. **Enable MFA** on AWS accounts
3. **Use least privilege IAM policies**
4. **Enable CloudTrail logging**
5. **Set up automatic credential rotation** in Secrets Manager
6. **Monitor AWS costs and usage**
7. **Use VPC endpoints** for Secrets Manager (optional, for enhanced security)
8. **Enable AWS Config** for compliance monitoring

## 📧 Support

If you need help with AWS setup:
1. **Setup Issues**: Run `python3 scripts/setup_s3_credentials.py`
2. **Testing Issues**: Run `python3 scripts/verify_security_fix.py`
3. **Check AWS documentation** for Secrets Manager and S3
4. **Use AWS support** (if you have a support plan)
5. **Review AWS security best practices guide**

## 🔄 Migration from Environment Variables

If you were previously using environment variables:

1. **Create secret**: Run `python3 scripts/setup_s3_credentials.py`
2. **Test integration**: Run `python3 scripts/setup_real_s3.py`
3. **Remove old env vars**: Unset environment variables
4. **Verify security**: Run `python3 scripts/verify_security_fix.py`

The system will now automatically use AWS Secrets Manager for all credential operations.