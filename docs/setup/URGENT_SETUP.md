# URGENT: Secure and Setup rex.com Server

## 1. FIRST - Change Your Password (CRITICAL!)

Connect via SSH or Console and immediately run:
```bash
passwd
```
Enter a new strong password twice.

## 2. Connect to Your Server

```bash
ssh root@165.227.69.235
# Enter the password (change it immediately after login!)
```

## 3. Quick Setup Commands

Once connected, copy and paste these commands:

```bash
# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y python3-pip python3-venv nginx git curl

# Clone your repository
cd /opt
git clone https://github.com/plturrell/cryptotrading.git rex.com
cd rex.com

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install flask flask-cors flask-restx sqlalchemy python-dotenv web3 eth-account

# Create simple test to verify
python3 -c "import flask; print('Flask OK')"

# Initialize database
python3 init_db.py

# Test run the application
python3 app.py
```

## 4. If Everything Works, Set Up as Service

Press Ctrl+C to stop the test, then:

```bash
# Create service file
cat > /etc/systemd/system/reks.service << 'EOF'
[Unit]
Description=rex.com Trading Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rex.com
Environment="PATH=/opt/rex.com/venv/bin"
ExecStart=/opt/rex.com/venv/bin/python /opt/rex.com/app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start the service
systemctl daemon-reload
systemctl enable reks
systemctl start reks

# Configure Nginx
cat > /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80 default_server;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# Restart Nginx
systemctl restart nginx
```

## 5. Verify Everything is Running

```bash
# Check services
systemctl status reks
systemctl status nginx

# Test the endpoints
curl http://localhost/health
```

## 6. Access Your Trading Platform

Open in browser: http://165.227.69.235

You should see the rex.com trading platform with:
- AI Analysis
- MetaMask Integration
- Real-time Data
- Trading Signals

## Important Security Steps

1. **Change the root password** (if you haven't already)
2. Consider setting up a firewall:
   ```bash
   ufw allow 22
   ufw allow 80
   ufw allow 443
   ufw enable
   ```

3. Remove this file after setup to protect credentials
   ```bash
   rm /Users/apple/projects/cryptotrading/URGENT_SETUP.md
   ```

Ready to teach trading on rex.com!