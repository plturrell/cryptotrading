# Fix рекс.com Server on DigitalOcean

Your droplet is running at:
- IPv4: `165.227.69.235`
- IPv6: `2604:a880:800:14:0:1:96ef:2000`
- Private IP: `10.108.0.2`

## Access Your Server

### Option 1: Use DigitalOcean Console
1. Click "Console" in your DigitalOcean dashboard
2. This opens a web-based terminal to your server
3. Login as `root` with your password

### Option 2: SSH from Terminal
```bash
ssh root@165.227.69.235
```

## Once Connected, Run These Commands:

### 1. Check Current Status
```bash
# See what's installed
which python3
which nginx
ls -la /opt/

# Check if any services are running
systemctl status nginx
ps aux | grep python
```

### 2. Quick Install Script
```bash
# Download and run setup
cd /tmp
wget https://raw.githubusercontent.com/plturrell/cryptotrading/main/app.py
wget https://raw.githubusercontent.com/plturrell/cryptotrading/main/requirements.txt

# Or clone entire repo
cd /opt
git clone https://github.com/plturrell/cryptotrading.git рекс.com
cd рекс.com
```

### 3. Install Dependencies
```bash
# Install system packages
apt update
apt install -y python3-pip python3-venv nginx git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install flask flask-cors flask-restx sqlalchemy python-dotenv
```

### 4. Start рекс.com
```bash
# Quick test
python3 app.py

# If it works, press Ctrl+C and set up as service
```

### 5. Create Service (for auto-start)
```bash
cat > /etc/systemd/system/reks.service << 'EOF'
[Unit]
Description=рекс.com Trading Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/рекс.com
ExecStart=/usr/bin/python3 /opt/рекс.com/app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable reks
systemctl start reks
```

### 6. Configure Nginx
```bash
cat > /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:5000/health;
    }
}
EOF

nginx -t
systemctl restart nginx
```

## Test Your Server

From your local machine:
```bash
# Test if it's working
curl http://165.227.69.235/health

# Or open in browser
open http://165.227.69.235
```

## Troubleshooting Commands

```bash
# Check logs
journalctl -u reks -n 50
tail -f /var/log/nginx/error.log

# Check ports
netstat -tlnp | grep -E '(80|5000)'

# Test Flask directly
cd /opt/рекс.com
python3 -c "import app; print('App imports OK')"
```

## Once It's Running

Your teaching platform will be available at:
- http://165.227.69.235 (direct IP)
- http://рекс.com (once DNS updates)

With features:
- AI Trading Analysis
- MetaMask Integration  
- Real-time Market Data
- Paper Trading System

Ready to teach trading together!