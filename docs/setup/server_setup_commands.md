# рекс.com Server Setup Commands

## Option 1: Access Your Existing Server

If you have SSH access to 165.227.69.235, run these commands:

```bash
# SSH into server
ssh root@165.227.69.235

# Check what's running
systemctl status рекс
systemctl status nginx
ps aux | grep python

# Restart services
systemctl restart рекс
systemctl restart nginx

# Check logs
journalctl -u рекс -n 50
```

## Option 2: Create New Droplet (Recommended)

Since the server seems to have issues, let's create a fresh one:

### 1. Create New Droplet on DigitalOcean

Go to DigitalOcean and create:
- Ubuntu 22.04
- Basic Plan ($12/month - 2GB RAM)
- NYC or SFO datacenter
- Add your SSH key

### 2. Initial Server Setup

Once created, SSH in and run:

```bash
# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y python3 python3-pip python3-venv nginx git postgresql redis-server

# Clone your repository
cd /opt
git clone https://github.com/plturrell/cryptotrading.git рекс.com
cd рекс.com

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python init_db.py
```

### 3. Create Service File

```bash
# Create systemd service
cat > /etc/systemd/system/рекс.service << 'EOF'
[Unit]
Description=рекс.com Trading Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/рекс.com
Environment="PATH=/opt/рекс.com/venv/bin"
Environment="FLASK_APP=app.py"
ExecStart=/opt/рекс.com/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable рекс
systemctl start рекс
```

### 4. Configure Nginx

```bash
# Create Nginx config
cat > /etc/nginx/sites-available/рекс.com << 'EOF'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

# Enable site
ln -s /etc/nginx/sites-available/рекс.com /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
systemctl restart nginx
```

### 5. Test the Installation

```bash
# Check services
systemctl status рекс
systemctl status nginx

# Test locally
curl http://localhost/health

# Check from outside
curl http://YOUR_SERVER_IP/health
```

## Quick Debugging Commands

```bash
# View application logs
journalctl -u рекс -f

# Check Python errors
cd /opt/рекс.com
source venv/bin/activate
python app.py

# Test database connection
python -c "from src.рекс.database import get_db; db = get_db(); print('DB OK')"

# Check port 5000
netstat -tlnp | grep 5000
```

## Update GoDaddy DNS

Once server is running, update DNS:
- A Record: @ → YOUR_NEW_SERVER_IP
- A Record: www → YOUR_NEW_SERVER_IP

---

## For Teaching Trading Together

Once the server is running, you'll have:

1. **Live Dashboard**: http://рекс.com
2. **API Endpoints**: 
   - `/api/ai/analyze` - AI market analysis
   - `/api/wallet/balance` - MetaMask balance
   - `/api/defi/opportunities` - DeFi yields
   - `/api/ai/signals/BTC` - Trading signals

3. **Features for Teaching**:
   - Real-time price data
   - AI-powered analysis
   - Risk management tools
   - Paper trading system (coming Day 15)

The platform will be perfect for teaching someone to trade with real tools and data!