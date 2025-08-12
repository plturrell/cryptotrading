#!/bin/bash
# Deploy рекс.com to DigitalOcean server

SERVER_IP="165.227.69.235"
echo "🚀 Deploying рекс.com to DigitalOcean server: $SERVER_IP"

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf deploy.tar.gz \
    app.py \
    requirements.txt \
    pyproject.toml \
    .htaccess \
    manifest.json \
    init_db.py \
    webapp/ \
    src/ \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".git"

# Copy files to server
echo "📤 Uploading files to server..."
scp -o StrictHostKeyChecking=no deploy.tar.gz root@$SERVER_IP:/tmp/

# Deploy and start services
echo "🔧 Installing and starting рекс.com..."
ssh -o StrictHostKeyChecking=no root@$SERVER_IP << 'EOF'
    # Stop any existing services
    systemctl stop рекс || true
    systemctl stop nginx || true
    
    # Extract files
    cd /opt
    rm -rf рекс.com.backup
    mv рекс.com рекс.com.backup || true
    mkdir -p рекс.com
    cd рекс.com
    tar -xzf /tmp/deploy.tar.gz
    
    # Install Python dependencies
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Initialize database
    python3 init_db.py
    
    # Create systemd service
    cat > /etc/systemd/system/рекс.service << 'SERVICE'
[Unit]
Description=рекс.com Trading Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/рекс.com
Environment="PATH=/opt/рекс.com/venv/bin"
ExecStart=/opt/рекс.com/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE

    # Configure Nginx
    cat > /etc/nginx/sites-available/рекс.com << 'NGINX'
server {
    listen 80;
    server_name рекс.com www.рекс.com 165.227.69.235;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /static/ {
        alias /opt/рекс.com/webapp/;
    }
}
NGINX

    # Enable services
    ln -sf /etc/nginx/sites-available/рекс.com /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Start services
    systemctl daemon-reload
    systemctl enable рекс
    systemctl start рекс
    systemctl restart nginx
    
    # Check status
    sleep 5
    systemctl status рекс --no-pager
    curl -I http://localhost:5000/health
EOF

# Clean up
rm deploy.tar.gz

echo "✅ Deployment complete!"
echo "🌐 Check: http://$SERVER_IP/health"
echo "📊 View logs: ssh root@$SERVER_IP 'journalctl -u рекс -f'"