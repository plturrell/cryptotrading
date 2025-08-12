#!/bin/bash
# Run рекс.com locally for teaching

echo "🚀 Starting рекс.com locally..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install flask flask-cors flask-restx sqlalchemy python-dotenv web3 eth-account

# Initialize database if needed
if [ ! -f "data/рекс.db" ]; then
    echo "🗄️ Initializing database..."
    python init_db.py
fi

# Start the application
echo "✨ Starting рекс.com on http://localhost:5000"
echo "📊 Access the platform at: http://localhost:5000"
echo "📚 API docs at: http://localhost:5000/api/"
echo ""
echo "Press Ctrl+C to stop"

python app.py