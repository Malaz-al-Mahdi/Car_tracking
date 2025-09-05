#!/bin/bash

echo "🚗 Car Detection Analysis - Environment Setup"
echo "=============================================="

# Create virtual environment if it doesn't exist
if [ ! -d "car_detection_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv car_detection_env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source car_detection_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch first (CPU version for better compatibility)
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To use the system:"
echo "1. Activate environment: source car_detection_env/bin/activate"
echo "2. Place your video.MOV file in this directory"
echo "3. Run: python car_detection_analysis.py"
echo ""
echo "📖 For help: python car_detection_analysis.py --help"