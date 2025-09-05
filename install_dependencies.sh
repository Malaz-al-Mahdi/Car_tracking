#!/bin/bash

echo "🚗 Installing Car Detection Analysis Dependencies"
echo "================================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "❌ No virtual environment detected!"
    echo "Please activate your virtual environment first:"
    echo "source car_detection_env/bin/activate"
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch (CPU version for better compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install ultralytics
pip install opencv-python
pip install easyocr
pip install scikit-learn
pip install numpy
pip install webcolors
pip install Pillow

# Verify installations
echo "🔍 Verifying installations..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully')"
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} installed successfully')"
python -c "import ultralytics; print('✅ Ultralytics installed successfully')"
python -c "import easyocr; print('✅ EasyOCR installed successfully')"
python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__} installed successfully')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__} installed successfully')"

echo ""
echo "🎉 All dependencies installed successfully!"
echo "You can now run: python car_detection_analysis.py"