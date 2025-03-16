# BasicOperationImageProcessing

This repository contains basic image processing operations implemented in Python. It includes functions for:
- Converting images to greyscale
- Applying binary thresholding using the Sobel operator and other methods
- Zoom functionality to inspect RGB, greyscale, and binary pixel values

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/your-username/BasicOperationImageProcessing.git
cd BasicOperationImageProcessing
```

### 2. Set Up a Virtual Environment

#### Windows (Command Prompt / PowerShell)
```sh
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux (Terminal)
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
Run the main script to start processing images:
```sh
python namefolder/namefile.py
```

## About the Assets Folder
The `Assets` folder contains various resources used in the project, such as images and sample data. If you want to add more files, please place them in this folder.

If you want to run the code with different assets, modify the main script to use a different image or dataset. Ensure that you only change the filename while keeping the directory structure unchanged.

Example:
```python
img = cv2.imread('../Assets/sunflower.jpg')
```
