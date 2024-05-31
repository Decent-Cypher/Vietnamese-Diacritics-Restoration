# Vietnamese Diacritics Restoration

This project implements N-gram models, BiLSTM model, and Transformer model to restore diacritics in Vietnamese texts. Additionally, it provides a user interface using the PyQt6 library.

## Features

- **N-gram Models**: Statistical models that predict the likelihood of a word given the previous words.
- **BiLSTM Model**: Bidirectional Long Short-Term Memory network for sequence prediction.
- **Transformer Model**: Advanced model leveraging self-attention mechanisms for better context understanding.
- **PyQt6 User Interface**: A graphical user interface for easy interaction with the models.


## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Decent-Cypher/Vietnamese-Diacritics-Restoration.git
    cd Vietnamese-Diacritics-Restoration
    ```

2. **Create a virtual environment (recommended)**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the main application**:
    ```sh
    python app.py
    ```

