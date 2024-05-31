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
1. Ensure you have completed the installation steps.

2. **Run the main application**:
    ```sh
    python app.py
    ```
    
3. On the left side of the newly created window, you will see a text box which will contain the messages you want to predict. In order to predict new text inputs, you can use the **Add text** and **Read text from file** buttons, after that, please press the **PREDICT** button in the middle. The output will appear on the right side of the app window. You can use our prepared text files contained in the demos folder. The model used in making predictions can also be switched using the **Choose a model** combobox.
       * You should see a window like this pop up when first running the application:
![App_User_Interface](pictures/App_User_Interface.png)

	   * You can choose one from 3 models listed in the combobox for making predictions:
![App_User_Interface_Algorithms](pictures/App_User_Interface_Models.png)

	   * After you have chosen a file or added some messages yourself and pushed the PREDICT button, a result like this will be shown:
![App_User_Interface2](pictures/App_User_Interface2.png)

