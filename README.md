# Dash Chat Interface for Databricks Models

This project provides a web-based interface for interacting with machine learning models served on Databricks, featuring functionalities to register new Hugging Face models, chat with these models, and view chat history. Built with Dash and leveraging Databricks for model serving, it offers a user-friendly platform for experimenting with and deploying machine learning models.

## Features

- **Model Registration**: Easily register new Hugging Face models with Databricks and create serving endpoints directly from the interface.
- **Chat Interface**: Engage in conversations with the registered models served on Databricks, providing a seamless chatting experience.
- **Chat History**: View the history of your interactions with the models, allowing for review and analysis of past conversations.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Databricks account
- Hugging Face models (for registration)

### Installation

1. Clone the repository to your local machine:

'''git clone https://github.com/yourgithubusername/yourrepositoryname.git'''


2. Navigate to the cloned directory:

cd yourrepositoryname


3. Install the required Python dependencies:

''' pip install -r requirements.txt '''


4. Set up your environment variables by creating a `.env` file in the root directory and adding your Databricks token and host:

DATABRICKS_TOKEN=your_databricks_token
DATABRICKS_HOST=https://your-databricks-workspace-url


### Running the Application

1. Start the Dash application:

python app.py


2. Open a web browser and navigate to `http://localhost:8050/` to access the application.

## Usage

### Registering New Models

- Navigate to the "Register New Models" tab.
- Fill in the form with the Hugging Face model name, a registered name for your model, maximum tokens for inference, and the default temperature.
- Click "Register Model with Databricks and Create Serving Endpoint" to complete the registration.

### Chatting with Models

- Go to the "Chat Interface" tab.
- Select a model from the "Select Endpoint" dropdown.
- Enter your message in the text area and click "Submit Text" to send.
- The model's response will appear in the chat interface.

### Viewing Chat History

- Switch to the "Chat History" tab to view the history of your conversations.

## Contributing

Contributions to improve the project are welcome. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - your@email.com

Project Link: [https://github.com/yourgithubusername/yourrepositoryname](https://github.com/yourgithubusername/yourrepositoryname)

