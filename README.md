# Dash Chat Interface for Databricks Models

This project provides a web-based interface for interacting with machine learning models served on Databricks, featuring functionalities to register new Hugging Face models, chat with these models, and view chat history. Built with Dash and leveraging Databricks for model serving, it offers a user-friendly platform for experimenting with and deploying machine learning models.

<div align="center">
  <a href="https://dash.plotly.com/project-maintenance">
    <img src="https://dash.plotly.com/assets/images/maintained-by-plotly.png" width="400px" alt="Maintained by Plotly">
  </a>
</div>


## Features

- **Model Registration**: Easily register new Hugging Face models with Databricks and create serving endpoints directly from the interface.
- **Chat Interface**: Engage in conversations with the registered models served on Databricks, providing a seamless chatting experience.
- **Chat History**: View the history of your interactions with the models, allowing for review and analysis of past conversations.

## Getting Started

### Prerequisites

In order to run this app, you will need to make sure you have the following items:

- Python 3.10 or later
- Databricks account
- Hugging Face models (for registration)

Databricks-specific items:
- A Databricks workspace with Unity Catalog enabled
        This workspace is where you'll be pulling models from Hugging Face, logging them, and deploying them. It's essential that your account has the permissions needed to create clusters, jobs, and manage models.
- A personal access token to authenticate with Databricks SDK workspace client
- A Python development environment running Python3.10
 Having a functional development environment for Plotly Dash is crucial. Dash is a Python framework for building data applications. Follow the instructions in this project’s README.md file on Github (link) in order to deploy this project’s Dash app.  If you are looking for a quickstart guide to Dash, check out our Dash in 20 minutes tutorial here. 


*** In addition, the user whose personal access token you choose to use must have CREATE MODEL permissions enabled. (see below) ***

<img width="616" alt="Screenshot 2024-03-11 at 10 02 30 AM" src="https://github.com/plotly/dbx_llm_app/assets/49540501/d8f9c9ee-8cf7-41af-80c4-67a86bee42b9">


### Installation

1. Clone the repository to your local machine:

```python 
git clone https://github.com/yourgithubusername/yourrepositoryname.git
```


2. Navigate to the cloned directory:

```cd dashdbxhf```


3. Install the required Python dependencies:

```pip install -r requirements.txt```


4. Set up your environment variables by creating a `.env` file in the root directory and adding your Databricks token and host:

```
DATABRICKS_TOKEN=your_databricks_token
DATABRICKS_HOST=https://your-databricks-workspace-url
```


### Running the Application

1. Start the Dash application:

```python app.py```


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

Sachin- sachin@plot.ly



