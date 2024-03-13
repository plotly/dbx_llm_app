# Standard Library Imports
import json
import os
import time  # Import time for simulating a delay
import requests

# Third-party Library Imports
from dash import dcc, html, callback, Input, Output, State
import dash
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import openai
from openai import OpenAI
from sqlalchemy import create_engine

# Databricks Imports
from databricks import sql, sqlalchemy
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

# Pyspark Imports
from pyspark.sql import DataFrame
from pyspark_ai import SparkAI


# Local/Application Specific Imports

  
import re

import pandas as pd
import re
import plotly.express as px

import json

from dash import dcc, html
import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State

from dash import html
from dash.dependencies import Input, Output, State, ClientsideFunction
import time  # For generating a timestamp
import plotly.graph_objects as go
from dash import dcc, html, Output, Input, State, no_update, callback_context
import json

from dash import no_update
import json

from dotenv import load_dotenv

load_dotenv()
# Note: Commented out import for 'langchain_openai.ChatOpenAI' as it seems to be unused or specific to an external module not provided here.
# from langchain_openai import ChatOpenAI

# Initialize the OpenAI client with Databricks API token and base URL (Placeholder for actual initialization code)


# Use the DATABRICKS_TOKEN
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
MAX_RESULTS = 10
w= WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
# Use the registered_models API to list the registered models.
serving_endpoints = w.serving_endpoints.list()
dropdown_options = [{'value': endpoint.name, 'label': endpoint.name} for endpoint in serving_endpoints]



model_type_map = {
    "databricks-llama-2-70b-chat": "chat",
    "databricks-mixtral-8x7b-instruct": "chat",
    "databricks-bge-large-en": "embedding",
    "databricks-mpt-30b-instruct": "completion",
    "databricks-mpt-7b-instruct": "completion",
    "starling7b": "hf-chat",
    "biomistral7b": "hf-chat",
    # Add more mappings as necessary
}





def save_chat_history(chat_history, file_path='./mount/chat_history.json'):
    """Save the chat history to a file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(chat_history, file)
    except Exception as e:
        print(f"Error saving chat history: {e}")


def load_chat_history(file_path='./mount/chat_history.json'):
    """Load the chat history from a file."""
    try:
        with open(file_path, 'r') as file:
            chat_history = json.load(file)
        return chat_history
    except FileNotFoundError:
        # If the file does not exist, return an empty chat history
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []





# Initialize the OpenAI client with Databricks API token and base URL
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints"
)

    


def score_model(selected_model_endpoint, text, model_type_override=None, **kwargs):
    """
    Queries a model endpoint with a dynamic payload based on the selected endpoint,
    allowing for a manual override of the model type, and defaults to 'chat' for unsupported types.
    
    :param selected_model_endpoint: str, The name of the serving endpoint.
    :param text: str, The main text input for the model.
    :param model_type_override: Optional[str], Manual override for the model type ('chat', 'embedding', 'completion').
    :param kwargs: dict, Optional keyword arguments for additional parameters like 'max_tokens', 'temperature', etc.
    :return: dict, A simplified, consistent response structure.
    """
    url = f"{DATABRICKS_HOST}/serving-endpoints/{selected_model_endpoint}/invocations"
    print(url)
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    
    data_json = json.dumps({
    "inputs": [
        {"role": "user", "content": text}
    ]
    
    })

    print(data_json)
    
    # Use the model type map with a default fallback to 'chat'
    model_type = model_type_map.get(selected_model_endpoint, "chat")

    # Check if there's a manual override for the model type
    if model_type_override:
        model_type = model_type_override

    if '-hf' in selected_model_endpoint:
        return 'hf-chat'

    try:
        # Initialize json_simplified_response to None
        json_simplified_response = None

        if model_type == 'chat':
            api_response = client.chat.completions.create(
                model=selected_model_endpoint,
                messages=[{"role": "user", "content": text}],
                **kwargs 
            )
            # Assuming api_response is an object with a method to convert to dict
            simplified_response = api_response.dict()

        elif model_type == 'embedding':
            api_response = client.embeddings.create(
                model=selected_model_endpoint,
                input=text,
                **kwargs
            )
            simplified_response = api_response.dict()

        elif model_type == 'hf-chat':
            response = requests.post(url=url, headers=headers, data=data_json)
            if response.status_code == 200:
                json_simplified_response = response.json()  # Correctly handling JSON response
            else:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            # No need to set simplified_response here since json_simplified_response is used
        

        elif model_type == 'completion':
            api_response = client.completions.create(
                model=selected_model_endpoint,
                prompt=text,
                **kwargs
            )
            simplified_response = api_response.dict()

        else:  
            api_response = client.chat.completions.create(
                model=selected_model_endpoint,
                messages=[{"role": "user", "content": text}],
                **kwargs
            )
            simplified_response = api_response.dict()

        # Determine which response to process based on model type
        if model_type == 'hf-chat':
            # For 'hf-chat', use the json_simplified_response
            print(json_simplified_response)  # For debugging
            return extract_response(json_simplified_response)
        else:
            # For all other model types, use the simplified_response
            print(simplified_response)  # For debugging
            return extract_response(simplified_response)

    except Exception as e:
        raise Exception(f"Query failed: {e}")


def extract_response(api_response):
    """
    Extracts text from various types of model responses, including handling of pandas DataFrames.
    """
    extracted_messages = []

    # Regular expression to match and exclude the initial {'role': 'user', 'content': '...'} part
    pattern = re.compile(r"\{'role': 'user', 'content': '[^']+'\}\s*")


    if isinstance(api_response, list):
        return api_response

    elif 'predictions' in api_response and isinstance(api_response['predictions'], list):
        for prediction in api_response['predictions']:
            cleaned_prediction = pattern.sub('', prediction, count=1)
            extracted_messages.append(cleaned_prediction)

    elif 'choices' in api_response and isinstance(api_response['choices'], list):
        for choice in api_response['choices']:
            if 'text' in choice:
                extracted_messages.append(choice['text'])
            elif 'message' in choice and 'content' in choice['message']:
                extracted_messages.append(choice['message']['content'])
            else:
                extracted_messages.append("Relevant content not found.")

    else:
        extracted_messages.append("Unexpected response format received.")

    return extracted_messages


 
def register_layout():
    return dmc.MantineProvider(
        children=dmc.NotificationsProvider(
            children=[
                dmc.Container(
                    className="container",
                    children=[
                        dmc.Paper(
                            className="paper-component",
                            shadow="xs",
                            children=[
                                dmc.Title("Register New Hugging Face Model", style={"fontSize": "24px", "marginTop": "5px"}),
                                html.Div(id="notifications-container"),
                                dmc.Space(h=10),
                                dmc.TextInput(
                                    id="model_name",
                                    label="Hugging Face Model Name:",
                                    type="text",
                                    placeholder="Enter name of the model huggingface/model",
                                    size="md",
                                    className="input-field",
                                ),
                                dmc.Space(h=10),
                                dmc.TextInput(
                                    id="registered_name",
                                    label="Assign a Registered Name:",
                                    type="text",
                                    placeholder="Enter what you want the registered name for your model to be",
                                    size="md",
                                    className="input-field",
                                ),
                                dmc.Space(h=10),
                                dmc.TextInput(
                                    id="max_tokens_inference",
                                    label="Maximum Number of New Tokens the Model Can Generate:",
                                    type="text",
                                    placeholder="max tokens",
                                    size="md",
                                    className="input-field",
                                ),
                                dmc.Space(h=10),
                                dmc.TextInput(
                                    id="temperature_inference",
                                    label="Default Temperature",
                                    type="text",
                                    placeholder="Enter your default temperature value",
                                    size="md",
                                    className="input-field",
                                ),
                                dmc.Group(
                                    position="left",
                                    mt="xl",
                                    children=[
                                        dmc.Button(
                                            "Register Model with Databricks and Create Serving Endpoint",
                                            rightIcon=DashIconify(
                                                icon="codicon:run-above",
                                            ),
                                            id="register-button",
                                            size="md",
                                        )
                                    ],
                                ),
                                dmc.Space(h=10),
                                dcc.Store(id="selected-profile-store", storage_type="memory"),
                                dcc.Store(id="engine-store", storage_type="memory"),
                                dcc.Store(id="profile-store", storage_type="local"),
                                html.Div(id="dummy", style={"display": "none"}),
                                dcc.Store(id="cluster-options-store", storage_type="memory"),
                                html.Div(
                                    id="success-message"),  # Hidden success message
                                dmc.Modal(
                                    id="user-modal",
                                    title="Select Profile",
                                    children=[
                                        dmc.Select(
                                            data=[],
                                            id="profile-radio",
                                        ),
                                        dmc.Space(h=20),
                                        dmc.Group(
                                            position="right",
                                            children=[
                                                dmc.Button(
                                                    id="switch-profile-button",
                                                    children="Switch Profile",
                                                    size="md",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    fluid=True
                )
            ]
        )
    )

def layout():
    return dmc.MantineProvider(
        children=dmc.NotificationsProvider(
            children=[
                dmc.Container(
                    className="container",
                    children=[
                        dmc.Title("Chat with Hugging Face Models Served On Databricks", style={"fontSize": "24px", "marginTop": "5px"}),
                        dmc.Space(h=20),
                        dmc.Divider(variant="solid"),
                        dmc.Space(h=20),
                         dmc.Group(
                            position="left",
                            children=[
                                dmc.Select(id='endpoint-select',label="Select Endpoint", data=dropdown_options),
                                dmc.TextInput(
                                    id="max_new_tokens",
                                    label="max new tokens:",
                                    placeholder= 600,
                                    className="input-field",
                                ),
                                dmc.TextInput(
                                    id="temperature",
                                    label="temperature:",
                                    placeholder="0.1",
                                    className="input-field"
                                ),
                                dmc.Badge(
                                    id='endpoint-state-display'
                                ),
                                
                            ],
                        ), 
                        dmc.Space(h=20),
                        html.Script("""
                                        document.addEventListener('DOMContentLoaded', function() {
                                            typeWriter('model-output', 'Hello, this is a typewriter effect!', 150);
                                        });
                                    """),
                        dmc.Grid(
                            justify="center",
                            children=[
                                dmc.Col(
                                    children=dmc.LoadingOverlay(
                                        loaderProps={"color": "#228be6", "variant": "bars"},
                                        overlayOpacity=0.6,
                                        overlayColor="#c5d9ed",
                                        id='model-output-overlay',  # Control the visibility
                                        children=dmc.Paper(
                                            className="paper-component",
                                            children=[],
                                            id='model-output'
                                            # dcc.Markdown is now a child of dmc.Paper
                                        )
                                    ),
                                )
                            ]
                        ),
                        dmc.Space(h=20),
                        html.Div(id='typewriter-source', style={'display': 'none'}),
                        html.Div(id='dummy-output', style={'display': 'none'}),
                        dmc.Grid(
                            children=[
                                dmc.Col(
                                    # span=6,
                                    # offset=3,
                                    children=dmc.Textarea(
                                        className="textarea",
                                        id='input-text',
                                        autosize=True,
                                        placeholder="Enter your text here...",   
                                    ), 
                                ),
                                dmc.Space(h=20),
                            
                                dmc.Paper(
                                    children=[
                                    ], 
                                    id='graph-ouput'),
                                dmc.Button('Submit Text', id='submit-button', variant="outline") 
                            ],
                            justify="center"
                        ),
                        # Assuming these components are defined within your app.layout
                        dcc.Store(id='chat-history', storage_type='memory'),
                        dcc.Store(id='chat-history', data=load_chat_history(), storage_type='memory'),
  # To store chat history
                       

                        dmc.Space(h=20),
                        
                    ],
                    fluid=True
                )
            ]
        )
    )


def chat_history_layout():
    return dmc.MantineProvider(
        children=dmc.NotificationsProvider(
            children=[
                dmc.Container(
                    className="container",
                    children=[
                        dmc.Title("Chat History",
                                  style={"fontSize": "24px", "marginTop": "5px"}),
                        dmc.Space(h=20),
                        dmc.Paper(
                            id='chat-window',
                            children=[],
                            style={'maxHeight': '500px', 'overflowY': 'auto', 'padding': '20px'}  # Adjusted for padding
                        ),  # To display chat
                        dmc.Space(h=20),
                        # Additional components like input fields and send button can be added here
                    ],
                    fluid=True
                )
            ]
        )
    )




app = dash.Dash(__name__)



app.layout = dmc.Tabs(
    [
        dmc.TabsList([
            dmc.Tab("Register New Models", value="register"),
            dmc.Tab(
                "Chat Interface", value="chat"
            ),
            dmc.Tab("Chat History", value="chat-history"), 
    ]),
        dmc.TabsPanel(layout(), value="chat"),
        dmc.TabsPanel(register_layout(), value="register"),
        dmc.TabsPanel(chat_history_layout(), value="chat-history"),
    ], 
    placement="left",
    orientation="vertical",
)



@app.callback(
    [
        Output('chat-history', 'data'),  # Continue to update the chat history
        Output('typewriter-source', 'children')  # Update the source for the typewriter effect
    ],
    [Input('submit-button', 'n_clicks')],
    [State('endpoint-select', 'value'),
     State('input-text', 'value'),
     State('max_new_tokens', 'value'),
     State('temperature', 'value'),
     State('chat-history', 'data')]
)
def update_output_and_typewriter(n_clicks, selected_model_endpoint, input_text, max_tokens, temperature, chat_history):
    if not n_clicks or not input_text.strip():
        return no_update, no_update

    kwargs = {}
    if max_tokens: kwargs['max_tokens'] = int(max_tokens)
    if temperature: kwargs['temperature'] = float(temperature)

    if chat_history is None: chat_history = []
    chat_history.append({'sender': 'user', 'text': input_text})

    response = score_model(selected_model_endpoint, input_text, **kwargs)
    typewriter_text = ""  # Initialize the variable for the typewriter effect

    try:
        if response:
            # Handle textual response
            if isinstance(response, list) and response:
                model_response = response[0]
            else:
                model_response = response

            if isinstance(model_response, dict):
                # Update chat history with model text
                chat_text = model_response.get('text', "Unexpected response format received.")
                chat_history.append({'sender': 'model', 'text': chat_text})
                typewriter_text = chat_text
            else:
                # Update with textual response if it's not a dictionary
                chat_history.append({'sender': 'model', 'text': model_response})
                typewriter_text = model_response

    except Exception as e:
        chat_history.append({'sender': 'model', 'text': f"Error processing model response: {str(e)}"})

    return chat_history, typewriter_text


app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='typeWriterEffect'
    ),
    Output('dummy-output', 'children'),  # Dummy output, as JS updates the content
    [Input('typewriter-source', 'children')]  # Triggered when source text updates
)


@app.callback(
    Output('endpoint-state-display', 'children'),
    Input('endpoint-select', 'value')
)
def update_endpoint_state(selected_endpoint_name):
    # Find the selected endpoint by name
    selected_endpoint = next((ep for ep in serving_endpoints if ep.name == selected_endpoint_name), None)
    
    # If an endpoint is selected, extract and format its state for display
    if selected_endpoint:
        # Extracting state details
        config_update_status = selected_endpoint.state.config_update.name  # For example: 'NOT_UPDATING'
        
        ready_status = selected_endpoint.state.ready.name  # For example: 'READY'
        
        # Formatting the state information into a string
        state_info = f"Config Update: {config_update_status}, Ready: {ready_status}"
        
        # Return the formatted state information within a Badge component
        return dmc.Badge(state_info, color="blue" if ready_status == "READY" else "red")
    else:
        return "Please select an endpoint."
    


@app.callback(
    Output('chat-window', 'children'),
    [Input('chat-history', 'data')]
)
def update_chat_window(chat_history):
    if chat_history is None:
        return []
    chat_window_content = []
    for message in chat_history:
        text = f"{message['sender'].title()}: {message['text']}"
        chat_window_content.append(html.P(text))
    return chat_window_content



from dash.exceptions import PreventUpdate

from components import NewCluster, NotebookTask, GitSource
import components



from dash.exceptions import PreventUpdate

@app.callback(
    Output("success-message", "children"),
    Input("register-button", "n_clicks"),
    [State("model_name", "value"),
     State("registered_name", "value"),
     State("max_tokens_inference", "value"),
     State("temperature_inference", "value")]
)
def register_model(n_clicks, model_name, registered_name, max_tokens, temperature):
    print("Callback triggered")
    if n_clicks is None:
        print("No clicks yet")
        raise PreventUpdate

    if not all([model_name, registered_name, max_tokens, temperature]):
        print("Some fields are empty")
        return "Please fill in all fields before registering."
    
    if not model_name.endswith("-hf"):
        model_name += "-hf"

    try:
        print("Preparing job configuration")
        base_parameters = {
            "model_name": model_name,
            "registered_name": registered_name,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        # Create the job
        created_job = w.jobs.create(
            name=f"{registered_name} Creation Run",
            git_source=GitSource(
                git_url="https://github.com/sachds/serve-hf-models-databricks-mlflow.git",
                git_provider="GITHUB",  # Corrected value
                git_branch="main"
            ),
            max_concurrent_runs=1,
            tasks=[
                jobs.Task(
                    task_key="Register_Model",
                    notebook_task=NotebookTask(
                        notebook_path="serve-hf-models-databricks-dash",  # Adjusted to a workspace path
                        base_parameters=base_parameters
                    ),
                    new_cluster=NewCluster(
                        "g4dn.2xlarge",
                        "14.3.x-gpu-ml-scala2.12",
                        3,
                        {"spark.databricks.delta.preview.enabled": "true"},
                        {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"},
                        True,
                    ),
                    timeout_seconds=0,
                )
            ],
            timeout_seconds=0,
        )

        job_id = created_job.job_id
        print(job_id)
        run_resp = w.jobs.run_now(job_id=created_job.job_id)
        print(run_resp)
        run_id = run_resp.response.run_id
        print(run_id)


        return f"Model registration initiated successfully! Job ID: {job_id}, Run ID: {run_id}"
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return f"An error occurred during model registration: {str(e)}"


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
