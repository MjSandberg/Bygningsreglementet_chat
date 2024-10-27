import os
import dash
from dash import html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from rag import Retriever, Generator
from scraper import WebScraper

# Initialize components
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize scraper and get data
scraper = WebScraper()
if not os.path.exists("bygningsreglementet_data.json"):
    print("No local data found. Starting scraping process...")
    scraper.scrape_all()
data = scraper.get_data()

# Initialize retriever and generator
retriever = Retriever(data)
generator = Generator(API_KEY)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom styles for markdown
markdown_styles = {
    'backgroundColor': 'white',
    'padding': '1rem',
    'borderRadius': '0.25rem',
}

# App layout
app.layout = dbc.Container([
    html.H1("Bygningsreglementet Chat", className="my-4"),
    
    # Chat history
    dbc.Card(
        dbc.CardBody(id="chat-history", children=[]),
        style={"height": "400px", "overflow-y": "auto", "margin-bottom": "20px"}
    ),
    
    # Input area with loading overlay
    dbc.Row([
        dbc.Col([
            dbc.Input(
                id="user-input",
                placeholder="Skriv dit spørgsmål her...",
                type="text",
            ),
        ], width=10),
        dbc.Col([
            dbc.Button(
                "Send",
                id="submit-button",
                color="primary",
                n_clicks=0,
            ),
        ], width=2),
    ]),
    
    dbc.Spinner(
        html.Div(id="loading-output"),
        color="primary",
        type="border",
    ),
    
    # Store components
    dcc.Store(id='chat-store', data=[]),
], fluid=True)

@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-store', 'data'),
     Output('user-input', 'value'),
     Output('submit-button', 'disabled'),
     Output('user-input', 'disabled'),
     Output('loading-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('chat-store', 'data')],
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, chat_data):
    if not user_input:
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    # Disable inputs while processing
    chat_display = dash.no_update
    if chat_data:
        chat_display = [
            dbc.Card(
                dbc.CardBody(message["user"], style={"background-color": "#f8f9fa"}),
                className="mb-2 ml-auto",
                style={"width": "70%", "margin-left": "30%"}
            ) if i % 2 == 0 else
            dbc.Card(
                dbc.CardBody([
                    dcc.Markdown(message["bot"], style=markdown_styles)
                ]),
                className="mb-2",
                style={"width": "70%"}
            )
            for message in chat_data
            for i in range(2)
        ]
    
    # Generate response
    response = generator.generate_answer(user_input, data, retriever)
    
    # Update chat data
    chat_data = chat_data or []
    chat_data.append({"user": user_input, "bot": response})
    
    # Create chat display
    chat_display = []
    for message in chat_data:
        chat_display.extend([
            dbc.Card(
                dbc.CardBody(message["user"], style={"background-color": "#f8f9fa"}),
                className="mb-2 ml-auto",
                style={"width": "70%", "margin-left": "30%"}
            ),
            dbc.Card(
                dbc.CardBody([
                    dcc.Markdown(message["bot"], style=markdown_styles)
                ]),
                className="mb-2",
                style={"width": "70%"}
            ),
        ])
    
    return chat_display, chat_data, "", False, False, ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
