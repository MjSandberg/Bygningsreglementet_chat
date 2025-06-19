"""Main Dash application for the Bygningsreglementet Chat Bot."""

from typing import List, Dict, Any, Optional
import dash
from dash import html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..config import Config
from ..utils.logging import get_logger
from ..rag import Retriever, Generator
from ..scraper import WebScraper


class ChatApp:
    """Main chat application class."""
    
    def __init__(self):
        """Initialize the chat application."""
        self.logger = get_logger("ui")
        self.app = None
        self.retriever = None
        self.generator = None
        self.data = None
        
    def initialize_components(self) -> None:
        """Initialize scraper, data, retriever, and generator."""
        try:
            self.logger.info("Initializing application components...")
            
            # Validate configuration
            Config.validate()
            
            # Initialize scraper and data
            scraper = WebScraper()
            if not scraper.data:
                self.logger.info("No local data found. Starting scraping process...")
                scraper.scrape_all()
            self.data = scraper.get_data()
            
            # Initialize retriever and generator
            self.retriever = Retriever(self.data)
            self.generator = Generator()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def create_layout(self) -> dbc.Container:
        """
        Create the application layout.
        
        Returns:
            Dash layout container
        """
        # Custom styles for markdown
        markdown_styles = {
            'backgroundColor': 'white',
            'padding': '1rem',
            'borderRadius': '0.25rem',
        }
        
        layout = dbc.Container([
            html.H1("Bygningsreglementet Chat", className="my-4"),
            
            # Chat history
            dbc.Card(
                dbc.CardBody(id="chat-history", children=[]),
                style={"height": "400px", "overflow-y": "auto", "margin-bottom": "20px"}
            ),
            
            # Input area
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
            
            # Loading spinner
            dbc.Spinner(
                html.Div(id="loading-output"),
                color="primary",
                type="border",
            ),
            
            # Store components
            dcc.Store(id='chat-store', data=[]),
        ], fluid=True)
        
        return layout

    def setup_callbacks(self) -> None:
        """Set up Dash callbacks."""
        
        @self.app.callback(
            [Output('chat-history', 'children'),
             Output('chat-store', 'data'),
             Output('user-input', 'value'),
             Output('submit-button', 'disabled'),
             Output('user-input', 'disabled'),
             Output('loading-output', 'children')],
            [Input('submit-button', 'n_clicks'),
             Input('user-input', 'n_submit')],
            [State('user-input', 'value'),
             State('chat-store', 'data')],
            prevent_initial_call=True
        )
        def update_chat(n_clicks: int, n_submit: Optional[int], 
                       user_input: str, chat_data: List[Dict[str, str]]):
            """Update chat interface with new messages."""
            try:
                if not user_input or not user_input.strip():
                    return no_update, no_update, no_update, no_update, no_update, no_update
                
                user_input = user_input.strip()
                
                # Generate response
                self.logger.info(f"Processing query: {user_input}")
                response = self.generator.generate_answer(user_input, self.data, self.retriever)
                
                # Update chat data
                chat_data = chat_data or []
                chat_data.append({"user": user_input, "bot": response})
                
                # Create chat display
                chat_display = self._create_chat_display(chat_data)
                
                return chat_display, chat_data, "", False, False, ""
                
            except Exception as e:
                self.logger.error(f"Error in chat callback: {e}")
                error_message = "Der opstod en fejl. Prøv venligst igen."
                
                chat_data = chat_data or []
                chat_data.append({"user": user_input, "bot": error_message})
                chat_display = self._create_chat_display(chat_data)
                
                return chat_display, chat_data, "", False, False, ""

    def _create_chat_display(self, chat_data: List[Dict[str, str]]) -> List[dbc.Card]:
        """
        Create chat display from chat data.
        
        Args:
            chat_data: List of chat messages
            
        Returns:
            List of Dash components for display
        """
        chat_display = []
        
        for message in chat_data:
            # User message
            user_card = dbc.Card(
                dbc.CardBody(message["user"], style={"background-color": "#f8f9fa"}),
                className="mb-2 ml-auto",
                style={"width": "70%", "margin-left": "30%"}
            )
            
            # Bot message
            bot_card = dbc.Card(
                dbc.CardBody([
                    dcc.Markdown(
                        message["bot"], 
                        style={
                            'backgroundColor': 'white',
                            'padding': '1rem',
                            'borderRadius': '0.25rem',
                        }
                    )
                ]),
                className="mb-2",
                style={"width": "70%"}
            )
            
            chat_display.extend([user_card, bot_card])
        
        return chat_display

    def create_app(self) -> dash.Dash:
        """
        Create and configure the Dash application.
        
        Returns:
            Configured Dash app
        """
        # Initialize components first
        self.initialize_components()
        
        # Create Dash app
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        
        # Set layout
        self.app.layout = self.create_layout()
        
        # Setup callbacks
        self.setup_callbacks()
        
        self.logger.info("Dash application created successfully")
        return self.app


def create_app() -> dash.Dash:
    """
    Factory function to create the Dash application.
    
    Returns:
        Configured Dash app
    """
    chat_app = ChatApp()
    return chat_app.create_app()
