"""Main Dash application for the Bygningsreglementet Chat Bot."""

from typing import List, Dict, Any, Optional
import dash
from dash import html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..config import Config
from ..utils.logging import get_logger
from ..scraper import WebScraper
from ..agents import AgenticRAGSystem


class ChatApp:
    """Main chat application class."""
    
    def __init__(self):
        """Initialize the chat application."""
        self.logger = get_logger("ui")
        self.app = None
        self.agentic_system = None
        self.data = None
        
    def initialize_components(self) -> None:
        """Initialize scraper, data, and agentic system."""
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
            
            # Initialize agentic RAG system
            self.agentic_system = AgenticRAGSystem(self.data)
            
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
        layout = dbc.Container([
            html.H1("Bygningsreglementet Chat", className="my-4"),
            
            # Chat history
            dbc.Card(
                dbc.CardBody(id="chat-history", children=[]),
                style={"height": "400px", "overflow-y": "auto", "margin-bottom": "20px"}
            ),
            
            # Citations modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Kilder og Referencer")),
                dbc.ModalBody(id="citations-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Luk", id="close-citations-modal", className="ms-auto", n_clicks=0)
                ),
            ], id="citations-modal", is_open=False, size="lg"),
            
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
            dcc.Store(id='citations-store', data={}),
        ], fluid=True)
        
        return layout

    def setup_callbacks(self) -> None:
        """Set up Dash callbacks."""
        
        @self.app.callback(
            [Output('chat-history', 'children'),
             Output('chat-store', 'data'),
             Output('citations-store', 'data'),
             Output('user-input', 'value'),
             Output('submit-button', 'disabled'),
             Output('user-input', 'disabled'),
             Output('loading-output', 'children')],
            [Input('submit-button', 'n_clicks'),
             Input('user-input', 'n_submit')],
            [State('user-input', 'value'),
             State('chat-store', 'data'),
             State('citations-store', 'data')],
            prevent_initial_call=True
        )
        def update_chat(n_clicks: int, n_submit: Optional[int], 
                       user_input: str, chat_data: List[Dict[str, str]], citations_data: Dict):
            """Update chat interface with new messages."""
            try:
                if not user_input or not user_input.strip():
                    return no_update, no_update, no_update, no_update, no_update, no_update, no_update
                
                user_input = user_input.strip()
                
                # Generate response using agentic system
                self.logger.info(f"Processing query: {user_input}")
                response = self.agentic_system.process_query(user_input)
                
                # Get citation mapping from the last query
                citation_mapping = self.agentic_system.get_last_citation_mapping()
                
                # Update chat data
                chat_data = chat_data or []
                message_index = len(chat_data)
                
                # Prepare citation data for this message
                message_citations = None
                if citation_mapping and citation_mapping.citations:
                    formatted_citations = self.agentic_system.citation_agent.format_citations_for_display(citation_mapping)
                    message_citations = formatted_citations
                    citations_data[str(message_index)] = formatted_citations
                
                chat_data.append({
                    "user": user_input, 
                    "bot": response,
                    "has_citations": message_citations is not None,
                    "citation_count": len(message_citations.get("citations", [])) if message_citations else 0
                })
                
                # Create chat display
                chat_display = self._create_chat_display(chat_data)
                
                return chat_display, chat_data, citations_data, "", False, False, ""
                
            except Exception as e:
                self.logger.error(f"Error in chat callback: {e}")
                error_message = "Der opstod en fejl. Prøv venligst igen."
                
                chat_data = chat_data or []
                chat_data.append({"user": user_input, "bot": error_message, "has_citations": False, "citation_count": 0})
                chat_display = self._create_chat_display(chat_data)
                
                return chat_display, chat_data, citations_data, "", False, False, ""
        
        # Citation modal callbacks
        @self.app.callback(
            [Output("citations-modal", "is_open"),
             Output("citations-modal-body", "children")],
            [Input({"type": "citation-button", "index": dash.dependencies.ALL}, "n_clicks"),
             Input("close-citations-modal", "n_clicks")],
            [State("citations-modal", "is_open"),
             State("citations-store", "data")],
            prevent_initial_call=True
        )
        def toggle_citations_modal(citation_clicks, close_clicks, is_open, citations_data):
            """Toggle the citations modal and populate with citation data."""
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return False, []
            
            trigger_id = ctx.triggered[0]["prop_id"]
            
            # Close modal
            if "close-citations-modal" in trigger_id:
                return False, []
            
            # Open modal with citations
            if "citation-button" in trigger_id and any(citation_clicks):
                # Find which button was clicked
                for i, clicks in enumerate(citation_clicks):
                    if clicks:
                        message_citations = citations_data.get(str(i))
                        if message_citations:
                            modal_content = self._create_citations_display(message_citations)
                            return True, modal_content
                        break
            
            return is_open, []

    def _create_chat_display(self, chat_data: List[Dict[str, str]]) -> List[dbc.Card]:
        """
        Create chat display from chat data.
        
        Args:
            chat_data: List of chat messages
            
        Returns:
            List of Dash components for display
        """
        chat_display = []
        
        for i, message in enumerate(chat_data):
            # User message
            user_card = dbc.Card(
                dbc.CardBody(message["user"], style={"background-color": "#f8f9fa"}),
                className="mb-2 ml-auto",
                style={"width": "70%", "margin-left": "30%"}
            )
            
            # Bot message content
            bot_content = [
                dcc.Markdown(
                    message["bot"], 
                    style={
                        'backgroundColor': 'white',
                        'padding': '1rem',
                        'borderRadius': '0.25rem',
                    }
                )
            ]
            
            # Add citation button if citations are available
            if message.get("has_citations", False):
                citation_count = message.get("citation_count", 0)
                citation_button = dbc.Button(
                    [
                        html.I(className="fas fa-quote-left me-2"),
                        f"Se kilder ({citation_count})"
                    ],
                    id={"type": "citation-button", "index": i},
                    color="outline-info",
                    size="sm",
                    className="mt-2"
                )
                bot_content.append(html.Div([citation_button]))
            
            bot_card = dbc.Card(
                dbc.CardBody(bot_content),
                className="mb-2",
                style={"width": "70%"}
            )
            
            chat_display.extend([user_card, bot_card])
        
        return chat_display
    
    def _create_citations_display(self, citations_data: Dict[str, Any]) -> List:
        """
        Create citations display for the modal.
        
        Args:
            citations_data: Formatted citation data
            
        Returns:
            List of Dash components for citations display
        """
        if not citations_data or not citations_data.get("citations"):
            return [html.P("Ingen kilder tilgængelige.")]
        
        citations_display = []
        
        # Add summary
        total_citations = citations_data.get("total_citations", 0)
        citations_display.append(
            html.H5(f"Fundet {total_citations} kilder:", className="mb-3")
        )
        
        # Add each citation
        for i, citation in enumerate(citations_data["citations"], 1):
            citation_card = dbc.Card([
                dbc.CardHeader([
                    html.H6(f"Kilde {i}: {citation['title']}", className="mb-0"),
                    dbc.Badge(
                        citation['source_type'].replace('_', ' ').title(),
                        color="info" if citation['source_type'] == "local_knowledge" else "success",
                        className="ms-2"
                    )
                ]),
                dbc.CardBody([
                    html.P(citation['excerpt'], className="mb-2"),
                    html.Small([
                        f"Tillid: {citation['confidence']:.2f} | ",
                        f"Relevans: {citation['relevance_score']:.2f}"
                    ], className="text-muted"),
                    # Add URL if available
                    html.Div([
                        html.A(
                            "Åbn kilde",
                            href=citation.get('url', '#'),
                            target="_blank",
                            className="btn btn-outline-primary btn-sm mt-2"
                        )
                    ]) if citation.get('url') else html.Div()
                ])
            ], className="mb-3")
            
            citations_display.append(citation_card)
        
        return citations_display

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
            external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"]
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
