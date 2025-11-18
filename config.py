# config.py

"""
Configuration settings for the Telegram bot
"""
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- IDS DE CANAUX PAR DÉFAUT (Supprimés, les vrais IDs sont maintenant dans config.json) ---
DEFAULT_TARGET_CHANNEL_ID = None 
DEFAULT_PREDICTION_CHANNEL_ID = None 

# --- CONSTANTES POUR LES CALLBACKS DE CONFIGURATION ---
CALLBACK_SOURCE = "config_source"
CALLBACK_PREDICTION = "config_prediction"
CALLBACK_CANCEL = "config_cancel"

class Config:
    """Configuration class for bot settings"""
    
    def __init__(self):
        # BOT_TOKEN - OBLIGATOIRE
        self.BOT_TOKEN = self._get_bot_token()
        
        # Détermination de l'URL du Webhook
        self.WEBHOOK_URL = self._determine_webhook_url()
        logger.info(f"🔗 Webhook URL configuré: {self.WEBHOOK_URL}")

        # Port pour le serveur (utilise PORT env ou 5000 par défaut)
        self.PORT = int(os.getenv('PORT') or 5000)
        
        # Canaux (Les vraies valeurs sont gérées par CardPredictor)
        self.TARGET_CHANNEL_ID = DEFAULT_TARGET_CHANNEL_ID
        self.PREDICTION_CHANNEL_ID = DEFAULT_PREDICTION_CHANNEL_ID
        
        # Mode Debug
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Validation finale
        self._validate_config()
    
    def _get_bot_token(self) -> str:
        """Récupère et valide le jeton du bot."""
        token = os.getenv('BOT_TOKEN')
        
        if not token:
            logger.error("❌ BOT_TOKEN non trouvé dans les variables d'environnement")
            raise ValueError("BOT_TOKEN environment variable is required")
        
        if len(token.split(':')) != 2:
            logger.error("❌ Format de token invalide")
            raise ValueError("Invalid bot token format")

        logger.info(f"✅ BOT_TOKEN configuré: {token[:10]}...")
        return token
    
    def _determine_webhook_url(self) -> str:
        """Détermine l'URL du webhook avec priorité à l'ENV."""
        webhook_url = os.getenv('WEBHOOK_URL')
        
        # Logique d'auto-génération (adaptée à Replit comme dans le schéma)
        if not webhook_url:
            if os.getenv('REPLIT_DOMAINS'):
                webhook_url = f"https://{os.getenv('REPLIT_DOMAINS')}"
            else:
                webhook_url = f'https://{os.getenv("REPL_SLUG", "")}.{os.getenv("REPL_OWNER", "")}.repl.co'
        
        return webhook_url
    
    def _validate_config(self) -> None:
        """Valide les paramètres de configuration."""
        if self.WEBHOOK_URL and not self.WEBHOOK_URL.startswith('https://'):
            logger.warning("⚠️ L'URL du webhook devrait utiliser HTTPS pour la production.")
        
        logger.info("✅ Configuration validée avec succès.")
    
    def get_webhook_url(self) -> str:
        """Renvoie l'URL complète du webhook (y compris /webhook)."""
        if self.WEBHOOK_URL:
            return f"{self.WEBHOOK_URL}/webhook"
        return ""
    
    def __str__(self) -> str:
        """Représentation textuelle de la configuration (sans données sensibles)."""
        return (
            f"Config(webhook_url={self.WEBHOOK_URL}, "
            f"port={self.PORT}, "
            f"debug={self.DEBUG})"
    )
            
