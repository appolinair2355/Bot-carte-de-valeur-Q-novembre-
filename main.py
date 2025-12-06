# main.py

"""
Main entry point for the Telegram bot deployment on render.com
"""
import os
import logging
import json
from flask import Flask, request, jsonify
import requests

# Importe la configuration et le bot
from config import Config
from bot import TelegramBot 

# --- IMPORTATION POUR LA PLANIFICATION ---
from apscheduler.schedulers.background import BackgroundScheduler
import pytz # Pour gérer les fuseaux horaires

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Définir le fuseau horaire du Bénin (UTC+1)
BENIN_TIMEZONE = pytz.timezone('Africa/Porto-Novo')

# Initialize bot and config
try:
    config = Config()
except ValueError as e:
    logger.error(f"❌ Erreur d'initialisation de la configuration: {e}")
    exit(1) 

# 'bot' est l'instance de la classe TelegramBot
bot = TelegramBot(config.BOT_TOKEN) 

# Initialize Flask app
app = Flask(__name__)


# --- LOGIQUE WEBHOOK ---

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook from Telegram"""
    try:
        update = request.get_json(silent=True)
        if not update:
            return jsonify({'status': 'ok'}), 200

        # Délégation du traitement complet à bot.handle_update
        if update:
            bot.handle_update(update)
        
        return 'OK', 200
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        return 'Error', 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for render.com"""
    return {'status': 'healthy', 'service': 'telegram-bot'}, 200

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return {'message': 'Telegram Bot is running', 'status': 'active'}, 200

# --- CONFIGURATION WEBHOOK ---

def setup_webhook():
    """Set up webhook on startup"""
    try:
        full_webhook_url = config.get_webhook_url()
        
        if full_webhook_url and not config.WEBHOOK_URL.startswith('https://.repl.co'):
            logger.info(f"🔗 Tentative de configuration webhook: {full_webhook_url}")

            success = bot.set_webhook(full_webhook_url)
            
            if success:
                logger.info(f"✅ Webhook configuré avec succès.")
                logger.info(f"🎯 Bot prêt pour prédictions automatiques et vérifications via webhook")
            else:
                logger.error("❌ Échec configuration webhook.")
        else:
            logger.warning("⚠️ WEBHOOK_URL non configurée ou non valide. Le webhook ne sera PAS configuré.")
    except Exception as e:
        logger.error(f"❌ Erreur critique lors du setup du webhook: {e}")

# --- LOGIQUE DE PLANIFICATION QUOTIDIENNE ---

def start_scheduler():
    """Démarre l'ordonnanceur pour la tâche quotidienne de réinitialisation."""
    scheduler = BackgroundScheduler(timezone=BENIN_TIMEZONE)
    
    # Récupérer l'instance de CardPredictor
    card_predictor = bot.handlers.card_predictor
    
    if card_predictor:
        # Tâche : Appeler daily_reset_all_data à 00h59 (heure du Bénin)
        scheduler.add_job(
            card_predictor.daily_reset_all_data, 
            'cron', 
            hour=0, 
            minute=59, 
            id='daily_reset_job',
            misfire_grace_time=600 # Permet une exécution jusqu'à 10 min de retard
        )
        logger.info("⏱️ Tâche de reset quotidien programmée pour 00h59 (heure du Bénin).")
        scheduler.start()
    else:
        logger.error("❌ Impossible de démarrer le scheduler : CardPredictor non initialisé.")

# --- DÉMARRAGE DU PROGRAMME PRINCIPAL ---

if __name__ == '__main__':
    # 1. Configurer le Webhook
    setup_webhook()

    # 2. Démarrer le Scheduler (planification)
    start_scheduler()
    
    # 3. Démarrer le serveur Flask
    port = config.PORT
    logger.info(f"🚀 Serveur Flask démarré sur le port {port}")
    app.run(host='0.0.0.0', port=port)
