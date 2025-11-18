# handlers.py

import logging
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import requests 
import time
import json # Assurez-vous que json est importé

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Importation de CardPredictor (Assurez-vous que card_predictor.py existe et est accessible)
try:
    from card_predictor import CardPredictor
except ImportError:
    # Fallback minimal pour éviter le crash
    class CardPredictor:
        def __init__(self):
            self.target_channel_id = None
            self.prediction_channel_id = None
            self.is_inter_mode_active = False
            self.inter_data = []
        def set_channel_id(self, *args):
            logger.error("CardPredictor non chargé, impossible de définir l'ID du canal.")
            return False
        def get_inter_status(self): 
            return "Système INTER non disponible.", None
        def analyze_and_set_smart_rules(self, *args): return []
        def _save_data(self, *args): pass
        # Ajoutez toutes les autres méthodes appelées si nécessaire
    logger.error("❌ Échec de l'importation de CardPredictor. Les fonctionnalités de prédiction seront désactivées.")
    

# Limites de débit (Logique conservée pour la robustesse)
user_message_counts = defaultdict(list)
MAX_MESSAGES_PER_MINUTE = 30
RATE_LIMIT_WINDOW = 60

# Messages
WELCOME_MESSAGE = """
🎭 **BIENVENUE DANS LE MONDE DE JOKER DEPLOY299999 !** 🔮

🎯 **COMMANDES DISPONIBLES:**
• `/start` - Accueil
• `/stat` - Statistiques de réussite (Dame Q)
• `/bilan` - Bilan des prédictions stockées
• `/inter` - Gérer le Mode Intelligent N-2 → Q à N

🎯 **Version DEPLOY299999 - Port 10000**
"""
# --- CONSTANTES POUR LES CALLBACKS DE CONFIGURATION ---
CALLBACK_SOURCE = "config_source"
CALLBACK_PREDICTION = "config_prediction"
CALLBACK_CANCEL = "config_cancel"

# --- CONSTANTES POUR LES CALLBACKS INTER ---
CALLBACK_INTER_APPLY = "inter_apply"
CALLBACK_INTER_DEFAULT = "inter_default"


# Fonction utilitaire pour l'Inline Keyboard de configuration
def get_config_keyboard() -> Dict:
    """Crée l'Inline Keyboard pour la configuration des canaux."""
    keyboard = [
        [
            {'text': "✅ OUI, Canal SOURCE (Lecture)", 'callback_data': CALLBACK_SOURCE},
            {'text': "✅ OUI, Canal PRÉDICTION (Écriture)", 'callback_data': CALLBACK_PREDICTION}
        ],
        [
            {'text': "❌ ANNULER", 'callback_data': CALLBACK_CANCEL}
        ]
    ]
    return {'inline_keyboard': keyboard}


class TelegramHandlers:
    """Handlers for Telegram bot using webhook approach"""

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Initialize advanced handlers
        self.card_predictor: Optional[CardPredictor] = None
        if CardPredictor:
            self.card_predictor = CardPredictor()


    # --- MÉTHODES D'INTERACTION TELEGRAM (requests) ---

    def send_message(self, chat_id: int, text: str, parse_mode='Markdown', message_id: Optional[int] = None, edit=False, reply_markup: Optional[Dict] = None) -> Optional[Dict]:
        """Envoie ou édite un message via requests."""
        if message_id or edit:
            method = 'editMessageText'
            payload = {'chat_id': chat_id, 'message_id': message_id, 'text': text, 'parse_mode': parse_mode}
        else:
            method = 'sendMessage'
            payload = {'chat_id': chat_id, 'text': text, 'parse_mode': parse_mode}
        
        if reply_markup:
             payload['reply_markup'] = reply_markup

        url = f"{self.base_url}/{method}"
        try:
            # Sérialiser reply_markup en JSON si présent
            if 'reply_markup' in payload:
                payload['reply_markup'] = json.dumps(payload['reply_markup'])
                
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur {method} Telegram à {chat_id}: {e}")
            return None

    def edit_message(self, chat_id: int, message_id: int, text: str, parse_mode='Markdown', reply_markup: Optional[Dict] = None) -> bool:
        """Fonction utilitaire pour l'édition de message."""
        result = self.send_message(chat_id, text, parse_mode, message_id, edit=True, reply_markup=reply_markup)
        return result.get('ok', False) if result else False
            
    def process_prediction_action(self, action: Dict):
        """Traite les actions de prédiction/vérification (envoi/édition)."""
        if not self.card_predictor or not self.card_predictor.prediction_channel_id:
             logger.warning("Prédiction ignorée: Canal de prédiction non configuré.")
             return
             
        predicted_game = action.get('predicted_game')
        new_message = action.get('new_message')
        chat_id = self.card_predictor.prediction_channel_id 

        if action.get('type') == 'new_prediction':
            result = self.send_message(chat_id=chat_id, text=new_message)
            
            if result and result.get('ok'):
                message_id = result['result']['message_id']
                if predicted_game in self.card_predictor.predictions:
                    self.card_predictor.predictions[predicted_game]['message_id'] = message_id
            
        elif action.get('type') == 'edit_message':
            prediction_data = self.card_predictor.predictions.get(predicted_game)
            message_id = prediction_data.get('message_id') if prediction_data else None

            if message_id:
                self.edit_message(
                    chat_id=chat_id, 
                    text=new_message,
                    message_id=message_id
                )
            else:
                self.send_message(chat_id=chat_id, text=new_message)
        
        # S'assurer que les données de prédiction sont sauvegardées après l'action
        if hasattr(self.card_predictor, '_save_all_data'):
            self.card_predictor._save_all_data()

    # --- GESTION DES COMMANDES (/start, /stat, /bilan, /inter) ---
    def _handle_start_command(self, chat_id: int) -> None:
        self.send_message(chat_id, WELCOME_MESSAGE)
    
    def _handle_stat_command(self, chat_id: int) -> None:
        if not self.card_predictor: return
        source_id = self.card_predictor.target_channel_id if self.card_predictor.target_channel_id else "❌ Non Configuré"
        pred_id = self.card_predictor.prediction_channel_id if self.card_predictor.prediction_channel_id else "❌ Non Configuré"
        
        text = (
            f"**📈 STATISTIQUES GLOBALES 📊**\n"
            f"Canal Source (Lecture): `{source_id}`\n"
            f"Canal Prédiction (Écriture): `{pred_id}`\n"
            f"Mode Intelligent Actif: {'✅ OUI' if self.card_predictor.is_inter_mode_active else '❌ NON'}"
        )
        self.send_message(chat_id, text)

    def _handle_bilan_command(self, chat_id: int) -> None:
        if not self.card_predictor: return
        text = f"**📋 BILAN 🛎️**\nPrédictions stockées: {len(self.card_predictor.predictions) if hasattr(self.card_predictor, 'predictions') else 0}"
        self.send_message(chat_id, text)

    def _handle_inter_command(self, chat_id: int) -> None:
        """Gère l'affichage du statut INTER et des boutons d'action."""
        if not self.card_predictor:
            self.send_message(chat_id, "⚠️ Le système de prédiction n'est pas initialisé.")
            return
        
        # Appel à la méthode mise à jour de CardPredictor
        message, keyboard = self.card_predictor.get_inter_status()
        
        self.send_message(chat_id, message, reply_markup=keyboard)
        
    # --- GESTION DE LA CONFIGURATION DYNAMIQUE ---

    def _send_config_prompt(self, chat_id: int, chat_title: str) -> None:
        """Envoie le message de configuration avec les boutons au chat où le bot a été ajouté."""
        keyboard = get_config_keyboard()

        message = (
            f"**🚨 Configuration du Canal 🚨**\n\n"
            f"Le bot a été ajouté au chat **`{chat_title}`** (ID: `{chat_id}`).\n\n"
            f"Veuillez confirmer le rôle de ce chat pour les prédictions Dame (Q):"
        )
        self.send_message(chat_id, message, reply_markup=keyboard)


    def _handle_callback_query(self, callback_query: Dict[str, Any]) -> None:
        """Gère les réponses des boutons de configuration et INTER."""
        data = callback_query['data']
        chat_id = callback_query['message']['chat']['id'] 
        message_id = callback_query['message']['message_id']
        chat_title = callback_query['message']['chat'].get('title', f'Chat ID: {chat_id}')
        callback_id = callback_query['id'] 

        if not self.card_predictor:
            self.edit_message(chat_id, message_id, "⚠️ Erreur: Système de prédiction non initialisé.")
            self._answer_callback(callback_id, "Erreur système.")
            return

        message = ""
        action_success = False

        # --- GESTION DES BOUTONS DE CONFIGURATION INITIALE ---
        if data == CALLBACK_SOURCE:
            self.card_predictor.set_channel_id(chat_id, 'source')
            message = (
                f"**🟢 CONFIGURATION RÉUSSIE : CANAL SOURCE**\n"
                f"Ce chat (`{chat_title}`) est maintenant le canal où le bot **LIRE** les jeux (ID: `{chat_id}`)."
            )
            action_success = True
        elif data == CALLBACK_PREDICTION:
            self.card_predictor.set_channel_id(chat_id, 'prediction')
            message = (
                f"**🔵 CONFIGURATION RÉUSSIE : CANAL DE PRÉDICTION**\n"
                f"Ce chat (`{chat_title}`) est maintenant le canal où le bot **ÉCRIRA** ses prédictions (ID: `{chat_id}`)."
            )
            action_success = True
        elif data == CALLBACK_CANCEL:
            message = f"**❌ CONFIGURATION ANNULÉE.** Le chat `{chat_title}` n'a pas été configuré."
            action_success = True

        # --- GESTION DES BOUTONS INTER ---
        elif data == CALLBACK_INTER_APPLY:
            # Re-analyse l'historique et définit les nouvelles règles
            self.card_predictor.analyze_and_set_smart_rules(initial_load=False)
            status_text, _ = self.card_predictor.get_inter_status() # Récupère le nouveau statut sans les boutons
            
            message = (
                f"**✅ RÈGLES INTELLIGENTES APPLIQUÉES!**\n\n"
                f"Le bot va maintenant prédire "
                f"en utilisant le TOP 3 des déclencheurs trouvés dans l'historique."
            )
            message += "\n\n---\n" + status_text
            self._answer_callback(callback_id, "Règles appliquées.")
            action_success = True


        elif data == CALLBACK_INTER_DEFAULT:
            # Désactive le mode intelligent
            self.card_predictor.is_inter_mode_active = False
            # Sauvegarde uniquement le statut (les règles restent en mémoire mais sont ignorées)
            self.card_predictor._save_data(self.card_predictor.is_inter_mode_active, 'inter_mode_status.json')
            
            message = "**❌ RÈGLE PAR DÉFAUT APPLIQUÉE!**\n\nLe bot utilise uniquement la logique statique (ex: Valets J) pour la prédiction."
            self._answer_callback(callback_id, "Mode Défaut activé.")
            action_success = True
            
        else:
            self._answer_callback(callback_id, "Action inconnue.")
            return
        
        # Édite le message de configuration/commande pour afficher le résultat final (retire les boutons si l'action est complète)
        if action_success:
             self.edit_message(chat_id, message_id, message) 
             if data not in (CALLBACK_INTER_APPLY, CALLBACK_INTER_DEFAULT): # Pour les configurations de canal
                 self._answer_callback(callback_id, "Configuration terminée!")


    def _answer_callback(self, callback_id: str, text: str):
        """Répond à une callback query pour afficher une notification."""
        url = f"{self.base_url}/answerCallbackQuery"
        payload = {'callback_query_id': callback_id, 'text': text}
        try:
            requests.post(url, json=payload)
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur answerCallbackQuery: {e}")

    # --- GESTION DES UPDATES PRINCIPALES ---
    
    def _handle_message(self, message: Dict[str, Any]) -> None:
        # Logique pour gérer les commandes et le traitement du canal source
        try:
            chat_id = message['chat']['id']
            if 'text' in message:
                text = message['text'].strip()
                if text.startswith('/'):
                    if text == '/start': self._handle_start_command(chat_id)
                    elif text == '/stat': self._handle_stat_command(chat_id)
                    elif text == '/bilan': self._handle_bilan_command(chat_id)
                    elif text.startswith('/inter'): self._handle_inter_command(chat_id)
                    return 

                if self.card_predictor and chat_id == self.card_predictor.target_channel_id: 
                    self._process_channel_message(message)
        except Exception as e:
            logger.error(f"❌ Erreur de traitement du message: {e}")

    def _handle_edited_message(self, message: Dict[str, Any]) -> None:
        # Logique pour gérer les messages édités du canal source
        try:
            chat_id = message['chat']['id']
            if self.card_predictor and chat_id == self.card_predictor.target_channel_id:
                self._process_channel_message(message, is_edited=True)
        except Exception as e:
            logger.error(f"❌ Erreur de traitement du message édité: {e}")

    def _process_channel_message(self, message: Dict[str, Any], is_edited: bool = False) -> None:
        # Logique unifiée de prédiction et de vérification pour les messages de canal (dépend de CardPredictor)
        if not self.card_predictor: return
        message_text = message.get('text', '')
        if not message_text: return
        
        # 1. Vérification des prédictions passées
        verification_action = self.card_predictor._verify_prediction_common(message_text, is_edited=is_edited)
        if verification_action:
            self.process_prediction_action(verification_action)
            
        # 2. Déclenchement de la nouvelle prédiction (inclut la collecte INTER)
        should_predict, game_number, predicted_value = self.card_predictor.should_predict(message_text)
        if should_predict:
            new_prediction_message = self.card_predictor.make_prediction(game_number, predicted_value)
            action = {
                'type': 'new_prediction',
                'predicted_game': game_number + 2,
                'new_message': new_prediction_message
            }
            self.process_prediction_action(action)


    def handle_update(self, update: Dict[str, Any]) -> None:
        """Point d'entrée principal pour traiter une mise à jour Telegram."""
        try:
            # 1. GESTION DES CALLBACKS (Boutons)
            if 'callback_query' in update:
                self._handle_callback_query(update['callback_query'])
                
            # 2. GESTION DE L'AJOUT DU BOT AU CANAL (my_chat_member)
            elif 'my_chat_member' in update:
                my_chat_member = update['my_chat_member']
                # Vérifie si le statut change pour le bot lui-même
                if my_chat_member['new_chat_member']['status'] in ['member', 'administrator']:
                    # Pour être sûr que c'est bien notre bot et non un autre
                    bot_id = int(self.bot_token.split(':')[0])
                    if my_chat_member['new_chat_member']['user']['id'] == bot_id:
                        chat_id = my_chat_member['chat']['id']
                        chat_title = my_chat_member['chat'].get('title', f'Chat ID: {chat_id}')
                        chat_type = my_chat_member['chat'].get('type', 'private')
                        
                        # Déclenche le prompt de configuration si c'est un groupe ou un canal
                        if chat_type in ['channel', 'group', 'supergroup']:
                            logger.info(f"✨ BOT AJOUTÉ/PROMU : Envoi du prompt de configuration à {chat_title} ({chat_id})")
                            self._send_config_prompt(chat_id, chat_title)
            
            # 3. GESTION DES MESSAGES/POSTS
            elif 'message' in update:
                self._handle_message(update['message'])
            elif 'edited_message' in update:
                self._handle_edited_message(update['edited_message'])
            elif 'channel_post' in update:
                self._handle_message(update['channel_post'])
            elif 'edited_channel_post' in update:
                self._handle_edited_message(update['edited_channel_post'])

        except Exception as e:
            logger.error(f"❌ Erreur critique lors du traitement de l'update: {e}")
