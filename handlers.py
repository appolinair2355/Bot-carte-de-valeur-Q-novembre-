# handlers.py

import logging
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
import requests 
import time
import json
import zipfile
import shutil

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
            self.smart_rules = [] # Initialise smart_rules pour éviter les erreurs
            self.predictions = {} # Initialise predictions
        def set_channel_id(self, *args):
            logger.error("CardPredictor non chargé, impossible de définir l'ID du canal.")
            return False
        def get_inter_status(self): 
            return "Système INTER non disponible.", None
        def analyze_and_set_smart_rules(self, *args): return []
        def _save_data(self, *args): pass
        def _verify_prediction_common(self, *args, **kwargs): return None # Ajout pour éviter les erreurs
        def should_predict(self, *args): return False, None, None # Ajout pour éviter les erreurs
        def make_prediction(self, *args): return "" # Ajout pour éviter les erreurs
        def _save_all_data(self): pass # Ajout pour éviter les erreurs
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
• `/deploy` - Télécharger le pack de déploiement

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
            try:
                self.card_predictor = CardPredictor()
            except Exception as e:
                logger.error(f"❌ Échec de l'initialisation de CardPredictor: {e}")
                self.card_predictor = None # Assure que ce reste None en cas d'échec


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
                # S'assurer que self.card_predictor.predictions existe et est un dict
                if not hasattr(self.card_predictor, 'predictions') or not isinstance(self.card_predictor.predictions, dict):
                    self.card_predictor.predictions = {}

                if predicted_game in self.card_predictor.predictions:
                    self.card_predictor.predictions[predicted_game]['message_id'] = message_id
                else:
                    # Initialiser si nécessaire, en s'assurant que la structure est correcte
                    self.card_predictor.predictions[predicted_game] = {'message_id': message_id}


        elif action.get('type') == 'edit_message':
            prediction_data = self.card_predictor.predictions.get(predicted_game) if hasattr(self.card_predictor, 'predictions') else None
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
        if not self.card_predictor: 
            self.send_message(chat_id, "⚠️ Le système de prédiction n'est pas initialisé.")
            return

        source_id = self.card_predictor.target_channel_id if self.card_predictor.target_channel_id else "❌ Non Configuré"
        pred_id = self.card_predictor.prediction_channel_id if self.card_predictor.prediction_channel_id else "❌ Non Configuré"
        inter_status = self.card_predictor.get_inter_status()[0] if hasattr(self.card_predictor, 'get_inter_status') else "Indisponible"

        text = (
            f"**📈 STATISTIQUES GLOBALES 📊**\n"
            f"Canal Source (Lecture): `{source_id}`\n"
            f"Canal Prédiction (Écriture): `{pred_id}`\n"
            f"Mode Intelligent Actif: {inter_status.splitlines()[0].split(' - ')[0].replace('Système INTER ', '')}" # Extrait le statut ON/OFF du message retourné par get_inter_status
        )
        self.send_message(chat_id, text)

    def _handle_bilan_command(self, chat_id: int) -> None:
        if not self.card_predictor: 
            self.send_message(chat_id, "⚠️ Le système de prédiction n'est pas initialisé.")
            return
        
        predictions_count = 0
        if hasattr(self.card_predictor, 'predictions') and isinstance(self.card_predictor.predictions, dict):
            predictions_count = len(self.card_predictor.predictions)

        text = f"**📋 BILAN 🛎️**\nPrédictions stockées: {predictions_count}"
        self.send_message(chat_id, text)

    def _handle_inter_command(self, chat_id: int) -> None:
        """Gère l'affichage du statut INTER et des boutons d'action."""
        if not self.card_predictor:
            self.send_message(chat_id, "⚠️ Le système de prédiction n'est pas initialisé.")
            return

        # Appel à la méthode mise à jour de CardPredictor
        message, keyboard = self.card_predictor.get_inter_status()

        # Utilisation de send_message pour envoyer le message avec le clavier
        self.send_message(chat_id, message, reply_markup=keyboard)

    def _handle_deploy_command(self, chat_id: int) -> None:
        """Génère et envoie le pack de déploiement ZIP."""
        try:
            self.send_message(chat_id, "📦 Génération du pack de déploiement en cours...")
            
            # Nom du fichier ZIP
            zip_filename = "fing1.zip"
            
            # Liste des fichiers à inclure pour le déploiement
            files_to_include = [
                "main.py",
                "bot.py",
                "handlers.py",
                "card_predictor.py",
                "config.py",
                "requirements.txt",
                "Procfile",
                "render.yaml"
            ]
            
            # Fichiers JSON de données (optionnels mais recommandés pour conserver l'état)
            data_files = [
                "channels_config.json",
                "inter_data.json",
                "inter_mode_status.json",
                "last_prediction_time.json",
                "predictions.json",
                "processed.json",
                "sequential_history.json",
                "smart_rules.json"
            ]
            
            # Créer le ZIP
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Ajouter les fichiers principaux
                for filename in files_to_include:
                    if os.path.exists(filename):
                        zipf.write(filename)
                        logger.info(f"✅ Ajouté au ZIP: {filename}")
                    else:
                        logger.warning(f"⚠️ Fichier non trouvé: {filename}")
                
                # Ajouter les fichiers de données s'ils existent
                for filename in data_files:
                    if os.path.exists(filename):
                        zipf.write(filename)
                        logger.info(f"✅ Données ajoutées au ZIP: {filename}")
                
                # Ajouter README.md s'il existe
                if os.path.exists("README.md"):
                    zipf.write("README.md")
                    logger.info(f"✅ Ajouté au ZIP: README.md")
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(zip_filename)
            logger.info(f"📦 Taille du pack: {file_size / 1024:.2f} KB")
            
            # Envoyer le fichier
            self._send_document(chat_id, zip_filename)
            
            # Message de confirmation avec instructions
            instructions = (
                "✅ **Pack de déploiement généré avec succès!**\n\n"
                "📋 **Instructions pour Render.com:**\n"
                "1. Extrayez le contenu du ZIP\n"
                "2. Créez un nouveau dépôt Git avec ces fichiers\n"
                "3. Connectez le dépôt à Render.com\n"
                "4. Configurez les variables d'environnement:\n"
                "   - `BOT_TOKEN`: Votre token Telegram\n"
                "   - `WEBHOOK_URL`: URL de votre app Render\n"
                "   - `PORT`: 10000\n"
                "5. Déployez!\n\n"
                "⚙️ Le port 10000 est préconfigué dans render.yaml"
            )
            self.send_message(chat_id, instructions)
            
            # Nettoyer
            if os.path.exists(zip_filename):
                os.remove(zip_filename)
                logger.info(f"🧹 Fichier ZIP temporaire supprimé")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du pack de déploiement: {e}")
            self.send_message(chat_id, f"❌ Erreur lors de la génération du pack: {str(e)}")

    def _send_document(self, chat_id: int, file_path: str) -> bool:
        """Envoie un document via l'API Telegram."""
        try:
            url = f"{self.base_url}/sendDocument"
            
            if not os.path.exists(file_path):
                logger.error(f"Fichier non trouvé: {file_path}")
                return False
            
            with open(file_path, 'rb') as file:
                files = {
                    'document': (os.path.basename(file_path), file, 'application/zip')
                }
                data = {
                    'chat_id': chat_id,
                    'caption': '📦 Pack de déploiement FING1 - Render.com (Port 10000)\n✅ Prêt pour déploiement avec toutes les modifications'
                }
                
                response = requests.post(url, data=data, files=files, timeout=60)
                result = response.json()
                
                if result.get('ok'):
                    logger.info(f"✅ Document envoyé avec succès à {chat_id}")
                    return True
                else:
                    logger.error(f"❌ Échec de l'envoi du document: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi du document: {e}")
            return False

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
            if self.card_predictor.set_channel_id(chat_id, 'source'):
                message = (
                    f"**🟢 CONFIGURATION RÉUSSIE : CANAL SOURCE**\n"
                    f"Ce chat (`{chat_title}`) est maintenant le canal où le bot **LIRE** les jeux (ID: `{chat_id}`)."
                )
                action_success = True
            else:
                message = f"**🔴 ERREUR CONFIGURATION : CANAL SOURCE**\nImpossible de définir ce chat comme canal source."
                
        elif data == CALLBACK_PREDICTION:
            if self.card_predictor.set_channel_id(chat_id, 'prediction'):
                message = (
                    f"**🔵 CONFIGURATION RÉUSSIE : CANAL DE PRÉDICTION**\n"
                    f"Ce chat (`{chat_title}`) est maintenant le canal où le bot **ÉCRIRA** ses prédictions (ID: `{chat_id}`)."
                )
                action_success = True
            else:
                message = f"**🔴 ERREUR CONFIGURATION : CANAL PRÉDICTION**\nImpossible de définir ce chat comme canal de prédiction."

        elif data == CALLBACK_CANCEL:
            message = f"**❌ CONFIGURATION ANNULÉE.** Le chat `{chat_title}` n'a pas été configuré."
            action_success = True

        # --- GESTION DES BOUTONS INTER ---
        elif data == CALLBACK_INTER_APPLY:
            # Re-analyse l'historique et définit les nouvelles règles
            if hasattr(self.card_predictor, 'analyze_and_set_smart_rules'):
                top_rules = self.card_predictor.analyze_and_set_smart_rules(initial_load=False)
            else:
                top_rules = []
            
            # Récupérer le statut mis à jour
            status_text, _ = self.card_predictor.get_inter_status() if hasattr(self.card_predictor, 'get_inter_status') else ("Erreur statut", None)

            if top_rules:
                message = (
                    f"**✅ RÈGLES INTELLIGENTES APPLIQUÉES!**\n\n"
                    f"Le bot utilise maintenant le TOP {len(top_rules)} des déclencheurs:\n"
                )
                for rule_str in top_rules:
                    message += f"• {rule_str}\n"
                message += f"\n✅ Mode INTER: **ACTIF**\n"
                message += "Les règles statiques sont désactivées."
            else:
                message = (
                    f"**⚠️ IMPOSSIBLE D'APPLIQUER LES RÈGLES INTELLIGENTES**\n\n"
                    f"Pas assez de données dans l'historique.\n"
                    f"Minimum requis: 3 occurrences du même déclencheur.\n\n"
                    f"❌ Mode INTER: **INACTIF**\n"
                    f"Les règles statiques restent actives."
                )
            
            message += "\n\n---\n" + status_text
            self._answer_callback(callback_id, "Analyse terminée.")
            action_success = True


        elif data == CALLBACK_INTER_DEFAULT:
            # Désactive le mode intelligent
            if hasattr(self.card_predictor, 'is_inter_mode_active'):
                self.card_predictor.is_inter_mode_active = False
            # Sauvegarde uniquement le statut (les règles restent en mémoire mais sont ignorées)
            if hasattr(self.card_predictor, '_save_data'):
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
                # Traiter TOUTES les commandes, peu importe le chat
                if text.startswith('/'):
                    if text == '/start': 
                        self._handle_start_command(chat_id)
                        return
                    elif text == '/stat': 
                        self._handle_stat_command(chat_id)
                        return
                    elif text == '/bilan': 
                        self._handle_bilan_command(chat_id)
                        return
                    elif text.startswith('/inter'): 
                        self._handle_inter_command(chat_id)
                        return
                    elif text == '/deploy': 
                        self._handle_deploy_command(chat_id)
                        return

                # Traiter les messages du canal source uniquement
                if self.card_predictor and self.card_predictor.target_channel_id and chat_id == self.card_predictor.target_channel_id: 
                    self._process_channel_message(message)
        except Exception as e:
            logger.error(f"❌ Erreur de traitement du message: {e}")

    def _handle_edited_message(self, message: Dict[str, Any]) -> None:
        # Logique pour gérer les messages édités du canal source
        try:
            chat_id = message['chat']['id']
            # Assurez-vous que card_predictor et target_channel_id sont valides
            if self.card_predictor and self.card_predictor.target_channel_id and chat_id == self.card_predictor.target_channel_id:
                self._process_channel_message(message, is_edited=True)
        except Exception as e:
            logger.error(f"❌ Erreur de traitement du message édité: {e}")

    def _process_channel_message(self, message: Dict[str, Any], is_edited: bool = False) -> None:
        # Logique unifiée de prédiction et de vérification pour les messages de canal (dépend de CardPredictor)
        if not self.card_predictor: return
        message_text = message.get('text', '')
        if not message_text: return

        # 1. Vérification des prédictions passées
        # Assurez-vous que _verify_prediction_common existe
        if hasattr(self.card_predictor, '_verify_prediction_common'):
            verification_action = self.card_predictor._verify_prediction_common(message_text, is_edited=is_edited)
            if verification_action:
                self.process_prediction_action(verification_action)

        # 2. Déclenchement de la nouvelle prédiction (inclut la collecte INTER)
        # Assurez-vous que should_predict et make_prediction existent
        if hasattr(self.card_predictor, 'should_predict') and hasattr(self.card_predictor, 'make_prediction'):
            should_predict, game_number, predicted_value = self.card_predictor.should_predict(message_text)
            
            # Nouvelle logique de prédiction :
            # Si le mode INTER est actif, on ne prédit que si should_predict est True ET que la règle correspondante est valide (3+ occurrences).
            # Sinon (mode INTER désactivé), on utilise la logique par défaut (statique).
            
            predict_now = False
            if self.card_predictor.is_inter_mode_active:
                # En mode INTER, on prédit uniquement si should_predict est True (règle INTER trouvée)
                if should_predict and predicted_value == "Q":
                    predict_now = True
            else:
                # Mode INTER désactivé : on utilise la logique statique par défaut de should_predict
                if should_predict:
                    predict_now = True

            if predict_now:
                new_prediction_message = self.card_predictor.make_prediction(game_number, predicted_value)
                action = {
                    'type': 'new_prediction',
                    'predicted_game': game_number + 2, # game_number est l'index, on veut le numéro du jeu
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