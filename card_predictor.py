# card_predictor.py

"""
Card prediction logic for Joker's Telegram Bot.
Version finale garantissant l'envoi :
1. Correction de la 'SyntaxError' dans la logique de nettoyage.
2. Correction de la 'Unpacking Error' en assurant le retour de 4 valeurs (statut, num, val, confiance) dans TOUS les cas.
3. Activation des règles statiques par défaut si l'INTER mode n'a pas encore de données.
"""
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import time
import os
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- CONSTANTES ---
HIGH_VALUE_CARDS = ["A", "K", "Q", "J"] 
CARD_SYMBOLS = [r"♠️", r"♥️", r"♦️", r"♣️", r"❤️"] 

class CardPredictor:
    """Gère la logique de prédiction de carte Dame (Q) et la vérification."""

    def __init__(self):
        # Données de persistance
        self.predictions = self._load_data('predictions.json') 
        self.processed_messages = self._load_data('processed.json', is_set=True) 
        self.last_prediction_time = self._load_data('last_prediction_time.json', is_scalar=True)
        
        # Configuration dynamique
        self.config_data = self._load_data('channels_config.json')
        self.target_channel_id = self.config_data.get('target_channel_id', None)
        self.prediction_channel_id = self.config_data.get('prediction_channel_id', None)
        
        # --- Logique INTER ---
        self.sequential_history: Dict[int, Dict] = self._load_data('sequential_history.json') 
        self.inter_data: List[Dict] = self._load_data('inter_data.json') 
        
        # Statut et Règles
        self.is_inter_mode_active = self._load_data('inter_mode_status.json', is_scalar=True)
        self.smart_rules = self._load_data('smart_rules.json') 
        self.prediction_cooldown = 30 
        
        # Tente d'analyser les règles si l'historique existe mais le mode n'est pas actif.
        if self.inter_data and not self.is_inter_mode_active:
             self.analyze_and_set_smart_rules(initial_load=True)

    # --- Persistance des Données ---
    def _load_data(self, filename: str, is_set: bool = False, is_scalar: bool = False) -> Any:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if is_set: return set(data)
                if is_scalar:
                    if filename == 'inter_mode_status.json': return data.get('active', False)
                    return int(data) if isinstance(data, (int, float)) else data
                if filename == 'inter_data.json': return data
                if filename == 'sequential_history.json': 
                    return {int(k): v for k, v in data.items()}
                if filename == 'smart_rules.json': return data
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"⚠️ Fichier {filename} non trouvé ou vide. Init par défaut.")
            if is_set: return set()
            if is_scalar and filename == 'inter_mode_status.json': return False
            if is_scalar: return 0.0
            if filename == 'inter_data.json': return []
            if filename == 'sequential_history.json': return {}
            if filename == 'smart_rules.json': return []
            return {}
        except Exception as e:
             logger.error(f"❌ Erreur critique chargement {filename}: {e}")
             return set() if is_set else (False if filename == 'inter_mode_status.json' else ([] if filename == 'inter_data.json' else {}))

    def _save_data(self, data: Any, filename: str):
        if filename == 'inter_mode_status.json':
            data_to_save = {'active': self.is_inter_mode_active}
        elif isinstance(data, set):
            data_to_save = list(data)
        else:
            data_to_save = data  
        try:
            with open(filename, 'w') as f:
                json.dump(data_to_save, f, indent=4)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde {filename}: {e}.")

    def _save_all_data(self):
        self._save_data(self.predictions, 'predictions.json')
        self._save_data(self.processed_messages, 'processed.json')
        self._save_data(self.last_prediction_time, 'last_prediction_time.json')
        self._save_data(self.inter_data, 'inter_data.json')
        self._save_data(self.sequential_history, 'sequential_history.json')
        self._save_data(self.is_inter_mode_active, 'inter_mode_status.json')
        self._save_data(self.smart_rules, 'smart_rules.json')

    def _save_channels_config(self):
        self.config_data['target_channel_id'] = self.target_channel_id
        self.config_data['prediction_channel_id'] = self.prediction_channel_id
        self._save_data(self.config_data, 'channels_config.json')

    def set_channel_id(self, channel_id: int, channel_type: str):
        if channel_type == 'source':
            self.target_channel_id = channel_id
        elif channel_type == 'prediction':
            self.prediction_channel_id = channel_id
        else:
            return False
        self._save_channels_config()
        return True

    # --- Extraction ---
    def extract_game_number(self, message: str) -> Optional[int]:
        match = re.search(r'#N(\d+)\.', message, re.IGNORECASE) 
        if not match: match = re.search(r'🔵(\d+)🔵', message)
        if match:
            try: return int(match.group(1))
            except ValueError: return None
        return None

    def extract_first_parentheses_content(self, message: str) -> Optional[str]:
        match = re.search(r'\(([^)]*)\)', message)
        return match.group(1).strip() if match else None

    def extract_card_details(self, content: str) -> List[Tuple[str, str]]:
        card_details = []
        normalized_content = content.replace("❤️", "♥️")
        matches = re.findall(r'(\d+|[AKQJ])(♠️|♥️|♦️|♣️)', normalized_content, re.IGNORECASE)
        for value, costume in matches:
            card_details.append((value.upper(), costume))
        return card_details

    def get_first_two_cards(self, content: str) -> List[str]:
        card_details = self.extract_card_details(content)
        return [f"{v}{c}" for v, c in card_details[:2]]

    def check_value_Q_in_first_parentheses(self, message: str) -> Optional[Tuple[str, str]]:
        first_content = self.extract_first_parentheses_content(message)
        if not first_content: return None
        for value, costume in self.extract_card_details(first_content):
            if value == "Q": return (value, costume)
        return None

    def _get_card_numeric_value(self, card_value: str) -> int:
        if card_value.isdigit(): return int(card_value)
        mapping = {'A': 14, 'K': 13, 'Q': 12, 'J': 11}
        return mapping.get(card_value, 0)

    # --- Logique INTER ---
    def collect_inter_data(self, game_number: int, message: str):
        first_group_content = self.extract_first_parentheses_content(message)
        if not first_group_content: return

        # 1. Enregistrer N
        first_two_cards = self.get_first_two_cards(first_group_content)
        if len(first_two_cards) == 2:
            self.sequential_history[game_number] = {
                'cartes': first_two_cards,
                'date': datetime.now().isoformat()
            }
        
        # 2. Vérifier si Q à N
        q_card_details = self.check_value_Q_in_first_parentheses(message)
        if q_card_details:
            n_minus_2_game = game_number - 2
            trigger_entry = self.sequential_history.get(n_minus_2_game)
            if trigger_entry:
                trigger_cards = trigger_entry['cartes']
                # Anti-doublon
                is_duplicate = any(entry.get('numero_resultat') == game_number for entry in self.inter_data)
                if not is_duplicate:
                    new_entry = {
                        'numero_resultat': game_number,
                        'declencheur': trigger_cards,
                        'numero_declencheur': n_minus_2_game,
                        'carte_q': f"{q_card_details[0]}{q_card_details[1]}",
                        'date_resultat': datetime.now().isoformat()
                    }
                    self.inter_data.append(new_entry)
                    self._save_all_data() 
                    logger.info(f"💾 INTER: Q à N={game_number} déclenché par N-2={n_minus_2_game}")
        
        obsolete_game_limit = game_number - 50 
        # CORRECTION SYNTAXE: Utilisation de `num, entry` pour le dictionnaire comprehension
        self.sequential_history = {num: entry for num, entry in self.sequential_history.items() if num >= obsolete_game_limit}

    def analyze_and_set_smart_rules(self, initial_load: bool = False) -> List[str]:
        declencheur_counts = {}
        for data in self.inter_data:
            key = tuple(data['declencheur']) 
            declencheur_counts[key] = declencheur_counts.get(key, 0) + 1

        sorted_declencheurs = sorted(declencheur_counts.items(), key=lambda item: item[1], reverse=True)
        top_3 = [{'cards': list(d), 'count': c} for d, c in sorted_declencheurs[:3]]
        self.smart_rules = top_3
        
        if top_3: self.is_inter_mode_active = True
        elif not initial_load: self.is_inter_mode_active = False 

        self._save_data(self.is_inter_mode_active, 'inter_mode_status.json')
        self._save_data(self.smart_rules, 'smart_rules.json')
        return [f"{c['cards'][0]} {c['cards'][1]} (x{c['count']})" for c in top_3]

    def get_inter_status(self) -> Tuple[str, Optional[Dict]]:
        status_lines = ["**📋 HISTORIQUE D'APPRENTISSAGE INTER 🧠**\n"]
        total = len(self.inter_data) 
        status_lines.append(f"**Mode Intelligent:** {'✅ ON' if self.is_inter_mode_active else '❌ OFF'}")
        status_lines.append(f"**Données Q collectées:** {total}\n")

        if total > 0:
            status_lines.append("**Derniers (N-2 → Q):**")
            for entry in self.inter_data[-10:]:
                line = f"• N{entry['numero_resultat']} ({entry['carte_q']}) → N{entry['numero_declencheur']} ({entry['declencheur'][0]} {entry['declencheur'][1]})"
                status_lines.append(line)
        
        if self.is_inter_mode_active and self.smart_rules:
            status_lines.append("\n**🎯 Règles Actives (Top 3):**")
            for rule in self.smart_rules:
                status_lines.append(f"- {rule['cards'][0]} {rule['cards'][1]} (x{rule['count']})")

        keyboard = None
        if total > 0:
            txt = f"🔄 Re-analyser (Actif)" if self.is_inter_mode_active else f"✅ Appliquer Règles ({total} données)"
            keyboard = {'inline_keyboard': [[{'text': txt, 'callback_data': 'inter_apply'}], [{'text': "➡️ Mode Défaut", 'callback_data': 'inter_default'}]]}
        
        return "\n".join(status_lines), keyboard

    def can_make_prediction(self) -> bool:
        if not self.last_prediction_time: return True
        return time.time() > (self.last_prediction_time + self.prediction_cooldown)

    # --- Indicateurs ---
    def has_pending_indicators(self, message: str) -> bool:
        return '🕐' in message or '⏰' in message
        
    def has_completion_indicators(self, message: str) -> bool:
        return '✅' in message or '🔰' in message

    # --- LOGIQUE PRINCIPALE (AVEC CONFIANCE) ---
    def should_predict(self, message: str) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
        """
        Détermine si une prédiction doit être faite.
        RETOURNE 4 VALEURS: (statut, numéro_jeu, valeur_prédite, confiance)
        """
        if not self.target_channel_id: 
            return False, None, None, None # Retourne 4
             
        game_number = self.extract_game_number(message)
        if not game_number: 
            return False, None, None, None # Retourne 4

        self.collect_inter_data(game_number, message) 
        
        if self.has_pending_indicators(message): 
            return False, None, None, None # Retourne 4
            
        if not self.has_completion_indicators(message): 
            return False, None, None, None # Retourne 4
            
        predicted_value = None
        confidence = None # Variable pour stocker le pourcentage
        
        # Extraction
        g1_content = self.extract_first_parentheses_content(message)
        if not g1_content: 
            return False, None, None, None # Retourne 4
        
        g1_details = self.extract_card_details(g1_content)
        g1_values = [v for v, c in g1_details]
        
        all_matches = re.findall(r'\(([^)]*)\)', message)
        g2_content = all_matches[1] if len(all_matches) > 1 else ""
        g2_details = self.extract_card_details(g2_content)
        g2_values = [v for v, c in g2_details]
        
        # --- PRIORITÉ 0: INTER (100% ou Intelligent) ---
        if self.is_inter_mode_active and self.smart_rules:
            current_trigger = tuple(self.get_first_two_cards(g1_content))
            if any(tuple(rule['cards']) == current_trigger for rule in self.smart_rules):
                predicted_value = "Q"
                confidence = "100% 🧠" 
                logger.info(f"🔮 PRÉDICTION INTER: Règle intelligente.")
        
        # --- LOGIQUE STATIQUE (Si pas de prédiction INTER) ---
        if not predicted_value:
            all_high = HIGH_VALUE_CARDS 
            has_j_in_g1 = 'J' in g1_values
            has_k_in_g1 = 'K' in g1_values
            is_g2_weak = not any(v in all_high for v in g2_values)
            
            # --- RÈGLE 1 (99%): J seul dans G1 + G2 faible ---
            is_j_solo_high_in_g1 = (g1_values.count('J') == 1 and 
                                    not any(v in ['A', 'K', 'Q'] for v in g1_values))
            
            if is_j_solo_high_in_g1 and is_g2_weak:
                predicted_value = "Q"
                confidence = "99%"
                logger.info("🔮 RÈGLE 1 (99%): J seul + G2 faible.")

            # --- RÈGLE 5 (67%): Deux Valets (J) dans G1 ---
            elif g1_values.count('J') >= 2:
                predicted_value = "Q"
                confidence = "67%"
                logger.info("🔮 RÈGLE 5 (67%): Deux Valets (J).")

            # --- RÈGLE 2 (55%): K + J dans G1 + G2 faible ---
            elif has_k_in_g1 and has_j_in_g1 and is_g2_weak:
                predicted_value = "Q"
                confidence = "55%"
                logger.info("🔮 RÈGLE 2 (55%): K + J + G2 faible.")

            # --- RÈGLE 3 (45%): Faibles consécutives dans G1 ---
            elif not predicted_value:
                is_g1_weak = not any(v in all_high for v in g1_values)
                if is_g1_weak and len(g1_values) >= 2:
                    nums = sorted([self._get_card_numeric_value(v) for v in g1_values])
                    is_consecutive = all(nums[i] == nums[i-1] + 1 for i in range(1, len(nums)))
                    
                    if is_consecutive:
                        predicted_value = "Q"
                        confidence = "45%"
                        logger.info(f"🔮 RÈGLE 3 (45%): Faibles consécutives.")
            
            # --- RÈGLE 4 (41%): Total Jeu (Somme des valeurs) ≥ 45 ---
            if not predicted_value:
                total_sum = sum(self._get_card_numeric_value(v) for v in g1_values + g2_values)
                
                if total_sum >= 45:
                    predicted_value = "Q"
                    confidence = "41%"
                    logger.info(f"🔮 RÈGLE 4 (41%): Total {total_sum} ≥ 45.")

        # VÉRIFICATION COOLDOWN
        if predicted_value and not self.can_make_prediction():
            logger.warning("⏳ PRÉDICTION ÉVITÉE: Cooldown actif.")
            return False, None, None, None # Retourne 4

        # ENVOI (Déclenchement du gestionnaire)
        if predicted_value:
            msg_hash = hash(message)
            if msg_hash not in self.processed_messages:
                self.processed_messages.add(msg_hash)
                self.last_prediction_time = time.time()
                self._save_all_data()
                # 🟢 Retourne 4 valeurs : Cela déclenche l'envoi dans le gestionnaire principal
                return True, game_number, predicted_value, confidence

        return False, None, None, None # Retourne 4
        
    def make_prediction(self, game_number: int, predicted_value: str, confidence: str = "") -> str:
        """Génère le message avec la confiance incluse."""
        target_game = game_number + 2
        conf_str = f"({confidence})" if confidence else ""
        prediction_text = f"🔵{target_game}🔵:Valeur Q {conf_str} statut :⏳"
        
        self.predictions[target_game] = {
            'predicted_costume': 'Q',
            'status': 'pending',
            'predicted_from': game_number,
            'verification_count': 0,
            'message_text': prediction_text,
            'message_id': None,
            'confidence': confidence 
        }
        self._save_all_data()
        return prediction_text
        
    def _verify_prediction_common(self, text: str, is_edited: bool = False) -> Optional[Dict]:
        """Vérifie si le message contient le résultat pour une prédiction en attente (Q)."""
        game_number = self.extract_game_number(text)
        if not game_number or not self.predictions:
            return None

        for predicted_game in sorted(self.predictions.keys()):
            prediction = self.predictions[predicted_game]

            if prediction.get('status') != 'pending' or prediction.get('predicted_costume') != 'Q':
                continue

            verification_offset = game_number - predicted_game
            
            if 0 <= verification_offset <= 2:
                status_symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
                q_found = self.check_value_Q_in_first_parentheses(text)
                
                # Récupérer la confiance pour le message final
                conf_saved = prediction.get('confidence', '')
                conf_str = f"({conf_saved})" if conf_saved else ""
                
                if q_found:
                    # SUCCÈS
                    status_symbol = status_symbol_map[verification_offset]
                    updated_message = f"🔵{predicted_game}🔵:Valeur Q {conf_str} statut :{status_symbol}"
                    
                    prediction['status'] = f'correct_offset_{verification_offset}'
                    prediction['verification_count'] = verification_offset
                    prediction['final_message'] = updated_message
                    self._save_all_data()
                    
                    logger.info(f"🔍 ✅ SUCCÈS OFFSET +{verification_offset} - Dame (Q) trouvée au jeu {game_number}")
                    
                    return {
                        'type': 'edit_message',
                        'predicted_game': predicted_game,
                        'new_message': updated_message,
                    }
                elif verification_offset == 2 and not q_found:
                    # ÉCHEC
                    updated_message = f"🔵{predicted_game}🔵:Valeur Q {conf_str} statut :❌"

                    prediction['status'] = 'failed'
                    prediction['final_message'] = updated_message
                    self._save_all_data()
                    
                    logger.info(f"🔍 ❌ ÉCHEC OFFSET +2 - Rien trouvé, prédiction marquée: ❌")

                    return {
                        'type': 'edit_message',
                        'predicted_game': predicted_game,
                        'new_message': updated_message,
                    }
        return None
