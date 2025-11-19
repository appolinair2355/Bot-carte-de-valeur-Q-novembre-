# card_predictor.py (corrigé) — PARTIE 1/2
# Basé sur ton code collé — corrections appliquées pour /inter et prédictions.
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
CARD_SYMBOLS = [r"♠️", r"♥️", r"♦️", r"♣️", r"❤️"]  # variantes
# Confiances pour les règles statiques (mapping fourni)
CONFIDENCE_RULES = {
    "2.1": 98,  # valet solitaire
    "2.2": 57,  # deux valets
    "2.3": 97,  # total points >=45
    "2.4": 60,  # 4 jeux sans Q
    "2.5": 70,  # combinaison 8-9-10
    "2.6": 70,  # bloc final
    "default_static": 70
}

class CardPredictor:
    """Gère la logique de prédiction de carte Dame (Q) et la vérification."""

    def __init__(self):
        # Données de persistance (Prédictions et messages)
        self.predictions = self._load_data('predictions.json')
        self.processed_messages = self._load_data('processed.json', is_set=True)
        self.last_prediction_time = self._load_data('last_prediction_time.json', is_scalar=True)

        # Configuration dynamique des canaux
        self.config_data = self._load_data('channels_config.json')
        self.target_channel_id = self.config_data.get('target_channel_id', None)
        self.prediction_channel_id = self.config_data.get('prediction_channel_id', None)

        # --- Logique INTER (N-2 -> Q à N) ---
        self.sequential_history: Dict[int, Dict] = self._load_data('sequential_history.json')
        self.inter_data: List[Dict] = self._load_data('inter_data.json')

        # Statut et Règles
        self.is_inter_mode_active = self._load_data('inter_mode_status.json', is_scalar=True)
        self.smart_rules = self._load_data('smart_rules.json')  # Stocke les Top 3 actifs
        self.prediction_cooldown = 30

        # si historique existe et smart_rules pas initialisées, calcule-les
        if self.inter_data and not self.smart_rules:
            try:
                self.analyze_and_set_smart_rules(initial_load=True)
            except Exception:
                logger.exception("Erreur analyse initiale smart_rules")

    # --- Persistance des Données (JSON) ---
    def _load_data(self, filename: str, is_set: bool = False, is_scalar: bool = False) -> Any:
        """Charge les données depuis un fichier JSON."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if is_set:
                    return set(data)
                if is_scalar:
                    if filename == 'inter_mode_status.json':
                        return data.get('active', False)
                    return data
                if filename == 'sequential_history.json':
                    # Convertir les clés string en int si nécessaire
                    if isinstance(data, dict):
                        try:
                            return {int(k): v for k, v in data.items()}
                        except Exception:
                            return data
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"⚠️ Fichier {filename} non trouvé ou vide. Initialisation par défaut.")
            if is_set: return set()
            if is_scalar and filename == 'inter_mode_status.json': return False
            if is_scalar: return 0.0
            if filename == 'inter_data.json': return []
            if filename == 'sequential_history.json': return {}
            if filename == 'smart_rules.json': return []
            if filename == 'predictions.json': return {}
            return {}
        except Exception as e:
            logger.error(f"❌ Erreur critique de chargement de {filename}: {e}")
            return set() if is_set else (False if filename == 'inter_mode_status.json' else {})

    def _save_data(self, data: Any, filename: str):
        """Sauvegarde les données dans un fichier JSON."""
        if filename == 'inter_mode_status.json':
            data_to_save = {'active': self.is_inter_mode_active}
        elif isinstance(data, set):
            data_to_save = list(data)
        else:
            data_to_save = data

        try:
            with open(filename, 'w') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erreur critique de sauvegarde de {filename}: {e}. Problème de permissions ou de disque.")

    def _save_all_data(self):
        """Sauvegarde tous les états persistants."""
        # Normaliser les clés des prédictions en string pour cohérence JSON
        normalized_predictions = {}
        for k, v in self.predictions.items():
            normalized_predictions[str(k)] = v
        self._save_data(normalized_predictions, 'predictions.json')
        self._save_data(self.processed_messages, 'processed.json')
        self._save_data(self.last_prediction_time, 'last_prediction_time.json')
        self._save_data(self.inter_data, 'inter_data.json')
        # sequential_history keys -> strings for JSON
        seq_save = {str(k): v for k, v in self.sequential_history.items()}
        self._save_data(seq_save, 'sequential_history.json')
        self._save_data(self.is_inter_mode_active, 'inter_mode_status.json')
        self._save_data(self.smart_rules, 'smart_rules.json')
        # channels config
        self.config_data['target_channel_id'] = self.target_channel_id
        self.config_data['prediction_channel_id'] = self.prediction_channel_id
        self._save_data(self.config_data, 'channels_config.json')

    def _save_channels_config(self):
        """Sauvegarde les IDs de canaux dans channels_config.json."""
        self.config_data['target_channel_id'] = self.target_channel_id
        self.config_data['prediction_channel_id'] = self.prediction_channel_id
        self._save_data(self.config_data, 'channels_config.json')

    def set_channel_id(self, channel_id: int, channel_type: str):
        """Met à jour les IDs de canal et sauvegarde."""
        if channel_type == 'source':
            self.target_channel_id = channel_id
            logger.info(f"💾 Canal SOURCE mis à jour: {channel_id}")
        elif channel_type == 'prediction':
            self.prediction_channel_id = channel_id
            logger.info(f"💾 Canal PRÉDICTION mis à jour: {channel_id}")
        else:
            return False
        self._save_channels_config()
        return True

    # --- Logique d'Extraction (Mise à jour pour #N et #n) ---
    def extract_game_number(self, message: str) -> Optional[int]:
        """Extrait le numéro du jeu, reconnaissant formats usuels (#n51., 🔵51🔵, etc.)."""
        if not message:
            return None
        # Emoji style
        m = re.search(r'🔵\s*(\d{1,6})\s*🔵', message)
        if m:
            return int(m.group(1))
        # #n51. ou #N51. (avec ou sans point)
        m = re.search(r'#\s*[nN]\s*\.?\s*(\d{1,6})\.?', message)
        if m:
            return int(m.group(1))
        # fallback #51
        m = re.search(r'#\s*(\d{1,6})', message)
        if m:
            return int(m.group(1))
        return None

    def extract_first_parentheses_content(self, message: str) -> Optional[str]:
        """Extrait le contenu de la première parenthèse, tolérant l'absence d'espaces."""
        if not message:
            return None
        start = message.find("(")
        if start == -1:
            return None
        end = message.find(")", start)
        if end == -1:
            return None
        return message[start+1:end].strip()

    def extract_card_details(self, content: str) -> List[Tuple[str, str]]:
        """Extrait la valeur et le costume des cartes, tolérant '❤️' -> '♥️'."""
        if not content:
            return []
        normalized = content.replace("❤️", "♥️").replace("❤", "♥️")
        # capture 10 or digits or letters and one of suit emojis (some variation selectors included)
        card_pattern = r'(10|[2-9]|[AKQJ])(♠️|♥️|♦️|♣️)'
        matches = re.findall(card_pattern, normalized)
        return [(v.upper(), s) for v, s in matches]

    def get_first_two_cards(self, content: str) -> List[str]:
        """Renvoie les deux premières cartes pour le déclencheur INTER."""
        card_details = self.extract_card_details(content)
        return [f"{v}{s}" for v, s in card_details[:2]]

    def check_value_Q_in_first_parentheses(self, message: str) -> Optional[str]:
        """Vérifie si la Dame (Q) est dans le premier groupe et retourne 'Q♣️' par ex."""
        content = self.extract_first_parentheses_content(message)
        if not content:
            return None
        for v, s in self.extract_card_details(content):
            if v == "Q":
                return f"{v}{s}"
        return None

    # --- Logique INTER (Mode Intelligent) - Collecte robuste + anti-doublon ---
    def collect_inter_data(self, game_number: int, message: str):
        """Collecte séquentielle: toujours mémoriser 2 premières cartes, puis si Q à N relier N-2."""
        if not isinstance(game_number, int):
            try:
                game_number = int(game_number)
            except Exception:
                return

        first_group = self.extract_first_parentheses_content(message)
        # Toujours mémoriser les deux premières cartes si trouvées
        if first_group:
            first_two = self.get_first_two_cards(first_group)
            if len(first_two) == 2:
                # store as int key
                self.sequential_history[int(game_number)] = {'cartes': first_two, 'date': datetime.now().isoformat()}
                # save progressive
                self._save_all_data()

        # Si Q présent (message finalisé est contrôlé par should_predict avant appel à make_prediction)
        q_card = self.check_value_Q_in_first_parentheses(message)
        if q_card:
            n_minus_2 = game_number - 2
            trigger_entry = self.sequential_history.get(n_minus_2)
            if trigger_entry:
                # anti doublon
                if any(entry.get('numero_resultat') == game_number for entry in self.inter_data):
                    logger.debug(f"INTER: doublon N={game_number}, ignore")
                    return
                new_entry = {
                    'numero_resultat': game_number,
                    'declencheur': trigger_entry['cartes'],
                    'numero_declencheur': n_minus_2,
                    'carte_q': q_card,
                    'date_resultat': datetime.now().isoformat()
                }
                self.inter_data.append(new_entry)
                self._save_all_data()
                logger.info(f"[INTER] Enregistré: N={game_number} déclencheur N-2={n_minus_2} ({trigger_entry['cartes']})")

        # nettoyage history -> garder dernières 200 entrées max (par sécurité)
        min_keep = max(0, game_number - 200)
        self.sequential_history = {k: v for k, v in self.sequential_history.items() if k >= min_keep}
        # save is already called on additions (safe)
        # card_predictor.py (corrigé) — PARTIE 2/2

    # ------------------------------------------------------------
    # Calcul des règles intelligentes (TOP3) et status / affichage
    # ------------------------------------------------------------
    def analyze_and_set_smart_rules(self, initial_load: bool = False) -> List[str]:
        """Analyse inter_data et construit self.smart_rules = [{'cards':[c1,c2],'count':n}, ...]"""
        counts = {}
        for entry in self.inter_data:
            key = tuple(entry.get('declencheur', []))
            if len(key) != 2:
                continue
            counts[key] = counts.get(key, 0) + 1
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        self.smart_rules = [{'cards': list(k), 'count': v} for k, v in sorted_items[:3]]
        # si des règles trouvées, on peut activer (ou conserver l'état d'initial_load)
        if self.smart_rules:
            self.is_inter_mode_active = True
        elif not initial_load:
            self.is_inter_mode_active = False
        self._save_all_data()
        return [f"{r['cards'][0]} {r['cards'][1]} (x{r['count']})" for r in self.smart_rules]

    def get_inter_status(self) -> Tuple[str, Optional[Dict]]:
        """Retour formaté pour /inter (message, keyboard)."""
        lines = ["**📋 HISTORIQUE INTER (N-2 → Q à N)**\n"]
        lines.append(f"Mode Intelligent: {'✅ OUI' if self.is_inter_mode_active else '❌ NON'}")
        lines.append(f"Données collectées: {len(self.inter_data)}\n")
        if self.inter_data:
            lines.append("Derniers enregistrements (max 10):")
            for entry in self.inter_data[-10:]:
                decl = ", ".join(entry['declencheur'])
                lines.append(f"N : {entry['numero_resultat']} — Déclencheur N{entry['numero_declencheur']}: {decl} — Carte: {entry['carte_q']}")
            keyboard = {
                'inline_keyboard': [
                    [{'text': '🧠 Appliquer la règle intelligente', 'callback_data': 'inter_apply'}],
                    [{'text': '📘 Règle par défaut', 'callback_data': 'inter_default'}]
                ]
            }
        else:
            lines.append("Aucun déclencheur enregistré.")
            keyboard = None
        return "\n".join(lines), keyboard

    # ------------------------------------------------------------
    # Helpers: cooldown et indicateurs
    # ------------------------------------------------------------
    def can_make_prediction(self) -> bool:
        if not self.last_prediction_time:
            return True
        try:
            return time.time() > (float(self.last_prediction_time) + float(self.prediction_cooldown))
        except Exception:
            return True

    def has_pending_indicators(self, message: str) -> bool:
        return '🕐' in message or '⏰' in message

    def has_completion_indicators(self, message: str) -> bool:
        return '✅' in message or '🔰' in message

    # ------------------------------------------------------------
    # check_static_rules: renvoie une confiance int ou None
    # ------------------------------------------------------------
    def check_static_rules(self, message: str, game_number: int) -> Optional[int]:
        """Implémente les règles 2.1..2.6 et retourne la confiance (int) si correspond."""
        first_group = self.extract_first_parentheses_content(message)
        if not first_group:
            return None
        card_details = self.extract_card_details(first_group)
        values = [v for v, s in card_details]

        # 2.1 Valet solitaire (exactement 1 J et aucune autre carte haute A/K/Q dans ce groupe)
        if values.count('J') == 1 and not any(v in ['A', 'K', 'Q'] for v in values if v != 'J'):
            return CONFIDENCE_RULES['2.1']

        # 2.2 Deux valets ou plus
        if values.count('J') >= 2:
            return CONFIDENCE_RULES['2.2']

        # 2.3 Total points (#T) >= 45
        total = None
        m = re.search(r'#T\s*(\d+)', message)
        if m:
            total = int(m.group(1))
        if total is not None and total >= 45:
            return CONFIDENCE_RULES['2.3']

        # 2.4 Manque consécutif de Q >=4 (se base sur inter_data historique)
        missing = 0
        for prev in range(game_number - 1, game_number - 5, -1):
            if not any(e.get('numero_resultat') == prev for e in self.inter_data):
                missing += 1
        if missing >= 4:
            return CONFIDENCE_RULES['2.4']

        # 2.5 combinaison 8-9-10 dans G1 ou G2
        groups = self.extract_all_parentheses_groups(message)
        g1_vals = [v for v, s in self.extract_card_details(groups[0])] if len(groups) >= 1 else []
        g2_vals = [v for v, s in self.extract_card_details(groups[1])] if len(groups) >= 2 else []
        if {'8', '9', '10'}.issubset(set(g1_vals + g2_vals)):
            return CONFIDENCE_RULES['2.5']

        # 2.6 Bloc final (K+J in G1) OR tag O/R OR double weakness
        condA = ('K' in values and 'J' in values)
        condB = bool(re.search(r'\bO\b|\bR\b', message))
        def is_weak(vals):
            return not any(v in HIGH_VALUE_CARDS for v in vals)
        condC = False
        prev = self.sequential_history.get(game_number - 1)
        if prev:
            prev_vals = []
            for c in prev.get('cartes', []):
                m = re.match(r'(\d+|[AKQJ])', c)
                if m:
                    prev_vals.append(m.group(1))
            condC = is_weak(values) and is_weak(prev_vals)
        if condA or condB or condC:
            return CONFIDENCE_RULES['2.6']

        return None

    def extract_all_parentheses_groups(self, message: str) -> List[str]:
        """Renvoie la liste de tous les contenus entre parenthèses."""
        return re.findall(r'\(([^)]*)\)', message)

    # ------------------------------------------------------------
    # should_predict: RENVOIE (bool, game_number, confidence:int)
    # ------------------------------------------------------------
    def should_predict(self, message: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        - collecte INTER en continu (séquential_history)
        - n'analyse pour prédiction que si message 'finalisé'
        - renvoie la confiance (int) si correspond à une règle -> handlers appelle make_prediction(game_number, confidence)
        """
        if not self.target_channel_id:
            return False, None, None

        game_number = self.extract_game_number(message)
        if not game_number:
            return False, None, None

        # Toujours mémoriser les deux premières cartes
        try:
            self.collect_inter_data(game_number, message)
        except Exception:
            logger.exception("Erreur collect_inter_data dans should_predict")

        # Ne pas prédire si message en attente
        if self.has_pending_indicators(message):
            return False, None, None

        # Considérer finalisé si indicateur explicite, OU (pas d'indicateur d'attente et présence de #T)
        finalized = self.has_completion_indicators(message) or ('#T' in message and not self.has_pending_indicators(message))
        if not finalized:
            logger.info("Prédiction bloquée: message non-finalisé (pas d'indicateur ✅/🔰 ou #T absent).")
            return False, None, None

        # Éviter duplicates
        msg_hash = hash(message)
        if msg_hash in self.processed_messages:
            return False, None, None

        # Cooldown
        if not self.can_make_prediction():
            logger.info("Prédiction bloquée: cooldown actif.")
            return False, None, None

        # Extraction G1 et G2
        g1 = self.extract_first_parentheses_content(message)
        if not g1:
            return False, None, None
        g1_cards = self.extract_card_details(g1)
        g1_values = [v for v, s in g1_cards]

        groups = self.extract_all_parentheses_groups(message)
        g2 = groups[1] if len(groups) > 1 else ""
        g2_values = [v for v, s in self.extract_card_details(g2)]

        # --- LOGIQUE INTER (prioritaire) ---
        if self.is_inter_mode_active and self.smart_rules:
            two = self.get_first_two_cards(g1)
            # compute total counts for confidence calc
            total_counts = sum(r.get('count', 0) for r in self.smart_rules) or 1
            for rule in self.smart_rules:
                if rule.get('cards') == two:
                    confidence = int(round((rule.get('count', 0) / total_counts) * 100))
                    # mark processed and save
                    self.processed_messages.add(msg_hash)
                    self.last_prediction_time = time.time()
                    self._save_all_data()
                    logger.info(f"✅ should_predict: INTER match {two} -> conf {confidence}%")
                    return True, game_number, confidence

        # --- LOGIQUE STATIQUE (si INTER n'a pas prédit) ---
        static_conf = self.check_static_rules(message, game_number)
        if static_conf:
            self.processed_messages.add(msg_hash)
            self.last_prediction_time = time.time()
            self._save_all_data()
            logger.info(f"✅ should_predict: STATIQUE matched -> conf {static_conf}%")
            return True, game_number, static_conf

        return False, None, None

    # ------------------------------------------------------------
    # make_prediction: prend confidence:int et enregistre la prédiction
    # ------------------------------------------------------------
    def make_prediction(self, game_number: int, confidence: int) -> str:
        """Génère le message de prédiction et l'enregistre (les handlers s'attendent au texte renvoyé)."""
        target_game = int(game_number) + 2
        key = str(target_game)  # use string key for JSON safety
        prediction_text = f"🔵{target_game}🔵:Valeur Q statut :⏳ ({int(confidence)}%)"

        self.predictions[key] = {
            'predicted_costume': 'Q',
            'status': 'pending',
            'predicted_from': int(game_number),
            'verification_count': 0,
            'message_text': prediction_text,
            'message_id': None,
            'confidence': int(confidence),
            'created_at': datetime.now().isoformat()
        }
        # update last prediction time
        self.last_prediction_time = time.time()
        self._save_all_data()
        logger.info(f"💬 make_prediction: sauvegardée prédiction pour {target_game} conf {confidence}%")
        return prediction_text

    # ------------------------------------------------------------
    # Verification / édition des prédictions (_verify_prediction_common)
    # ------------------------------------------------------------
    def _verify_prediction_common(self, text: str, is_edited: bool = False) -> Optional[Dict]:
        """Vérifie un message finalisé pour confirmer/infirmer une prédiction."""
        game_number = self.extract_game_number(text)
        if not game_number:
            return None

        # Parcourir une copie pour éviter mutation lors d'itération
        for key_str, prediction in list(self.predictions.items()):
            try:
                predicted_game = int(key_str)
            except Exception:
                # si la clé est déjà int, handle it
                try:
                    predicted_game = int(key_str)
                except:
                    continue

            if prediction.get('status') != 'pending' or prediction.get('predicted_costume') != 'Q':
                continue

            offset = game_number - predicted_game
            if offset < 0 or offset > 2:
                continue

            q_found = self.check_value_Q_in_first_parentheses(text)
            conf = prediction.get('confidence', CONFIDENCE_RULES.get('default_static', 70))

            # Succès
            if q_found:
                symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
                sym = symbol_map.get(offset, "✅")
                new_msg = f"🔵{predicted_game}🔵:Valeur Q statut :{sym} ({conf}%)"
                prediction['status'] = f'correct_offset_{offset}'
                prediction['verification_count'] = offset
                prediction['final_message'] = new_msg
                prediction['verified_at'] = datetime.now().isoformat()
                self._save_all_data()
                logger.info(f"🔍 Vérification: SUCCÈS pour prédiction {predicted_game} offset {offset}")
                # Retourne une action d'édition (handlers éditera le message existant en se basant sur message_id stocké)
                return {'type': 'edit_message', 'predicted_game': predicted_game, 'new_message': new_msg}

            # Echec à offset 2
            if offset == 2 and not q_found:
                new_msg = f"🔵{predicted_game}🔵:Valeur Q statut :❌ ({conf}%)"
                prediction['status'] = 'failed'
                prediction['final_message'] = new_msg
                prediction['verified_at'] = datetime.now().isoformat()
                self._save_all_data()
                logger.info(f"🔍 Vérification: ÉCHEC pour prédiction {predicted_game} (offset 2)")
                return {'type': 'edit_message', 'predicted_game': predicted_game, 'new_message': new_msg}

        return None

    # ------------------------------------------------------------
    # Utilitaires divers
    # ------------------------------------------------------------
    def reset_inter(self):
        self.inter_data = []
        self.smart_rules = []
        self.is_inter_mode_active = False
        self._save_all_data()
        logger.info("INTER reset effectué.")
        return True

# --- Fin du fichier corrigé ---
