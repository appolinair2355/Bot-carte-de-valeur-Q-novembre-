# -*- coding: utf-8 -*-
"""
Card Predictor – Version Complète Mise à Jour

Fonctionnalités principales :
- Règles statiques 2.1 → 2.6 intégrées avec niveau de confiance
- Le bot n’analyse les messages QUE lorsqu’ils sont finalisés (✅ ou 🔰)
- La confiance (%) apparaît dans les prédictions et dans les mises à jour
- Mode intelligent INTER conservé
- Apprentissage N-2 → N (Q)
- Vérification offset 0/1/2 avec émoji + confiance
"""

import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# CONSTANTES
# -----------------------------
HIGH_VALUE_CARDS = ["A", "K", "Q", "J"]

CARD_SYMBOLS = [r"♠️", r"♥️", r"♦️", r"♣️", r"❤️"]

CONFIDENCE_RULES = {
    "2.1": 98,   # Valet solitaire
    "2.2": 57,   # Deux Valets
    "2.3": 97,   # Total de points élevé (>=45)
    "2.4": 60,   # Manque consécutif de Q
    "2.5": 70,   # Combinaison 8-9-10
    "2.6": 70,   # Bloc final (K+J, Tag O/R, Double faiblesse)
}


# ================================================================
#                       CLASSE PRINCIPALE
# ================================================================

class CardPredictor:
    """Gestion complète de la prédiction Q."""

    def __init__(self):
        # --------- Stockage local JSON ---------
        self.predictions: Dict = self._load_data("predictions.json")
        self.processed_messages: set = self._load_data("processed.json", is_set=True)
        self.last_prediction_time: float = self._load_data("last_prediction_time.json", is_scalar=True)

        # --------- Configuration canaux ---------
        self.config_data = self._load_data("channels_config.json")
        self.target_channel_id = self.config_data.get("target_channel_id")
        self.prediction_channel_id = self.config_data.get("prediction_channel_id")

        # --------- Mode INTER / Historique ---------
        self.sequential_history: Dict[int, Dict] = self._load_data("sequential_history.json")
        self.inter_data: List[Dict] = self._load_data("inter_data.json")
        self.is_inter_mode_active = self._load_data("inter_mode_status.json", is_scalar=True)
        self.smart_rules: List[Dict] = self._load_data("smart_rules.json")

        self.prediction_cooldown = 30  # anti-spam

        if self.inter_data and not self.is_inter_mode_active:
            self.analyze_and_set_smart_rules(initial_load=True)

    # ================================================================
    #                        JSON PERSISTENCE
    # ================================================================

    def _load_data(self, filename: str, is_set=False, is_scalar=False):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            if is_set:
                return set(data)

            if is_scalar:
                if filename == "inter_mode_status.json":
                    return data.get("active", False)
                return float(data)

            if filename == "sequential_history.json":
                return {int(k): v for k, v in data.items()}

            return data

        except (FileNotFoundError, json.JSONDecodeError):
            if is_set:
                return set()
            if is_scalar:
                return 0
            if filename == "inter_data.json":
                return []
            if filename == "sequential_history.json":
                return {}
            return {}

    def _save_data(self, data, filename: str):
        if filename == "inter_mode_status.json":
            data = {"active": self.is_inter_mode_active}

        if isinstance(data, set):
            data = list(data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _save_all(self):
        self._save_data(self.predictions, "predictions.json")
        self._save_data(self.processed_messages, "processed.json")
        self._save_data(self.last_prediction_time, "last_prediction_time.json")
        self._save_data(self.inter_data, "inter_data.json")
        self._save_data(self.sequential_history, "sequential_history.json")
        self._save_data(self.smart_rules, "smart_rules.json")
        self._save_data(self.is_inter_mode_active, "inter_mode_status.json")

    # ================================================================
    #                           EXTRACTION
    # ================================================================

    def extract_game_number(self, message: str) -> Optional[int]:
        match = re.search(r"#N(\d+)\.", message, re.IGNORECASE)
        if not match:
            match = re.search(r"🔵(\d+)🔵", message)
        if match:
            return int(match.group(1))
        return None

    def extract_first_group(self, message: str) -> Optional[str]:
        m = re.search(r"\(([^)]*)\)", message)
        return m.group(1).strip() if m else None

    def extract_all_groups(self, message: str) -> List[str]:
        return re.findall(r"\(([^)]*)\)", message)

    def extract_cards(self, content: str) -> List[Tuple[str, str]]:
        if not content:
            return []
        content = content.replace("❤️", "♥️")
        matches = re.findall(r"(\d+|[AKQJ])(♠️|♥️|♦️|♣️)", content)
        return [(v.upper(), s) for v, s in matches]

    def get_first_two_cards(self, content: str) -> List[str]:
        cards = self.extract_cards(content)
        return [f"{v}{s}" for v, s in cards[:2]]

    def extract_total_points(self, message: str) -> Optional[int]:
        m = re.search(r"#T(\d+)", message)
        return int(m.group(1)) if m else None

    def has_Q_in_group1(self, message: str):
        g1 = self.extract_first_group(message)
        cards = self.extract_cards(g1)
        for v, s in cards:
            if v == "Q":
                return v, s
        return None

    # ================================================================
    #               CONDITION : MESSAGE FINALISÉ ? (OBLIGATOIRE)
    # ================================================================

    def is_finalized(self, message: str) -> bool:
        return "✅" in message or "🔰" in message

    # ================================================================
    #                    INTER — APPRENTISSAGE N-2 → N
    # ================================================================

    def collect_inter_data(self, game_number: int, message: str):
        """INTER NE DOIT SE FAIRE QUE SUR MESSAGES FINALISÉS."""
        if not self.is_finalized(message):
            return

        g1 = self.extract_first_group(message)
        if not g1:
            return

        first_two = self.get_first_two_cards(g1)
        if len(first_two) == 2:
            self.sequential_history[game_number] = {
                "cartes": first_two,
                "date": datetime.now().isoformat(),
            }

        q_info = self.has_Q_in_group1(message)
        if not q_info:
            return

        trig_game = game_number - 2
        if trig_game not in self.sequential_history:
            return

        # Anti-doublon
        if any(entry["numero_resultat"] == game_number for entry in self.inter_data):
            return

        self.inter_data.append({
            "numero_resultat": game_number,
            "numero_declencheur": trig_game,
            "declencheur": self.sequential_history[trig_game]["cartes"],
            "carte_q": f"{q_info[0]}{q_info[1]}",
            "date_resultat": datetime.now().isoformat()
        })

        # Nettoyage
        limit = game_number - 60
        self.sequential_history = {
            k: v for k, v in self.sequential_history.items() if k >= limit
        }

        self._save_all()

    # ================================================================
    #                  ANALYSE DES TOP 3 RÈGLES INTER
    # ================================================================

    def analyze_and_set_smart_rules(self, initial_load=False):
        counts = {}
        for entry in self.inter_data:
            trig = tuple(entry["declencheur"])
            counts[trig] = counts.get(trig, 0) + 1

        sorted_rules = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_rules[:3]

        self.smart_rules = [{"cards": list(k), "count": v} for k, v in top3]

        if top3:
            self.is_inter_mode_active = True
        elif not initial_load:
            self.is_inter_mode_active = False

        self._save_all()

        return [
            f"{r['cards'][0]} {r['cards'][1]} (x{r['count']})"
            for r in self.smart_rules
]
        # ================================================================
    #                      UTILITAIRES
    # ================================================================

    def can_make_prediction(self) -> bool:
        """Retourne True si le cooldown est passé."""
        if not self.last_prediction_time:
            return True
        try:
            return time.time() > (float(self.last_prediction_time) + self.prediction_cooldown)
        except Exception:
            return True

    def mark_processed(self, message: str):
        """Marque un message comme traité en stockant son hash."""
        h = hash(message)
        self.processed_messages.add(h)
        self._save_all()

    # ================================================================
    #                    LOGIQUE DE PRÉDICTION (PARTIE 2)
    # ================================================================

    def should_predict(self, message: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Analyse un message FINALISÉ et décide si une prédiction Q doit être faite.
        Retour: (do_predict, game_number, confidence_percent) — si do_predict True, faire make_prediction(game_number, confidence)
        """
        # 0. Vérifier canal
        if not self.target_channel_id:
            return False, None, None

        # 1. Extraction du numéro
        game_number = self.extract_game_number(message)
        if not game_number:
            return False, None, None

        # 2. N'ANALYSE QUE SI MESSAGE FINALISÉ
        if not self.is_finalized(message):
            # Collecte INTER seulement sur messages finalisés (déjà géré ailleurs)
            return False, None, None

        # 3. Collecter données INTER (N-2 -> N)
        self.collect_inter_data(game_number, message)

        # 4. Anti-duplication: message déjà traité ?
        msg_hash = hash(message)
        if msg_hash in self.processed_messages:
            return False, None, None

        # 5. En période de cooldown ?
        if not self.can_make_prediction():
            logger.warning("⏳ PRÉDICTION BLOQUÉE: cooldown actif.")
            return False, None, None

        # 6. Extractions utiles
        g1_content = self.extract_first_group(message)
        all_groups = self.extract_all_groups(message)
        g2_content = all_groups[1] if len(all_groups) > 1 else ""
        g1_cards = [v for v, s in self.extract_cards(g1_content)]
        g2_cards = [v for v, s in self.extract_cards(g2_content)]

        # 7. Priorité : Mode INTER (règles apprises)
        if self.is_inter_mode_active and self.smart_rules:
            current_trigger = self.get_first_two_cards(g1_content)
            current_trigger_tuple = tuple(current_trigger)
            for rule in self.smart_rules:
                if tuple(rule["cards"]) == current_trigger_tuple:
                    confidence = 100  # les règles INTER sont considérées très fiables ; tu peux ajuster
                    logger.info(f"🔮 PRÉDICTION INTER: déclencheur trouvé {current_trigger} -> Q (conf {confidence}%)")
                    # marquer processed et enregistrer prédiction
                    self.mark_processed(message)
                    self.last_prediction_time = time.time()
                    self._save_all()
                    return True, game_number, confidence

        # 8. Bloc Règles Statiques dans l'ordre demandé

        # Règle 2.1 : Valet solitaire (G1 contient exactement 1 J et aucune A/K/Q)
        if g1_cards.count("J") == 1 and not any(v in ["A", "K", "Q"] for v in g1_cards if v != "J"):
            confidence = CONFIDENCE_RULES["2.1"]
            logger.info(f"🔮 PRÉDICTION STATIQUE 2.1: Valet solitaire -> conf {confidence}%")
            self.mark_processed(message)
            self.last_prediction_time = time.time()
            self._save_all()
            return True, game_number, confidence

        # Règle 2.2 : Deux Valets ou plus (G1)
        if g1_cards.count("J") >= 2:
            confidence = CONFIDENCE_RULES["2.2"]
            logger.info(f"🔮 PRÉDICTION STATIQUE 2.2: Deux Valets -> conf {confidence}%")
            self.mark_processed(message)
            self.last_prediction_time = time.time()
            self._save_all()
            return True, game_number, confidence

        # Règle 2.3 : Total des points du jeu (#T >= 45)
        total_points = self.extract_total_points(message)
        if total_points is not None and total_points >= 45:
            confidence = CONFIDENCE_RULES["2.3"]
            logger.info(f"🔮 PRÉDICTION STATIQUE 2.3: Total #T{total_points} >=45 -> conf {confidence}%")
            self.mark_processed(message)
            self.last_prediction_time = time.time()
            self._save_all()
            return True, game_number, confidence

        # Règle 2.4 : Manque consécutif de Q (>=4 jeux N-1..N-4 sans Q en G1)
        missing_q_count = 0
        for prev in range(game_number - 1, game_number - 5, -1):
            prev_entry = self.sequential_history.get(prev)
            if not prev_entry:
                # si pas d'info, on considère absence de Q (estimation conservative)
                missing_q_count += 1
            else:
                # vérifier si Q était présent dans le message correspondant (si on a stocké ça)
                # On ne stocke pas le texte original; on utilise inter_data pour connaître les résultats
                # Si inter_data contient un entry pour prev where carte_q exists -> Q was found
                found_q = any(e["numero_resultat"] == prev for e in self.inter_data)
                if not found_q:
                    missing_q_count += 1

        if missing_q_count >= 4:
            confidence = CONFIDENCE_RULES["2.4"]
            logger.info(f"🔮 PRÉDICTION STATIQUE 2.4: Manque consécutif de Q ({missing_q_count}) -> conf {confidence}%")
            self.mark_processed(message)
            self.last_prediction_time = time.time()
            self._save_all()
            return True, game_number, confidence

        # Règle 2.5 : Présence des 8,9,10 dans G1 ou G2 (peut être réparti)
        found_vals = set(g1_cards + g2_cards)
        if {"8", "9", "10"}.issubset(found_vals):
            confidence = CONFIDENCE_RULES["2.5"]
            logger.info(f"🔮 PRÉDICTION STATIQUE 2.5: Combinaison 8-9-10 -> conf {confidence}%")
            self.mark_processed(message)
            self.last_prediction_time = time.time()
            self._save_all()
            return True, game_number, confidence

        # Règle 2.6 : Bloc de fin (au moins une des sous-conditions)
        # A) K & J dans G1
        condA = ("K" in g1_cards) and ("J" in g1_cards)
        # B) tags O ou R dans le message
        condB = bool(re.search(r"\b[OR]\b", message)) or (" O " in message) or (" R " in message)
        # C) Double faiblesse consécutive: G1 current weak and previous G1 weak
        def is_group_weak(cards_list):
            return not any(v in HIGH_VALUE_CARDS for v in cards_list)

        condC = False
        if is_group_weak(g1_cards):
            prev_entry = self.sequential_history.get(game_number - 1)
            if prev_entry:
                prev_values = [re.match(r"(\d+|[AKQJ])", c).group(1) for c in prev_entry["cartes"] if re.match(r"(\d+|[AKQJ])", c)]
                condC = is_group_weak(prev_values)

        if condA or condB or condC:
            confidence = CONFIDENCE_RULES["2.6"]
            logger.info(f"🔮 PRÉDICTION STATIQUE 2.6: Bloc final (A:{condA},B:{condB},C:{condC}) -> conf {confidence}%")
            self.mark_processed(message)
            self.last_prediction_time = time.time()
            self._save_all()
            return True, game_number, confidence

        # Aucune règle déclenchée
        return False, None, None

    # ================================================================
    #                    CRÉATION / ENREGISTREMENT PRÉDICTION
    # ================================================================

    def make_prediction(self, game_number: int, confidence: int) -> str:
        """
        Enregistre la prédiction pour game_number+2 avec la confiance.
        Retourne le texte du message à poster.
        """
        target_game = game_number + 2
        message_text = f\"🔵{target_game}🔵:Valeur Q statut :⏳ ({confidence}%)\"

        # Stocker en tant que clé str pour JSON convivial
        key = str(target_game)
        self.predictions[key] = {
            "predicted_costume": "Q",
            "status": "pending",
            "predicted_from": game_number,
            "verification_count": 0,
            "message_text": message_text,
            "message_id": None,
            "confidence": int(confidence),
            "created_at": datetime.now().isoformat(),
        }
        self._save_all()
        logger.info(f\"💾 Prédiction créée pour {target_game} (depuis {game_number}) conf {confidence}%\")
        return message_text

    # ================================================================
    #                    VÉRIFICATION DES PRÉDICTIONS
    # ================================================================

    def _verify_prediction_common(self, text: str, is_edited: bool = False) -> Optional[Dict]:
        """
        Vérifie si le message (finalisé) correspond au résultat d'une prédiction en attente.
        Retourne dict d'action si une mise à jour est nécessaire:
        { 'type': 'edit_message', 'predicted_game': X, 'new_message': '...' }
        """
        # Ne rien faire si message non finalisé
        if not self.is_finalized(text):
            return None

        game_number = self.extract_game_number(text)
        if not game_number:
            return None

        # Parcourir prédictions en attente
        keys = list(self.predictions.keys())
        for k in keys:
            try:
                predicted_game = int(k)
            except ValueError:
                continue
            prediction = self.predictions.get(str(predicted_game))
            if not prediction:
                continue

            if prediction.get("status") != "pending" or prediction.get("predicted_costume") != "Q":
                continue

            verification_offset = game_number - predicted_game  # 0..2 expected
            if 0 <= verification_offset <= 2:
                q_found = self.has_Q_in_group1(text)
                confidence = prediction.get("confidence", None) or 0

                status_symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
                if q_found:
                    status_symbol = status_symbol_map.get(verification_offset, "✅")
                    updated_message = f\"🔵{predicted_game}🔵:Valeur Q statut :{status_symbol} ({confidence}%)\"

                    prediction["status"] = f\"correct_offset_{verification_offset}\"
                    prediction["verification_count"] = verification_offset
                    prediction["final_message"] = updated_message
                    prediction["finalized_at"] = datetime.now().isoformat()
                    self._save_all()

                    logger.info(f\"🔍 ✅ SUCCÈS OFFSET +{verification_offset} - Dame trouvée au jeu {game_number} (pred {predicted_game})\")
                    return {
                        "type": "edit_message",
                        "predicted_game": predicted_game,
                        "new_message": updated_message,
                    }
                else:
                    # Si on est à offset 2 et toujours rien => échec
                    if verification_offset == 2:
                        updated_message = f\"🔵{predicted_game}🔵:Valeur Q statut :❌ ({confidence}%)\"
                        prediction["status"] = "failed"
                        prediction["final_message"] = updated_message
                        prediction["finalized_at"] = datetime.now().isoformat()
                        self._save_all()
                        logger.info(f\"🔍 ❌ ÉCHEC OFFSET +2 - Rien trouvé pour prédiction {predicted_game}\")
                        return {
                            "type": "edit_message",
                            "predicted_game": predicted_game,
                            "new_message": updated_message,
                        }
        return None

    # ================================================================
    #                         FIN DE LA CLASSE
    # ================================================================
