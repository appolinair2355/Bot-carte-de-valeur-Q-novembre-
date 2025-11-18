# -*- coding: utf-8 -*-
"""
Card Predictor – Version Propre et Corrigée
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
    "2.1": 98,
    "2.2": 57,
    "2.3": 97,
    "2.4": 60,
    "2.5": 70,
    "2.6": 70,
}


# ================================================================
#                       CLASSE PRINCIPALE
# ================================================================

class CardPredictor:
    """Gestion complète de la prédiction Q."""

    def __init__(self):
        self.predictions: Dict = self._load_data("predictions.json")
        self.processed_messages: set = self._load_data("processed.json", is_set=True)
        self.last_prediction_time: float = self._load_data("last_prediction_time.json", is_scalar=True)

        self.config_data = self._load_data("channels_config.json")
        self.target_channel_id = self.config_data.get("target_channel_id")
        self.prediction_channel_id = self.config_data.get("prediction_channel_id")

        self.sequential_history: Dict[int, Dict] = self._load_data("sequential_history.json")
        self.inter_data: List[Dict] = self._load_data("inter_data.json")
        self.is_inter_mode_active = self._load_data("inter_mode_status.json", is_scalar=True)
        self.smart_rules: List[Dict] = self._load_data("smart_rules.json")

        self.prediction_cooldown = 30

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
        match = re.search(r"#N(\d+)\.", message)
        if not match:
            match = re.search(r"🔵(\d+)🔵", message)
        return int(match.group(1)) if match else None

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
    #                    MESSAGE FINALISÉ ?
    # ================================================================

    def is_finalized(self, message: str) -> bool:
        return "✅" in message or "🔰" in message

    # ================================================================
    #           INTER — APPRENTISSAGE N-2 → N (Q)
    # ================================================================

    def collect_inter_data(self, game_number: int, message: str):
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

        if any(entry["numero_resultat"] == game_number for entry in self.inter_data):
            return

        self.inter_data.append({
            "numero_resultat": game_number,
            "numero_declencheur": trig_game,
            "declencheur": self.sequential_history[trig_game]["cartes"],
            "carte_q": f"{q_info[0]}{q_info[1]}",
            "date_resultat": datetime.now().isoformat()
        })

        cutoff = game_number - 60
        self.sequential_history = {
            k: v for k, v in self.sequential_history.items() if k >= cutoff
        }

        self._save_all()

    # ================================================================
    #                    TOP 3 RÈGLES INTELLIGENTES
    # ================================================================

    def analyze_and_set_smart_rules(self, initial_load=False):
        counts = {}
        for entry in self.inter_data:
            trig = tuple(entry["declencheur"])
            counts[trig] = counts.get(trig, 0) + 1

        sorted_rules = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_rules[:3]

        self.smart_rules = [{"cards": list(k), "count": v} for k, v in top3]
        self.is_inter_mode_active = bool(top3 or initial_load)

        self._save_all()

        return [
            f"{r['cards'][0]} {r['cards'][1]} (x{r['count']})"
            for r in self.smart_rules
        ]
        # ================================================================
    #                        RÈGLES STATIQUES
    # ================================================================

    def is_group_weak(self, cards: List[Tuple[str, str]]) -> bool:
        """Un groupe est faible s'il ne contient aucune carte A, K, Q, J."""
        for v, s in cards:
            if v in HIGH_VALUE_CARDS:
                return False
        return True

    def count_missing_consecutive_Q(self, current_game: int) -> int:
        """Compte le nombre de jeux consécutifs sans Q dans G1 avant le jeu actuel."""
        count = 0
        game = current_game - 1

        while game >= current_game - 20:
            if game not in self.sequential_history:
                break
            msg_group = self.sequential_history[game].get("cartes", [])
            q_found = any("Q" in card for card in msg_group)
            if q_found:
                break
            count += 1
            game -= 1

        return count

    # ================================================================
    #             should_predict — Règles 2.1 → 2.6
    # ================================================================

    def should_predict(self, message: str, game_number: int) -> Optional[int]:
        """Renvoie la confiance (%) si une règle déclenche, sinon None."""

        if not self.is_finalized(message):
            return None

        # Extraction
        g1_str = self.extract_first_group(message)
        if not g1_str:
            return None

        cards = self.extract_cards(g1_str)
        values_only = [v for v, s in cards]
        total_points = self.extract_total_points(message)

        # Règle 2.1 — Valet solitaire
        if values_only.count("J") == 1 and not any(v in ["A", "K", "Q"] for v in values_only):
            return CONFIDENCE_RULES["2.1"]

        # Règle 2.2 — Deux Valets
        if values_only.count("J") >= 2:
            return CONFIDENCE_RULES["2.2"]

        # Règle 2.3 — Total de points élevé
        if total_points and total_points >= 45:
            return CONFIDENCE_RULES["2.3"]

        # Règle 2.4 — Manque consécutif de Q
        miss_count = self.count_missing_consecutive_Q(game_number)
        if miss_count >= 4:
            return CONFIDENCE_RULES["2.4"]

        # Règle 2.5 — Combinaison 8-9-10
        if {"8", "9", "10"}.issubset(set(values_only)):
            return CONFIDENCE_RULES["2.5"]

        # Règle 2.6 — Bloc final
        weak_current = self.is_group_weak(cards)
        prev_cards = self.sequential_history.get(game_number - 1, {}).get("cartes", [])
        weak_prev = all(v[0] not in HIGH_VALUE_CARDS for v in [tuple([c[0], c[1:]]) for c in prev_cards]) if prev_cards else False

        if ("K" in values_only and "J" in values_only) \
            or ("O" in message or "R" in message) \
            or (weak_current and weak_prev):
            return CONFIDENCE_RULES["2.6"]

        return None

    # ================================================================
    #                make_prediction — AJOUT AVEC CONFIANCE
    # ================================================================

    def make_prediction(self, game_number: int, confidence: int) -> str:
        """Crée la prédiction Q pour game_number+2 avec affichage confiance."""
        target_game = game_number + 2
        msg = f"🔵{target_game}🔵:Valeur Q statut :⏳ ({confidence}%)"

        key = str(target_game)
        self.predictions[key] = {
            "predicted_costume": "Q",
            "status": "pending",
            "predicted_from": game_number,
            "verification_count": 0,
            "message_text": msg,
            "message_id": None,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
        }

        self._save_all()

        logger.info(
            f"💾 Prédiction créée pour {target_game} (depuis {game_number}) conf {confidence}%"
        )

        return msg

    # ================================================================
    #                  VÉRIFICATION DES PRÉDICTIONS
    # ================================================================

    def _verify_prediction_common(self, text: str, is_edited: bool = False) -> Optional[Dict]:

        if not self.is_finalized(text):
            return None

        game_number = self.extract_game_number(text)
        if not game_number:
            return None

        for k in list(self.predictions.keys()):
            predicted_game = int(k)
            prediction = self.predictions.get(k)

            if not prediction:
                continue
            if prediction.get("status") != "pending":
                continue
            if prediction.get("predicted_costume") != "Q":
                continue

            # Offset
            offset = game_number - predicted_game
            if offset < 0 or offset > 2:
                continue

            q_found = self.has_Q_in_group1(text)
            confidence = prediction.get("confidence", 0)

            status_map = {
                0: "0️⃣",
                1: "1️⃣",
                2: "2️⃣",
            }

            # ---- SUCCÈS ----
            if q_found:
                symbol = status_map.get(offset, "0️⃣")
                updated = f"🔵{predicted_game}🔵:Valeur Q statut :{symbol} ({confidence}%)"

                prediction["status"] = f"correct_offset_{offset}"
                prediction["verification_count"] = offset
                prediction["final_message"] = updated
                prediction["finalized_at"] = datetime.now().isoformat()

                self._save_all()

                logger.info(
                    f"✔️ SUCCÈS +{offset} – Q trouvée au jeu {game_number} (Prédiction {predicted_game})"
                )

                return {
                    "type": "edit_message",
                    "predicted_game": predicted_game,
                    "new_message": updated,
                }

            # ---- ÉCHEC +2 ----
            if offset == 2 and not q_found:
                updated = f"🔵{predicted_game}🔵:Valeur Q statut :❌ ({confidence}%)"

                prediction["status"] = "failed"
                prediction["final_message"] = updated
                prediction["finalized_at"] = datetime.now().isoformat()

                self._save_all()

                logger.info(
                    f"❌ ÉCHEC +2 – aucune Dame trouvée (Prédiction {predicted_game})"
                )

                return {
                    "type": "edit_message",
                    "predicted_game": predicted_game,
                    "new_message": updated,
                }

        return None

# ================================================================
#                         FIN DU FICHIER
# ================================================================
