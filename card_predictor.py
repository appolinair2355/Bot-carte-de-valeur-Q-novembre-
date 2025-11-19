# card_predictor.py
# Version corrigée (partie 1/2)
# Compatible avec handlers.py fourni et format du canal source #n51. 18(8♣️Q♦️7♣️) ...

import os
import re
import json
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HIGH_VALUE_CARDS = ["A", "K", "Q", "J"]

CONFIDENCE_RULES = {
    "2.1": 98,
    "2.2": 57,
    "2.3": 97,
    "2.4": 60,
    "2.5": 70,
    "2.6": 70,
}

# -------------------------
# Helpers JSON
# -------------------------
def _safe_load(filename: str, default):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_save(filename: str, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Erreur sauvegarde {filename}: {e}")

# -------------------------
# Card parsing helpers (robustes pour emojis)
# -------------------------
# We will accept card tokens like "Q♣️", "10♠️", "7♦️", "A❤️" etc.
_card_regex = re.compile(r"(10|[2-9]|[AKQJ])\s*(♠️|♣️|♥️|♦️)")

def parse_cards_from_group(group_text: str) -> List[str]:
    """
    Extracts normalized card strings from group text.
    Accepts formats with or without spaces, e.g.:
      (8♣️Q♦️7♣️) or (8 ♣️ Q ♦️ 7 ♣️) or (8♣ Q♦ 7♣)
    Returns list like ['8♣️', 'Q♦️', '7♣️']
    """
    if not group_text:
        return []
    # Normalize variant hearts emoji (some sources use different heart codepoints)
    normalized = group_text.replace("❤", "♥️").replace("❤️", "♥️")
    # Find all matches; allow missing variation selector by using fallback
    matches = _card_regex.findall(normalized)
    cards = []
    for value, suit in matches:
        # ensure suit kept as the same emoji character that appears (suit may contain variation selector)
        cards.append(f"{value}{suit}")
    return cards

# -------------------------
# Classe principale
# -------------------------
class CardPredictor:
    def __init__(self):
        # persistence files
        self.predictions: Dict[str, Dict[str, Any]] = _safe_load("predictions.json", {})
        self.processed: set = set(_safe_load("processed.json", []))
        self.last_prediction_time: float = _safe_load("last_prediction_time.json", 0.0)

        cfg = _safe_load("channels_config.json", {})
        self.target_channel_id = cfg.get("target_channel_id")
        self.prediction_channel_id = cfg.get("prediction_channel_id")

        # inter / sequential history
        self.inter_data: List[Dict[str, Any]] = _safe_load("inter_data.json", [])
        raw_seq = _safe_load("sequential_history.json", {})
        try:
            self.sequential_history = {int(k): v for k, v in raw_seq.items()}
        except Exception:
            self.sequential_history = raw_seq if isinstance(raw_seq, dict) else {}

        self.smart_rules: List[Dict[str, Any]] = _safe_load("smart_rules.json", [])
        self.is_inter_mode_active: bool = _safe_load("inter_mode_status.json", {"active": False}).get("active", False)

        self.prediction_cooldown = 30  # seconds

        # compatibility
        if not isinstance(self.predictions, dict):
            self.predictions = {}
        if not isinstance(self.inter_data, list):
            self.inter_data = []
        if not isinstance(self.smart_rules, list):
            self.smart_rules = []

        # compute smart rules from inter_data if needed
        if self.inter_data and not self.smart_rules:
            try:
                self.analyze_and_set_smart_rules(initial_load=True)
            except Exception:
                logger.exception("Erreur initial analyze_and_set_smart_rules")

    # -------------------------
    # Persistence API (used by handlers)
    # -------------------------
    def _save_data(self, data, filename: str):
        """
        Generic save utility (kept for handlers compatibility)
        """
        try:
            if filename in ("inter_mode_status.json", "inter_mode.json"):
                if isinstance(data, bool):
                    _safe_save(filename, {"active": data})
                else:
                    _safe_save(filename, data)
            else:
                _safe_save(filename, data)
        except Exception as e:
            logger.error(f"_save_data failed for {filename}: {e}")

    def _save_all_data(self):
        try:
            _safe_save("predictions.json", self.predictions)
            _safe_save("processed.json", list(self.processed))
            _safe_save("last_prediction_time.json", self.last_prediction_time)
            _safe_save("channels_config.json", {"target_channel_id": self.target_channel_id, "prediction_channel_id": self.prediction_channel_id})
            _safe_save("inter_data.json", self.inter_data)
            _safe_save("sequential_history.json", {str(k): v for k, v in self.sequential_history.items()})
            _safe_save("smart_rules.json", self.smart_rules)
            _safe_save("inter_mode_status.json", {"active": bool(self.is_inter_mode_active)})
        except Exception as e:
            logger.exception(f"_save_all_data error: {e}")

    # -------------------------
    # Channel setters
    # -------------------------
    def set_channel_id(self, channel_id: int, channel_type: str):
        if channel_type == "source":
            self.target_channel_id = channel_id
            logger.info(f"Canal SOURCE défini: {channel_id}")
        elif channel_type == "prediction":
            self.prediction_channel_id = channel_id
            logger.info(f"Canal PREDICTION défini: {channel_id}")
        else:
            logger.warning(f"Type de canal inconnu: {channel_type}")
            return False
        self._save_all_data()
        return True

    # -------------------------
    # Extraction robustes
    # -------------------------
    def extract_game_number(self, text: str) -> Optional[int]:
        """
        Accept formats:
         - '#n51.' or '#n51' or '#N51.' or '#N51'
         - '🔵51🔵' if present
         - fallback to 'n51' or 'N51' or standalone number patterns '#51'
        """
        if not text:
            return None
        # try emoji style first
        m = re.search(r"🔵\s*(\d{1,6})\s*🔵", text)
        if m:
            return int(m.group(1))
        # try hash-n style: #n51. or #n51
        m = re.search(r"#n\s*\.?\s*(\d{1,6})\.?", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # try compact '#n51.' or '#n51'
        m = re.search(r"#n(\d{1,6})", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # try generic '#N51' or '#51'
        m = re.search(r"#N?(\d{1,6})", text)
        if m:
            return int(m.group(1))
        return None

    def extract_first_group(self, text: str) -> Optional[str]:
        """
        Extracts the FIRST parenthesis group content, tolerant to no spaces:
        e.g. "18(8♣️Q♦️7♣️) - ..." -> returns "8♣️Q♦️7♣️"
        """
        if not text:
            return None
        # find first '(' and matching ')'
        start = text.find("(")
        if start == -1:
            return None
        # find closing ')'
        end = text.find(")", start)
        if end == -1:
            return None
        return text[start+1:end]

    def extract_all_groups(self, text: str) -> List[str]:
        """
        Returns list of all parenthesis contents in order.
        """
        if not text:
            return []
        groups = []
        idx = 0
        while True:
            s = text.find("(", idx)
            if s == -1:
                break
            e = text.find(")", s)
            if e == -1:
                break
            groups.append(text[s+1:e])
            idx = e + 1
        return groups

    def extract_cards(self, group: str) -> List[Tuple[str, str]]:
        """
        Returns list of tuples (value, suit) for group content.
        Uses parse_cards_from_group internally.
        """
        cards = parse_cards_from_group(group)
        result = []
        for c in cards:
            # split value and suit (suit is last char(s))
            m = re.match(r"^(10|[2-9A KQJ]+?)(.+)$", c)
            # safe split: value is digits or A K Q J, suit is emoji (non-ascii)
            m = re.match(r"^(10|[2-9]|[AKQJ])(.+)$", c)
            if m:
                val = m.group(1).upper()
                suit = m.group(2)
                result.append((val, suit))
        return result

    def get_first_two_cards(self, group: str) -> List[str]:
        cards = self.extract_cards(group)
        return [f"{v}{s}" for v, s in cards[:2]]


# End of PARTIE 1/2
# ============================================================
    #  INTER MODE — Collecte automatique des déclencheurs
    # ============================================================
    def collect_inter_data(self, game_number: int, group_text: str):
        """
        Analyse la première main d’un jeu pour détecter la présence d’une Dame (Q).
        Format du canal source :
            #n51. 18(8♣️Q♦️7♣️) - 14(K♠️K♣️6♠️)
        On collecte N-2 → Q(N)
        """
        if not group_text:
            return

        cards = self.extract_cards(group_text)
        if not cards:
            return

        # Vérifier si Dame présente
        q_card = None
        for v, s in cards:
            if v == "Q":
                q_card = f"{v}{s}"
                break

        if not q_card:
            return

        # Les deux premières cartes = déclencheur
        triggers = []
        for v, s in cards[:2]:
            triggers.append(f"{v}{s}")

        entry = {
            "game_number": game_number,
            "triggers": triggers,
            "queen": q_card
        }

        self.inter_data.append(entry)
        self._save_data(self.inter_data, "inter_data.json")
        logger.info(f"[INTER] Ajout déclencheur N={game_number} : {triggers} → {q_card}")

    # ============================================================
    #  INTER — Calcul TOP 3
    # ============================================================
    def analyze_and_set_smart_rules(self, initial_load=False):
        """
        Analyse inter_data et détermine les déclencheurs les plus fréquents
        pour produire les règles intelligentes (TOP3).
        """
        if not self.inter_data:
            self.smart_rules = []
            self.is_inter_mode_active = False
            self._save_all_data()
            return []

        freq = {}
        for entry in self.inter_data:
            key = tuple(entry["triggers"])
            freq[key] = freq.get(key, 0) + 1

        sorted_rules = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_rules[:3]

        rules = []
        rank = 1
        for triggers, count in top3:
            rules.append({
                "rank": rank,
                "triggers": list(triggers),
                "count": count
            })
            rank += 1

        self.smart_rules = rules
        self.is_inter_mode_active = True
        self._save_all_data()

        logger.info(f"[INTER] Nouvelles règles intelligentes : {self.smart_rules}")
        return self.smart_rules

    # ============================================================
    #  INTER — Affichage pour la commande /inter
    # ============================================================
    def format_inter_list(self) -> str:
        """Retourne un texte complet de la liste des déclencheurs enregistrés."""
        if not self.inter_data:
            return "📬 Aucun déclencheur Dame (Q) enregistré."

        lines = []
        for item in self.inter_data:
            N = item["game_number"]
            t = item["triggers"]
            q = item["queen"]
            lines.append(f"N : {N}\nDéclencheur : {t[0]}, {t[1]}\nCarte : {q}\n")

        return "\n".join(lines)

    def get_inter_status(self):
        """
        Retourne :
          - message
          - inline keyboard
        utilisable directement par handlers.py
        """
        list_text = self.format_inter_list()

        status = "Mode Intelligent Actif: " + ("✅ OUI" if self.is_inter_mode_active else "❌ NON")
        message = f"{status}\n\n{list_text}"

        keyboard = {
            "inline_keyboard": [
                [{"text": "🧠 Appliquer la règle intelligente", "callback_data": "inter_apply"}],
                [{"text": "📘 Règle par défaut", "callback_data": "inter_default"}]
            ]
        }

        return message, keyboard

    # ============================================================
    # Should Predict (appelé par handlers)
    # ============================================================
    def should_predict(self, message_text: str):
        """
        Retourne :
            should_predict, game_number, predicted_value
        """
        game_number = self.extract_game_number(message_text)
        if not game_number:
            return False, None, None

        first_group = self.extract_first_group(message_text)
        if not first_group:
            return False, None, None

        # Collecte INTER
        self.collect_inter_data(game_number, first_group)

        cards = self.extract_cards(first_group)
        if not cards:
            return False, game_number, None

        # MODE INTELLIGENT
        if self.is_inter_mode_active and self.smart_rules:
            two = [f"{v}{s}" for v, s in cards[:2]]

            for r in self.smart_rules:
                if r["triggers"] == two:
                    return True, game_number, "Q"

        # MODE PAR DÉFAUT
        v1 = cards[0][0]
        if v1 == "J":
            return True, game_number, "Q"

        return False, game_number, None

    # ============================================================
    # Prédiction + pourcentage
    # ============================================================
    def make_prediction(self, game_number: int, predicted_value: str) -> str:
        if not predicted_value:
            return ""

        rule_name = "2.1" if self.is_inter_mode_active else "2.2"
        confidence = CONFIDENCE_RULES.get(rule_name, 70)

        next_game = game_number + 2

        text = (
            f"🔮 *Prédiction pour #n{next_game}*\n"
            f"➡️ Carte attendue : *{predicted_value}*\n"
            f"📊 Confiance : *{confidence}%*\n"
            f"⚙️ Mode : {'Intelligent' if self.is_inter_mode_active else 'Défaut'}"
        )

        self.predictions[str(next_game)] = {
            "game_number": next_game,
            "predicted_value": predicted_value,
            "confidence": confidence,
            "timestamp": time.time(),
            "message_id": None
        }

        self._save_all_data()
        return text

    # ============================================================
    # Vérification / confirmation / édition
    # ============================================================
    def _verify_prediction_common(self, message_text: str, is_edited=False):
        """
        Vérifie si un message contient la réponse d’un jeu prédit.
        """
        game_number = self.extract_game_number(message_text)
        if not game_number:
            return None

        key = str(game_number)
        if key not in self.predictions:
            return None

        # On analyse la première main
        first_group = self.extract_first_group(message_text)
        if not first_group:
            return None

        cards = self.extract_cards(first_group)
        if not cards:
            return None

        real = cards[0][0]  # valeur première carte réelle
        predicted = self.predictions[key]["predicted_value"]

        if real == predicted:
            final_text = f"✅ *#n{game_number}* — Prédiction correcte !"
        else:
            final_text = f"❌ *#n{game_number}* — Prédiction ratée."

        return {
            "type": "edit_message",
            "predicted_game": game_number,
            "new_message": final_text
        }
