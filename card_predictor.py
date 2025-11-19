# card_predictor.py (corrigé) — PARTIE 1/2
# Reconstructed and corrected version (Part 1)

import re
import logging
import json
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------
# Constants & Rules
# -------------------------
HIGH_VALUE_CARDS = ["A", "K", "Q", "J"]
CONFIDENCE_RULES = {
    "2.1": 98,
    "2.2": 57,
    "2.3": 97,
    "2.4": 60,
    "2.5": 70,
    "2.6": 70,
    "default_static": 70,
}

# -------------------------
# Helpers: file IO
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
# Card parsing helpers
# -------------------------
# Accept tokens like "Q♣️", "10♠️", "7♦️", "A❤️"
_CARD_PATTERN = re.compile(r'(10|[2-9]|[AKQJ])(♠️|♥️|♦️|♣️)')

def parse_cards_from_group(group_text: str) -> List[str]:
    if not group_text:
        return []
    normalized = group_text.replace("❤️", "♥️").replace("❤", "♥️")
    matches = _CARD_PATTERN.findall(normalized)
    return [f"{v}{s}" for v, s in matches]

# -------------------------
# Main class
# -------------------------
class CardPredictor:
    def __init__(self):
        # persistence files
        self.predictions: Dict[str, Dict[str, Any]] = _safe_load("predictions.json", {})
        # processed messages hashes (to avoid duplicates)
        self.processed_messages = set(_safe_load("processed.json", []))
        self.last_prediction_time: float = _safe_load("last_prediction_time.json", 0.0)

        cfg = _safe_load("channels_config.json", {})
        self.target_channel_id = cfg.get("target_channel_id")
        self.prediction_channel_id = cfg.get("prediction_channel_id")

        # inter history: sequential_history stores {game_number(int): {"cartes":[card1,card2], "date":...}}
        self.sequential_history: Dict[int, Dict[str, Any]] = {}
        raw_seq = _safe_load("sequential_history.json", {})
        try:
            # convert keys to int
            self.sequential_history = {int(k): v for k, v in raw_seq.items()}
        except Exception:
            self.sequential_history = raw_seq if isinstance(raw_seq, dict) else {}

        self.inter_data: List[Dict[str, Any]] = _safe_load("inter_data.json", [])
        self.smart_rules: List[Dict[str, Any]] = _safe_load("smart_rules.json", [])
        self.is_inter_mode_active: bool = _safe_load("inter_mode_status.json", {"active": False}).get("active", False)

        self.prediction_cooldown = 30  # seconds

        # normalize types
        if not isinstance(self.predictions, dict):
            self.predictions = {}
        if not isinstance(self.inter_data, list):
            self.inter_data = []
        if not isinstance(self.smart_rules, list):
            self.smart_rules = []

        # if there is inter_data but no smart_rules, compute them
        if self.inter_data and not self.smart_rules:
            try:
                self.analyze_and_set_smart_rules(initial_load=True)
            except Exception:
                logger.exception("Erreur initiale analyse_and_set_smart_rules")

    # -------------------------
    # Persistence helpers
    # -------------------------
    def _save_all_data(self):
        try:
            # ensure prediction keys are strings
            preds = {str(k): v for k, v in self.predictions.items()}
            _safe_save("predictions.json", preds)
            _safe_save("processed.json", list(self.processed_messages))
            _safe_save("last_prediction_time.json", self.last_prediction_time)
            _safe_save("inter_data.json", self.inter_data)
            # sequential_history keys -> strings
            seq_serial = {str(k): v for k, v in self.sequential_history.items()}
            _safe_save("sequential_history.json", seq_serial)
            _safe_save("smart_rules.json", self.smart_rules)
            _safe_save("inter_mode_status.json", {"active": bool(self.is_inter_mode_active)})
            _safe_save("channels_config.json", {"target_channel_id": self.target_channel_id, "prediction_channel_id": self.prediction_channel_id})
        except Exception as e:
            logger.exception(f"_save_all_data error: {e}")

    # -------------------------
    # Channel config
    # -------------------------
    def set_channel_id(self, channel_id: int, channel_type: str):
        if channel_type == "source":
            self.target_channel_id = channel_id
        elif channel_type == "prediction":
            self.prediction_channel_id = channel_id
        else:
            return False
        self._save_all_data()
        return True

    # -------------------------
    # Extraction utilities
    # -------------------------
    def extract_game_number(self, text: str) -> Optional[int]:
        if not text:
            return None
        # emoji style
        m = re.search(r"🔵\s*(\d{1,6})\s*🔵", text)
        if m:
            return int(m.group(1))
        # #n51. or #n51
        m = re.search(r"#\s*[nN]\s*\.?\s*(\d{1,6})\.?", text)
        if m:
            return int(m.group(1))
        # fallback #51
        m = re.search(r"#\s*(\d{1,6})", text)
        if m:
            return int(m.group(1))
        return None

    def extract_first_group(self, text: str) -> Optional[str]:
        if not text:
            return None
        s = text.find("(")
        if s == -1:
            return None
        e = text.find(")", s)
        if e == -1:
            return None
        return text[s+1:e]

    def extract_cards(self, group: str) -> List[Tuple[str, str]]:
        if not group:
            return []
        cards = parse_cards_from_group(group)
        result = []
        for c in cards:
            m = re.match(r"^(10|[2-9]|[AKQJ])(.+)$", c)
            if m:
                val = m.group(1).upper()
                suit = m.group(2)
                result.append((val, suit))
        return result

    def get_first_two_cards(self, group: str) -> List[str]:
        cards = self.extract_cards(group)
        return [f"{v}{s}" for v, s in cards[:2]]

    def has_Q_in_group(self, group: str) -> Optional[str]:
        cards = self.extract_cards(group)
        for v, s in cards:
            if v == "Q":
                return f"{v}{s}"
        return None

    def extract_all_groups(self, text: str) -> List[str]:
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

    # -------------------------
    # INTER collection logic
    # -------------------------
    def collect_inter_data(self, game_number: int, message_text: str):
        """
        - Always store first two cards of G1 into sequential_history[game_number]
        - If message contains Q in G1, attempt to find N-2 in sequential_history and
          if found, append inter_data entry linking trigger -> Q at N
        """
        try:
            if not isinstance(game_number, int):
                game_number = int(game_number)
        except Exception:
            return

        g1 = self.extract_first_group(message_text)
        if g1:
            first_two = self.get_first_two_cards(g1)
            if len(first_two) == 2:
                # store as int key
                self.sequential_history[int(game_number)] = {"cartes": first_two, "date": datetime.now().isoformat()}
                # persist progressively
                self._save_all_data()

        # If Q in group1, attempt to link to N-2
        q_card = self.has_Q_in_group(g1) if g1 else None
        if q_card:
            n_minus_2 = int(game_number) - 2
            trigger = self.sequential_history.get(n_minus_2)
            if trigger:
                # avoid duplicate for same result number
                if any(e.get("numero_resultat") == int(game_number) for e in self.inter_data):
                    return
                entry = {
                    "numero_resultat": int(game_number),
                    "numero_declencheur": n_minus_2,
                    "declencheur": trigger["cartes"],
                    "carte_q": q_card,
                    "date_resultat": datetime.now().isoformat()
                }
                self.inter_data.append(entry)
                self._save_all_data()
                logger.info(f"[INTER] enregistré: N={game_number} déclencheur N-2={n_minus_2} -> {trigger['cartes']}")

        # trim sequential_history to recent window (keep last 500)
        try:
            min_keep = max(0, int(game_number) - 500)
            self.sequential_history = {k: v for k, v in self.sequential_history.items() if k >= min_keep}
        except Exception:
            pass

    # -------------------------
    # Smart rules analysis
    # -------------------------
    def analyze_and_set_smart_rules(self, initial_load: bool = False):
        counts = {}
        for entry in self.inter_data:
            key = tuple(entry.get("declencheur", []))
            if len(key) != 2:
                continue
            counts[key] = counts.get(key, 0) + 1
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        self.smart_rules = [{"cards": list(k), "count": v} for k, v in sorted_items[:3]]
        if self.smart_rules:
            self.is_inter_mode_active = True
        elif not initial_load:
            self.is_inter_mode_active = False
        self._save_all_data()
        return self.smart_rules
        # ---------------------------------------------------------
    # INTER status for /inter command
    # ---------------------------------------------------------
    def get_inter_status(self):
        lines = []
        lines.append("📋 **HISTORIQUE INTER (Déclencheur N-2 → Q à N)**\n")
        lines.append(f"Mode Intelligent : {'🟢 ACTIVÉ' if self.is_inter_mode_active else '🔴 DÉSACTIVÉ'}")
        lines.append(f"Entrées enregistrées : {len(self.inter_data)}\n")

        if not self.inter_data:
            lines.append("Aucun déclencheur enregistré.")
            keyboard = {
                "inline_keyboard": [
                    [{"text": "📘 Activer règles par défaut", "callback_data": "inter_default"}]
                ]
            }
            return "\n".join(lines), keyboard

        # show last 10 entries
        lines.append("Dernières entrées :")
        for entry in self.inter_data[-10:]:
            decl = ", ".join(entry["declencheur"])
            lines.append(
                f"N : {entry['numero_resultat']} — Déclencheur N-2 ({entry['numero_declencheur']}) : {decl} — Carte : {entry['carte_q']}"
            )

        keyboard = {
            "inline_keyboard": [
                [{"text": "🧠 Appliquer règle intelligente", "callback_data": "inter_apply"}],
                [{"text": "📘 Règles par défaut", "callback_data": "inter_default"}],
            ]
        }
        return "\n".join(lines), keyboard

    # ---------------------------------------------------------
    # Helpers: cooldown and indicators
    # ---------------------------------------------------------
    def can_make_prediction(self) -> bool:
        if not self.last_prediction_time:
            return True
        try:
            return time.time() > float(self.last_prediction_time) + self.prediction_cooldown
        except Exception:
            return True

    def has_pending_indicators(self, text: str) -> bool:
        return "🕐" in text or "⏰" in text

    def has_completion_indicators(self, text: str) -> bool:
        return "🔰" in text or "✅" in text

    # ---------------------------------------------------------
    # Static rule evaluation
    # ---------------------------------------------------------
    def check_static_rules(self, text: str, game_number: int):
        """
        Returns confidence if a static rule matches.
        """
        g1 = self.extract_first_group(text)
        if not g1:
            return None

        cards = self.extract_cards(g1)
        values = [v for v, s in cards]

        # 2.1 — Single J, no A/K/Q
        if values.count("J") == 1 and not any(v in ["A", "K", "Q"] for v in values if v != "J"):
            return CONFIDENCE_RULES["2.1"]

        # 2.2 — Two or more J
        if values.count("J") >= 2:
            return CONFIDENCE_RULES["2.2"]

        # 2.3 — total >= 45
        m = re.search(r"#T\s*(\d+)", text)
        if m:
            total = int(m.group(1))
            if total >= 45:
                return CONFIDENCE_RULES["2.3"]

        # 2.4 — Missing Q >= 4
        missing = 0
        for prev in range(game_number - 1, game_number - 5, -1):
            if not any(e.get("numero_resultat") == prev for e in self.inter_data):
                missing += 1
        if missing >= 4:
            return CONFIDENCE_RULES["2.4"]

        # 2.5 — 8,9,10 present in G1 or G2
        groups = self.extract_all_groups(text)
        g_values = []
        for g in groups:
            g_values.extend([v for v, _ in self.extract_cards(g)])
        if {"8", "9", "10"}.issubset(set(g_values)):
            return CONFIDENCE_RULES["2.5"]

        # 2.6 — K+J in G1 or weaknesses etc
        if "K" in values and "J" in values:
            return CONFIDENCE_RULES["2.6"]

        return None

    # ---------------------------------------------------------
    # Main: should_predict   ✔ FIXED: NO DOUBLE PREDICTION
    # ---------------------------------------------------------
    def should_predict(self, message_text: str):
        if not self.target_channel_id:
            return False, None, None

        game_number = self.extract_game_number(message_text)
        if not game_number:
            return False, None, None

        # Always collect INTER data
        try:
            self.collect_inter_data(game_number, message_text)
        except Exception:
            logger.exception("Erreur collect_inter_data")

        # Ignore non-final messages
        if self.has_pending_indicators(message_text):
            return False, None, None

        finalized = (
            self.has_completion_indicators(message_text)
            or ("#T" in message_text and not self.has_pending_indicators(message_text))
        )
        if not finalized:
            return False, None, None

        # Avoid duplicates
        h = hash(message_text)
        if h in self.processed_messages:
            return False, None, None

        # Check cooldown
        if not self.can_make_prediction():
            return False, None, None

        # ---------------------------
        # STOP DOUBLE PREDICTION HERE
        # ---------------------------
        target_game = game_number + 2
        if str(target_game) in self.predictions:
            logger.info(f"‼️ Prédiction déjà existante pour {target_game}, pas de double.")
            return False, None, None

        # ---------------------------
        # INTER MODE PRIORITY
        # ---------------------------
        if self.is_inter_mode_active and self.smart_rules:
            g1 = self.extract_first_group(message_text)
            two_cards = self.get_first_two_cards(g1)
            total_count = sum(r["count"] for r in self.smart_rules) or 1

            for r in self.smart_rules:
                if r["cards"] == two_cards:
                    confidence = int(round((r["count"] / total_count) * 100))
                    self.processed_messages.add(h)
                    self.last_prediction_time = time.time()
                    self._save_all_data()
                    return True, game_number, confidence

        # ---------------------------
        # STATIC RULES
        # ---------------------------
        conf = self.check_static_rules(message_text, game_number)
        if conf:
            self.processed_messages.add(h)
            self.last_prediction_time = time.time()
            self._save_all_data()
            return True, game_number, conf

        return False, None, None

    # ---------------------------------------------------------
    # Make prediction (no double)
    # ---------------------------------------------------------
    def make_prediction(self, game_number: int, confidence: int):
        target_game = int(game_number) + 2
        key = str(target_game)

        text = f"🔵{target_game}🔵:Valeur Q statut :⏳ ({confidence}%)"

        self.predictions[key] = {
            "predicted_costume": "Q",
            "status": "pending",
            "predicted_from": int(game_number),
            "verification_count": 0,
            "message_text": text,
            "message_id": None,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
        }

        self.last_prediction_time = time.time()
        self._save_all_data()
        return text

    # ---------------------------------------------------------
    # Verification — EDIT ONLY (NO NEW MESSAGE)
    # ---------------------------------------------------------
    def _verify_prediction_common(self, message_text: str):
        game_number = self.extract_game_number(message_text)
        if not game_number:
            return None

        for key_str, pred in list(self.predictions.items()):
            predicted_game = int(key_str)

            if pred["status"] != "pending":
                continue

            if pred["predicted_costume"] != "Q":
                continue

            offset = game_number - predicted_game
            if offset < 0 or offset > 2:
                continue

            q_found = self.has_Q_in_group(
                self.extract_first_group(message_text)
            )

            conf = pred.get("confidence", 70)

            # SUCCESS
            if q_found:
                sym = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}.get(offset, "✅")
                new_msg = f"🔵{predicted_game}🔵:Valeur Q statut :{sym} ({conf}%)"

                pred["status"] = f"correct_offset_{offset}"
                pred["final_message"] = new_msg
                pred["verification_count"] = offset
                pred["verified_at"] = datetime.now().isoformat()
                self._save_all_data()

                return {
                    "type": "edit_message",
                    "message_id": pred.get("message_id"),
                    "new_text": new_msg,
                }

            # FAIL at offset 2
            if offset == 2 and not q_found:
                new_msg = f"🔵{predicted_game}🔵:Valeur Q statut :❌ ({conf}%)"

                pred["status"] = "failed"
                pred["final_message"] = new_msg
                pred["verified_at"] = datetime.now().isoformat()
                self._save_all_data()

                return {
                    "type": "edit_message",
                    "message_id": pred.get("message_id"),
                    "new_text": new_msg,
                }

        return None

    # ---------------------------------------------------------
    # Reset INTER
    # ---------------------------------------------------------
    def reset_inter(self):
        self.inter_data = []
        self.smart_rules = []
        self.is_inter_mode_active = False
        self._save_all_data()
        return True
