# card_predictor.py — PARTIE 1/2
# Version encapsulée dans une classe CardPredictor compatible avec bot.py / handlers.py
# Basée sur ton fichier original, INTER intact, règles statiques ajoutées (avec pourcentage)
# Ne touche pas à /inter si tu veux préserver le comportement intelligent.

import re
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------
# Configuration des confiances (règles statiques)
# -------------------------
CONFIDENCE_RULES = {
    "rule_1_single_J_g2_weak": 99,   # 1 J in G1 and G2 weak
    "rule_2_KJ_g2_weak": 55,         # K+J in G1 and G2 weak
    "rule_3_consecutive_weak": 45,   # consecutive weak groups
    "rule_4_total_ge_45": 41,        # total #T >= 45
    "rule_5_two_J": 67,              # existing two J rule
    "default_static": 70,
}

CARD_RE = re.compile(r'(10|[2-9]|[AKQJ])(♠️|♥️|♦️|♣️)')

# -------------------------
# Helper functions (module-level helpers used inside the class)
# -------------------------
def parse_cards_from_text(group_text: str) -> List[str]:
    if not group_text:
        return []
    normalized = group_text.replace("❤️", "♥️").replace("❤", "♥️")
    matches = CARD_RE.findall(normalized)
    return [f"{v}{s}" for v, s in matches]

def split_parentheses_groups(text: str) -> List[str]:
    return re.findall(r'\(([^)]*)\)', text)

def extract_first_parentheses(text: str) -> Optional[str]:
    m = re.search(r'\(([^)]*)\)', text)
    return m.group(1) if m else None

def extract_game_number(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'🔵\s*(\d{1,6})\s*🔵', text)
    if m:
        return int(m.group(1))
    m = re.search(r'#\s*[nN]\s*\.?\s*(\d{1,6})\.?', text)
    if m:
        return int(m.group(1))
    m = re.search(r'#\s*(\d{1,6})', text)
    if m:
        return int(m.group(1))
    return None

def has_Q_in_group_text(group_text: str) -> Optional[str]:
    cards = parse_cards_from_text(group_text or "")
    for c in cards:
        if c.startswith("Q"):
            return c
    return None

# -------------------------
# CardPredictor class
# -------------------------
class CardPredictor:
    """
    Encapsulation du code de prédiction. Fournit l'API attendue par handlers.py:
    - set_channel_id(channel_id, 'source'|'prediction')
    - get_inter_status()
    - analyze_and_set_smart_rules(initial_load=False)
    - should_predict(message_text) -> (bool, game_number, confidence)
    - make_prediction(game_number, confidence) -> text
    - _verify_prediction_common(message_text, is_edited=False) -> action dict or None
    - _save_all_data(), _save_data(data, filename)
    Attributs publics: predictions (dict), target_channel_id, prediction_channel_id, is_inter_mode_active
    """

    def __init__(self):
        # persistence files
        self._predictions_file = "predictions.json"
        self._processed_file = "processed.json"
        self._last_pred_time_file = "last_prediction_time.json"
        self._channels_file = "channels_config.json"
        self._inter_data_file = "inter_data.json"
        self._seq_history_file = "sequential_history.json"
        self._smart_rules_file = "smart_rules.json"
        self._inter_mode_file = "inter_mode_status.json"

        # load persistent data
        self.predictions: Dict[str, Dict[str, Any]] = self._load_json(self._predictions_file, {})
        processed_list = self._load_json(self._processed_file, [])
        self.processed_hashes = set(processed_list if isinstance(processed_list, list) else [])
        self.last_prediction_time = self._load_json(self._last_pred_time_file, 0.0)

        channels_config = self._load_json(self._channels_file, {})
        self.target_channel_id = channels_config.get("target_channel_id")
        self.prediction_channel_id = channels_config.get("prediction_channel_id")

        self.inter_data: List[Dict[str, Any]] = self._load_json(self._inter_data_file, [])
        seq_raw = self._load_json(self._seq_history_file, {})
        self.sequential_history: Dict[int, Dict[str, Any]] = {}
        try:
            for k, v in dict(seq_raw).items():
                self.sequential_history[int(k)] = v
        except Exception:
            # keep as-is if not dict-like
            if isinstance(seq_raw, dict):
                self.sequential_history = seq_raw
        self.smart_rules: List[Dict[str, Any]] = self._load_json(self._smart_rules_file, [])
        self.is_inter_mode_active = self._load_json(self._inter_mode_file, {"active": False}).get("active", False)

        # configurable cooldown
        self.PREDICTION_COOLDOWN = 30

        # Ensure smart rules computed if needed
        if self.inter_data and not self.smart_rules:
            try:
                self.analyze_and_set_smart_rules(initial_load=True)
            except Exception:
                logger.exception("Erreur lors de l'analyse initiale des smart_rules")

    # -------------------------
    # JSON persistence helpers
    # -------------------------
    def _load_json(self, path: str, default: Any):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, data: Any):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Erreur sauvegarde {path}: {e}")

    def _save_all_data(self):
        try:
            seq = {str(k): v for k, v in self.sequential_history.items()}
            self._save_json(self._seq_history_file, seq)
            self._save_json(self._inter_data_file, self.inter_data)
            self._save_json(self._smart_rules_file, self.smart_rules)
            self._save_json(self._inter_mode_file, {"active": bool(self.is_inter_mode_active)})
            self._save_json(self._predictions_file, self.predictions)
            self._save_json(self._processed_file, list(self.processed_hashes))
            self._save_json(self._last_pred_time_file, self.last_prediction_time)
            self._save_json(self._channels_file, {"target_channel_id": self.target_channel_id, "prediction_channel_id": self.prediction_channel_id})
        except Exception:
            logger.exception("Erreur lors de _save_all_data()")

    def _save_data(self, data: Any, filename: str):
        """Compatibility helper used by handlers (they sometimes call _save_data directly)."""
        try:
            # if filename is path-like, write that file directly
            self._save_json(filename, data)
        except Exception:
            logger.exception(f"Erreur _save_data pour {filename}")

    # -------------------------
    # Channel configuration
    # -------------------------
    def set_channel_id(self, channel_id: int, channel_type: str):
        if channel_type == 'source':
            self.target_channel_id = channel_id
            logger.info(f"💾 Canal SOURCE mis à jour: {channel_id}")
        elif channel_type == 'prediction':
            self.prediction_channel_id = channel_id
            logger.info(f"💾 Canal PRÉDICTION mis à jour: {channel_id}")
        else:
            return False
        self._save_all_data()
        return True

    # -------------------------
    # INTER (collecte) — NE PAS TOUCHER LA LOGIQUE (implémentation préservée)
    # -------------------------
    def collect_inter_data(self, game_number: int, message_text: str):
        try:
            if not isinstance(game_number, int):
                game_number = int(game_number)
        except Exception:
            return

        g1 = extract_first_parentheses(message_text)
        if g1:
            first_two = parse_cards_from_text(g1)[:2]
            if len(first_two) == 2:
                self.sequential_history[int(game_number)] = {"cartes": first_two, "date": datetime.now().isoformat()}
                self._save_all_data()

        q_card = has_Q_in_group_text(g1) if g1 else None
        if q_card:
            n_minus_2 = int(game_number) - 2
            trigger = self.sequential_history.get(n_minus_2)
            if trigger:
                if any(e.get("numero_resultat") == int(game_number) for e in self.inter_data):
                    return
                entry = {
                    "numero_resultat": int(game_number),
                    "numero_declencheur": n_minus_2,
                    "declencheur": trigger.get("cartes", []),
                    "carte_q": q_card,
                    "date_resultat": datetime.now().isoformat()
                }
                self.inter_data.append(entry)
                self._save_all_data()
                logger.info(f"[INTER] Enregistré: N={game_number} déclencheur N-2={n_minus_2} -> {trigger.get('cartes')}")

    def analyze_and_set_smart_rules(self, initial_load: bool = False) -> List[Dict[str, Any]]:
        counts: Dict[Tuple[str, str], int] = {}
        for e in self.inter_data:
            key = tuple(e.get("declencheur", []))
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

    def get_inter_status(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        lines = ["📋 **HISTORIQUE INTER (Déclencheur N-2 → Q à N)**\n"]
        lines.append(f"Mode Intelligent : {'🟢 ACTIVÉ' if self.is_inter_mode_active else '🔴 DÉSACTIVÉ'}")
        lines.append(f"Entrées enregistrées : {len(self.inter_data)}\n")
        if not self.inter_data:
            lines.append("Aucun déclencheur enregistré.")
            keyboard = {
                "inline_keyboard": [
                    [{"text": "📘 Règles par défaut", "callback_data": "inter_default"}]
                ]
            }
            return "\n".join(lines), keyboard
        lines.append("Dernières entrées :")
        for e in self.inter_data[-10:]:
            decl = ", ".join(e.get("declencheur", []))
            lines.append(f"N : {e.get('numero_resultat')} — Déclencheur N-2 ({e.get('numero_declencheur')}): {decl} — Carte: {e.get('carte_q')}")
        keyboard = {
            "inline_keyboard": [
                [{"text": "🧠 Appliquer la règle intelligente", "callback_data": "inter_apply"}],
                [{"text": "📘 Règle par défaut", "callback_data": "inter_default"}],
            ]
        }
        return "\n".join(lines), keyboard
        # card_predictor.py — PARTIE 2/2
# Suite et fin de la classe CardPredictor

    # -------------------------
    # Helpers: indicators / cooldown
    # -------------------------
    def can_make_prediction(self) -> bool:
        try:
            if not self.last_prediction_time:
                return True
            return time.time() > float(self.last_prediction_time) + float(self.PREDICTION_COOLDOWN)
        except Exception:
            return True

    def has_pending_indicator(self, text: str) -> bool:
        return "🕐" in text or "⏰" in text

    def has_completion_indicator(self, text: str) -> bool:
        return "✅" in text or "🔰" in text

    # -------------------------
    # STATIC RULES — fonction dédiée (ajout des règles statiques demandées)
    # -------------------------
    def check_static_rules(self, message_text: str, game_number: int) -> Optional[int]:
        g1 = extract_first_parentheses(message_text)
        if not g1:
            return None

        g1_cards = parse_cards_from_text(g1)
        g1_ranks = []
        for c in g1_cards:
            m = re.match(r'^(10|[2-9]|[AKQJ])', c)
            if m:
                g1_ranks.append(m.group(1))

        groups = split_parentheses_groups(message_text)
        g2 = groups[1] if len(groups) > 1 else ""
        g2_cards = parse_cards_from_text(g2)
        g2_ranks = []
        for c in g2_cards:
            m = re.match(r'^(10|[2-9]|[AKQJ])', c)
            if m:
                g2_ranks.append(m.group(1))

        def is_group_weak(ranks: List[str]) -> bool:
            return not any(r in ["A", "K", "Q", "J"] for r in ranks)

        # Règle 1: 1 J in G1 and G2 weak -> 99%
        if g1_ranks.count("J") == 1 and is_group_weak(g2_ranks):
            return CONFIDENCE_RULES["rule_1_single_J_g2_weak"]

        # Règle 2: K + J in G1 and G2 weak -> 55%
        if "K" in g1_ranks and "J" in g1_ranks and is_group_weak(g2_ranks):
            return CONFIDENCE_RULES["rule_2_KJ_g2_weak"]

        # Règle existante: Deux J dans G1 -> 67%
        if g1_ranks.count("J") >= 2:
            return CONFIDENCE_RULES["rule_5_two_J"]

        # Règle 3: Faiblesse consécutive (G1 faible at N and N-1) -> 45%
        prev_entry = self.sequential_history.get(game_number - 1)
        prev_ranks = []
        if prev_entry:
            for c in prev_entry.get("cartes", []):
                m = re.match(r'^(10|[2-9]|[AKQJ])', c)
                if m:
                    prev_ranks.append(m.group(1))
        if is_group_weak(g1_ranks) and is_group_weak(prev_ranks):
            return CONFIDENCE_RULES["rule_3_consecutive_weak"]

        # Règle 4: Total #T >= 45 -> 41%
        m = re.search(r'#T\s*(\d+)', message_text)
        if m and int(m.group(1)) >= 45:
            return CONFIDENCE_RULES["rule_4_total_ge_45"]

        return None

    # -------------------------
    # should_predict: logique principale
    # -------------------------
    def should_predict(self, message_text: str) -> Tuple[bool, Optional[int], Optional[int]]:
        if not self.target_channel_id:
            return False, None, None

        game_number = extract_game_number(message_text)
        if not game_number:
            return False, None, None

        try:
            self.collect_inter_data(game_number, message_text)
        except Exception:
            logger.exception("Erreur collect_inter_data dans should_predict")

        if self.has_pending_indicator(message_text):
            return False, None, None

        finalized = self.has_completion_indicator(message_text) or ("#T" in message_text and not self.has_pending_indicator(message_text))
        if not finalized:
            return False, None, None

        h = hash(message_text)
        if h in self.processed_hashes:
            return False, None, None

        if not self.can_make_prediction():
            return False, None, None

        target_game = game_number + 2
        if str(target_game) in self.predictions:
            logger.info(f"Prédiction déjà existante pour {target_game}, pas de double.")
            return False, None, None

        # INTER priority
        if self.is_inter_mode_active and self.smart_rules:
            g1 = extract_first_parentheses(message_text)
            two_cards = parse_cards_from_text(g1)[:2] if g1 else []
            total_count = sum(r.get("count", 0) for r in self.smart_rules) or 1
            for r in self.smart_rules:
                if r.get("cards") == two_cards:
                    confidence = int(round((r.get("count", 0) / total_count) * 100))
                    self.processed_hashes.add(h)
                    self.last_prediction_time = time.time()
                    self._save_all_data()
                    return True, game_number, confidence

        # Static rules fallback
        conf = self.check_static_rules(message_text, game_number)
        if conf:
            self.processed_hashes.add(h)
            self.last_prediction_time = time.time()
            self._save_all_data()
            return True, game_number, conf

        return False, None, None

    # -------------------------
    # make_prediction: enregistre la prediction et renvoie le texte à envoyer
    # -------------------------
    def make_prediction(self, game_number: int, confidence: int) -> str:
        target = int(game_number) + 2
        key = str(target)
        message_text = f"🔵{target}🔵:Valeur Q statut :⏳ ({int(confidence)}%)"
        self.predictions[key] = {
            "predicted_costume": "Q",
            "status": "pending",
            "predicted_from": int(game_number),
            "verification_count": 0,
            "message_text": message_text,
            "message_id": None,
            "confidence": int(confidence),
            "created_at": datetime.now().isoformat(),
        }
        self.last_prediction_time = time.time()
        self._save_all_data()
        logger.info(f"Prédiction créée pour {target} depuis {game_number} conf {confidence}%")
        return message_text

    # -------------------------
    # _verify_prediction_common: retourne uniquement l'action d'édition
    # -------------------------
    def _verify_prediction_common(self, message_text: str, is_edited: bool = False) -> Optional[Dict[str, Any]]:
        game_number = extract_game_number(message_text)
        if not game_number:
            return None

        for key_str, pred in list(self.predictions.items()):
            try:
                predicted_game = int(key_str)
            except Exception:
                continue

            if pred.get("status") != "pending":
                continue
            if pred.get("predicted_costume") != "Q":
                continue

            offset = game_number - predicted_game
            if offset < 0 or offset > 2:
                continue

            g1 = extract_first_parentheses(message_text)
            q_found = has_Q_in_group_text(g1) if g1 else None
            conf = pred.get("confidence", CONFIDENCE_RULES.get("default_static", 70))
            message_id = pred.get("message_id")

            if q_found:
                symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
                sym = symbol_map.get(offset, "✅")
                new_text = f"🔵{predicted_game}🔵:Valeur Q statut :{sym} ({conf}%)"

                pred["status"] = f"correct_offset_{offset}"
                pred["verification_count"] = offset
                pred["final_message"] = new_text
                pred["verified_at"] = datetime.now().isoformat()
                self._save_all_data()
                return {"type": "edit_message", "message_id": message_id, "new_text": new_text}

            if offset == 2 and not q_found:
                new_text = f"🔵{predicted_game}🔵:Valeur Q statut :❌ ({conf}%)"
                pred["status"] = "failed"
                pred["final_message"] = new_text
                pred["verified_at"] = datetime.now().isoformat()
                self._save_all_data()
                return {"type": "edit_message", "message_id": message_id, "new_text": new_text}

        return None

    # -------------------------
    # Reset helpers
    # -------------------------
    def reset_inter(self):
        self.inter_data = []
        self.smart_rules = []
        self.is_inter_mode_active = False
        self._save_all_data()
        return True

    def reset_predictions(self):
        self.predictions = {}
        self.processed_hashes = set()
        self._save_all_data()
        return True

# End of card_predictor.py
