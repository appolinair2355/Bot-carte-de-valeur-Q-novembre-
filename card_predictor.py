# card_predictor.py
# Version: Compatible avec handlers.py (structure conservée)
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


class CardPredictor:
    def __init__(self):
        self.predictions: Dict[str, Dict[str, Any]] = _safe_load("predictions.json", {})
        self.processed: set = set(_safe_load("processed.json", []))
        self.last_prediction_time = _safe_load("last_prediction_time.json", 0.0)

        cfg = _safe_load("channels_config.json", {})
        self.target_channel_id = cfg.get("target_channel_id")
        self.prediction_channel_id = cfg.get("prediction_channel_id")

        self.inter_data = _safe_load("inter_data.json", [])
        raw_seq = _safe_load("sequential_history.json", {})

        try:
            self.sequential_history = {int(k): v for k, v in raw_seq.items()}
        except Exception:
            self.sequential_history = raw_seq if isinstance(raw_seq, dict) else {}

        self.smart_rules = _safe_load("smart_rules.json", [])
        self.is_inter_mode_active = _safe_load(
            "inter_mode_status.json",
            {"active": False}
        ).get("active", False)

        self.prediction_cooldown = 30

        if self.inter_data and not self.smart_rules:
            try:
                self.analyze_and_set_smart_rules(initial_load=True)
            except:
                logger.exception("Erreur analyse smart rules")

    def _save_data(self, data, filename: str):
        try:
            if filename == "inter_mode_status.json":
                if isinstance(data, bool):
                    _safe_save(filename, {"active": data})
                else:
                    _safe_save(filename, data)
            else:
                _safe_save(filename, data)
        except Exception as e:
            logger.error(f"_save_data failed {filename}: {e}")

    def _save_all_data(self):
        try:
            _safe_save("predictions.json", self.predictions)
            _safe_save("processed.json", list(self.processed))
            _safe_save("last_prediction_time.json", self.last_prediction_time)
            _safe_save("channels_config.json", {
                "target_channel_id": self.target_channel_id,
                "prediction_channel_id": self.prediction_channel_id
            })
            _safe_save("inter_data.json", self.inter_data)
            _safe_save("sequential_history.json", {
                str(k): v for k, v in self.sequential_history.items()
            })
            _safe_save("smart_rules.json", self.smart_rules)
            _safe_save("inter_mode_status.json", {"active": self.is_inter_mode_active})
        except Exception as e:
            logger.exception(f"_save_all_data: {e}")

    def save_all(self):
        self._save_all_data()

    def set_channel_id(self, chat_id: int, channel_type: str):
        if channel_type == "source":
            self.target_channel_id = chat_id
        elif channel_type == "prediction":
            self.prediction_channel_id = chat_id
        self._save_all_data()
        return True

    # ----------------------
    # Extraction outils
    # ----------------------
    def extract_game_number(self, text: str):
        if not text:
            return None
        m = re.search(r"🔵(\d+)🔵", text)
        if m:
            return int(m.group(1))
        m = re.search(r"#N(\d+)", text)
        return int(m.group(1)) if m else None

    def extract_first_group(self, text: str) -> Optional[str]:
        m = re.search(r"\(([^)]*)\)", text)
        return m.group(1).strip() if m else None

    def extract_all_groups(self, text: str):
        return re.findall(r"\(([^)]*)\)", text)

    def extract_cards(self, group: str):
        if not group:
            return []
        group = group.replace("❤️", "♥️")
        matches = re.findall(r"(\d+|[AKQJ])(♠️|♥️|♦️|♣️)", group)
        return [(v.upper(), s) for v, s in matches]

    def get_first_two_cards(self, group: str):
        cards = self.extract_cards(group)
        return [f"{v}{s}" for v, s in cards[:2]]

    def extract_total_points(self, text):
        m = re.search(r"#T(\d+)", text)
        return int(m.group(1)) if m else None

    def is_finalized(self, text: str) -> bool:
        return "🟩" in text or "🟦" in text or "🟨" in text or "🟧" in text

    def has_Q_in_group1(self, text: str):
        g1 = self.extract_first_group(text)
        if not g1:
            return None
        for v, s in self.extract_cards(g1):
            if v == "Q":
                return f"{v}{s}"
        return None

    # ----------------------
    # INTER COLLECT (N-2 → N)
    # ----------------------
    def collect_inter_data(self, N: int, text: str):
        g1 = self.extract_first_group(text)
        if g1:
            first = self.get_first_two_cards(g1)
            if len(first) == 2:
                self.sequential_history[N] = {
                    "cartes": first,
                    "date": datetime.now().isoformat()
                }

        if not self.is_finalized(text):
            return

        qcard = self.has_Q_in_group1(text)
        if not qcard:
            return

        trigger = self.sequential_history.get(N - 2)
        if not trigger:
            return

        if any(e["numero_resultat"] == N for e in self.inter_data):
            return

        self.inter_data.append({
            "numero_resultat": N,
            "numero_declencheur": N - 2,
            "declencheur": trigger["cartes"],
            "carte_resultat": qcard,
            "date_resultat": datetime.now().isoformat()
        })

        self._save_all_data()

    # ----------------------
    # INTER TOP3
    # ----------------------
    def analyze_and_set_smart_rules(self, initial_load=False):
        counts = {}
        for entry in self.inter_data:
            key = tuple(entry["declencheur"])
            counts[key] = counts.get(key, 0) + 1

        top3 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        self.smart_rules = [{"cards": list(k), "count": v} for k, v in top3]

        self._save_all_data()
        return [f"{a} {b} (x{c})" for (a, b), c in top3]

    def get_top3_rules_text(self):
        if not self.smart_rules:
            return "❌ Aucune règle intelligente."
        lines = ["🧠 **TOP 3 RÈGLES INTELLIGENTES**\n"]
        for i, r in enumerate(self.smart_rules[:3], start=1):
            lines.append(f"{i}. {r['cards'][0]}, {r['cards'][1]} ({r['count']} déclenchements)")
        return "\n".join(lines)

    # ----------------------
    # get_inter_status (ATTENDU PAR handlers.py)
    # ----------------------
    def get_inter_status(self):
        if not self.inter_data:
            msg = "📭 Aucun déclencheur Dame (Q) enregistré."
        else:
            lines = ["📘 **LISTE DES DÉCLENCHEURS Q**\n"]
            for e in self.inter_data:
                lines.append(
                    f"N : {e['numero_resultat']}\n"
                    f"Déclencheur : {', '.join(e['declencheur'])}\n"
                    f"Carte : {e['carte_resultat']}\n"
                )
            msg = "\n".join(lines)

        keyboard = {
            "inline_keyboard": [
                [{"text": "🧠 Appliquer la règle intelligente", "callback_data": "inter_apply"}],
                [{"text": "📘 Règle par défaut", "callback_data": "inter_default"}]
            ]
        }
        return msg, keyboard
        # ----------------------
    # MODE INTELLIGENT ON/OFF
    # ----------------------
    def activate_intelligent_mode(self):
        self.is_inter_mode_active = True
        self._save_all_data()

    def deactivate_intelligent_mode(self):
        self.is_inter_mode_active = False
        self._save_all_data()

    # ----------------------
    # RÈGLES STATIQUES
    # ----------------------
    def check_static_rules(self, text: str, N: int):
        if not self.is_finalized(text):
            return None

        g1 = self.extract_first_group(text)
        if not g1:
            return None

        cards = self.extract_cards(g1)
        values = [v for v, s in cards]

        # 2.1 Valet solitaire
        if values.count("J") == 1 and not any(v in ["A", "K", "Q"] for v in values if v != "J"):
            return CONFIDENCE_RULES["2.1"]

        # 2.2 Deux Valets ou plus
        if values.count("J") >= 2:
            return CONFIDENCE_RULES["2.2"]

        # 2.3 Total de points ≥45
        total = self.extract_total_points(text)
        if total and total >= 45:
            return CONFIDENCE_RULES["2.3"]

        # 2.4 Manque consécutif de Q ≥4
        missing = 0
        for prev in range(N - 1, N - 5, -1):
            found = any(e["numero_resultat"] == prev for e in self.inter_data)
            if not found:
                missing += 1
        if missing >= 4:
            return CONFIDENCE_RULES["2.4"]

        # 2.5 Combinaison 8-9-10
        groups = self.extract_all_groups(text)
        g1_vals = [v for v, s in self.extract_cards(groups[0])] if len(groups) else []
        g2_vals = [v for v, s in self.extract_cards(groups[1])] if len(groups) > 1 else []
        if {"8", "9", "10"}.issubset(set(g1_vals + g2_vals)):
            return CONFIDENCE_RULES["2.5"]

        # 2.6 Bloc final
        condA = ("K" in values and "J" in values)
        condB = bool(re.search(r"\bO\b|\bR\b", text))

        def weak(vals):
            return not any(v in HIGH_VALUE_CARDS for v in vals)

        condC = False
        prev_entry = self.sequential_history.get(N - 1)
        if prev_entry:
            prev_vals = [re.match(r"(\d+|[AKQJ])", c).group(1) for c in prev_entry["cartes"]]
            condC = weak(values) and weak(prev_vals)

        if condA or condB or condC:
            return CONFIDENCE_RULES["2.6"]

        return None

    # ----------------------
    # MODE INTELLIGENT SEUL
    # ----------------------
    def check_intelligent_rules(self, text: str, N: int):
        if not self.smart_rules:
            return None
        g1 = self.extract_first_group(text)
        if not g1:
            return None
        first2 = self.get_first_two_cards(g1)
        for rule in self.smart_rules:
            if rule["cards"] == first2:
                total = sum(r["count"] for r in self.smart_rules) or 1
                return int((rule["count"] / total) * 100)
        return None

    # ----------------------
    # should_predict — APPEL EXACT DE handlers.py
    # ----------------------
    def should_predict(self, text: str):
        N = self.extract_game_number(text)
        if N is None:
            return False, None, None

        # Collecte même si pas finalisé
        self.collect_inter_data(N, text)

        if not self.is_finalized(text):
            return False, None, None

        # Éviter doublons
        if hash(text) in self.processed:
            return False, None, None

        # Cooldown
        if time.time() < self.last_prediction_time + self.prediction_cooldown:
            return False, None, None

        # Mode intelligent
        if self.is_inter_mode_active:
            c = self.check_intelligent_rules(text, N)
            if c:
                return True, N, c
            return False, None, None

        # Mode statique
        c = self.check_static_rules(text, N)
        if c:
            return True, N, c

        return False, None, None

    # ----------------------
    # CRÉATION PRÉDICTION
    # ----------------------
    def make_prediction(self, game_number: int, confidence: int) -> str:
        target = game_number + 2
        key = str(target)

        msg = f"🔵{target}🔵:Valeur Q statut :⏳ ({confidence}%)"

        self.predictions[key] = {
            "predicted_costume": "Q",
            "status": "pending",
            "predicted_from": game_number,
            "verification_count": 0,
            "message_text": msg,
            "message_id": None,
            "confidence": int(confidence),
            "created_at": datetime.now().isoformat(),
        }

        self.last_prediction_time = time.time()
        self._save_all_data()

        return msg

    # ----------------------
    # VÉRIFICATION (APPELÉ PAR handlers)
    # ----------------------
    def _verify_prediction_common(self, text: str, is_edited=False):
        if not self.is_finalized(text):
            return None

        N = self.extract_game_number(text)
        if N is None:
            return None

        for key, pred in self.predictions.items():
            if pred["status"] != "pending":
                continue

            target = int(key)
            offset = N - target
            if offset < 0 or offset > 2:
                continue

            q = self.has_Q_in_group1(text)
            conf = pred["confidence"]

            # Succès
            if q:
                symbol = {0: "0️⃣", 1: "1️⃣", 2: "2️⃣"}.get(offset, "0️⃣")
                new_msg = f"🔵{target}🔵:Valeur Q statut :✅{symbol} ({conf}%)"

                pred["status"] = f"correct_offset_{offset}"
                pred["final_message"] = new_msg
                pred["verified_at"] = datetime.now().isoformat()
                self._save_all_data()

                return {
                    "type": "edit_message",
                    "predicted_game": target,
                    "new_message": new_msg
                }

            # Échec au dernier essai
            if offset == 2 and not q:
                new_msg = f"🔵{target}🔵:Valeur Q statut :❌ ({conf}%)"

                pred["status"] = "failed"
                pred["final_message"] = new_msg
                pred["verified_at"] = datetime.now().isoformat()
                self._save_all_data()

                return {
                    "type": "edit_message",
                    "predicted_game": target,
                    "new_message": new_msg
                }

        return None
