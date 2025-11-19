import os
import re
import json
import time
import logging
from datetime import datetime

HIGH_VALUE_CARDS = ["A", "K", "Q", "J"]

CONFIDENCE_RULES = {
    "2.1": 98,
    "2.2": 57,
    "2.3": 97,
    "2.4": 60,
    "2.5": 70,
    "2.6": 70,
}

def safe_load_json(filename, default):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_save_json(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ ERROR saving {filename}: {e}")


class CardPredictor:

    def __init__(self):

        # Prédictions pour N+2
        self.predictions = safe_load_json("predictions.json", {})

        # Historique des messages déjà traités
        self.processed = set(safe_load_json("processed.json", []))

        # Dernier timestamp de prédiction
        self.last_prediction_time = safe_load_json("last_prediction_time.json", 0)

        # Mode intelligent ON/OFF
        self.is_inter_mode_active = safe_load_json("inter_mode.json", {"active": False}).get("active", False)

        # Liste des déclencheurs trouvés (N-2)
        self.inter_data = safe_load_json("inter_data.json", [])

        # Historique séquentiel → deux premières cartes
        raw_hist = safe_load_json("sequential_history.json", {})
        self.sequential_history = {int(k): v for k, v in raw_hist.items()} if raw_hist else {}

        # Top 3 règles intelligentes
        self.smart_rules = safe_load_json("smart_rules.json", [])

        # Cooldown de prédiction
        self.cooldown = 30


    # ---------------------------- UTILITAIRES ----------------------------

    def save_all(self):
        safe_save_json("predictions.json", self.predictions)
        safe_save_json("processed.json", list(self.processed))
        safe_save_json("last_prediction_time.json", self.last_prediction_time)
        safe_save_json("inter_data.json", self.inter_data)
        safe_save_json("inter_mode.json", {"active": self.is_inter_mode_active})
        safe_save_json("smart_rules.json", self.smart_rules)
        safe_save_json("sequential_history.json", {str(k): v for k, v in self.sequential_history.items()})


    def extract_game_number(self, msg: str):
        if not msg:
            return None
        m = re.search(r"🔵(\d+)🔵", msg)
        return int(m.group(1)) if m else None


    def extract_first_group(self, msg: str):
        m = re.search(r"\(([^)]*)\)", msg)
        return m.group(1).strip() if m else None


    def extract_cards(self, group: str):
        if not group:
            return []
        return [(v.upper(), s) for v, s in re.findall(r"(\d+|[AKQJ])(♠️|♥️|♦️|♣️)", group)]


    def extract_first_two_cards(self, group: str):
        cards = self.extract_cards(group)
        return [f"{v}{s}" for v, s in cards[:2]]


    def extract_total_points(self, msg: str):
        m = re.search(r"#T(\d+)", msg)
        return int(m.group(1)) if m else None


    def is_finalized(self, msg: str):
        return "✅" in msg or "🔰" in msg


    def has_Q_in_group1(self, msg: str):
        g1 = self.extract_first_group(msg)
        if not g1:
            return None
        for v, s in self.extract_cards(g1):
            if v == "Q":
                return f"{v}{s}"
        return None


    # ---------------------------- MODE INTELLIGENT ----------------------------

    def collect_inter_data(self, N: int, msg: str):
        """Enregistre les deux premières cartes pour toute réception de message."""

        g1 = self.extract_first_group(msg)
        if g1:
            first_two = self.extract_first_two_cards(g1)
            if len(first_two) == 2:
                self.sequential_history[N] = {
                    "cartes": first_two,
                    "date": datetime.now().isoformat()
                }

        # Enregistrer le déclencheur si la Dame apparaît
        if not self.is_finalized(msg):
            return

        q_card = self.has_Q_in_group1(msg)
        if not q_card:
            return

        N2 = N - 2
        trigger = self.sequential_history.get(N2)
        if not trigger:
            return

        if any(e["numero_resultat"] == N for e in self.inter_data):
            return

        self.inter_data.append({
            "numero_resultat": N,
            "numero_declencheur": N2,
            "declencheur": trigger["cartes"],
            "carte_q": q_card,
            "date_resultat": datetime.now().isoformat(),
        })

        self.save_all()


    def analyze_smart_rules(self):
        """Analyse les déclencheurs et génère les Top 3 règles intelligentes."""

        counts = {}
        for entry in self.inter_data:
            key = tuple(entry["declencheur"])
            counts[key] = counts.get(key, 0) + 1

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]

        self.smart_rules = [
            {"cards": list(k), "count": v} for k, v in ranked
        ]

        self.save_all()

        return [
            f"{r['cards'][0]} {r['cards'][1]} (x{r['count']})"
            for r in self.smart_rules
        ]


    def format_inter_list(self):
        """Affichage demandé : format propre"""

        if not self.inter_data:
            return "Aucune donnée INTER enregistrée."

        lines = ["📋 **Liste des déclencheurs INTER enregistrés**\n"]

        for e in self.inter_data:
            lines.append(f"N : {e['numero_resultat']}")
            lines.append(f"Déclencheur : {', '.join(e['declencheur'])}")
            lines.append(f"Carte : {e['carte_q']}\n")

        return "\n".join(lines)


    def activate_inter(self):
        self.is_inter_mode_active = True
        self.save_all()
        return (
            "🧠 **Mode Intelligent Activé**\n"
            "Les règles statiques sont désactivées.\n"
            "Le bot prédira uniquement avec les déclencheurs intelligents."
        )


    def deactivate_inter(self):
        self.is_inter_mode_active = False
        self.save_all()
        return (
            "📘 **Mode statique activé**\n"
            "Les règles intelligentes sont désactivées."
        )


    # ---------------------------- SUITE DANS PARTIE 2 ----------------------------
    # ---------------------------- RÈGLES STATIQUES ----------------------------

    def check_static_rules(self, msg: str, N: int):

        if not self.is_finalized(msg):
            return None

        g1 = self.extract_first_group(msg)
        if not g1:
            return None

        cards = self.extract_cards(g1)
        values = [v for v, s in cards]

        # 2.1 — Valet solitaire
        if values.count("J") == 1 and not any(v in ["A", "K", "Q"] for v in values if v != "J"):
            return CONFIDENCE_RULES["2.1"]

        # 2.2 — 2 Valets ou plus
        if values.count("J") >= 2:
            return CONFIDENCE_RULES["2.2"]

        # 2.3 — Total des points ≥ 45
        total = self.extract_total_points(msg)
        if total is not None and total >= 45:
            return CONFIDENCE_RULES["2.3"]

        # 2.4 — 4 jeux consécutifs sans Q
        missing = 0
        for prev in range(N - 1, N - 5, -1):
            has_q = any(e["numero_resultat"] == prev for e in self.inter_data)
            if not has_q:
                missing += 1
        if missing >= 4:
            return CONFIDENCE_RULES["2.4"]

        # 2.5 — combinaison 8-9-10 dans groupe 1 ou 2
        groups = re.findall(r"\(([^)]*)\)", msg)
        g1_vals = [v for v, s in self.extract_cards(groups[0])] if len(groups) >= 1 else []
        g2_vals = [v for v, s in self.extract_cards(groups[1])] if len(groups) >= 2 else []

        if {"8", "9", "10"}.issubset(set(g1_vals + g2_vals)):
            return CONFIDENCE_RULES["2.5"]

        # 2.6 — Bloc final (70%)
        # A : K et J dans g1
        condA = ("K" in values and "J" in values)

        # B : Tag O ou R
        condB = bool(re.search(r"\bO\b|\bR\b", msg))

        # C : Faiblesse consécutive
        def group_weak(vals):
            return not any(v in HIGH_VALUE_CARDS for v in vals)

        condC = False
        prev = self.sequential_history.get(N - 1)
        if prev:
            prev_cards = prev["cartes"]
            prev_vals = [re.match(r"(\d+|[AKQJ])", c).group(1) for c in prev_cards]
            condC = group_weak(prev_vals) and group_weak(values)

        if condA or condB or condC:
            return CONFIDENCE_RULES["2.6"]

        return None


    # ---------------------------- RÈGLES INTELLIGENTES ----------------------------

    def check_intelligent_rules(self, msg: str):
        if not self.smart_rules:
            return None

        g1 = self.extract_first_group(msg)
        if not g1:
            return None

        first_two = self.extract_first_two_cards(g1)
        if len(first_two) != 2:
            return None

        for rule in self.smart_rules:
            if rule["cards"] == first_two:
                total = sum(r["count"] for r in self.smart_rules)
                confidence = int((rule["count"] / total) * 100)
                return confidence

        return None


    # ---------------------------- DÉCISION DE PRÉDICTION ----------------------------

    def should_predict(self, msg: str, N: int):

        # Toujours collecter les données INTER
        self.collect_inter_data(N, msg)

        if not self.is_finalized(msg):
            return None

        # Mode intelligent = ON
        if self.is_inter_mode_active:
            return self.check_intelligent_rules(msg)

        # Mode statique
        return self.check_static_rules(msg, N)


    # ---------------------------- CRÉATION PREDICTION ----------------------------

    def make_prediction(self, N: int, confidence: int):
        target = N + 2
        key = str(target)

        msg = f"🔵{target}🔵:Valeur Q statut :⏳ ({confidence}%)"

        self.predictions[key] = {
            "predicted_costume": "Q",
            "status": "pending",
            "predicted_from": N,
            "verification_count": 0,
            "message_text": msg,
            "message_id": None,
            "confidence": int(confidence),
            "created_at": datetime.now().isoformat(),
        }

        self.last_prediction_time = time.time()
        self.save_all()

        return msg


    # ---------------------------- VÉRIFICATION PREDICTION ----------------------------

    def verify_prediction(self, msg: str, chat_id: int, message_id: int):

        if not self.is_finalized(msg):
            return None

        N = self.extract_game_number(msg)
        if N is None:
            return None

        for key, pred in self.predictions.items():

            if pred["status"] != "pending":
                continue

            target = int(key)
            offset = N - target

            if offset < 0 or offset > 2:
                continue

            q_found = self.has_Q_in_group1(msg)
            conf = pred["confidence"]

            # Succès
            if q_found:
                symbol = {0: "0️⃣", 1: "1️⃣", 2: "2️⃣"}.get(offset, "?")
                new_msg = f"🔵{target}🔵:Valeur Q statut :✅{symbol} ({conf}%)"

                pred["status"] = f"correct_offset_{offset}"
                pred["final_message"] = new_msg
                pred["verified_at"] = datetime.now().isoformat()

                self.save_all()

                return {"game": target, "text": new_msg}

            # Échec à offset 2
            if offset == 2:
                new_msg = f"🔵{target}🔵:Valeur Q statut :❌ ({conf}%)"

                pred["status"] = "failed"
                pred["final_message"] = new_msg
                pred["verified_at"] = datetime.now().isoformat()

                self.save_all()

                return {"game": target, "text": new_msg}

        return None
