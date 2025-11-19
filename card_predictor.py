# card_predictor.py — PARTIE 1/2
# Fichier corrigé — collecte N-2, règles statiques, smart INTER, pas de double-prédiction.
# Remplace entièrement ton ancien card_predictor.py par ces deux parties collées.

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
    "2.1": 98,  # Valet J solitaire
    "2.2": 57,  # Deux valets
    "2.3": 97,  # Total #T >= 45
    "2.4": 60,  # 4 jeux consécutifs sans Q
    "2.5": 70,  # Combo 8-9-10
    "2.6": 70,  # Bloc final
    "default_static": 70,
}

# -------------------------
# Helpers persistences JSON
# -------------------------
def _load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception(f"Erreur sauvegarde {path}: {e}")

# -------------------------
# Card parsing helpers
# -------------------------
CARD_RE = re.compile(r'(10|[2-9]|[AKQJ])(♠️|♥️|♦️|♣️)')

def parse_cards_from_text(group_text: str) -> List[str]:
    """Retourne liste de cartes formatées 'Q♣️', '10♠️', ..."""
    if not group_text:
        return []
    normalized = group_text.replace("❤️", "♥️").replace("❤", "♥️")
    matches = CARD_RE.findall(normalized)
    return [f"{v}{s}" for v, s in matches]

def split_parentheses_groups(text: str) -> List[str]:
    """Retourne le contenu de toutes les parenthèses (ordre d'apparition)."""
    return re.findall(r'\(([^)]*)\)', text)

def extract_first_parentheses(text: str) -> Optional[str]:
    m = re.search(r'\(([^)]*)\)', text)
    return m.group(1) if m else None

def extract_game_number(text: str) -> Optional[int]:
    """Extrait le numéro du jeu depuis formats usuels (#n51., 🔵51🔵, #51...)."""
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
# Chargement initial des données persistantes
# -------------------------
predictions: Dict[str, Dict[str, Any]] = _load_json("predictions.json", {})  # keys are str(target_game)
processed_hashes = set(_load_json("processed.json", []))
last_prediction_time = _load_json("last_prediction_time.json", 0.0)

channels_config = _load_json("channels_config.json", {})
target_channel_id = channels_config.get("target_channel_id")
prediction_channel_id = channels_config.get("prediction_channel_id")

inter_data: List[Dict[str, Any]] = _load_json("inter_data.json", [])
sequential_history_raw = _load_json("sequential_history.json", {})  # keys likely strings
# normalize sequential_history keys to int
sequential_history: Dict[int, Dict[str, Any]] = {}
try:
    for k, v in sequential_history_raw.items():
        sequential_history[int(k)] = v
except Exception:
    sequential_history = sequential_history_raw if isinstance(sequential_history_raw, dict) else {}

smart_rules: List[Dict[str, Any]] = _load_json("smart_rules.json", [])
inter_mode_active = _load_json("inter_mode_status.json", {"active": False}).get("active", False)

PREDICTION_COOLDOWN = 30  # seconds

# -------------------------
# Persistence helper to save all
# -------------------------
def _save_all():
    try:
        # convert sequential_history keys to strings
        seq = {str(k): v for k, v in sequential_history.items()}
        _save_json("sequential_history.json", seq)
        _save_json("inter_data.json", inter_data)
        _save_json("smart_rules.json", smart_rules)
        _save_json("inter_mode_status.json", {"active": bool(inter_mode_active)})
        _save_json("predictions.json", predictions)
        _save_json("processed.json", list(processed_hashes))
        _save_json("last_prediction_time.json", last_prediction_time)
        _save_json("channels_config.json", {"target_channel_id": target_channel_id, "prediction_channel_id": prediction_channel_id})
    except Exception:
        logger.exception("Erreur lors de _save_all()")

# -------------------------
# INTER: collecte - NE PAS TOUCHER LA LOGIQUE SI TU AS DÉSIRÉ LA GARDER
# (Ici j'intègre une version robuste et compatible: mémorise les 2 premières cartes de G1,
#  relie N -> N-2 si Q apparait, évite doublons, persiste)
# -------------------------
def collect_inter_data(game_number: int, message_text: str):
    """Doit être appelé pour chaque message (final ou non) pour mémoriser G1 et créer inter_data si Q trouvé."""
    global sequential_history, inter_data
    try:
        if not isinstance(game_number, int):
            game_number = int(game_number)
    except Exception:
        return

    g1 = extract_first_parentheses(message_text)
    if g1:
        first_two = parse_cards_from_text(g1)[:2]
        if len(first_two) == 2:
            sequential_history[int(game_number)] = {"cartes": first_two, "date": datetime.now().isoformat()}
            _save_all()

    # si Q dans G1 ET N-2 existe dans sequential_history -> créer inter_data
    q_card = has_Q_in_group_text(g1) if g1 else None
    if q_card:
        n_minus_2 = int(game_number) - 2
        trigger = sequential_history.get(n_minus_2)
        if trigger:
            # éviter doublon pour meme numero_resultat
            if any(e.get("numero_resultat") == int(game_number) for e in inter_data):
                return
            entry = {
                "numero_resultat": int(game_number),
                "numero_declencheur": n_minus_2,
                "declencheur": trigger.get("cartes", []),
                "carte_q": q_card,
                "date_resultat": datetime.now().isoformat()
            }
            inter_data.append(entry)
            _save_all()
            logger.info(f"[INTER] Enregistré: N={game_number} déclencheur N-2={n_minus_2} -> {trigger.get('cartes')}")

# -------------------------
# Smart rules (top 3) — calcul à partir de inter_data
# -------------------------
def analyze_and_set_smart_rules(initial_load: bool = False) -> List[Dict[str, Any]]:
    global smart_rules, inter_mode_active
    counts: Dict[Tuple[str, str], int] = {}
    for e in inter_data:
        key = tuple(e.get("declencheur", []))
        if len(key) != 2:
            continue
        counts[key] = counts.get(key, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    smart_rules = [{"cards": list(k), "count": v} for k, v in sorted_items[:3]]
    if smart_rules:
        inter_mode_active = True
    elif not initial_load:
        inter_mode_active = False
    _save_all()
    return smart_rules

# End of PART 1/2
# card_predictor.py — PARTIE 2/2
# Suite et fin — règles statiques, should_predict, make & verify, utilitaires.

# -------------------------
# INTER status UI (commande /inter)
# -------------------------
def get_inter_status() -> Tuple[str, Optional[Dict[str, Any]]]:
    """Retourne message + keyboard inline pour /inter."""
    lines: List[str] = []
    lines.append("📋 **HISTORIQUE INTER (Déclencheur N-2 → Q à N)**\n")
    lines.append(f"Mode Intelligent : {'🟢 ACTIVÉ' if inter_mode_active else '🔴 DÉSACTIVÉ'}")
    lines.append(f"Entrées enregistrées : {len(inter_data)}\n")
    if not inter_data:
        lines.append("Aucun déclencheur enregistré.")
        keyboard = {
            "inline_keyboard": [
                [{"text": "📘 Règles par défaut", "callback_data": "inter_default"}]
            ]
        }
        return "\n".join(lines), keyboard

    lines.append("Dernières entrées :")
    for e in inter_data[-10:]:
        decl = ", ".join(e.get("declencheur", []))
        lines.append(f"N : {e.get('numero_resultat')} — Déclencheur N-2 ({e.get('numero_declencheur')}): {decl} — Carte: {e.get('carte_q')}")
    keyboard = {
        "inline_keyboard": [
            [{"text": "🧠 Appliquer la règle intelligente", "callback_data": "inter_apply"}],
            [{"text": "📘 Règle par défaut", "callback_data": "inter_default"}],
        ]
    }
    return "\n".join(lines), keyboard

# -------------------------
# Helpers: indicators / cooldown
# -------------------------
def can_make_prediction() -> bool:
    global last_prediction_time
    try:
        if not last_prediction_time:
            return True
        return time.time() > float(last_prediction_time) + PREDICTION_COOLDOWN
    except Exception:
        return True

def has_pending_indicator(text: str) -> bool:
    return "🕐" in text or "⏰" in text

def has_completion_indicator(text: str) -> bool:
    return "✅" in text or "🔰" in text

# -------------------------
# STATIC RULES — fonction dédiée
# -------------------------
def check_static_rules(message_text: str, game_number: int) -> Optional[int]:
    """
    Implémente les règles 2.1..2.6 et retourne la confiance (int) si match.
    """
    # G1 obligatoire
    g1 = extract_first_parentheses(message_text)
    if not g1:
        return None
    g1_cards = parse_cards_from_text(g1)
    g1_values = [c[:-2] if len(c) > 2 else c[0:-1] for c in g1_cards]  # safe extract values like 'Q' or '10'
    # Normaliser: values like '10' or '8', or 'Q','J','K','A'
    # convert to just ranks:
    ranks = []
    for c in g1_cards:
        m = re.match(r'^(10|[2-9]|[AKQJ])', c)
        if m:
            ranks.append(m.group(1))

    # 2.1 Valet Solitaire (J exactly 1 and no other high A/K/Q)
    if ranks.count("J") == 1 and not any(r in ["A", "K", "Q"] for r in ranks if r != "J"):
        return CONFIDENCE_RULES["2.1"]

    # 2.2 Deux Valets
    if ranks.count("J") >= 2:
        return CONFIDENCE_RULES["2.2"]

    # 2.3 Total points (#T) >= 45
    m = re.search(r'#T\s*(\d+)', message_text)
    if m and int(m.group(1)) >= 45:
        return CONFIDENCE_RULES["2.3"]

    # 2.4 Manque consécutif de Q >= 4
    missing = 0
    for prev in range(game_number - 1, game_number - 5, -1):
        if not any(e.get("numero_resultat") == prev for e in inter_data):
            missing += 1
    if missing >= 4:
        return CONFIDENCE_RULES["2.4"]

    # 2.5 Combinaison 8-9-10 dans G1 ou G2
    groups = split_parentheses_groups(message_text)
    found_vals = []
    for g in groups:
        for c in parse_cards_from_text(g):
            m2 = re.match(r'^(10|[2-9]|[AKQJ])', c)
            if m2:
                found_vals.append(m2.group(1))
    if {"8", "9", "10"}.issubset(set(found_vals)):
        return CONFIDENCE_RULES["2.5"]

    # 2.6 Bloc final (K and J in G1) OR tags O/R OR double weakness
    condA = ("K" in ranks and "J" in ranks)
    condB = bool(re.search(r'\bO\b|\bR\b', message_text))
    def is_weak_list(vals: List[str]) -> bool:
        return not any(v in ["A", "K", "Q", "J"] for v in vals)
    # previous game weak?
    prev_entry = sequential_history.get(game_number - 1)
    prev_ranks = []
    if prev_entry:
        for card in prev_entry.get("cartes", []):
            m3 = re.match(r'^(10|[2-9]|[AKQJ])', card)
            if m3:
                prev_ranks.append(m3.group(1))
    condC = is_weak_list(ranks) and is_weak_list(prev_ranks)
    if condA or condB or condC:
        return CONFIDENCE_RULES["2.6"]

    return None

# -------------------------
# should_predict: règle principale (renvoie tuple (bool, game_number, confidence))
# -------------------------
def should_predict(message_text: str) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    1) collecte inter_data (mémoire N-2)
    2) vérifie finalisation du message
    3) empêche double-prediction en vérifiant si predictions contient déjà N+2
    4) applique d'abord smart_rules (si inter actif), ensuite règles statiques
    """
    global last_prediction_time, processed_hashes

    if not target_channel_id:
        return False, None, None

    game_number = extract_game_number(message_text)
    if not game_number:
        return False, None, None

    # collect INTER data always
    try:
        collect_inter_data(game_number, message_text)
    except Exception:
        logger.exception("Erreur collect_inter_data dans should_predict")

    # block if pending indicators
    if has_pending_indicator(message_text):
        return False, None, None

    # consider finalized if explicit or includes #T
    finalized = has_completion_indicator(message_text) or ("#T" in message_text and not has_pending_indicator(message_text))
    if not finalized:
        return False, None, None

    # avoid duplicates processing
    h = hash(message_text)
    if h in processed_hashes:
        return False, None, None

    # cooldown
    if not can_make_prediction():
        return False, None, None

    # STOP double prediction: if prediction for N+2 already exists, do NOT create another
    target_game = game_number + 2
    if str(target_game) in predictions:
        logger.info(f"Prédiction déjà existante pour {target_game}, pas de double.")
        return False, None, None

    # INTER mode priority
    if inter_mode_active and smart_rules:
        g1 = extract_first_parentheses(message_text)
        two_cards = parse_cards_from_text(g1)[:2] if g1 else []
        total_count = sum(r.get("count", 0) for r in smart_rules) or 1
        for r in smart_rules:
            if r.get("cards") == two_cards:
                confidence = int(round((r.get("count", 0) / total_count) * 100))
                processed_hashes.add(h)
                last_prediction_time = time.time()
                _save_all()
                return True, game_number, confidence

    # Static rules fallback
    conf = check_static_rules(message_text, game_number)
    if conf:
        processed_hashes.add(h)
        last_prediction_time = time.time()
        _save_all()
        return True, game_number, conf

    return False, None, None

# -------------------------
# make_prediction: enregistre la prediction et renvoie le texte à envoyer
# Format EXACT requis:
# 🔵(N+2)🔵:Valeur Q statut :⏳(XX%)
# -------------------------
def make_prediction(game_number: int, confidence: int) -> str:
    global predictions, last_prediction_time
    target = int(game_number) + 2
    key = str(target)
    message_text = f"🔵{target}🔵:Valeur Q statut :⏳ ({int(confidence)}%)"
    predictions[key] = {
        "predicted_costume": "Q",
        "status": "pending",
        "predicted_from": int(game_number),
        "verification_count": 0,
        "message_text": message_text,
        "message_id": None,
        "confidence": int(confidence),
        "created_at": datetime.now().isoformat(),
    }
    last_prediction_time = time.time()
    _save_all()
    logger.info(f"Prédiction créée pour {target} depuis {game_number} conf {confidence}%")
    return message_text

# -------------------------
# _verify_prediction_common: vérifie les messages entrants et ne retourne QUE des actions d'édition
# (handlers doit prendre cette action et appeler editMessage en utilisant message_id stocké)
# -------------------------
def _verify_prediction_common(message_text: str, is_edited: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parcourt predictions en attente et si le message confirme (ou infirme au offset 2),
    renvoie {'type':'edit_message', 'message_id':..., 'new_text':...}
    """
    global predictions
    game_number = extract_game_number(message_text)
    if not game_number:
        return None

    # iterate over copy to avoid mutation during iteration
    for key_str, pred in list(predictions.items()):
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

        # SUCCESS
        if q_found:
            symbol_map = {0: "✅0️⃣", 1: "✅1️⃣", 2: "✅2️⃣"}
            sym = symbol_map.get(offset, "✅")
            new_text = f"🔵{predicted_game}🔵:Valeur Q statut :{sym} ({conf}%)"

            pred["status"] = f"correct_offset_{offset}"
            pred["verification_count"] = offset
            pred["final_message"] = new_text
            pred["verified_at"] = datetime.now().isoformat()
            _save_all()
            # always return edit action – handler must edit using message_id
            return {"type": "edit_message", "message_id": message_id, "new_text": new_text}

        # FAIL at offset == 2
        if offset == 2 and not q_found:
            new_text = f"🔵{predicted_game}🔵:Valeur Q statut :❌ ({conf}%)"
            pred["status"] = "failed"
            pred["final_message"] = new_text
            pred["verified_at"] = datetime.now().isoformat()
            _save_all()
            return {"type": "edit_message", "message_id": message_id, "new_text": new_text}

    return None

# -------------------------
# Reset helpers
# -------------------------
def reset_inter():
    global inter_data, smart_rules, inter_mode_active
    inter_data = []
    smart_rules = []
    inter_mode_active = False
    _save_all()
    return True

def reset_predictions():
    global predictions, processed_hashes
    predictions = {}
    processed_hashes = set()
    _save_all()
    return True

# -------------------------
# End of file
# -------------------------
