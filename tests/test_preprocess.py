"""Golden-case tests for src.preprocess, taken from student_resource/eda/EDA_REPORT.md."""
from __future__ import annotations

from rapidfuzz.fuzz import ratio

from src.preprocess import normalize


def test_slv_marketing_group_name_core_matches_modulo_typo():
    a = normalize("5LV Marketing Private", "", "India")
    b = normalize("SLV Private Marketing (Limited)", "", "India")
    c = normalize("SLV Marketing Prívate Límited", "", "India")

    assert b["name_core"] == c["name_core"] == "slv marketing"
    # "5LV" differs from "SLV" by the known 5/S typo only.
    assert a["name_core"] == "5lv marketing"
    assert ratio(a["name_core"], b["name_core"]) >= 90


def test_prospect_brothers_name_sorted_matches_despite_word_order():
    a = normalize("Prospect Limited Private Brothers", "", "India")
    b = normalize("Prospect Brothers Private Limited", "", "India")

    assert a["name_sorted"] == b["name_sorted"] == "brothers prospect"


def test_vaughn_tailwind_name_compact_matches_domain_form():
    a = normalize("Vaughn Tailwind Inc", "", "US")
    b = normalize("vaughntailwind.com", "", "US")

    assert a["name_compact"] == b["name_compact"] == "vaughntailwind"
    assert b["is_domain"] is True


def test_france_street_abbreviation_expands_to_rue():
    rec = normalize("Ets Test SAS", "63 R. DE DIEPPE, LILLE", "France")
    assert "rue de dieppe" in rec["addr_norm"]


def test_france_house_number_prefix_extracted():
    rec = normalize("Azaé Transports SA", "N°237 R. DES BOIS BLANCS", "France")
    assert rec["addr_numbers"] == ["237"]


def test_leading_zeros_collapse_to_same_addr_number():
    a = normalize("Test Co", "001122 MARSHALL ST", "US")
    b = normalize("Test Co", "1122 Marshall Street", "US")
    assert set(a["addr_numbers"]) & set(b["addr_numbers"])
    assert "1122" in a["addr_numbers"]
    assert "1122" in b["addr_numbers"]


def test_hyphenated_letter_prefixed_number_shares_canonical_number():
    a = normalize("Anand Food Private Limited", "18 2 279, Jangammet, Hyderabad, Telangana", "India")
    b = normalize("Anand Food Pirvre Limited", "Hyderabad, No G-18 2 279, Hyderabad, TG, Jangammet", "India")
    assert set(a["addr_numbers"]) & set(b["addr_numbers"])


def test_telugu_name_translit_close_to_english_and_legal_form_detected():
    rec = normalize("ఆనంద్ ఫుడ్ ప్రైవేట్ లిమిటెడ్", "", "India")
    assert rec["legal_form"] == "pvt_ltd"
    assert ratio(rec["name_translit"], "anand food") >= 80


def test_germany_unseen_country_runs_without_error():
    rec = normalize("Schmidt GmbH", "12 Hauptstrasse, Berlin", "Germany")
    assert rec["country"] == "Germany"
    assert rec["name_norm"]
    assert isinstance(rec["addr_numbers"], list)


def test_indic_script_kept_in_name_norm_not_stripped():
    rec = normalize("ఆనంద్ ఫుడ్ ప్రైవేట్ లిమిటెడ్", "", "India")
    assert rec["script"] == "telugu"
    assert "ఆనంద్" in rec["name_norm"]


def test_zero_width_char_deleted_not_spaced():
    # ZWNJ (U+200C) inside a word must be deleted, not turned into a space,
    # or it would wrongly split a single Devanagari word into two tokens.
    rec = normalize("अच‌ल", "", "India")
    assert rec["name_norm"] == "अचल"


def test_empty_address_does_not_crash():
    rec = normalize("Some Co", "", "US")
    assert rec["addr_numbers"] == []
    assert rec["addr_components"] == []
    assert rec["state_canon"] == ""
    assert rec["city_canon"] == ""
