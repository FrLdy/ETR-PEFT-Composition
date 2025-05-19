from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ETRRules(Enum):
    ESSENTIAL_INFO_ONLY = "Donne uniquement les informations essentielles. Évite la surcharge d’informations."
    LOGICAL_ORDER = (
        "Présente les informations dans un ordre logique et facile à suivre."
    )
    HIGHLIGHT_MAIN_INFO = "Mets en avant l’information principale dès le début."
    GROUP_RELATED_INFO = (
        "Regroupe ensemble les informations qui parlent du même sujet."
    )
    REPEAT_IMPORTANT_INFO = (
        "Répète les informations importantes si cela aide à la compréhension."
    )
    SHORT_SIMPLE_SENTENCES = "Utilise des phrases courtes et simples."
    SIMPLE_WORDS = "Choisis des mots faciles à comprendre."
    EXPLAIN_DIFFICULT_WORDS = "Explique clairement les mots difficiles, et répète l’explication si besoin."
    AUDIENCE_APPROPRIATE_LANGUAGE = (
        "Utilise un langage adapté aux personnes concernées."
    )
    CONSISTENT_TERMINOLOGY = "Emploie le même mot pour parler de la même chose tout au long du texte."
    AVOID_ABSTRACTIONS = "Évite les idées abstraites, les métaphores et les comparaisons complexes."
    AVOID_FOREIGN_TERMS = (
        "Ne recours pas à des mots étrangers ou peu connus sans explication."
    )
    NO_TEXT_SLANG = 'Ne pas utiliser de mots contractés ou de style "texto".'
    DIRECT_ADDRESS = (
        "Adresse-toi directement au lecteur, de manière claire et accessible."
    )
    CLEAR_PRONOUNS = (
        "Veille à ce que les pronoms soient toujours clairs et non ambigus."
    )
    POSITIVE_PHRASES = (
        "Privilégie des formulations positives plutôt que négatives."
    )
    USE_ACTIVE_VOICE = "Utilise la voix active autant que possible."
    SIMPLE_PUNCTUATION = "Choisis une ponctuation simple."
    USE_LISTS = "Utilise des puces ou des numéros pour les listes, plutôt que des virgules."
    NUMBERS_AS_DIGITS = (
        "Écris les nombres en chiffres (ex. : 1, 2, 3), pas en lettres."
    )
    EXPLAIN_ACRONYMS = "Explique les sigles dès leur première apparition."
    NO_UNEXPLAINED_ABBREVIATIONS = (
        "N’utilise pas d’abréviations non expliquées."
    )
    WRITE_DATES_FULL = "Écris les dates en toutes lettres pour plus de clarté."
    EXPLAIN_NUMBERS = "Limite l’usage des pourcentages ou grands nombres, et explique-les simplement."
    NO_SPECIAL_CHARACTERS = "N’utilise pas de caractères spéciaux inutiles."
    USE_CONCRETE_EXAMPLES = (
        "Utilise des exemples concrets pour illustrer les idées complexes."
    )
    EVERYDAY_EXAMPLES = "Privilégie des exemples issus de la vie quotidienne."


@dataclass
class PromptTemplate:
    system_prompt: str
    input_prompt: str
    shot_template: Optional[str] = None
    output_prefix: Optional[str] = None


class PromptTemplates(Enum):
    ZERO_SHOT = PromptTemplate(
        system_prompt="\n".join(
            (
                "Tu es un assistant chargé de rendre un texte plus clair et accessible.",
                "Réécris le texte ci-dessous en suivant les consignes suivantes :",
                "{}".format("\n".join([f"- {e.value}" for e in ETRRules])),
                "Réponds uniquement par le texte réécrit, en français.",
            )
        ),
        input_prompt="{input}",
    )
    FEW_SHOT = PromptTemplate(
        system_prompt="\n".join(
            (
                "Tu es un assistant chargé de rendre un texte plus clair et accessible.",
                "Réécris le texte ci-dessous en suivant les consignes suivantes :",
                "{}".format("\n".join([f"- {e.value}" for e in ETRRules])),
                "Voici une série de d'examples provenant de tâches proches de ce que tu dois faire :",
                "{shots}",
                "Maintenant, compléte l'example suivant en français.",
                "Garde le contexte tel quel, n'ajoute ni titre ni section supplémentaire ni saut de ligne."
                "Entoure ta réponse par des balises '@@@' comme dans les exemples précédents.",
            )
        ),
        shot_template="\n".join(
            (
                "### Exemple {i}",
                "Tâche: {task}",
                "Entrée: {input}",
                "Sortie: @@@{output}@@@",
            )
        ),
        input_prompt="\n".join(
            (
                "Tâche: {task}",
                "Entrée: {input}",
                "Sortie: ",
            )
        ),
        output_prefix="@@@",
    )

    COT = PromptTemplate(
        system_prompt="\n".join(
            (
                "Tu es un assistant chargé de rendre un texte plus clair et accessible.",
                "Procède en suivant les étapes suivantes :",
                "1. Analyse le texte pour identifier ce qui peut être simplifié ou clarifié.",
                "2. Note brièvement les points à améliorer (syntaxe, vocabulaire, structure...).",
                "3. Réécris le texte en appliquant les consignes suivantes :",
                "{}".format("\n".join([f"- {e.value}" for e in ETRRules])),
                "4. Vérifie que la version réécrite est plus claire, plus accessible et respecte bien toutes les consignes.",
                "Commence par réfléchir étape par étape, puis termine en donnant la version finale du texte en français entourée par les balises '@@@'.",
            )
        ),
        input_prompt="{input}",
        output_prefix="@@@",
    )
