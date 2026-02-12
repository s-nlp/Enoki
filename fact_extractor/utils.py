"""Utility functions for fact extraction."""

from typing import Dict, List, Optional
import re

from .models import QuotedEntityMapping, ContrastiveParse, ListParse


def preprocess_quoted_entities(text: str) -> QuotedEntityMapping:
    """
    Replace quoted entities with placeholders before parsing.

    Example:
        Input: '"Attention Is All You Need" was published in 2017'
        Output: QuotedEntityMapping(
            processed_text='__ENTITY_0__ was published in 2017',
            mapping={'__ENTITY_0__': 'Attention Is All You Need'}
        )
    """
    entities = {}
    counter = 0

    patterns = [
        r'"([^"]+)"',      # "double quotes"
        r"'([^']+)'",      # 'single quotes'
        r'«([^»]+)»',      # «guillemets»
        r'"([^"]+)"',      # "smart quotes"
        r'\u2018([^\u2019]+)\u2019',  # 'smart single quotes'
    ]

    def replace_match(match):
        nonlocal counter
        entity = match.group(1)
        placeholder = f"__ENTITY_{counter}__"
        entities[placeholder] = entity
        counter += 1
        return placeholder

    processed_text = text
    for pattern in patterns:
        processed_text = re.sub(pattern, replace_match, processed_text)

    return QuotedEntityMapping(processed_text=processed_text, mapping=entities)


def restore_quoted_entities(text: str, mapping: Dict[str, str]) -> str:
    """Restore quoted entities from placeholders."""
    result = text
    for placeholder, entity in mapping.items():
        result = result.replace(placeholder, entity)
    return result


def parse_contrastive_construction(sentence: str) -> Optional[ContrastiveParse]:
    """
    Parse contrastive constructions like "X on A, B but not on C, D".

    Example:
        Input: "solid ground can be found on Earth, Mars but not on Neptune, Sun"
        Output: ContrastiveParse(
            base_phrase='solid ground can be found',
            preposition='on',
            positive_items=['Earth', 'Mars'],
            negative_items=['Neptune', 'Sun']
        )
    """
    # Pattern: "... PREP X, Y but not PREP Z, W"
    pattern = r'(.+?)\s+(on|in|at|to|from|with)\s+([^,]+(?:,\s*[^,]+)*(?:\s+and\s+[^,]+)?)\s+but\s+not\s+(?:on|in|at|to|from|with)\s+(.+?)(?:\.|$|,\s+(?:which|that|where|when))'

    match = re.search(pattern, sentence, re.IGNORECASE)
    if not match:
        pattern2 = r'(.+?)\s+(on|in|at|to|from|with)\s+([^,]+(?:,\s*[^,]+)*(?:\s+and\s+[^,]+)?)\s+but\s+not\s+(.+?)(?:\.|$|,\s+(?:which|that|where|when))'
        match = re.search(pattern2, sentence, re.IGNORECASE)

    if not match:
        return None

    base_phrase = match.group(1).strip()
    preposition = match.group(2).strip()
    positive_text = match.group(3).strip()
    negative_text = match.group(4).strip()

    positive_items = split_list_items(positive_text)
    negative_items = split_list_items(negative_text)

    return ContrastiveParse(
        base_phrase=base_phrase,
        preposition=preposition,
        positive_items=positive_items,
        negative_items=negative_items
    )


def split_list_items(text: str) -> List[str]:
    """Split text by commas and 'and', removing empty items."""
    text = re.sub(r'\s+and\s+', ', ', text, flags=re.IGNORECASE)
    items = [item.strip() for item in text.split(',')]
    items = [item for item in items if item]
    return items


def detect_list_pattern(sentence: str) -> Optional[ListParse]:
    """
    Detect if sentence is a list/enumeration and parse it.

    Examples:
        "The justices are: A, B, C" -> ListParse(context="The justices", items=["A", "B", "C"])
        "Brands include A, B and C" -> ListParse(context="Brands", items=["A", "B", "C"])
    """
    patterns = [
        # Pattern: "The X are: A, B, C"
        (r'(.+?)\s+(?:are|is):\s*(.+)', 'colon'),

        # Pattern: "X include/includes A, B"
        (r'(.+?)\s+includes?\s+(.+)', 'include'),

        # Pattern: "X: 1. A 2. B"
        (r'(.+?):\s+\d+\.\s+(.+)', 'numbered'),
    ]

    for pattern, pattern_type in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            context = match.group(1).strip()
            items_text = match.group(2).strip()

            items = split_enumeration(items_text, pattern_type)

            if items:
                return ListParse(context=context, items=items)

    return None


def split_enumeration(items_text: str, pattern_type: str) -> List[str]:
    """
    Split enumeration text into individual items.

    Handles: "A, B and C", "A - details, B - details", "1. A 2. B 3. C"
    """
    items_text = re.sub(r'(\d+)\.\s+', r'||\1. ', items_text)

    if '||' in items_text:
        items = items_text.split('||')
    else:
        protected = []
        def protect_quotes(match):
            protected.append(match.group(0))
            return f'__PROTECTED_{len(protected)-1}__'

        items_text = re.sub(r'"[^"]*"', protect_quotes, items_text)

        items_text = re.sub(r'\s+and\s+', ',', items_text, flags=re.IGNORECASE)
        items = items_text.split(',')

        for i, item in enumerate(items):
            for j, prot in enumerate(protected):
                item = item.replace(f'__PROTECTED_{j}__', prot)
            items[i] = item

    cleaned = []
    for item in items:
        if not item.strip():
            continue

        item = re.sub(r'^\d+\.\s*', '', item)
        item = re.sub(r'\s*-\s*[^-]+$', '', item)

        item = item.strip()
        if item:
            cleaned.append(item)

    return cleaned


def extract_name_from_context(context: str) -> str:
    """
    Extract clean subject name from context (singularize).

    Example:
        "The female U.S. Supreme Court justices" -> "female U.S. Supreme Court justice"
    """
    context = re.sub(r'^(the|a|an)\s+', '', context, flags=re.IGNORECASE).strip()

    if context.endswith('ies'):
        context = context[:-3] + 'y'
    elif context.endswith('ses'):
        context = context[:-2]
    elif context.endswith('s') and not context.endswith('ss'):
        context = context[:-1]

    return context
