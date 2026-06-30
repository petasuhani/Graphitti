import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Allowed entity types
VALID_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART"
}

# Pronouns to ignore
PRONOUNS = {
    "he", "she", "it", "they", "them",
    "his", "her", "hers", "him",
    "their", "theirs", "its",
    "we", "us", "our", "ours",
    "you", "your", "yours",
    "i", "me", "my", "mine"
}


def build_entity_map(doc):
    """
    Maps every token belonging to a named entity
    to the complete entity text.
    """

    entity_map = {}

    for ent in doc.ents:

        if ent.label_ not in VALID_ENTITY_TYPES:
            continue

        entity_text = ent.text.strip()

        if entity_text.lower() in PRONOUNS:
            continue

        for token in ent:
            entity_map[token.i] = entity_text

    return entity_map
def extract_triplets(doc):

    triplets = []

    entity_map = build_entity_map(doc)

    for token in doc:

        # Only verbs become relationships
        if token.pos_ != "VERB":
            continue

        subject = None
        obj = None

        # -------------------------
        # Find Subject
        # -------------------------
        for child in token.children:

            if child.dep_ in ("nsubj", "nsubjpass"):

                # Prefer a named entity
                for tok in child.subtree:
                    if tok.i in entity_map:
                        subject = entity_map[tok.i]
                        break

                # Otherwise use noun phrase
                if subject is None:
                    continue

                    if (
                        phrase
                        and phrase.lower() not in PRONOUNS
                        and len(phrase.split()) <= 4
                    ):
                        subject = phrase

        # -------------------------
        # Find Object
        # -------------------------
        for child in token.children:

            if child.dep_ in ("dobj", "pobj", "attr", "dative", "oprd"):

                # Prefer named entity
                for tok in child.subtree:
                    if tok.i in entity_map:
                        obj = entity_map[tok.i]
                        break

                # Otherwise noun phrase
                if obj is None:
                    continue

                    if (
                        phrase
                        and phrase.lower() not in PRONOUNS
                        and len(phrase.split()) <= 4
                    ):
                        obj = phrase

        # -------------------------
        # Skip bad triplets
        # -------------------------
        if subject is None or obj is None:
            continue

        if subject == obj:
            continue

        relation = token.lemma_.lower().strip()

        if len(relation) == 0:
            continue

        triplets.append({

            "subject": subject,

            "relation": relation,

            "object": obj

        })

    # -------------------------
    # Remove duplicates
    # -------------------------

    unique = []
    seen = set()

    for t in triplets:

        key = (
            t["subject"],
            t["relation"],
            t["object"]
        )

        if key not in seen:

            seen.add(key)

            unique.append(t)

    return unique
def run_nlp_pipeline(documents):

    results = []

    print("\n==============================")
    print("Starting NLP Pipeline")
    print("==============================\n")

    for document in documents:

        print(f"Processing: {document['url']}")

        doc = nlp(document["text"])

        # Extract entities
        entities = []

        for ent in doc.ents:

            if ent.label_ not in VALID_ENTITY_TYPES:
                continue

            if ent.text.lower() in PRONOUNS:
                continue

            entities.append({
                "text": ent.text,
                "label": ent.label_
            })

        # Remove duplicate entities
        seen_entities = set()
        unique_entities = []

        for ent in entities:

            key = (ent["text"], ent["label"])

            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(ent)

        # Extract triplets
        triplets = extract_triplets(doc)

        print(f"Entities : {len(unique_entities)}")
        print(f"Triplets : {len(triplets)}")
        print("-" * 50)

        results.append({

            "url": document["url"],

            "entities": unique_entities,

            "triplets": triplets

        })

    print("\nNLP Pipeline Completed Successfully!\n")

    return results


# ---------------------------------------
# Testing
# ---------------------------------------

if __name__ == "__main__":

    sample_documents = [

        {
            "url": "sample",

            "text": """
            Spider-Man is a superhero created by Stan Lee and Steve Ditko.
            Marvel Comics publishes Spider-Man.
            Peter Parker lives in New York City.
            Tony Stark founded Stark Industries.
            Google acquired YouTube.
            """
        }

    ]

    output = run_nlp_pipeline(sample_documents)

    from pprint import pprint

    pprint(output)