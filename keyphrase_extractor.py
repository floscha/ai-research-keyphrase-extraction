from wildfire import Wildfire

from launch import load_local_embedding_distributor
from launch import load_local_pos_tagger
from launch import extract_keyphrases as _extract_keyphrases


class KeyphraseExtractor:
    """A simple wrapper class to expose an API for keyphrase extraction."""
    def __init__(self):
        # Initialize certain objects here once, instead of every method call.
        self.embedding_distributor = load_local_embedding_distributor('en')
        self.pos_tagger = load_local_pos_tagger('en')

    def extract_keyphrases(self, text, num_keyphrases, language='en'):
        # Simply delegate to the `extract_keyphrases` method from `launch`.
        return _extract_keyphrases(self.embedding_distributor,
                                   self.pos_tagger,
                                   text,
                                   num_keyphrases,
                                   language)


api = Wildfire(KeyphraseExtractor())
