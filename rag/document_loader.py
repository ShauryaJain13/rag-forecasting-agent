from pathlib import Path


class DocumentLoader:
    """
    This class loads the filepath and stores it as a document
    """
    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def load(self):
        """
        Loading filepath as document
        """
        doc = self.filepath.read_text(encoding='utf-8')
        return doc
