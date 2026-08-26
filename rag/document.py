from pathlib import Path
from pypdf import PdfReader
import pandas as pd


class DocumentLoader:
    """
    This class loads the filepath and stores it as a document
    """

    def load(self, filepath):
        """
        Loading filepath as document
        """
        doc_type = Path(filepath).suffix.lower()
        if doc_type == ".pdf":
            pdf = self.load_pdf(filepath)
            return pdf
        elif doc_type == ".txt":
            txt = self.load_txt(filepath)
            return txt
        elif doc_type == ".csv":
            csv = self.load_csv(filepath)
            return csv
        else:
            raise ValueError(f"Document type {doc_type} is unknown")

    def load_pdf(self, filepath):
        """
        If the document uploaded is a pdf, it reads and stores that
        """
        documents = []
        pdf = PdfReader(filepath)  # .read(filepath)
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            doc = Document(metadata={"source": str(filepath),
                                     "page": page_number + 1,
                                     "type": "pdf"},
                           text=text)
            documents.append(doc)
        return documents

    def load_txt(self, filepath):
        """
        If the document uploaded is a text file, it reads and stores that
        """
        text = None
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        doc = Document(metadata={"source": str(filepath),
                                 "page": None,
                                 "type": "txt"}, text=text)
        return [doc]

    def load_csv(self, filepath):
        """
        If the document uploaded is a csv, it reads and stores that
        """
        dataframe = pd.read_csv(filepath)
        text = dataframe.to_csv(index=False)
        doc = Document(metadata={"source": filepath,
                                 "page": None,
                                 "type": "dataset"},
                       text=text)
        return [doc]


class Document:
    """
    This class is information about the actual document
    """
    def __init__(self, metadata, text):
        self.metadata = metadata
        self.text = text

    def __repr__(self):
        return f"Document(source: {self.metadata.get('source')})"

# text -> whatever is in the document
# metadata -> dictionary of source and page
