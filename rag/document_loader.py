from pathlib import Path
from pypdf import PdfReader
import pandas as pd


class DocumentLoader:
    """
    This class loads the filepath and stores it as a document
    """
    def __init__(self, filepath):
        self.filepath = Path(filepath)

    def load(self, filepath):
        """
        Loading filepath as document
        """
        doc_type = filepath[-4:]
        if doc_type == ".pdf":
            self.load_pdf(filepath)
        elif doc_type == ".txt":
            self.load_txt(filepath)
        elif doc_type == ".csv":
            self.load_csv(filepath)
        else:
            raise ValueError(f"Document type {doc_type} is unknown")

        # doc = self.filepath.read_text(encoding='utf-8')
        # return doc

    def load_pdf(self, filepath):
        """
        If the document uploaded is a pdf, it reads and stores that
        """
        documents = []
        pdf = PdfReader.read(filepath)
        for page in range(len(pdf.pages)):
            text = page.extract_text()
            doc = Document(metadata={"source": filepath,
                                     "page": page},
                           text=text)
            documents.append(doc)
        return documents

    def load_txt(self, filepath):
        """
        If the document uploaded is a text file, it reads and stores that
        """
        text = None
        with open(filepath) as f:
            text = f.read()

        doc = Document(metadata={"source": filepath,
                                 "page": None}, text=text)
        return [doc]

    def load_csv(self, filepath):
        """
        If the document uploaded is a csv, it reads and stores that
        """
        csv = pd.read_csv(filepath)
        txt = csv.to_csv('output.txt', sep='\t', index=False)
        doc = Document(metadata={"source": filepath,
                                 "type": "dataset"},
                       text=txt)
        return doc

    # def load_markdown(self, filepath):
    #     """
    #     If the document uploaded is a markdown, it reads and stores that
    #     """


class Document:
    """
    This class is information about the actual document
    """
    def __init__(self, metadata, text):
        self.metadata = metadata
        self.text = text

# text -> whatever is in the document
# metadata -> dictionary of source and page
