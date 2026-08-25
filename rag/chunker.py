from rag.document_loader import Document


class TextChunker:
    """
    This class reads in text data from the input documents and converts them
    into chunks
    """

    def __init__(self, document, chunk_size, chunk_overlap):
        self.document = document
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents):
        """
        Chunking of documents
        """
        all_chunks = []
        for document in documents:
            chunks = self.chunk_doc(document)
            all_chunks.append(chunks)
        return all_chunks

    def chunk_doc(self, document):
        """
        Actually splitting the document into chunks based on
        chunk size and overlap
        """
        all_chunks = []
        text = document.text
        chunks = self.split(text, self.chunk_size, self.chunk_overlap)
        for chunk in chunks:
            doc = Document(metadata=document.metadata, text=chunk)
            # self.metadata = {"chunk_id": }
            all_chunks.append(doc)
        return chunks

    def split(self, text, chunk_size, chunk_overlap):
        """
        Splits the document into the chunks, given the specifics
        """
        chunks = []
        begin, end = 0, chunk_size

        while end <= len(text):
            chunk = text[begin:end]
            chunks.append(chunk)

            begin = end - (chunk_overlap * chunk_size)
            while text.charAt(begin) != " ":
                begin -= 1

            end = end + chunk_size
            if end < len(text):
                while text.charAt(end) != " ":
                    end -= 1

        return chunks
