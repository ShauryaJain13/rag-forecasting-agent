# from rag.document import Document


# class TextChunker:
#     """
#     This class reads in text data from the input documents and converts them
#     into chunks
#     """

#     def __init__(self, document, chunk_size, chunk_overlap):
#         self.document = document
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#         if chunk_overlap >= chunk_size:
#             raise ValueError("chunk_size must be greater than chunk_overlap")

#     def chunk_documents(self, documents):
#         """
#         Chunking of documents
#         """
#         all_chunks = []
#         for document in documents:
#             chunks = self.chunk_doc(document)
#             all_chunks.extend(chunks)
#         return all_chunks

#     def chunk_doc(self, document):
#         """
#         Actually splitting the document into chunks based on
#         chunk size and overlap
#         """
#         all_chunks = []
#         text = document.text
#         chunks = self.split(text, self.chunk_size, self.chunk_overlap)
#         for chunk_index, chunk in enumerate(chunks):
#             metadata = document.metadata.copy()
#             metadata["chunk_data"] = chunk_index
#             chunk_doc = Document(metadata=metadata, text=chunk)
#             # self.metadata = {"chunk_id": }
#             all_chunks.append(chunk_doc)
#         return all_chunks

#     def split(self, text, chunk_size, chunk_overlap):
#         """
#         Splits the document into the chunks, given the specifics
#         """
#         chunks = []
#         begin = 0  # , end = 0, chunk_size

#         while begin <= len(text):
#             end = begin + chunk_size
#             chunk = text[begin:end]

#             if chunk.strip():
#                 chunks.append(chunk)

#             if end >= len(text):
#                 break

#             begin = end - chunk_overlap

#             # chunks.append(chunk)

#             # begin = end - (chunk_overlap - chunk_size)
#             # while text[begin] != " ":
#             # begin -= 1

#             # end = end + chunk_size
#             # if end < len(text):
#             #     while text[end] != " ":
#             #         end -= 1

#         return chunks

from rag.document import Document


class TextChunker:
    """
    This class reads in text data from the input documents and converts them
    into chunks
    """

    def __init__(self, chunk_size, chunk_overlap):
        # CHANGED: removed the `document` constructor argument. It was
        # stored as self.document but never read anywhere -- chunking
        # actually happens per-document via chunk_documents(documents)/
        # chunk_doc(document), which take the document(s) as method
        # arguments. The unused param just forced callers to have a
        # document on hand before they could even construct a chunker.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be greater than chunk_overlap")

    def chunk_documents(self, documents):
        """
        Chunking of documents
        """
        all_chunks = []
        for document in documents:
            chunks = self.chunk_doc(document)
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_doc(self, document):
        """
        Actually splitting the document into chunks based on
        chunk size and overlap
        """
        all_chunks = []
        text = document.text
        chunks = self.split(text, self.chunk_size, self.chunk_overlap)
        for chunk_index, chunk in enumerate(chunks):
            metadata = document.metadata.copy()
            metadata["chunk_data"] = chunk_index
            chunk_doc = Document(metadata=metadata, text=chunk)
            all_chunks.append(chunk_doc)
        return all_chunks

    def split(self, text, chunk_size, chunk_overlap):
        """
        Splits the document into the chunks, given the specifics
        """
        chunks = []
        begin = 0

        while begin <= len(text):
            end = begin + chunk_size
            chunk = text[begin:end]

            if chunk.strip():
                chunks.append(chunk)

            if end >= len(text):
                break

            begin = end - chunk_overlap

        return chunks