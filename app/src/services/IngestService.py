from dotenv import load_dotenv
import os
import sys
import pandas as pd
import json
import numpy as np
import warnings
import logging
import ast

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
import time

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from transformers import AutoTokenizer
from pymilvus import (
FieldSchema,
CollectionSchema,
DataType,
Collection,
AnnSearchRequest,
RRFRanker,
WeightedRanker,
connections,
MilvusClient
)

# remove if debugging
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

########## Functions to initialize environment ##########

def init_environment(collection_name, chunk_type):
    try:
        load_dotenv()
        config = {
            'wxd_milvus_host': os.getenv("WXD_MILVUS_HOST"),
            'wxd_milvus_port': os.getenv("WXD_MILVUS_PORT"),          
            'wxd_milvus_user': os.getenv("WXD_MILVUS_USER"),
            'wxd_milvus_password': os.getenv("WXD_MILVUS_PASSWORD"),
            'collection_name': collection_name,
            'chunk_type': chunk_type
        }
        validate_environment(config)
        logging.info("Environment initialized successfully")
        return config
    except Exception as e:
        logging.exception("Error initializing environment")
        raise

def validate_environment(config):
    """
    Validates required environment variables and logs warnings for missing variables.
    """
    missing_vars = [key for key, value in config.items() if not value]
    
    if missing_vars:
        error_msg = f"Required environment variables are missing: {', '.join(missing_vars)}"
        logging.error(error_msg)
        raise EnvironmentError(error_msg)
    
########## Main functions ##########    

def ingest_files(config, files):
    # set embedding model and tokenizer
    embedding_model='BAAI/bge-m3'
    ef = BGEM3EmbeddingFunction(model_name=embedding_model, use_fp16=False, device="cpu")
    ef_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # ef_max_tokens = 1024

    # convert docs to docling documents and use hybrid chunker
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use ACCURATE if you prefer more accurate TableFormer model, however it's slower

    document_converter = DocumentConverter(format_options=
                                        {InputFormat.PDF: PdfFormatOption(
                                            pipeline_options=pipeline_options)})
    
    print(1)
    
    EXPORT_TYPE = config['chunk_type']
    print(EXPORT_TYPE)
    if EXPORT_TYPE == "DOCLING_DOCS":
        def extract_metadata(chunk):
            metadata = {
                    "page_no": None,
                    "filename": None
                }
            if hasattr(chunk, 'meta'):

                # Extract page information and content type
                if hasattr(chunk.meta, 'doc_items'):
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'prov') and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, 'page_no'):
                                    metadata["page_no"] = prov.page_no
                
                if hasattr(chunk.meta, 'origin'):
                    if hasattr(chunk.meta.origin, 'filename'):
                        metadata['filename'] = chunk.meta.origin.filename
                
                return metadata

        def convert_and_hybrid_chunk(file):
            file_full_path = file['full_path']
            print(f"Converting {file_full_path} to a docling document and using hybrid chunker...")
            start_time = time.time()
            doc = document_converter.convert(file_full_path).document

            chunker = HybridChunker(tokenizer=ef_tokenizer)
            chunk_iter = chunker.chunk(dl_doc=doc)

            # iterate over chunks, creating a chunk document object per chunk with the content and metadata
            chunked_document_objects = []
            for chunk in chunk_iter:
                chunked_document_object = {"page_content": chunker.serialize(chunk=chunk), "metadata": extract_metadata(chunk)}
                chunked_document_objects.append(chunked_document_object)
            end_time = time.time()
            print(f"Converted {file_full_path} to a docling document and chunked\nNumber of chunks: {len(chunked_document_objects)}\nExecution time: {round(end_time - start_time, 2)}")
            return chunked_document_objects
        
        document_objects = []
        for file in files:
            chunked_document_objects = convert_and_hybrid_chunk(file)
            document_objects += chunked_document_objects

    if EXPORT_TYPE == "MARKDOWN":
        def convert_and_markdown_split(file):
            file_full_path = file['full_path']
            filename = file['filename']
            print(f"Converting {filename} to a docling document and using markdown splitter...")
            start_time = time.time()

            doc = document_converter.convert(file_full_path).document.export_to_markdown()
            splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
            strip_headers=False
            )
            chunks = [split for split in splitter.split_text(doc)]

            # chunk_size = 250
            # chunk_overlap = 30
            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=chunk_size, chunk_overlap=chunk_overlap
            # )

            # # Split
            # splits = text_splitter.split_documents(md_header_chunks)
            # chunk

            # iterate over chunks, creating a chunk document object per chunk with the content and metadata
            chunked_document_objects = []
            for chunk in chunks:
                print(chunk)
                chunked_document_object = {"page_content": chunk.page_content}
                chunked_document_object['metadata'] = chunk.metadata
                chunked_document_object['metadata']['filename'] = filename
                chunked_document_objects.append(chunked_document_object)
            end_time = time.time()
            print(f"Converted {file_full_path} to a markdown document and split\nNumber of chunks: {len(chunked_document_objects)}\nExecution time: {round(end_time - start_time, 2)}")
            return chunked_document_objects
        
        document_objects = []
        for file in files:
            chunked_document_objects = convert_and_markdown_split(file)
            document_objects += chunked_document_objects
            

    print(EXPORT_TYPE)
        
    # show all chunks    
    for i, chunk in enumerate(document_objects):
        print(f"===Chunk {i}===")
        print(f"chunk page content:\n{chunk['page_content']}\n")
        print(f"chunk metadata:\n{chunk['metadata']}\n")

    # get fields into lists
    docs_list = [doc['page_content'] for doc in document_objects]
    filename_list = [doc['metadata']['filename'] for doc in document_objects]
    if EXPORT_TYPE == "DOCLING_DOCS":
        page_no_list = [str(doc['metadata']['page_no']) for doc in document_objects]
    elif EXPORT_TYPE == "MARKDOWN":
        page_no_list = ["None"] * len(document_objects)

    # create embeddings
    dense_dim = ef.dim["dense"]
    docs_embeddings = ef(docs_list)

    # load into milvus collection
    connections.connect(host=config['wxd_milvus_host'],
                        port=config['wxd_milvus_port'],
                        secure=True, user=config['wxd_milvus_user'],
                        password=config['wxd_milvus_password'])

    client = MilvusClient(
    uri=f"https://{config['wxd_milvus_host']}:{config['wxd_milvus_port']}",
    user=config['wxd_milvus_user'],
    password=config['wxd_milvus_password'])

    # defining collection
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="id", dtype=DataType.VARCHAR, description='IDs', is_primary=True, auto_id=True, max_length=100
        ),
    FieldSchema(name='text', dtype=DataType.VARCHAR, description='text', max_length=65535),
        FieldSchema(name='page_no', dtype=DataType.VARCHAR, description='page_no', max_length=100),
        FieldSchema(name='filename', dtype=DataType.VARCHAR, description='filename', max_length=300),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, description='sparse embedding vectors'),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, description='dense embedding vectors', dim=dense_dim),
    ]

    collection_name = config['collection_name']
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
        print(f"Dropped existing collection: {collection_name}")
    schema = CollectionSchema(fields, "")

    col = Collection(collection_name, schema, consistentcy_level="Strong")

    # creating indexes for vectors
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    dense_index = {"index_type": "FLAT", "metric_type": "COSINE"}
    col.create_index("sparse_vector", sparse_index)
    col.create_index("dense_vector", dense_index)

    # insert data into collection
    entities = [
        docs_list,
        page_no_list,
        filename_list,
        docs_embeddings["sparse"],
        docs_embeddings["dense"],
    ]
    col.insert(entities)
    col.load()

    return len(docs_list)