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
from docling.document_converter import DocumentConverter, PdfFormatOption # import extra support for .json, markdown, HTML
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
import time

# new packages
from docling.document_converter import MarkdownFormatOption, FormatOption
from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.pipeline.simple_pipeline import SimplePipeline
from langchain_ibm.embeddings import WatsonxEmbeddings
from docling_core.types.doc import DoclingDocument, TextItem, GroupItem, ContentLayer, RefItem, DocumentOrigin

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter # more splitter support / code-level custom splitters


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

def init_environment(collection_name, chunk_type): # chunk type is where you can choose between a markdown and a recursive text splitter
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


# MAIN INGESTION PIPELINE

# THINGS TO CHANGE: 
'''
DENSE EMBEDDING SUPPORT: ibm/slate-125m-english-rtrvr
    parameter: choosing hybrid, dense+sparse, or just dense
SPARSE EMBEDDING SUPPORT: BM25 sparse embedding model with BM25BuiltInFunction
LOADER SUPPORT: including just markdown loaders for now
DOCLING SUPPORT: vs langchain splitters
CODE LEVEL CUSTOMIZATIONS (CUSTOM SCHEMA)
'''

# Step 1: look at main pipeline functions and see where they are called
# check the functions to see what functionalities they have
# Step 2: modify the functions so that they work one step at a time
# Step 3: test the functions

def json_document_converter(filename: str, doc_name: str = None) -> DoclingDocument:
    # code-level, some level of custom code for only 'insight headings', create separate file for custom code
    with open(filename, 'r') as file:
        json_data = json.load(file)

    text_items = []
    paragraph_texts = []
    text_refs = []

    for key, val in json_data.items():
        if isinstance(val, list):
            for item in val:
                paragraph_texts.append(f'{key}: {item}')
        else: 
            paragraph_texts.append(f'{key}: {val}')
    
    for i in range(len(paragraph_texts)):
        text_items.append(TextItem(self_ref=f'#/texts/{i}', label='text', orig=filename, text=paragraph_texts[i]))
        text_refs.append(RefItem(cref=f'#/texts/{i}'))
    
    body_group = GroupItem(
        name="_root_",
        self_ref="#/body",
        content_layer=ContentLayer.BODY,
        children=[]
        )   
    
    origin_ = DocumentOrigin(mimetype='application/json', filename=filename, binary_hash=0) # set binary_hash = 0 just for functionality -- import hash function only if proceeding with this
    
    doc = DoclingDocument(schema_name='DoclingDocument', version='1.5.0', name=doc_name, body=body_group, groups=[body_group], texts=text_items, origin=origin_)

    for i in range(len(text_items)):
        body_group._add_child(doc=doc, stack=[], new_ref=text_refs[i])

    #for i in range(len(text_refs)):
    #    text_refs[i].parent = body_group

    return doc

def ingest_files(config, files):
    # set embedding model and tokenizer
    # add parameter here so that you can determine hybrid, dense, dense+sparse

    embedding_model='BAAI/bge-m3' # SET AS PARAMETER

    ef = BGEM3EmbeddingFunction(model_name=embedding_model, use_fp16=False, device="cpu")
    ef_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # ef_max_tokens = 1024

    # convert docs to docling documents and use hybrid chunker
    # here, add other loaders like json loader support -> convert into docling doc
    # Markdown
    pipeline_options = PdfPipelineOptions() # for specific 
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use ACCURATE if you prefer more accurate TableFormer model, however it's slower

    document_converter = DocumentConverter(format_options=
                                        {
                                            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                                            #InputFormat.MD: MarkdownFormatOption()
                                            })
    
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
            filename = file['filename']
            print(f"Converting {file_full_path} to a docling document and using hybrid chunker...")
            start_time = time.time()

            file_ext = filename.lower().split('.')[-1]
            #print(file_ext)

            start_time = time.time()
            
            if file_ext == 'json':
                doc = json_document_converter(file_full_path, 'json_doc')
            else:
                doc = document_converter.convert(file_full_path).document

            chunker = HybridChunker(tokenizer=ef_tokenizer)
            chunk_iter = chunker.chunk(dl_doc=doc)

            # iterate over chunks, creating a chunk document object per chunk with the content and metadata
            chunked_document_objects = []
            for chunk in chunk_iter:
                chunked_document_object = {"page_content": chunker.contextualize(chunk=chunk), "metadata": extract_metadata(chunk)}
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
            file_ext = filename.lower().split('.')[-1]
            #print(file_ext)

            start_time = time.time()
            
            if file_ext == 'json':
                doc = json_document_converter(file_full_path, 'json_doc').export_to_markdown()
            else: 
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
        password=config['wxd_milvus_password']
    )
    

    # defining collection
    # CODE LEVEL: INSERT YOUR OWN COLLECTION SCHEMA
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="id", dtype=DataType.VARCHAR, description='IDs', is_primary=True, auto_id=True, max_length=100
        ),
        FieldSchema(name='text', dtype=DataType.VARCHAR, description='text', max_length=65535),
        FieldSchema(name='page_no', dtype=DataType.VARCHAR, description='page_no', max_length=100),    # comment out these since not every collection needs to include this
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