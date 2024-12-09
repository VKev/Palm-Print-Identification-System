from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import NotFoundError, ConnectionError
import numpy as np

def create_index_if_not_exists(es_client, index_name, index_mapping):
    try:
        if es_client.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists.")
            return True

        es_client.indices.create(index=index_name, body=index_mapping)
        print(f"Created index '{index_name}'")
        return True

    except ConnectionError:
        print("Failed to connect to Elasticsearch. Check connection settings.")
        return False
    except Exception as e:
        print(f"Error creating index: {e}")
        return False



def list_all_indices(es_client):
    try:
        indices = es_client.cat.indices(format='json')
        
        print("Existing Indices:")
        for index in indices:
            print(f"Index: {index['index']} | "
                  f"Docs: {index['docs.count']} | "
                  f"Storage Size: {index['store.size']}")
        
        return [index['index'] for index in indices]
    
    except Exception as e:
        print(f"Error listing indices: {e}")
        return []

def get_total_document_count(es, index_name):
    try:
        # Get the total number of documents in the index
        count_response = es.count(index=index_name)
        return count_response['count']
    except Exception as e:
        print(f"Error getting document count: {e}")
        return None

def list_documents_in_index(es_client,index_name, max_docs=10, debug=True):
    try:
        # Search query to retrieve documents
        search_query = {
            "size": max_docs,  # Limit number of documents
            "query": {"match_all": {}}  # Retrieve all documents
        }
        
        results = es_client.search(
            index=index_name, 
            body=search_query
        )
        
        if debug:
            print(f"\nDocuments in Index '{index_name}':")
            for hit in results['hits']['hits']:
                print(f"Document ID: {hit['_id']}")
                print("Source:")
                print(hit['_source'])
                print("-" * 50)
        
        return results['hits']['hits']
    
    except Exception as e:
        print(f"Error retrieving documents from {index_name}: {e}")
        return []

def delete_index(es_client, index_name):
    try:
        # Perform index deletion
        response = es_client.indices.delete(index=index_name)
        
        print(f"Successfully deleted index: {index_name}")
        return True
    
    except NotFoundError:
        print(f"Index '{index_name}' does not exist")
        return False
    except Exception as e:
        print(f"Error deleting index {index_name}: {e}")
        return False

def insert_document(es, index_name, document, auto_id=False):
    try:
        # If auto_id is True, omit the ID to let Elasticsearch generate one
        if auto_id:
            response = es.index(
                index=index_name,
                body=document
            )
        else:
            # If auto_id is False, require a document ID in the input
            if 'id' not in document:
                raise ValueError("Document ID is required when auto_id is False")
            
            response = es.index(
                index=index_name,
                id=document['id'],
                body=document
            )
            print("Inserted success to Index ", index_name)
        return response
    
    except Exception as e:
        print(f"Error inserting document: {e}")
        raise

def delete_all_documents_in_index(es_client, index_name):
    """
    Delete all documents in an index while preserving the index structure
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index to clear
    
    Returns:
        dict: Deletion operation results
    """
    try:
        # Delete by query to remove all documents
        delete_query = {
            "query": {
                "match_all": {}  # Matches all documents
            }
        }
        
        # Perform delete by query operation
        response = es_client.delete_by_query(
            index=index_name,
            body=delete_query
        )
        
        # Print deletion summary
        print(f"Deletion Summary for Index '{index_name}':")
        print(f"Total documents deleted: {response['deleted']}")
        print(f"Deletion took: {response['took']} ms")
        
        return response
    
    except Exception as e:
        print(f"Error deleting documents from index {index_name}: {e}")
        return None

def delete_document_by_id(es_client, index_name, document_id):
    """
    Delete a specific document from an index by its ID
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index containing the document
        document_id: Unique ID of the document to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Perform document deletion
        response = es_client.delete(
            index=index_name,
            id=document_id
        )
        
        print(f"Successfully deleted document {document_id} from index {index_name}")
        return True
    
    except NotFoundError:
        print(f"Document {document_id} not found in index {index_name}")
        return False
    except Exception as e:
        print(f"Error deleting document {document_id}: {e}")
        return False