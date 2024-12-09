from elastic_search_util import *

es = Elasticsearch(
        "https://es01:9200",
        basic_auth=("elastic", "1Q2uCUMactIRZdk_uE1m"),
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

def cosine_similarity_search(es, index_name, query_vector, top_k=10):
    search_query = {
        "knn": {
            "field": "feature_vector",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 5  # Increases search accuracy
        },
        "sort": [
            {
                "_score": {
                    "order": "desc"  
                }
            }
        ]
    }
    
    try:
        # Execute the search
        results = es.search(
            index=index_name,
            body=search_query
        )
        
        # Process and return formatted results
        processed_results = []
        for hit in results['hits']['hits']:
            processed_results.append({
                'id': hit['_id'],
                'score': hit['_score'],
                'source': hit['_source']
            })
        
        return processed_results
    
    except Exception as e:
        print(f"Error performing vector similarity search: {e}")
        raise


def euclidean_distance_search(es, index_name, query_vector, top_k=10):
    search_query = {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": {"match_all": {}}
                    }
                },
                "script": {
                    "source": "1 / (1 + l2norm(params.queryVector, 'feature_vector'))",
                    "params": {
                        "queryVector": query_vector
                    },
                    "lang": "painless"
                }
            }
        },
        "size": top_k,
        "_source": True,
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            }
        ]
    }
    
    try:
        results = es.search(
            index=index_name,
            body=search_query
        )
        
        processed_results = []
        for hit in results['hits']['hits']:
            processed_results.append({
                'id': hit['_id'],
                'similarity_score': hit['_score'],
                'source': hit['_source']
            })
        
        return processed_results
    
    except Exception as e:
        print(f"Error performing L2 norm Euclidean distance search: {e}")
        raise

def bulk_index_vectors(es, index_name, student_id, feature_vectors):
    """
    Bulk index feature vectors for a single student, with each vector as a separate document.
    
    :param es: Elasticsearch client
    :param index_name: Name of the index to insert documents
    :param student_id: Single student identifier
    :param feature_vectors: List of feature vectors to index
    :return: Bulk indexing response
    """
    # Validate inputs
    if not feature_vectors:
        raise ValueError("No feature vectors provided for indexing")
    
    # Prepare bulk index operations
    bulk_body = []
    for i, vector in enumerate(feature_vectors):
        # Create document with student ID and feature vector
        doc = {
            "id": student_id,
            "feature_vector": vector,
        }
        
        # Add index operation
        bulk_body.append({"index": {}})
        bulk_body.append(doc)
    
    try:
        # Perform bulk indexing
        response = es.bulk(
            index=index_name,
            body=bulk_body
        )
        
        # Print insertion summary
        successful_count = sum(1 for item in response['items'] if item['index']['status'] in [200, 201])
        failed_count = len(response['items']) - successful_count
        
        print(f"Bulk Indexing Summary for Student {student_id}:")
        print(f"Total Vectors: {len(feature_vectors)}")
        print(f"Successfully Indexed: {successful_count}")
        print(f"Failed Indexings: {failed_count}")
        
        # Optionally print details of failed indexings
        if failed_count > 0:
            print("Failed Indexing Details:")
            for item in response['items']:
                if item['index']['status'] not in [200, 201]:
                    print(item['index'])
        
        return response
    
    except Exception as e:
        print(f"Error during bulk indexing: {e}")
        raise

def bulk_euclidean_similarity_search(es, index_name, query_vectors):
    processed_results = []
    
    for query_vector in query_vectors:
        search_query = {
            "size": 1,  # Top 1 result
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "1 / (1 + l2norm(params.query_vector, 'feature_vector'))",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
        }
        
        try:
            result = es.search(index=index_name, body=search_query)
            
            if result['hits']['total']['value'] > 0:
                hit = result['hits']['hits'][0]
                processed_results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
            else:
                processed_results.append(None)
        
        except Exception as e:
            print(f"Error in exhaustive search: {e}")
            processed_results.append(None)
    
    return processed_results

def bulk_cosine_similarity_search(es, index_name, query_vectors):
    processed_results = []
    
    for query_vector in query_vectors:
        search_query = {
            "size": 1,  # Top 1 result
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "(cosineSimilarity(params.query_vector, 'feature_vector') + 1.0)/2.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
        }
        
        try:
            result = es.search(index=index_name, body=search_query)
            
            if result['hits']['total']['value'] > 0:
                hit = result['hits']['hits'][0]
                processed_results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                })
            else:
                processed_results.append(None)
        
        except Exception as e:
            print(f"Error in exhaustive search: {e}")
            processed_results.append(None)
    
    return processed_results

def create_palm_print_index():
    index_mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},  # For exact matching of IDs
                "feature_vector": {
                    "type": "dense_vector",  # Supports vector similarity search
                    "dims": 640,  # Specify the dimension of your feature vector
                    "index": True,  # Allow indexing
                    "similarity": "cosine"  # Use cosine similarity for vector search
                }
            }
        }
    }

    index_name= "palm-print-index"
    create_index_if_not_exists(es, index_name, index_mapping)

from collections import Counter

def verify_palm_print(results):
    # Extract IDs
    ids = [result['source']['id'] for result in results if result is not None]
    
    # Find the most common ID
    id_counts = Counter(ids)
    most_common_id = id_counts.most_common(1)[0][0]
    
    # Calculate average score for the most common ID
    most_common_id_results = [
        result for result in results 
        if result is not None and result['source']['id'] == most_common_id
    ]
    
    avg_score = sum(result['score'] for result in most_common_id_results) / len(most_common_id_results)
    
    return {
        'most_common_id': most_common_id,
        'occurrence_count': id_counts[most_common_id],
        'average_similarity_score': avg_score,
        'average_occurrence_score': id_counts[most_common_id]/id_counts.total(),
        'accept': (avg_score + id_counts[most_common_id]/id_counts.total())/2 > 0.698555148144563
    }

if __name__ == "__main__":
    create_palm_print_index()
    list_documents_in_index(es , "palm-print-index")
    delete_all_documents_in_index(es,"palm-print-index")