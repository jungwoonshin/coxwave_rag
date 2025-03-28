import requests
import json
import sseclient  # pip install sseclient-py


def stream_chat(query, history=None, session_id=None):
    if history is None:
        history = []
    
    # Prepare the request body
    request_body = {
        "query": query,
        "history": history,
        "stream": True,
        "session_id": session_id
    }
    
    # Make the API call
    response = requests.post(
        "http://localhost:8000/api/chat/stream",
        json=request_body,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    
    # Create SSE client
    client = sseclient.SSEClient(response)
    
    # Full response to collect all chunks
    full_response = ""
    
    # Process the stream
    for event in client.events():
        chunk = json.loads(event.data)["data"]
        full_response += chunk
        
        # Print each chunk as it arrives
        print(chunk, end='', flush=True)
    
    print()  # Print newline at end
    return full_response

# Example usage
response = stream_chat("스마트 스토어 배송은 얼마나 걸리나요?")