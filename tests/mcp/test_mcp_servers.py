# /// script
# dependencies = [
#   "requests",
# ]
# ///

import json
import queue
import sys
import threading
import time
import urllib.parse
import uuid

try:
    import pytest
except ImportError:
    pytest = None  # Allow running standalone without pytest

if pytest:
    requests = pytest.importorskip("requests")
else:
    import requests

# Color codes for output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _endpoint_reachable(url: str) -> bool:
    try:
        response = requests.get(url, timeout=1.0)
        return response.status_code < 500
    except Exception:
        return False


if pytest and not (
    _endpoint_reachable("http://localhost:11235/health")
    and _endpoint_reachable("http://localhost:38081/mcp")
):
    pytestmark = pytest.mark.skip(
        reason="MCP integration services are not reachable; start docker compose first"
    )

def log(msg, color=RESET):
    print(f"{color}{msg}{RESET}")

class MCPSSEClient:
    def __init__(self, url):
        self.url = url
        self.session_id = None
        self.post_endpoint = None
        self.response_stream = None
        self.listen_thread = None
        self.running = False
        self.responses = queue.Queue()

    def connect(self):
        log(f"Connecting to SSE endpoint: {self.url}...")
        try:
            # We must accept text/event-stream
            headers = {
                "Accept": "text/event-stream, application/json"
            }
            self.response_stream = requests.get(self.url, stream=True, timeout=10, headers=headers)
            
            if self.response_stream.status_code != 200:
                log(f"Failed to connect. Status: {self.response_stream.status_code}. Body: {self.response_stream.text}", RED)
                return False

            # Start a background thread to listen for events
            self.running = True
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)
            self.listen_thread.start()
            
            # Wait for the initial 'endpoint' event
            start_time = time.time()
            while time.time() - start_time < 5:
                if self.post_endpoint:
                    log(f"Received post_endpoint: {self.post_endpoint}", GREEN)
                    return True
                time.sleep(0.1)
                
            log("Timed out waiting for 'endpoint' event from SSE stream.", RED)
            return False
                
        except Exception as e:
            log(f"Connection failed: {e}", RED)
            return False

    def _listen(self):
        try:
            current_event = None
            for line in self.response_stream.iter_lines():
                if not self.running:
                    break
                if not line:
                    continue
                
                decoded_line = line.decode('utf-8')
                # log(f"DEBUG RECV: {decoded_line}")
                
                if decoded_line.startswith('event: '):
                    current_event = decoded_line[7:].strip()
                elif decoded_line.startswith('data: '):
                    data = decoded_line[6:].strip()
                    
                    if current_event == 'endpoint':
                        # Handle endpoint event
                        if data.startswith('/'):
                            from urllib.parse import urlparse
                            parsed = urlparse(self.url)
                            self.post_endpoint = f"{parsed.scheme}://{parsed.netloc}{data}"
                        else:
                            self.post_endpoint = data
                            
                        # Extract session ID
                        if 'session_id=' in self.post_endpoint:
                            parsed_ep = urllib.parse.urlparse(self.post_endpoint)
                            qs = urllib.parse.parse_qs(parsed_ep.query)
                            if 'session_id' in qs:
                                self.session_id = qs['session_id'][0]
                    
                    elif current_event == 'message':
                        # This is a JSON-RPC response or notification
                        try:
                            msg = json.loads(data)
                            self.responses.put(msg)
                        except Exception as e:
                            log(f"Failed to parse JSON message: {e}", RED)
                            
                    current_event = None # Reset for next event
        except Exception as e:
            if self.running:
                log(f"SSE Listener error: {e}", RED)

    def send_request(self, method, params=None, wait_response=True):
        if not self.post_endpoint:
            log("No POST endpoint established.", RED)
            return None
            
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
            "params": params or {}
        }
        
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json"
        }
        
        try:
            # log(f"Sending POST to {self.post_endpoint}...")
            response = requests.post(self.post_endpoint, json=payload, headers=headers)
            
            if response.status_code not in [200, 202]:
                log(f"POST Error: {response.status_code} {response.text}", RED)
                return None
                
            if wait_response:
                # If we get a 200 with a body, that's the response (some servers do this)
                # If we get a 202, the response MUST come via SSE
                if response.status_code == 200 and response.text.strip():
                    try:
                        return response.json()
                    except Exception:
                        pass # Continue to wait for SSE if JSON parse fails
                
                # Wait for response in queue
                start_time = time.time()
                while time.time() - start_time < 10:
                    try:
                        msg = self.responses.get(timeout=0.1)
                        if msg.get("id") == request_id:
                            return msg
                        # If it's a notification or another message, keep it? 
                        # For this simple test, we just discard or log it.
                    except queue.Empty:
                        continue
                log(f"Timed out waiting for response to {method}", RED)
                return None
            else:
                return {"status": "sent", "code": response.status_code}
                
        except Exception as e:
            log(f"Request failed: {e}", RED)
            return None
            
    def initialize(self):
        log("Sending initialize...")
        res = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0"
            }
        })
        if res and "result" in res:
             log("Initialize successful!", GREEN)
             # log(f"Initialize result: {json.dumps(res['result'], indent=2)}", GREEN)
             # Send initialized notification
             self.send_request("notifications/initialized", wait_response=False)
             return True
        else:
             log(f"Initialize failed: {res}", RED)
        return False

    def list_tools(self):
        log("Listing tools...")
        res = self.send_request("tools/list")
        if res and "result" in res:
            tools = res["result"].get("tools", [])
            log(f"Found {len(tools)} tools:", GREEN)
            for t in tools:
                log(f" - {t['name']}: {t.get('description', 'No description')[:100]}...")
            return True
        else:
            log(f"List tools failed: {res}", RED)
        return False
    
    def close(self):
        self.running = False
        if self.response_stream:
            self.response_stream.close()

def test_crawl4ai():
    log("=== Testing Crawl4AI ===")
    client = MCPSSEClient("http://localhost:11235/mcp/sse")
    try:
        if client.connect():
            if client.initialize():
                client.list_tools()
    finally:
        client.close()
            
def test_searxng():
    log("\n=== Testing SearXNG ===")
    # SearXNG MCP server from the previous logs showed it needed a session_id
    session_id = str(uuid.uuid4())
    # The logs indicated the server expects 'sessionId' (based on the variable name in the log)
    url = f"http://localhost:38081/mcp?sessionId={session_id}"
    
    client = MCPSSEClient(url)
    try:
        if client.connect():
            if client.initialize():
                client.list_tools()
        else:
            log("Trying without session_id as fallback...", YELLOW)
            client_fallback = MCPSSEClient("http://localhost:38081/mcp")
            if client_fallback.connect():
                if client_fallback.initialize():
                    client_fallback.list_tools()
                client_fallback.close()
    finally:
        client.close()

if __name__ == "__main__":
    test_crawl4ai()
    test_searxng()
