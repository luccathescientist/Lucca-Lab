import sys
import json
import http.client

def send_reply(text):
    data = json.dumps({"text": text})
    conn = http.client.HTTPConnection("localhost", 8889)
    headers = {'Content-type': 'application/json'}
    conn.request("POST", "/api/chat/ai", data, headers)
    response = conn.getresponse()
    print(response.read().decode())
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_reply(" ".join(sys.argv[1:]))
