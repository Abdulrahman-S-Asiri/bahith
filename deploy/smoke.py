"""Model-free container smoke test; uses only the Python standard library."""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

base = sys.argv[1].rstrip('/')


def fetch(path, **kwargs):
    request = urllib.request.Request(base + path, **kwargs)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


for attempt in range(30):
    try:
        if fetch('/health')[0] == 200:
            break
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(1)
else:
    raise SystemExit('Container did not become live')

status, body = fetch('/')
assert status == 200 and 'باحث · المعرفة تبدأ بسؤال'.encode() in body
assert fetch('/', headers={'Host': 'untrusted.example'})[0] == 400
assert fetch('/documents', method='POST', data=b'')[0] in (403, 405)
assert fetch('/feedback', method='POST', data=b'')[0] in (403, 405)
status, body = fetch('/api/query?' + urllib.parse.urlencode({'q': 'القراءة', 'method': 'keyword'}))
assert status == 200 and json.loads(body)['results']
assert fetch('/ready')[0] == 503, 'Unloaded model must not report search readiness'
print('Container smoke passed: Bahith page, keyword retrieval, host checks, mutation protection, honest readiness.')
