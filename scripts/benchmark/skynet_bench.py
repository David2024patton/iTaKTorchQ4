import requests, time

long_prompt = "You are an expert. " * 150 + "Write a haiku."
print("Running Run 1 (Uncached)...")
t0 = time.time()
try:
    r1 = requests.post('http://localhost:8085/v1/chat/completions', json={'model':'test','messages':[{'role':'user','content':long_prompt}],'max_tokens':50}).json()
    t1 = time.time()
    print('Run 1 (Uncached):', round(t1-t0, 3), 'seconds')
except Exception as e:
    print("Run 1 failed:", e)

time.sleep(1)

print("Running Run 2 (Cached)...")
t2 = time.time()
try:
    r2 = requests.post('http://localhost:8085/v1/chat/completions', json={'model':'test','messages':[{'role':'user','content':long_prompt}],'max_tokens':50}).json()
    t3 = time.time()
    print('Run 2 (Cached):', round(t3-t2, 3), 'seconds')
except Exception as e:
    print("Run 2 failed:", e)
