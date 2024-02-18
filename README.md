# HCMUT Chatbot

## How to install
```
pip install requirements.txt
```

## Developing mode
```
python main.py --dev
```

## Production mode
- **Step 1:** Reserve VRAM for embedding model
```
python reserve_mem.py
```
- **Step 2:** Run TGI
- **Step 3:** Go back to terminal at Step 1 and input 'q'
- **Step 4:** Start the application
```
uvicorn main:app --host 0.0.0.0 \
                 --port 8000 \
                 --workers 4 \
                 --root-path llama/haystack

or

python main.py
```
