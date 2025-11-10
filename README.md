# Aurora Applied AI/ML Engineer — QA Service

## What this does
This service fetches a public member messages API and exposes a single endpoint:

`GET /ask?q=When is Layla planning her trip to London?`


## Follow the below steps to run the system
git clone <repo>
cd aurora-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python fetch_messages.py     # populate data/messages.json
python indexer.py            # optional: builds embeddings
python app.py                # start server on :8080
# then test:
Just input the question, and it will answer.
