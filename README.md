# Copernicus Offshore Wave Gateway for Render

Files:
- gateway.py
- requirements.txt
- render.yaml

The matching standalone HTML expects this gateway.

## Deploy on Render

1. Put these three files in a GitHub repository.
2. In Render choose New > Web Service and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn gateway:app --host 0.0.0.0 --port $PORT`
5. Deploy.
6. Copy the resulting `https://...onrender.com` URL.
7. Paste that URL in the HTML under Advanced connection settings.

No Copernicus credentials are stored in the repository. Users enter their own credentials in the HTML. The gateway keeps them in server memory only for the temporary session.
