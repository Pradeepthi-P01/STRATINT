# Deployment Guide & Cloud Setup

This guide provides instructions for deploying **STRATINT** to various hosting environments.

---

## Option 1: Streamlit Community Cloud (Free & Easiest)

Streamlit Community Cloud is free and deploys directly from your GitHub repository.

### Steps:
1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy STRATINT dashboard"
   git push origin main
   ```
2. **Open Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. **Deploy New App**:
   - Click **New app**.
   - Select your repository: `your-username/OSINT_Project`.
   - Branch: `main`.
   - Main file path: `app.py`.
4. **Configure Secrets**:
   - In **Advanced settings** -> **Secrets**, paste your environment variable:
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key_here"
     ```
5. Click **Deploy!** Your app will be live with a public URL in 2–3 minutes.

---

## Option 2: Docker Container Deployment (Universal)

The included [`Dockerfile`](file:///d:/OSINT_Project/Dockerfile) can be deployed to **Render**, **Railway**, **AWS ECS**, **Google Cloud Run**, or **DigitalOcean**.

### Local Docker Build & Run:
```bash
# 1. Build the Docker image
docker build -t stratint-osint .

# 2. Run the container
docker run -d -p 8501:8501 --env GEMINI_API_KEY="your_api_key" --name stratint stratint-osint
```
Visit `http://localhost:8501`.

---

## Option 3: Linux VPS (Ubuntu / Debian Server)

For deploying on an AWS EC2, DigitalOcean Droplet, or Linode instance:

### 1. System Setup
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone <repo-url> /var/www/osint_project
cd /var/www/osint_project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Configure Systemd Service
Create `/etc/systemd/system/stratint.service`:
```ini
[Unit]
Description=STRATINT OSINT Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/var/www/osint_project
Environment="PATH=/var/www/osint_project/.venv/bin"
EnvironmentFile=/var/www/osint_project/.env
ExecStart=/var/www/osint_project/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3. Start & Enable Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stratint
```
