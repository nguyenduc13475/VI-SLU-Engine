# 🇻🇳 Vi-SLU: Vietnamese Spoken Language Understanding API for Smart Homes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B)

**Author:** Nguyễn Văn Đức  
**Project:** Extracting Intents and Time Slots from natural Vietnamese spoken commands and compiling them into a **Stateless JSON Execution Plan** for IoT/Smart Home systems.

---

## 📺 Project Demo

Below is a quick demonstration of the **Vi-SLU Engine** in action, parsing complex Vietnamese commands into structured JSON execution plans.

![Vi-SLU Demo](assets/demo.gif)

---

## 🌟 Key Features

* **Stateless API Architecture (Microservices):** The heart of the system. Instead of the API holding threads (thread-blocking) to count down hardware execution times, the NLP system solely handles semantic parsing and returns a standardized `execution_plan` JSON array. Time countdowns and hardware delays are offloaded to Edge Devices/Gateways.
* **Dual-Engine NLP:**
    * **Core AI (BiGRU):** A Bidirectional Gated Recurrent Unit neural network combined with Vietnamese Word2Vec, capable of learning complex contexts and accurately identifying nested time intervals.
    * **Fallback (Rule-Based):** An ultra-lightweight N-gram and Sliding Window mechanism that runs in a flash, ensuring the system always provides a response even when the AI encounters Out-Of-Vocabulary (OOV) words.
* **Smart Time Parser:** Automatically converts natural language time phrases (e.g., "sau 1 tiếng rưỡi" - after an hour and a half, "trong 5 phút" - within 5 minutes) into precise total seconds for seamless hardware execution.
* **Built-in UI/UX:** Comes with auto-generated API documentation (Swagger UI) and an intuitive Web Demo interface (Streamlit).

---

## 🏗 Project Structure

```text
Vi_SLU_Engine/
├── assets/                # Project assets
├── data/                  # Vietnamese Dataset (Train & Validation)
├── models/                # Pre-trained weights (BiGRU) & Word2Vec
├── src/                   # Main source code
│   ├── api/               # FastAPI layer (Routing, Schemas)
│   ├── core/              # Central Config & Hyperparameters
│   ├── engine/            # NLU Models (BiGRU, Rule-based, Time Parser, Interpreter)
│   └── utils/             # Helper functions (Metrics, Audio processing)
├── scripts/               # CLI scripts (Train, Evaluate, Infer)
├── ui/                    # Streamlit Web App interface
└── requirements.txt       # Dependencies list
```

---

## 🚀 Installation & Execution Guide

Since the repository already includes the AI weight files (`models/bigru_model.pth`) and the embedding dictionary, you can clone and run it immediately without wasting time retraining!

### 1. Environment Setup

Open a terminal at the project's root directory and run:

```bash
# Create and activate a virtual environment (Recommended)
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the REST API Server (FastAPI)

This is the core service serving IoT devices:

```bash
python -m src.api.main
```

* The server runs at: `http://localhost:8000`
* **👉 Open the browser to view the API Docs (Swagger UI):** `http://localhost:8000/docs`

### 3. Start the Web UI Demo (Streamlit)

Open a new Terminal (remember to activate `venv` again), ensure the API Server from step 2 is still running, and execute:

```bash
streamlit run ui/app.py
```

A web interface will automatically open, allowing you to input commands and visually observe the JSON breakdown.

### 4. Other CLI Support Commands

* **Quick interactive test via Terminal:** `python -m scripts.infer_cli`
* **Evaluate accuracy:** `python -m scripts.evaluate`
* **Retrain the Model:** `python -m scripts.train`

---

## 💡 JSON Execution Plan Example

When a user says: *"tắt đèn sau 5 giây rồi sau đó bật quạt nhanh lên sau 10 giây nữa"*

The system **does not** force the Server to wait. It immediately analyzes and returns a plan like the one below for the Gateway to handle:

**Request (POST `/api/v1/parse`)**

```json
{
  "text": "tắt đèn sau 5 giây rồi sau đó bật quạt nhanh lên sau 10 giây nữa"
}
```

**Response (200 OK)**

```json
{
  "raw_text": "tắt đèn sau 5 giây rồi sau đó bật quạt nhanh lên sau 10 giây nữa",
  "intents": [
    "TatDen",
    "BatQuat",
    "QuatNhanh",
  ],
  "execution_plan": [
    {
      "action": "LED_OFF",
      "delay_seconds": 5,
      "duration_seconds": null,
      "interval_seconds": null,
      "hold_seconds": null
    },
    {
      "action": "FAN_ON",
      "delay_seconds": 15.0,
      "duration_seconds": null,
      "interval_seconds": null,
      "hold_seconds": null
    },
    {
      "action": "FAN_SPEED_UP",
      "delay_seconds": 15.0,
      "duration_seconds": null,
      "interval_seconds": null,
      "hold_seconds": null
    },
  ]
}
```

*This design philosophy allows the backend system to serve millions of IoT devices simultaneously without the risk of thread exhaustion.*

---

*Developed with ❤️ by Nguyễn Văn Đức.*