# Diavoc

Diavoc is a unique project that combines the power of machine learning (ML) and modern web development. Its primary goal is to provide insights into diabetes from voice inputs. The repository is divided into two key components:

- **Frontend**: Built using React and Vite to deliver a seamless and responsive user interface.
- **Backend**: Powered by a machine learning model to analyze and interpret voice data related to diabetes.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
    - [Frontend](#frontend)
    - [Backend](#backend)
- [Setup Instructions](#setup-instructions)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

### Purpose
Diavoc interprets voice input to gather health insights related to diabetes. It's designed to make health data interpretation more accessible.

### Architecture
The application is structured as follows:
1. **Frontend**: React application built with Vite.
2. **Backend**: Python-powered application leveraging a machine learning model to analyze audio inputs.

---

## Features

### Frontend
- Real-time interaction with the backend.
- Clean and fast user interface tailored for voice interactions.

### Backend
- **Voice-to-Diabetes** model processes voice inputs.
- Secure communication with the frontend.

---

## Technologies Used

### Frontend
- React
- Vite
- ESLint (customizable rules)

### Backend
- Python (FastAPI)
- Machine Learning frameworks
   - Random Forest
   - Ensemble Learning

---

## Setup Instructions

### Prerequisites
- Node.js and npm (for the frontend)
- Python 3.x (for the backend)

### Frontend Setup

1. Navigate to the `frontend` folder:
   ```sh
   cd frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```

### Backend Setup

1. Navigate to the `backend` folder:
   ```sh
   cd backend/app/ml_model
   ```
2. Install required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the backend application:
   ```sh
   fastapi app/main.py
   ```

---

## Impact

### Why This Matters
- **Early, Accessible Screening**:
  - Reduces preventable deaths by enabling early diagnosis and intervention.
- **Inclusive Healthcare Innovation**:
  - Focus on underserved groups and rural areas where lab access is limited.
- **Aligned with SDGs**:
  - Strengthens digital health systems and drives worldwide health equity.

