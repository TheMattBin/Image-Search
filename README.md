## Setup Steps

1. **Create a new Conda environment:**
    ```bash
    conda create -n myenv python=3.10
    ```

2. **Activate the environment:**
    ```bash
    conda activate myenv
    ```

3. **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI server with Uvicorn (development mode):**
    ```bash
    uvicorn main.py --reload
    ```

5. **Start the frontend (development mode):**
    ```bash
    npm start
    ```

6. **Build the frontend for production:**
    ```bash
    npm run build
    ```