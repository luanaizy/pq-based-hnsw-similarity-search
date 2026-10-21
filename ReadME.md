## How to Run the Project

Follow these steps to set up and run the project on your machine:

1. **Prerequisite:** Make sure you have **Python** installed.
2. **Clone the repository:** Clone the project to your local machine.
3. **Open the terminal:** Navigate into the project folder.
4. **Set up the Environment:**
   * **Create the virtual environment:**
     ```bash
     python -m venv .venv
     ```
   * **Activate the environment:**
     * **Git Bash:** `source .venv/Scripts/activate`
     * **CMD:** `.venv\Scripts\activate`
     
     *(To exit the environment at any time, just type `deactivate`)*

5. **Configure the code:** In `main.py`, adjust the `USE_GPU` flag to `True` if you have an NVIDIA GPU.
6. **Execute:** Run the application with:
   ```bash
   python main.py
   ```