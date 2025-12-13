CRITICAL ERROR: PYTHON 3.14 IS UNSTABLE
=======================================

The "Connection Error" you are seeing is because Python 3.14 is currently in a testing phase and causes the 'numpy' library to crash on Windows. Streamlit cannot run without numpy.

HOW TO FIX:
-----------
1. Download Python 3.11 (Stable) from:
   https://www.python.org/downloads/release/python-3119/

2. Run the installer and make sure to check "Add Python to PATH".

3. Once installed, come back to this folder and run the following command in your terminal:

   py -3.11 -m pip install -r requirements.txt
   py -3.11 -m streamlit run app.py

This will force the project to use the stable Python 3.11 version instead of the crashing 3.14 version.
