@echo off
setlocal
set "PYTHONUTF8=1"
rem Use the environment that already contains Streamlit, TensorFlow, and Keras.
set "APP_PYTHON=%USERPROFILE%\anaconda3\envs\penv\python.exe"
if not exist "%APP_PYTHON%" (
    echo The penv Python environment was not found at "%APP_PYTHON%".
    echo See README.md for environment setup.
    exit /b 1
)
echo Python: %APP_PYTHON%
pushd "%~dp0"
"%APP_PYTHON%" -m streamlit run "%~dp0streamlit_app.py" %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %APP_EXIT_CODE%
