@ECHO OFF
:: This script downloads and extracts the GTZAN dataset
TITLE Extract GTZAN dataset
@ECHO Please wait, this may take a while...
:: Download the dataset from http://opihi.cs.uvic.ca/sound/genres.tar.gz
rem powershell -command "iex ((new-object net.webclient).downloadstring('http://opihi.cs.uvic.ca/sound/genres.tar.gz'))"
:: Extract the tar file to the current directory using 7-zip
7z x -y -o"%~dp0" genres.tar.gz
:: Delete the tar file
del genres.tar.gz
:: Create data folder  if it does not exist
if not exist "data" mkdir data

:: Move genres to data folder and delete the rest
move genres\*.* data
rd /s /q genres


