cd code
pyinstaller --name NeuroGuessr --windowed --add-data "../data:data"  --add-data "../code:code" -i "neuroguessr5.icns" neuroguessr_2d.py
