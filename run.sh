source .env

export LD_LIBRARY_PATH=`.venv/bin/python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`

.venv/bin/python claxy.py | tee "./logs/$(date +%F-%H-%M-%S)-logfile.txt"
