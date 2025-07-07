[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-6.8.8-Q16\magick.exe"
[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:294: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  lines_video = [l for l in lines if ' Video: ' in l and re.search('\d+x\d+', l)]
[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:367: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\d+$', l)]
[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:370: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-27T12:23:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  match = re.search('\d+$', rotation_line)
[2m2025-05-27T12:23:47Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Started server process [628]
[2m2025-05-27T12:23:47Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Waiting for application startup.
[2m2025-05-27T12:23:47Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Application startup complete.
[2m2025-05-27T12:23:47Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[2m2025-05-27T12:29:13Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]App mobius-api has excess capacity, autostopping machine 908027d3be7938. 1 out of 2 machines left running (region=dfw, process group=app)
[2m2025-05-27T12:29:13Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Sending signal SIGINT to main child process w/ PID 628
[2m2025-05-27T12:29:13Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Shutting down
[2m2025-05-27T12:29:13Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Waiting for application shutdown.
[2m2025-05-27T12:29:13Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Application shutdown complete.
[2m2025-05-27T12:29:13Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Finished server process [628]
[2m2025-05-27T12:29:14Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Main child exited normally with code: 0
[2m2025-05-27T12:29:14Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Starting clean up.
[2m2025-05-27T12:29:14Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][33m WARN[0m could not unmount /rootfs: EINVAL: Invalid argument
[2m2025-05-27T12:29:14Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][  335.305799] reboot: Restarting system
[2m2025-05-27T12:29:26Z[0m proxy[e286e73ef45768] [32mdfw[0m [[34minfo[0m]App mobius-api has excess capacity, autostopping machine e286e73ef45768. 0 out of 1 machines left running (region=dfw, process group=app)
[2m2025-05-27T12:29:26Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m][32m INFO[0m Sending signal SIGINT to main child process w/ PID 629
[2m2025-05-27T12:29:26Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m]INFO:     Shutting down
[2m2025-05-27T12:29:26Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m]INFO:     Waiting for application shutdown.
[2m2025-05-27T12:29:26Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m]INFO:     Application shutdown complete.
[2m2025-05-27T12:29:26Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m]INFO:     Finished server process [629]
[2m2025-05-27T12:29:27Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m][32m INFO[0m Main child exited normally with code: 0
[2m2025-05-27T12:29:27Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m][32m INFO[0m Starting clean up.
[2m2025-05-27T12:29:27Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m][33m WARN[0m could not unmount /rootfs: EINVAL: Invalid argument
[2m2025-05-27T12:29:27Z[0m app[e286e73ef45768] [32mdfw[0m [[34minfo[0m][  359.332019] reboot: Restarting system
[2m2025-05-28T00:45:43Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]Starting machine
[2m2025-05-28T00:45:44Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]2025-05-28T00:45:44.049239376 [01JW8V52E9CBY1D8FDNX693HN9:main] Running Firecracker v1.7.0
[2m2025-05-28T00:45:44Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Starting init (commit: a1a45272)...
[2m2025-05-28T00:45:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Preparing to run: `.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000` as root
[2m2025-05-28T00:45:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m [fly api proxy] listening at /.fly/api
[2m2025-05-28T00:45:45Z[0m runner[908027d3be7938] [32mdfw[0m [[34minfo[0m]Machine started in 1.144s
[2m2025-05-28T00:45:45Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]machine started in 1.148960666s
[2m2025-05-28T00:45:45Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]2025/05/28 00:45:45 INFO SSH listening listen_address=[fdaa:14:377c:a7b:41d:f0d:f44c:2]:22
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/config_defaults.py:47: SyntaxWarning: invalid escape sequence '\P'
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-6.8.8-Q16\magick.exe"
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:294: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  lines_video = [l for l in lines if ' Video: ' in l and re.search('\d+x\d+', l)]
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:367: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\d+$', l)]
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:370: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-28T00:45:50Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  match = re.search('\d+$', rotation_line)
[2m2025-05-28T00:45:50Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]waiting for machine to be reachable on 0.0.0.0:8000 (waited 5.231660101s so far)
[2m2025-05-28T00:45:52Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Started server process [628]
[2m2025-05-28T00:45:52Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Waiting for application startup.
[2m2025-05-28T00:45:52Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Application startup complete.
[2m2025-05-28T00:45:52Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[2m2025-05-28T00:45:52Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]machine became reachable in 7.237015006s
[2m2025-05-28T00:45:52Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:47646 - "OPTIONS /gpt HTTP/1.1" 200 OK
[2m2025-05-28T00:45:58Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:47656 - "POST /gpt HTTP/1.1" 200 OK
[2m2025-05-28T00:46:25Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:52378 - "OPTIONS /google HTTP/1.1" 200 OK
[2m2025-05-28T00:46:25Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:52382 - "POST /google HTTP/1.1" 200 OK
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Error converting audio .cache/tts/onyx_1.0_Hello__This_is_a_test_of_the_text_to_speech_functi.mp3 to video
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Traceback (most recent call last):
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/utils.py", line 421, in convert_audio_to_waveform_video
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    audio: AudioSegment = AudioSegment.from_file(audio_file_path)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/.venv/lib/python3.12/site-packages/pydub/audio_segment.py", line 728, in from_file
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    info = mediainfo_json(orig_file, read_ahead_limit=read_ahead_limit)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/.venv/lib/python3.12/site-packages/pydub/utils.py", line 274, in mediainfo_json
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    res = Popen(command, stdin=stdin_parameter, stdout=PIPE, stderr=PIPE)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/usr/local/lib/python3.12/subprocess.py", line 1026, in __init__
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    self._execute_child(args, executable, preexec_fn, close_fds,
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/usr/local/lib/python3.12/subprocess.py", line 1955, in _execute_child
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    raise child_exception_type(errno_num, err_msg, err_filename)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Error generating speech for text: Hello! This is a test of the text-to-speech functionality.
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Traceback (most recent call last):
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/utils.py", line 421, in convert_audio_to_waveform_video
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    audio: AudioSegment = AudioSegment.from_file(audio_file_path)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/.venv/lib/python3.12/site-packages/pydub/audio_segment.py", line 728, in from_file
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    info = mediainfo_json(orig_file, read_ahead_limit=read_ahead_limit)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/.venv/lib/python3.12/site-packages/pydub/utils.py", line 274, in mediainfo_json
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    res = Popen(command, stdin=stdin_parameter, stdout=PIPE, stderr=PIPE)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/usr/local/lib/python3.12/subprocess.py", line 1026, in __init__
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    self._execute_child(args, executable, preexec_fn, close_fds,
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/usr/local/lib/python3.12/subprocess.py", line 1955, in _execute_child
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    raise child_exception_type(errno_num, err_msg, err_filename)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]During handling of the above exception, another exception occurred:
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Traceback (most recent call last):
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/api.py", line 201, in generate_speech
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    waveform_video_file_path: Path = await services.generate_speech(text, voice, speed)
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/services.py", line 188, in generate_speech
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    convert_audio_to_waveform_video(
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/utils.py", line 553, in convert_audio_to_waveform_video
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    raise AudioProcessingError("Failed to convert audio to waveform video")
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]utils.AudioProcessingError: Failed to convert audio to waveform video
[2m2025-05-28T00:46:35Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:52902 - "POST /say HTTP/1.1" 500 Internal Server Error
