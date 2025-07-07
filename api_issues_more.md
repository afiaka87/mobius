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
[2m2025-05-28T00:53:17Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]App mobius-api has excess capacity, autostopping machine 908027d3be7938. 0 out of 1 machines left running (region=dfw, process group=app)
[2m2025-05-28T00:53:17Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Sending signal SIGINT to main child process w/ PID 628
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Shutting down
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Waiting for application shutdown.
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Application shutdown complete.
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Finished server process [628]
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Main child exited normally with code: 0
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Starting clean up.
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][33m WARN[0m could not unmount /rootfs: EINVAL: Invalid argument
[2m2025-05-28T00:53:18Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][  454.486978] reboot: Restarting system
[2m2025-05-28T01:04:54Z[0m runner[e286e73ef45768] [32mdfw[0m [[34minfo[0m]Pulling container image registry.fly.io/mobius-api:deployment-01JWA6NJNJ4Y3C4W3T0FMTP07X
[2m2025-05-28T01:04:54Z[0m runner[908027d3be7938] [32mdfw[0m [[34minfo[0m]Pulling container image registry.fly.io/mobius-api:deployment-01JWA6NJNJ4Y3C4W3T0FMTP07X
[2m2025-05-28T01:05:04Z[0m runner[908027d3be7938] [32mdfw[0m [[34minfo[0m]Successfully prepared image registry.fly.io/mobius-api:deployment-01JWA6NJNJ4Y3C4W3T0FMTP07X (10.073677549s)
[2m2025-05-28T01:05:04Z[0m runner[908027d3be7938] [32mdfw[0m [[34minfo[0m]Configuring firecracker
[2m2025-05-28T01:05:06Z[0m runner[e286e73ef45768] [32mdfw[0m [[34minfo[0m]Successfully prepared image registry.fly.io/mobius-api:deployment-01JWA6NJNJ4Y3C4W3T0FMTP07X (12.820216276s)
[2m2025-05-28T01:05:07Z[0m runner[e286e73ef45768] [32mdfw[0m [[34minfo[0m]Configuring firecracker
[2m2025-05-28T01:07:14Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]Starting machine
[2m2025-05-28T01:07:15Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]2025-05-28T01:07:15.001601803 [01JWA6QESWMPD6T428Q9MKS69Y:main] Running Firecracker v1.7.0
[2m2025-05-28T01:07:15Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Starting init (commit: a1a45272)...
[2m2025-05-28T01:07:15Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m Preparing to run: `.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000` as root
[2m2025-05-28T01:07:15Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m][32m INFO[0m [fly api proxy] listening at /.fly/api
[2m2025-05-28T01:07:16Z[0m runner[908027d3be7938] [32mdfw[0m [[34minfo[0m]Machine started in 1.103s
[2m2025-05-28T01:07:16Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]machine started in 1.11019057s
[2m2025-05-28T01:07:16Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]2025/05/28 01:07:16 INFO SSH listening listen_address=[fdaa:14:377c:a7b:41d:f0d:f44c:2]:22
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/config_defaults.py:47: SyntaxWarning: invalid escape sequence '\P'
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-6.8.8-Q16\magick.exe"
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:294: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  lines_video = [l for l in lines if ' Video: ' in l and re.search('\d+x\d+', l)]
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:367: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\d+$', l)]
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/moviepy/video/io/ffmpeg_reader.py:370: SyntaxWarning: invalid escape sequence '\d'
[2m2025-05-28T01:07:21Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  match = re.search('\d+$', rotation_line)
[2m2025-05-28T01:07:21Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]waiting for machine to be reachable on 0.0.0.0:8000 (waited 5.406952268s so far)
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/pydub/utils.py:300: SyntaxWarning: invalid escape sequence '\('
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  m = re.match('([su]([0-9]{1,2})p?) \(([0-9]{1,2}) bit\)$', token)
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/pydub/utils.py:301: SyntaxWarning: invalid escape sequence '\('
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  m2 = re.match('([su]([0-9]{1,2})p?)( \(default\))?$', token)
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/pydub/utils.py:310: SyntaxWarning: invalid escape sequence '\('
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  elif re.match('(flt)p?( \(default\))?$', token):
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]/app/.venv/lib/python3.12/site-packages/pydub/utils.py:314: SyntaxWarning: invalid escape sequence '\('
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  elif re.match('(dbl)p?( \(default\))?$', token):
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Started server process [628]
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Waiting for application startup.
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Application startup complete.
[2m2025-05-28T01:07:22Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
[2m2025-05-28T01:07:23Z[0m proxy[908027d3be7938] [32mdfw[0m [[34minfo[0m]machine became reachable in 7.410629921s
[2m2025-05-28T01:07:23Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:55656 - "OPTIONS /anthropic HTTP/1.1" 200 OK
[2m2025-05-28T01:07:29Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:55660 - "POST /anthropic HTTP/1.1" 200 OK
[2m2025-05-28T01:07:39Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:58024 - "OPTIONS /gpt HTTP/1.1" 200 OK
[2m2025-05-28T01:07:44Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:58036 - "POST /gpt HTTP/1.1" 200 OK
[2m2025-05-28T01:08:19Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:63526 - "POST /anthropic HTTP/1.1" 200 OK
[2m2025-05-28T01:08:29Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:64976 - "OPTIONS /o1 HTTP/1.1" 200 OK
[2m2025-05-28T01:08:33Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:64980 - "POST /o1 HTTP/1.1" 200 OK
[2m2025-05-28T01:08:36Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:33396 - "OPTIONS /flux HTTP/1.1" 200 OK
[2m2025-05-28T01:08:46Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:33406 - "POST /flux HTTP/1.1" 200 OK
[2m2025-05-28T01:09:09Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:37276 - "POST /say HTTP/1.1" 200 OK
[2m2025-05-28T01:09:25Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:41568 - "OPTIONS /youtube HTTP/1.1" 200 OK
[2m2025-05-28T01:09:25Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:41570 - "POST /youtube HTTP/1.1" 200 OK
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:46686 - "OPTIONS /t2v HTTP/1.1" 200 OK
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]ComfyUI libraries (comfy_api_simplified) are not available.
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Error generating t2v for prompt: A cat playing with a ball of yarn in a sunny garden
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]Traceback (most recent call last):
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/api.py", line 475, in text_to_video
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    video_path: Path = await services.t2v(
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]                       ^^^^^^^^^^^^^^^^^^^
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]  File "/app/services.py", line 568, in t2v
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]    raise ValueError("ComfyUI integration is not properly installed or imported.")
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]ValueError: ComfyUI integration is not properly installed or imported.
[2m2025-05-28T01:10:01Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:46700 - "POST /t2v HTTP/1.1" 500 Internal Server Error
[2m2025-05-28T01:10:05Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:47168 - "OPTIONS /gptimg/generate HTTP/1.1" 200 OK
[2m2025-05-28T01:11:06Z[0m app[908027d3be7938] [32mdfw[0m [[34minfo[0m]INFO:     172.16.2.98:47170 - "POST /gptimg/generate HTTP/1.1" 200 OK
