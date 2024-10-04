# Description
VoxComputeTranscription is a simple client that can connect to Vox's API in order to obtain audio files, process and upload the transcription.

It is using [WhisperX](https://github.com/m-bain/whisperX) to generate the transcription.

# Installation
1. Install WhisperX using the [instructions from their repo](https://github.com/m-bain/whisperX?tab=readme-ov-file#setup-%EF%B8%8F)
2. Clone this repo, create a `secret_keys.py` file, copy the below variables and fill in with your data:
```python
vox_api_key = '' # Your Vox API key (generated from Vox > profile > API clients list
vox_hf_personal_token = '' # Your Hugging Face personal token
vox_api_online_server_url = '<Vox server address>/api/computing-server-online' # Replace <Vox server address> with your server address
vox_api_pending_transcriptions_url = '<Vox server address>/api/pending_transcriptions'
vox_api_transcribe_url = '<Vox server address>/api/transcript'
```
3. Install dependencies inside the env that you created in step 1
```commandline
pip install plyer pystray
```
4. Run VoxComputeTranscription: `python compute_whisperx.py`. To turn on the transcribing process (which will start processing the queue) right click the icon in system tray and click `Transcribe server`, to turn it off click it again. Before turning the transcribing server on, you can always check the pending queue by clicking the `Check pending queue` from tray icon.

## Known issues
1. If WhisperX takes too much to process the diarization. [Check this fix](https://github.com/m-bain/whisperX/issues/688#issuecomment-2028626119).