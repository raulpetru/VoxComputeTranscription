from multiprocessing import Process
import requests
import json
from datetime import datetime

from plyer.utils import platform
from plyer import notification

from secret_keys import vox_api_key, vox_hf_personal_token, vox_api_online_server_url, \
    vox_api_pending_transcriptions_url, vox_api_transcribe_url

import time

import pystray
from PIL import Image

# Global vars
server_status = False


# Notification
def pending_notification(pending_transcriptions_number):
    notification.notify(
        title='Pending transcriptions',
        message=f'{pending_transcriptions_number} transcriptions pending',
        app_name='Vox Notification',
        app_icon='../static/favicon.ico'.format(
            # On Windows, app_icon has to be a path to a file in .ICO format.
            'ico' if platform == 'win' else 'png'
        )
    )


# Tray image
tray_img = Image.open('../static/favicon.png')


def after_click(tray, query):
    global server_status, p2, p3
    if str(query) == 'Check pending queue':
        pending_transcriptions()
    elif str(query) == 'Transcribe server':
        if not server_status:
            p2.start()
            p3.start()
            server_status = not query.checked
        else:
            p2.terminate()
            p3.terminate()

            p2 = Process(target=online_server)
            p3 = Process(target=transcribe)
            server_status = not query.checked
    elif str(query) == 'Exit':
        tray.stop()
        p2.terminate()
        p3.terminate()
        exit()


# Import API key and API urls
api_key = vox_api_key
api_online_server_url = vox_api_online_server_url
api_pending_transcriptions_url = vox_api_pending_transcriptions_url
api_transcribe_url = vox_api_transcribe_url


def online_server():
    while True:
        r = requests.post(api_online_server_url, headers={'X-API-KEY': api_key})
        # print(r.text)
        print(f'Server online! Time: {datetime.now()}')
        time.sleep(30)


def pending_transcriptions():
    r = requests.get(api_pending_transcriptions_url, headers={'X-API-KEY': api_key})
    if r.status_code == 200 and r.json() != '{}':
        recording_id, audio_file = next(iter(r.json().items()))
        pending_notification(len(r.json()))
        return recording_id, audio_file
    else:
        return None


def transcribe():
    while True:
        pending_transcription = pending_transcriptions()
        if pending_transcription is not None:
            import torch
            import whisperx
            import gc

            recording = pending_transcription
            print(recording)

            # Unpack API response
            recording_id, audio_file = recording

            device = 'cuda'

            batch_size = 6  # reduce if low on GPU mem
            compute_type = 'float16'  # change to "int8" if low on GPU mem (may reduce accuracy)

            # 1. Transcribe with original whisper (batched)
            model = whisperx.load_model('large-v3', device, compute_type=compute_type)

            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size)
            # print(result["segments"])  # before alignment
            print(f'Step 1. Transcribe complete! Time: {datetime.now()}')

            # delete model if low on GPU resources
            # gc.collect()
            # torch.cuda.empty_cache()
            # del model

            # 2. Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result['language'], device=device)
            result = whisperx.align(result['segments'], model_a, metadata, audio, device, return_char_alignments=False)
            print(f'Step 2. Alignment complete! Time: {datetime.now()}')

            # gc.collect()
            # torch.cuda.empty_cache()
            # del model_a

            # 3. Assign speaker labels
            hf_token = vox_hf_personal_token
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            # print(result['segments'])  # segments are now assigned speaker IDs
            print(f'Step 3. Diarization complete! Time: {datetime.now()}')

            prev_speaker = ''
            result_text_string = ''
            for segment in result['segments']:
                if 'speaker' in segment:
                    if segment['speaker'] != prev_speaker:
                        if result_text_string == '':
                            result_text_string += f'- {segment["text"]}'
                        else:
                            result_text_string += f'\n- {segment["text"]}'
                    else:
                        if result_text_string[-1] in ['.', ',', '?', '!'] and segment['text'][0] not in [' ', '\n']:
                            result_text_string += ' '
                        result_text_string += f'{segment["text"]}'
                    prev_speaker = segment['speaker']
                else:
                    # If current segment doesn't have a speaker, append it to the last speaker's text
                    if result_text_string[-1] in ['.', ',', '?', '!'] and segment['text'][0] not in [' ', '\n']:
                        result_text_string += ' '
                    result_text_string += f'{segment["text"]}'
                    print(segment["text"])

            gc.collect()
            torch.cuda.empty_cache()
            del model, model_a, diarize_model

            # Build payload and send through API
            payload = {'id': recording_id, 'data': result_text_string}
            # Send transcription
            r = requests.post(api_transcribe_url, headers={'X-API-KEY': api_key},
                              data=json.dumps(payload))
            print(r.text)
        else:
            time.sleep(30)


if __name__ == '__main__':
    p2 = Process(target=online_server)
    p3 = Process(target=transcribe)
    os_tray = pystray.Icon('Vox', tray_img, 'Vox',
                           menu=pystray.Menu(pystray.MenuItem('Check pending queue', after_click),
                                             pystray.MenuItem('Transcribe server',
                                                              after_click, checked=lambda query: server_status),
                                             pystray.MenuItem('Exit', after_click)))
    p1 = Process(target=os_tray.run())
    p1.start()
