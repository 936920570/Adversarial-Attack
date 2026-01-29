# coding=utf-8
import argparse
import glob
import io
import time
import google.cloud.speech_v1p1beta1 as speech
from google.protobuf.json_format import MessageToDict

from .account import ACCOUNT
from .utils import *


def google_decode_multi(audio_files: List[str], save_result: bool = True, max_workers: int = None, model: str = None, output_format: str = "transaction", language='zh') -> List[str]:
    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []
        for audio_file in audio_files:
            jobs.append(
                _executor_.submit(
                    google_decode, audio_file, model, save_result, output_format, language=language
                )
            )
        results = wait_for_jobs(jobs, _executor_, "[Google] Decode Sample Progress: ")
        return results


def google_find(json_file: str, expected_string: str, delete_stop_words: bool = False) -> Tuple[bool, Union[str, None], Union[float, None]]:
    with open(json_file, 'r') as _file_:
        decode_result = json.load(_file_)

    did_find, decode_string, find_confidence = False, None, None
    if decode_result['success'] is True and isinstance(decode_result['result'], dict):
        result = decode_result['result']  # type: dict
        alternatives = result.get('alternatives') or []
        for alternative in alternatives:  # type: dict
            if alternative.get('transcript') is None or alternative.get('confidence') is None:
                continue
            transaction = check_transaction(alternative['transcript'], delete_stop_words)
            split_transaction = transaction.split()
            confidence = alternative['confidence']
            if all(expected_word in split_transaction for expected_word in expected_string.split()):
                did_find = True
                if find_confidence is None or find_confidence < confidence:
                    find_confidence = confidence
                    decode_string = transaction

    return did_find, decode_string, find_confidence


def google_result_2_prob(json_result, transaction: str, _delete_stop_words_: bool = False, _error_prob_: float = float("-inf")) -> float:
    transaction = check_transaction(transaction, _delete_stop_words_)

    if not json_result['success']:
        return _error_prob_
    result = json_result['result']  # type: [None, dict]
    if result is None:
        return _error_prob_
    alternatives = result.get('alternatives')
    if alternatives is None:
        return _error_prob_

    _is_transaction_in_ = False
    max_prob = 0.0
    for alternative in alternatives:
        transcript = alternative['transcript']
        transcript = check_transaction(transcript, _delete_stop_words_)
        confidence = alternative['confidence']
        if transaction in transcript:
            _is_transaction_in_ = True
            max_prob = max(confidence, max_prob)

    if _is_transaction_in_:
        return max_prob
    else:
        return _error_prob_


@exception_printer
def google_decode(speech_file: str, model: str='command_and_search', save_result: bool = False, _max_alternatives_: int = 10, language: str = "en-US"):
    import os
    if "Google" in ACCOUNT and os.path.exists(ACCOUNT["Google"]):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ACCOUNT["Google"]
    else:
        # Fallback or keep existing if user hasn't set it in ACCOUNT
        if os.path.exists(r""):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r""
        else:
            print("[Warning] Google credentials not found. Please set ACCOUNT['Google'] in cloud_decode/account.py")


    __google_client__ = speech.SpeechClient()
    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.types.RecognitionAudio(content=content)

    config = speech.types.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        max_alternatives=_max_alternatives_,
        language_code=language,
        model=model
    )

    max_retry = 100
    for i in range(max_retry):
        try:
            response = __google_client__.recognize(config=config,audio=audio)
            break
        except Exception as e:
            if i == max_retry - 1:
                print(f"[Google API Error] {e}")
                return "", False
            time.sleep(1)

    if len(response.results):
        result = response.results[0].alternatives[0].transcript
        # Returns the result string and True for success
        return result, True
    else:
        # Returns empty string and False for failure
        return "", False

    if save_result:
        with open(speech_file.replace('.wav', "." + model + GOOGLE_JSON_SUFFIX), 'w') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False)

    return wav_result['result'],True



def api_recognize():
    logger.info("Start Decoding with Google.")
    wav_files = glob.glob(os.path.join(args.wav_folder, "**", "*.wav"), recursive=True)
    for wav_file in wav_files:
        wav_decode_file = wav_file.replace('.wav', "." + args.model + GOOGLE_JSON_SUFFIX)
        if os.path.exists(wav_decode_file):
            try:
                with open(wav_decode_file, 'r') as _file_:
                    wav_result = json.load(_file_)
                    if wav_result['success'] and not args.re_decode_all:  # Success, no re-decode
                        google_decode(wav_file, args.model, True, language=args.language)
                    elif not wav_result['success'] and not args.re_decode_failed:  # Failed but no re-decode
                        google_decode(wav_file, args.model, True, language=args.language)
            except json.decoder.JSONDecodeError:
                logger.warning("Load wav_decode_file Error. Re-decode the wav file using Google.")
        else:
            google_decode(wav_file, args.model, True, language=args.language)
    logger.info("Decode Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google Speech to Text.')
    parser.add_argument('wav_folder', type=str, help="Where is the wav_folder.")
    parser.add_argument('--re_decode_all', action="store_true", default=False, help="Whether re-decode all the wave files. Default is False.")
    parser.add_argument('--re_decode_failed', action="store_true", default=False, help="Whether to re-decode the failed decoding wave files. Default is False.")
    parser.add_argument('--max_workers', default=1, type=int, help="The number of the multi-process. Default is MAX.")
    parser.add_argument("--model", default='command_and_search', choices=['command_and_search', 'phone_call', 'video', 'default'])
    parser.add_argument("--language", "-l", default='en-US', choices=['en-US', 'zh'])
    args = parser.parse_args()
    google_decode('',save_result=True)

    # api_recognize()
