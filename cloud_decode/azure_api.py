# coding=utf-8
import glob

import azure.cognitiveservices.speech as speech_sdk
import click

from .utils import *
from .account import ACCOUNT

__api_key__ = ACCOUNT["Microsoft"]["speech_key"]
__api_region__ = ACCOUNT["Microsoft"]["location"]


def azure_decode_multi(audio_files: List[str], key: str = __api_key__, region: str = __api_region__, save_result: bool = True, max_workers: int = None, max_repeat_time: int = 1, wait_seconds: int = 10, output_format="transaction", language="zh-CN") -> List[str]:
    """
    using multi-process to decode audio files asynchronously.
    Because of the CancelException of cortana api, you can use max_repeat_time to control the re-decode times, and use wait_seconds in seconds to control the wait time between two queries.
    :return: transactions of every audio file.
    """
    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []
        for audio_file in audio_files:
            jobs.append(
                _executor_.submit(
                    azure_decode, audio_file, key, region, save_result, max_repeat_time, wait_seconds, output_format, language
                )
            )
        results = wait_for_jobs(jobs, _executor_, "[Azure] Decode Sample Progress: ")
    return results


def azure_find(json_file: str, expected_string: str, delete_stop_words: bool = False) -> Tuple[bool, Union[str, None], Union[float, None]]:
    with open(json_file, 'r') as _file_:
        decode_result = json.load(_file_)

    did_find, decode_string, find_confidence = False, None, None
    if decode_result['success'] is True and isinstance(decode_result['result'], dict):
        result = decode_result['result']
        n_best = result['NBest']
        for result in n_best:
            display = check_transaction(result['Display'], delete_stop_words)
            split_display = display.split()
            confidence = result['Confidence']
            if all([expected_word in split_display for expected_word in expected_string.split()]):
                did_find = True
                if find_confidence is None or find_confidence < confidence:
                    find_confidence = confidence
                    decode_string = display

    return did_find, decode_string, find_confidence


def azure_result_2_prob(json_result, transaction, _delete_stop_words_: bool = False, _error_prob_: float = float('-inf')) -> float:
    transaction = check_transaction(transaction, _delete_stop_words_)

    if not json_result['success']:
        return _error_prob_
    result = json_result['result']
    if result == "Miss Error" or not isinstance(result, dict):
        return _error_prob_
    n_best = result['NBest']

    _is_transaction_in_ = False
    max_prob = 0.0
    for k_result in n_best:
        confidence = k_result['Confidence']
        display = k_result['Display']
        display = check_transaction(display, _delete_stop_words_)
        if transaction in display:
            _is_transaction_in_ = True
            max_prob = max(confidence, max_prob)

    if _is_transaction_in_:
        return max_prob
    else:
        return _error_prob_


@exception_printer
def azure_decode(audio_file: str, key: str = __api_key__, region: str = __api_region__, save_result: bool = False, _max_repeat_time_: int = 100, _wait_seconds_: int = 10, language="en-US"):
    speech_config = speech_sdk.SpeechConfig(subscription=key, region=region)
    speech_config.output_format = speech_sdk.OutputFormat.Detailed


    try:
        wav_result = {"success": False}
        _flag_ = True
        repeat_time = 0
        audio_config = speech_sdk.AudioConfig(filename=audio_file)
        speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config, language=language)
        while _flag_ and repeat_time < _max_repeat_time_:
            _flag_ = False
            repeat_time += 1
            response = speech_recognizer.recognize_once()
            if response.reason == speech_sdk.ResultReason.RecognizedSpeech:
                wav_result = {
                    "success": True,
                    "result": json.loads(response.json)['DisplayText']
                }
            elif response.reason == speech_sdk.ResultReason.NoMatch:
                wav_result = {
                    "success": True,
                    "result": "No Match"
                }
            elif response.reason == speech_sdk.ResultReason.Canceled:
                _flag_ = True
                if repeat_time < _max_repeat_time_:
                    time.sleep(_wait_seconds_)  # wait for 30 seconds
                    logger.warning("Decode wave {} failed. Repeat Again {}.".format(audio_file, repeat_time))
                else:
                    logger.warning("Wav {} recognition canceled: {}".format(audio_file, response.cancellation_details.reason))
                    wav_result = {
                        "success": False,
                        "result": "Canceled"
                    }
            else:
                logger.error("Azure recognize wav file {} failed: {}".format(audio_file, response.reason))
                wav_result = {
                    "success": False,
                    "result": response.reason
                }
                print(response.reason)
    except InterruptedError as _err_:
        raise _err_
    except Exception:
        traceback.print_exc()
        logger.error("Azure recognize wav file {} failed.".format(audio_file))
        wav_result = {
            "success": False,
            "result": "Miss Error"
        }

    if save_result:
        audio_folder = os.path.dirname(audio_file)
        wav_name = os.path.splitext(os.path.basename(audio_file))[0]
        wav_decode_file = os.path.join(audio_folder, "{}{}".format(wav_name, AZURE_JSON_SUFFIX))
        with open(wav_decode_file, 'w') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False)

    return wav_result['result'],wav_result['success']

@click.command("API Recognize Script for Azure Cortana.")
@click.argument("wav_folder", type=click.Path(exists=True))
@click.option("--region", default=__api_region__, type=str, help="Region for decoding API. Default is centralus.")
@click.option("--key", default=__api_key__, type=str, help="Key for decoding API. Default is ***.")
@click.option("--language", '-l', default="en-US", type=click.Choice(['en-US', 'zh-CN'], case_sensitive=True), help="Language code in ['en-US', 'zh-CN']")
@click.option("--re_decode_all", expose_value=True, is_flag=True, default=False, help="Whether re-decode all the wav files. Default is False.")
@click.option("--re_decode_failed", expose_value=True, is_flag=True, default=False, help="Whether to re-decode the failed decoding wav files. Default is False.")
@click.option("--max_workers", default=4, type=int, help="The number of the multi-process. Default is MAX.")
def api_recognize(wav_folder: str, region: str, key: str, language: str = "zh-CN", re_decode_all: bool = False, re_decode_failed: bool = False, max_workers: int = 4):
    logger.info("Start decode folder {} using Azure Cortana.".format(wav_folder))

    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []

        wav_files = glob.glob(os.path.join(wav_folder, "**", "*.wav"), recursive=True)
        wav_files = filter_irrelevant_wav(wav_files)
        for wav_file in wav_files:
            wav_decode_file = wav_file.replace(".wav", AZURE_JSON_SUFFIX)

            if os.path.exists(wav_decode_file):
                try:
                    with open(wav_decode_file, 'r') as _file_:
                        wav_result = json.load(_file_)

                        if wav_result['success'] and not re_decode_all:  # Success, no re-decode
                            continue
                        elif not wav_result['success'] and not re_decode_failed:  # Failed but no re-decode
                            continue
                except json.decoder.JSONDecodeError:
                    logger.warning("Load wav_decode_file '{}' Error. Re-decode the wav file using Azure.".format(wav_decode_file))

            jobs.append(
                _executor_.submit(azure_decode, wav_file, key, region, language=language)
            )

        wait_for_jobs(jobs, _executor_)

    logger.info('Decoding folder {} Done!'.format(wav_folder))


if __name__ == '__main__':
    api_recognize()
