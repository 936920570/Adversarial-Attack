# coding=utf-8
import sys
import glob
import argparse
import json
import os
from typing import List
import time
import click
from .utils import *
from .openai_function import openai_recong


WHISPER_JSON_SUFFIX = ".whisper.json"

def openai_decode(audio_file: str, model: str = "whisper-1", language: str = None, prompt: str = None) -> tuple:

    try:
        file_name = os.path.basename(audio_file)
        
        result_text, success = openai_recong(audio_file, language, model, prompt)
        
        return result_text, success, file_name
                
    except Exception as e:
        error_msg = f"OpenAI Whisper API error: {str(e)}"
        print(f"Error processing {audio_file}: {error_msg}")
        return error_msg, False, os.path.basename(audio_file)


def whisper_decode_multi(audio_files: List[str], args=None, save_result: bool = True, 
                        output_format="transaction", re_decode_failed: bool = False) -> List[str]:

    if re_decode_failed:
        temp_audio_files = []
        for audio_file in audio_files:
            if not os.path.exists(audio_file.replace('.wav', WHISPER_JSON_SUFFIX)):
                temp_audio_files.append(audio_file)

        if len(temp_audio_files) == 0:
            print('All wav files are already decoded!\n')
            sys.exit()

        audio_files = temp_audio_files

    for i, audio_file in enumerate(audio_files):
        whisper_decode(audio_file, args)

        time.sleep(0.5)


@exception_printer
def whisper_decode(audio_file: str, args=None, save_result: bool = True, 
                  output_format="transaction", language=None):

    if (args is not None) and (args.re_decode_all == False):
        json_path = audio_file.replace('.wav', WHISPER_JSON_SUFFIX)
        if os.path.exists(json_path):
            print(f'#################### {audio_file} Is Already Decoded #################')
            return

    model = "whisper-1"
    target_language = None
    prompt = None
    
    if args is not None:
        if hasattr(args, 'model'):
            model = args.model
        if hasattr(args, 'language'):
            target_language = args.language
        if hasattr(args, 'prompt'):
            prompt = args.prompt
    elif language:
        target_language = language

    if target_language:
        if 'en' in target_language.lower():
            target_language = 'en'
        elif 'cn' in target_language.lower() or 'zh' in target_language.lower():
            target_language = 'zh'

    res, success, file_name = openai_decode(audio_file, model=model, 
                                           language=target_language, prompt=prompt)

    if success:
        wav_result = {
            "success": True,
            "errorMSG": None,
            "result": res,
            "confidence": None,
            "model": model
        }
    else:
        wav_result = {
            "success": False,
            "errorMSG": res,
            "result": "Recognition Failed",
            "confidence": None,
            "model": model
        }


    if save_result:
        with open(audio_file.replace('.wav', WHISPER_JSON_SUFFIX), 'w', encoding='utf-8') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False, indent=2)
    
    print(audio_file.split('/')[-1])
    print('RESULT:\t' + str(wav_result) + '\n')
    
    if output_format == "transaction":
        return wav_result['result'], success
        
    if output_format == "json":
        return wav_result, success


def api_recognize(args):

    print(f"Start decode folder {args.wav_folder} using OpenAI Whisper.\n")

    wav_files = glob.glob(os.path.join(args.wav_folder, "**", "*.wav"), recursive=True)
    wav_files = filter_irrelevant_wav(wav_files)
     
    whisper_decode_multi(wav_files, args, re_decode_failed=args.re_decode_failed)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='OpenAI Whisper Speech to Text.')
    parser.add_argument('wav_folder', type=str, help="Where is the wav_folder.")
    parser.add_argument('--re_decode_all', action="store_true", default=False, 
                       help="Whether re-decode all the wave files. Default is False.")
    parser.add_argument('--re_decode_failed', action="store_true", default=False, 
                       help="Whether to re-decode the failed decoding wave files. Default is False.")
    parser.add_argument('--max_workers', default=1, type=int, 
                       help="The number of the multi-process. Default is 1.")
    parser.add_argument("--language", "-l", default='auto', 
                       choices=['auto', 'en', 'zh', 'zh-CN', 'en-US'])
    parser.add_argument("--model", "-m", default='whisper-1', 
                       choices=['whisper-1'], help="Whisper model to use")
    parser.add_argument("--prompt", "-p", default=None, type=str,
                       help="Optional prompt to improve accuracy")
    
    args = parser.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    api_recognize(args)
