# coding=utf-8
import sys
import glob
import argparse

from cloud_decode.tencentyun_function import tencent_recong
import click
import time
from utils import *


def tencentyun_decode_multi(audio_files: List[str], args = None, save_result: bool = True,output_format="transaction", re_decode_failed: bool = False) -> List[str]:
    if re_decode_failed == True:
        Temp_audio_files = []
        for audio_file in audio_files:
            if os.path.exists(audio_file.replace('.wav', TENCENT_JSON_SUFFIX)) == False:
                Temp_audio_files.append(audio_file)

        if len(Temp_audio_files) == 0:
            print('All wav files are already decoded!\n')
            sys.exit()

        audio_files = Temp_audio_files

    for i,audio_file in enumerate(audio_files):
        tencentyun_decode(audio_file, args)



@exception_printer
def tencentyun_decode(audio_file: str, args=None, save_result: bool = True, output_format="transaction", language='en-us'):
    if (args!=None) and (args.re_decode_all==False):
        json_path = audio_file.replace('.wav', TENCENT_JSON_SUFFIX)
        if os.path.exists(json_path)== True:
            print(f'#################### {audio_file} Is Already Decoded #################')
            return

    if args!=None:
        res, success, file_name = tencent_recong(audio_file, args.language)
    else:
        if 'en' in language.lower():
            language='en-US'
        elif 'cn' in language.lower():
            language='zh-CN'
        res, success, file_name = tencent_recong(audio_file, language) 

    if success==True:
        wav_result = {
            "success": True,
            "errorMSG": None,
            "result": res,
            "confidence": None
        }
    else:
        wav_result = {
            "success": False,
            "errorMSG": "Miss Error",
            "result": "Miss Error",
            "confidence": None
        }
        wav_result['errorMSG']=res

    if save_result:
        with open(audio_file.replace('.wav', TENCENT_JSON_SUFFIX), 'w') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False)
    
    print(audio_file.split('/')[-1])
    print('RESULT:\t'+str(wav_result)+'\n')
    

    if output_format == "transaction":
        return wav_result['result'],success
        
    if output_format == "json":
        return wav_result,success

def api_recognize(args):
    logger.info("Start decode folder {} using tencentyun.\n".format(args.wav_folder))

    wav_files = glob.glob(os.path.join(args.wav_folder, "**", "*.wav"), recursive=True)
    wav_files = filter_irrelevant_wav(wav_files)
     
    tencentyun_decode_multi(wav_files, args, re_decode_failed = args.re_decode_failed)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Google Speech to Text.')
    parser.add_argument('wav_folder', type=str, help="Where is the wav_folder.")
    parser.add_argument('--re_decode_all', action="store_true", default=False, help="Whether re-decode all the wave files. Default is False.")
    parser.add_argument('--re_decode_failed', action="store_true", default=False, help="Whether to re-decode the failed decoding wave files. Default is False.")
    parser.add_argument('--max_workers', default=1, type=int, help="The number of the multi-process. Default is MAX.")
    parser.add_argument("--language", "-l", default='en-US', choices=['en-US', 'zh-CN'])
    args = parser.parse_args()

    # Generate tencentyun json files
    api_recognize(args)
 
   