# -*- coding: utf-8 -*-
import requests
import hmac
import hashlib
import base64
import time
import random
import os
import json
import sys
import threading
from datetime import datetime
from .account import ACCOUNT

class Credential:
    def __init__(self, secret_id, secret_key):
        self.secret_id = secret_id
        self.secret_key = secret_key

class FlashRecognitionRequest:
    def __init__(self, engine_type):
        self.engine_type = engine_type
        self.speaker_diarization = 0
        self.filter_dirty = 0
        self.filter_modal = 0
        self.filter_punc = 0
        self.convert_num_mode = 1
        self.word_info = 0
        self.hotword_id = ""
        self.voice_format = ""
        self.first_channel_only = 1

    def set_first_channel_only(self, first_channel_only):
        self.first_channel_only = first_channel_only

    def set_speaker_diarization(self, speaker_diarization):
        self.speaker_diarization = speaker_diarization

    def set_filter_dirty(self, filter_dirty):
        self.filter_dirty = filter_dirty

    def set_filter_modal(self, filter_modal):
        self.filter_modal = filter_modal

    def set_filter_punc(self, filter_punc):
        self.filter_punc = filter_punc

    def set_convert_num_mode(self, convert_num_mode):
        self.convert_num_mode = convert_num_mode

    def set_word_info(self, word_info):
        self.word_info = word_info

    def set_hotword_id(self, hotword_id):
        self.hotword_id = hotword_id

    def set_voice_format(self, voice_format):
        self.voice_format = voice_format

class FlashRecognizer:

    def __init__(self, appid, credential):
        self.credential = credential
        self.appid = appid

    def _format_sign_string(self, param):
        signstr = "POSTasr.cloud.tencent.com/asr/flash/v1/"
        for t in param:
            if 'appid' in t:
                signstr += str(t[1])
                break
        signstr += "?"
        for x in param:
            tmp = x
            if 'appid' in x:
                continue
            for t in tmp:
                signstr += str(t)
                signstr += "="
            signstr = signstr[:-1]
            signstr += "&"
        signstr = signstr[:-1]
        return signstr

    def _build_header(self):
        header = dict()
        header["Host"] = "asr.cloud.tencent.com"
        return header

    def _sign(self, signstr, secret_key):
        hmacstr = hmac.new(secret_key.encode('utf-8'),
                           signstr.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def _build_req_with_signature(self, secret_key, params, header):
        query = sorted(params.items(), key=lambda d: d[0])
        signstr = self._format_sign_string(query)
        signature = self._sign(signstr, secret_key)
        header["Authorization"] = signature
        requrl = "https://"
        requrl += signstr[4::]
        return requrl

    def _create_query_arr(self, req):
        query_arr = dict()
        query_arr['appid'] = self.appid
        query_arr['secretid'] = self.credential.secret_id
        query_arr['timestamp'] = str(int(time.time()))
        query_arr['engine_type'] = req.engine_type
        query_arr['voice_format'] = req.voice_format
        query_arr['speaker_diarization'] = req.speaker_diarization
        query_arr['hotword_id'] = req.hotword_id
        query_arr['filter_dirty'] = req.filter_dirty
        query_arr['filter_modal'] = req.filter_modal
        query_arr['filter_punc'] = req.filter_punc
        query_arr['convert_num_mode'] = req.convert_num_mode
        query_arr['word_info'] = req.word_info
        query_arr['first_channel_only'] = req.first_channel_only
        return query_arr

    def recognize(self, req, data):
        header = self._build_header()
        query_arr = self._create_query_arr(req)
        req_url = self._build_req_with_signature(self.credential.secret_key, query_arr, header)
        r = requests.post(req_url, headers=header, data=data)
        return r.text





def tencent_recong_standard(path, language='en-US'):
    """
    Tencent Cloud Recording File Recognition (CreateRecTask) implementation.
    """
    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        from tencentcloud.asr.v20190614 import asr_client, models
    except ImportError:
        print("Please install tencentcloud-sdk-python: pip install tencentcloud-sdk-python")
        return "Missing Dependency", False

    # Check dependencies
    APPID= ACCOUNT["Tencent"]["appid"]
    secret_id = ACCOUNT["Tencent"]["secret_id"]
    secret_key = ACCOUNT["Tencent"]["secret_key"]

    if not secret_id or not secret_key:
        print("Please set Tencent secret_id and secret_key in cloud_decode/account.py")
        return "Missing Credentials", False

    max_retries = 100
    last_error = None
    
    for attempt in range(max_retries):
        try:
            cred = credential.Credential(secret_id, secret_key)
            httpProfile = HttpProfile()
            httpProfile.endpoint = "asr.tencentcloudapi.com"
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            client = asr_client.AsrClient(cred, "ap-shanghai", clientProfile)

            with open(path, "rb") as f:
                data = f.read()
                base64_data = base64.b64encode(data).decode("utf-8")

            # Recording File Recognition - CreateRecTask
            req = models.CreateRecTaskRequest()
            
            # Determine engine type
            engine_type = "16k_en" 
            if 'zh' in language.lower() or 'cn' in language.lower():
                engine_type = "16k_zh"
            
            # Use Data (Base64) instead of URL
            params = {
                "EngineModelType": engine_type,
                "ChannelNum": 1,
                "ResTextFormat": 0, # 0: Basic recognition result
                "SourceType": 1,
                "Data": base64_data
            }
            req.from_json_string(json.dumps(params))
            resp = client.CreateRecTask(req)
            
            task_id = resp.Data.TaskId
            
            # Polling for result
            start_time = time.time()
            while True:
                if time.time() - start_time > 600: # 10 minutes timeout
                     return "Timeout", False
                     
                time.sleep(1) # Frequency of polling
                req_status = models.DescribeTaskStatusRequest()
                params_status = {"TaskId": task_id}
                req_status.from_json_string(json.dumps(params_status))
                resp_status = client.DescribeTaskStatus(req_status)
                
                status_str = resp_status.Data.StatusStr
                if status_str == "success":
                    # Result contains the text
                    return resp_status.Data.Result, True
                
                if status_str == "failed":
                    return resp_status.Data.ErrorMsg, False

                    
        except TencentCloudSDKException as err:
            print(f"Tencent Cloud SDK Error (Attempt {attempt+1}/{max_retries}): {err}")
            last_error = err
            time.sleep(1)
        except Exception as e:
            print(f"Error (Attempt {attempt+1}/{max_retries}): {e}")
            last_error = e
            time.sleep(1)
            
    return str(last_error), False

def tencent_recong(path, language='en-US'):
    # Uncomment the following line to use the Standard Recording File Recognition (CreateRecTask)
    return tencent_recong_standard(path, language)
    


if __name__ == "__main__":
    path = ""

    re,success,na = tencent_recong(path)
    print(re,na)
