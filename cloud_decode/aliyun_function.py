import http.client
import json
import time
from cloud_decode.account import ACCOUNT
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

def aliyun_recong(audio_path):
    appKey =  ACCOUNT["Alibaba"]["app_key"]
    access_key_id=ACCOUNT["Alibaba"]["access_key_id"]
    access_key_secret=ACCOUNT["Alibaba"]["access_key_secret"]

    client = AcsClient(access_key_id, access_key_secret, "cn-shanghai")

    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')
    response = client.do_action_with_exception(request)

    token = str(response, 'utf-8')
    token = json.loads(token)
    token = token['Token']['Id']

    url = 'http://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr'

    audioFile = audio_path
    format = 'wav'
    sampleRate = 16000
    enablePunctuationPrediction  = True
    enableInverseTextNormalization = True
    enableVoiceDetection  = False

    request = url + '?appkey=' + appKey
    request = request + '&format=' + format
    request = request + '&sample_rate=' + str(sampleRate)

    if enablePunctuationPrediction :
        request = request + '&enable_punctuation_prediction=' + 'true'

    if enableInverseTextNormalization :
        request = request + '&enable_inverse_text_normalization=' + 'true'

    if enableVoiceDetection :
        request = request + '&enable_voice_detection=' + 'true'

    # print('Request: ' + request)

    asr_result = process(request, token, audioFile)

    return asr_result


def process(request, token, audioFile) :
    with open(audioFile, mode = 'rb') as f:
        audioContent = f.read()

    host = 'nls-gateway.cn-shanghai.aliyuncs.com'

    httpHeaders = {
        'X-NLS-Token': token,
        'Content-type': 'application/octet-stream',
        'Content-Length': len(audioContent)
        }


    success_flag = False
    result = 'Recognizer failed!'

    for attempt in range(100):
        try:
            conn = http.client.HTTPConnection(host)

            conn.request(method='POST', url=request, body=audioContent, headers=httpHeaders)

            response = conn.getresponse()
            # print('Response status and response reason:')
            # print(response.status ,response.reason)

            body = response.read()
            conn.close()

            try:
                # print('Recognize response is:')
                body = json.loads(body)
                # print(body)

                status = body['status']
                if status == 20000000 :
                    result = body['result']
                    success_flag=True
                    # print('Recognize result: ' + result)
                else :
                    print(body)
                    result = 'Recognizer failed!'
                break # Loop exit on success

            except ValueError:
                result = "The response is not json format string"
                break 

        except (ConnectionResetError, http.client.IncompleteRead, OSError, Exception) as e:
            print(f"Connection error: {e}. Retrying {attempt+1}/5...")
            time.sleep(2)
            if attempt == 4:
                print("Max retries reached.")

    return result,success_flag


if __name__ == "__main__":
    aliyun_recong('',0)