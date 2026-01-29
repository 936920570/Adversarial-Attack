# coding=utf-8
import os
import json
import time
import whisper
import torch
from typing import Tuple

_whisper_model = None
_current_model_name = None

def openai_recong(audio_file: str, language: str = 'en', model: str = "large-v3", 
                 prompt: str = None) -> Tuple[str, bool]:

    global _whisper_model, _current_model_name
    
    try:

        if not os.path.exists(audio_file):
            return f"Audio file not found: {audio_file}", False
        

        if _whisper_model is None or _current_model_name != model:
            print(f"Loading Whisper model: {model}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _whisper_model = whisper.load_model(model, device=device)
            _current_model_name = model
            print(f"Model loaded successfully on {device}")
        

        target_language = None
        if not language or language.lower() == 'auto':
            target_language = "en"
        if language and language != 'auto':
            if 'zh' in language.lower() or 'cn' in language.lower():
                target_language = "zh"
            elif 'en' in language.lower():
                target_language = "en"
            else:
                target_language = language
 

        transcribe_options = {}
        if target_language:
            transcribe_options["language"] = target_language
        if prompt:
            transcribe_options["initial_prompt"] = prompt
        

        result = _whisper_model.transcribe(audio_file, **transcribe_options)
        

        result_text = result["text"].strip()
        
        if result_text:
            print(f"Recognition successful: {result_text}")
            return result_text, True
        else:
            print("Empty transcription result")
            return "", True
                
    except Exception as e:
        error_msg = f"Whisper transcription error: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg, False


def openai_decode(audio_file: str, language: str = None, model: str = "large-v3") -> Tuple[str, bool]:

    return openai_recong(audio_file, language, model)



def whisper_decode(audio_file: str, language: str = None) -> Tuple[str, bool]:

    return openai_recong(audio_file, language)


if __name__ == "__main__":

    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python openai_function.py <audio_file> [language]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None
    
    result, success = openai_recong(audio_file, language)
    
    print(f"Result: {result}")
    print(f"Success: {success}")
