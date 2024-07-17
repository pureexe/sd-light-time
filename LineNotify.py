import requests 
import socket
import os 

class LineNotify:
    def __init__(self, token=None, with_hostname=True):
        self.with_hostname = with_hostname
        if token is None:
            with open(".line_token") as file:
                token = file.read().strip()
        self.token = token

    def append_hostname(self, text, with_hostname=False):
        if with_hostname or self.with_hostname:
            gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES", "0")  
            text = f" {socket.gethostname()}g{gpu_ids}: {text}"
        return text

    def send(self,text, with_hostname=False):
        text = self.append_hostname(text, with_hostname)
        url = "https://notify-api.line.me/api/notify"
        payload = {"message": text}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "content-type": "application/x-www-form-urlencoded",
         }
        response = requests.post(url, headers=headers, params=payload)
        return response
    
    def send_image(self, text, image_path, with_hostname=False):
        url = "https://notify-api.line.me/api/notify"
        text = self.append_hostname(text, with_hostname)
        payload = {"message": text}
        img = {"imageFile": open(image_path, "rb")}
        
        headers = {
            "Authorization": f"Bearer {self.token}",
         }
        response = requests.post(url, headers=headers, data=payload, files=img)
        return response
    
    def call(self):
        try:
            with open(".call_api") as file:
                url = file.read().strip()
            response = requests.get(url)
            return response
        except:
            pass


def notify(func):
    def wrapper(*args, **kwargs):
        line = LineNotify()
        try:
            result = func(*args, **kwargs)
            file_path = os.path.realpath(__file__)
            line.send(f":{file_path}:{func.__name__} finished successfully", with_hostname=True)
            return result
        except Exception as e:
            file_path = os.path.realpath(__file__)
            line.send(f":{file_path}:{func.__name__} FAILED", with_hostname=True)
            raise e
    return wrapper

def catch_call(func):
    def wrapper(*args, **kwargs):
        line = LineNotify()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            line.call()
            raise e
    return wrapper

def always_call(func):
    def wrapper(*args, **kwargs):
        line = LineNotify()
        try:
            result = func(*args, **kwargs)
            line.call()
            return result
        except Exception as e:
            line.call()
            raise e
    return wrapper