import requests 
import socket

class LineNotify:
    def __init__(self, token=None):
        if token is None:
            with open(".line_token") as file:
                token = file.read().strip()
        self.token = token

    def send(self,text, with_hostname=False):
        if with_hostname:
            text = f" {socket.gethostname()}: {text}"
        url = "https://notify-api.line.me/api/notify"
        payload = {"message": text}
        headers = {
            "Authorization": f"Bearer {self.token}",
            "content-type": "application/x-www-form-urlencoded",
         }
        response = requests.post(url, headers=headers, params=payload)
        return response
    
    def send_image(self, text, image_path):
        url = "https://notify-api.line.me/api/notify"
        #payload = {"message": text}
        payload = {"message": text}
        img = {"imageFile": open(image_path, "rb")}
        
        headers = {
            "Authorization": f"Bearer {self.token}",
         }
        response = requests.post(url, headers=headers, data=payload, files=img)
        return response