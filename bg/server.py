import torch
import numpy as np
from model import BG

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

handler = None

@app.route('/',methods=['GET'])
def getQuestion():
    r = int(request.args.get("r"))
    g = int(request.args.get("g"))
    b = int(request.args.get("b"))
    out = 1 if handler.guess(r, g, b)[0] < 0.5 else 0
    print(out)
    response = jsonify({"output": out})
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response

class AIHandler():
    def __init__(self):
        self.model = BG()
        self.model.load_state_dict(torch.load("models/color_2.pt"))
        self.model.eval()

    def guess(self, r, g, b):
        color = torch.tensor(np.array([r/255, g/255, b/255])).float()
        g = self.model(color)
        return g

if __name__ == "__main__":
    handler = AIHandler()
    app.run()


    