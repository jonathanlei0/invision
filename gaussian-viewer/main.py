from typing import Annotated
import shutil
import urllib.parse
import html.parser
import re
import os
import json
import logging
import time
import traceback
#from starlette.middleware.cors import CORSMiddleware
import requests
from fastapi.responses import JSONResponse, HTMLResponse
#import Response
from fastapi import Request, Body

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, VideoStreamTrack
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.WARN)

from src import Camera, GaussianModel, Renderer, get_ice_servers

model_path = "/home/user/gaussian-viewer/models/output/TIM/point_cloud/iteration_2000/point_cloud.ply"
camera_path = "/home/user/gaussian-viewer/models/output/TIM/cameras.json"

sessions = {}

# load gaussian model
gaussian_model = GaussianModel().load(model_path)


# load camera info
with open(camera_path, "r") as f:
    cam_info = json.load(f)[12]


# initialize server
origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/files/")
async def image(video: UploadFile = File(...), iterations: int = Body(...)):
    print(iterations)
    with open("destination.mov", "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    print("Storing file")
    time.sleep(1)
    os.system("rm -rf /home/user/gaussian-splatting/Data/TIM/stills/*")
    os.system("mkdir /home/user/gaussian-splatting/Data/TIM/stills/input")
    time.sleep(1)
    os.system("ffmpeg -i /home/user/gaussian-viewer/destination.mov -c copy -movflags +faststart /home/user/gaussian-viewer/destination.mp4 -y")
    print("Using ffmpeg to convert to mp4")
    time.sleep(1)
    os.system('ffmpeg -i /home/user/gaussian-viewer/destination.mp4 -vf "select=not(mod(n\,15))" -vsync vfr -q:v 2 /home/user/gaussian-splatting/Data/TIM/stills/input/%02d.jpg')
    time.sleep(1)
    os.system('python3 /home/user/gaussian-splatting/convert.py --source_path /home/user/gaussian-splatting/Data/TIM/stills --no_gpu')
    time.sleep(1)
    os.system('/home/user/gaussian-splatting/venv/bin/python /home/user/gaussian-splatting/train.py -s /home/user/gaussian-splatting/Data/TIM/stills --iterations '+str(iterations)+'')
    time.sleep(1)

    
    content = """
    <body>
    <script>setTimeout(function(){ window.location = 'https://basementhost.com/'; }, 1000);</script>
    </body>
    """
    return HTMLResponse(content=content)

@app.get("/upload")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="video" type="file">
<input type="number" id="iterations" name="iterations" placeholder="Its" min="100" max="10000" value="2000">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


@app.get("/items/", response_class=JSONResponse)
async def read_items():
    r = requests.get("https://viewer.dylanebert.com/ice-servers")
    return r.json()

@app.post("/items/test/{e}", response_class=JSONResponse)
async def read_items2(e, info: Request):
    data = await info.body()
    r = requests.post("https://viewer.dylanebert.com/ice-candidate?session_id=${e}", json=json.loads(data.decode('utf-8')))
    return r.json()

def parse_frame(container, data):
    try:
        packets = container.parse(data)
        for packet in packets:
            frames = container.decode(packet)
            for frame in frames:
                return frame
    except Exception as e:
        logging.error(e)
        traceback.print_exc()

    return None


def create_session(session_id, pc):
    camera = Camera().load(cam_info)
    renderer = Renderer(gaussian_model, camera, logging=False)
    session = Session(session_id, renderer, pc)
    sessions[session_id] = session
    return session


class Offer(BaseModel):
    sdp: str
    type: str


class IceCandidate(BaseModel):
    candidate: str
    sdpMLineIndex: int
    sdpMid: str
    usernameFragment: str


class Session:
    session_id: str
    renderer: Renderer
    pc: RTCPeerConnection

    def __init__(self, session_id: str, renderer: Renderer, pc: RTCPeerConnection):
        self.session_id = session_id
        self.renderer = renderer
        self.pc = pc


class FrameProducer(VideoStreamTrack):
    kind = "video"

    def __init__(self, session: Session):
        super().__init__()
        self.session = session

        container = av.CodecContext.create("h264", "r")
        container.pix_fmt = "yuv420p"
        container.width = session.renderer.camera.image_width
        container.height = session.renderer.camera.image_height
        container.bit_rate = 14000000
        container.options = {"preset": "ultrafast", "tune": "zerolatency"}

        self.container = container

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        failed_attempts = 0
        max_failed_attempts = 10

        while True:
            try:
                start_time = time.time()
                data = self.session.renderer.render()
                logging.info(f"Render time: {time.time() - start_time}")
                if data is not None and len(data) > 0:
                    frame = parse_frame(self.container, data)
                    if frame is not None:
                        break
                    else:
                        raise Exception("Error parsing frame")
            except Exception as e:
                logging.error(e)
                logging.debug(traceback.format_exc())
                failed_attempts += 1
                if failed_attempts >= max_failed_attempts:
                    logging.error(f"Failed to render frame after {failed_attempts} attempts")
                    break

        frame.pts = pts
        frame.time_base = time_base

        return frame


@app.post("/ice-candidate")
async def add_ice_candidate(candidate: IceCandidate, session_id: str = Query(...)):
    logging.info(f"Adding ICE candidate for session {session_id}")
    pc = sessions[session_id].pc
    pattern = r"candidate:(\d+) (\d+) (\w+) (\d+) (\S+) (\d+) typ (\w+)"
    match = re.match(pattern, candidate.candidate)
    if match:
        foundation, component, protocol, priority, ip, port, typ = match.groups()
        ice_candidate = RTCIceCandidate(
            component=int(component),
            foundation=foundation,
            ip=ip,
            port=int(port),
            priority=int(priority),
            protocol=protocol,
            type=typ,
            sdpMid=candidate.sdpMid,
            sdpMLineIndex=candidate.sdpMLineIndex,
        )
        await pc.addIceCandidate(ice_candidate)
    else:
        logging.error(f"Failed to parse ICE candidate: {candidate.candidate}")

@app.post("/offer")
async def create_offer(offer: Offer, session_id: str = Query(...)):
    logging.info(f"Creating offer for session {session_id}")

    pc = RTCPeerConnection()
    pc.configuration = RTCConfiguration(iceServers=get_ice_servers())
    session = create_session(session_id, pc)
    track = FrameProducer(session)
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            del sessions[session_id]

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            payload = json.loads(message)
            logging.info(f"Received payload: {payload}")

            if payload["type"] == "camera_update":
                position = payload["position"]
                rotation = payload["rotation"]
                rotation = Rotation.from_euler("xyz", rotation, degrees=True).as_matrix()
                track.session.renderer.update(position, rotation)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

    answer = await pc.createAnswer()

    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.get("/ice-servers")
async def get_ice():
    return get_ice_servers()


@app.get("/models", response_class=FileResponse)
async def download_models():
    return FileResponse("models/models.zip")


app.mount("/", StaticFiles(directory="gaussian-viewer-frontend/public", html=True), name="public")
