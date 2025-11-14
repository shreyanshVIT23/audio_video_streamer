import asyncio
from fastapi import FastAPI, Request
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from fastapi.responses import JSONResponse
from .processing import AudioProcessor, VideoProcessor
from fastapi.middleware.cors import CORSMiddleware

PCS = set()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8080", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]
    )
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(configuration=config)
    PCS.add(pc)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        # if track.kind == "audio":
        #     ap = AudioProcessor(track=track)
        #     asyncio.create_task(ap.process_audio())

        if track.kind == "video":
            vp = VideoProcessor(track=track)
            asyncio.create_task(vp.start())

        #
        @track.on("ended")
        async def on_ended():
            PCS.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


@app.on_event("shutdown")
async def on_shutdown(app):
    coros = [pc.close() for pc in PCS]
    await asyncio.gather(*coros)
    PCS.clear()


@app.post("/close")
async def close(request):
    for pc in list(PCS):
        await pc.close()
        PCS.discard(pc)
    return JSONResponse({"status": "closed"})


#
# app = web.Application()
# app.router.add_post("/offer", offer)
# app.router.add_post("/close", close)
# app.on_shutdown.append(on_shutdown)
# if __name__ == "__main__":
#     web.run_app(app=app, port=8000)
