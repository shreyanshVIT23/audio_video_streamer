import asyncio
import json
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaPlayer
import aiohttp

SIGNALLING_SERVER = "https://dev.shreyanshpande.work/offer"


async def client_request():
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls="stun:stun.l.google.com:19302",
            )
        ]
    )
    pc = RTCPeerConnection(configuration=config)
    player = MediaPlayer(
        "/dev/video0", format="v4l2", options={"video_size": "640x480"}
    )
    mic = MediaPlayer("./2 people conversation.opus")
    pc.addTrack(player.video)
    pc.addTrack(mic.audio)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            SIGNALLING_SERVER,
            data=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        ) as response:
            answer = await response.json()

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )
    await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(client_request())
