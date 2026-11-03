"""Runs on a Raspberry Pi wired to a relay -> siren.   pip install fastapi uvicorn gpiozero
Run: SIREN_KEY=change-me-too uvicorn siren_server:app --host 0.0.0.0 --port 8001
Wiring: GPIO17 -> relay IN, relay switches the siren's own power supply (never power it from the Pi)."""
import os
import threading

from fastapi import FastAPI, Header, HTTPException
from gpiozero import OutputDevice
from pydantic import BaseModel

relay = OutputDevice(17, active_high=False, initial_value=False)
app = FastAPI()
timer = None


class Cmd(BaseModel):
    state: str
    seconds: int = 30


@app.post("/siren")
def siren(cmd: Cmd, x_key: str = Header(default="")):
    global timer
    if x_key != os.getenv("SIREN_KEY", ""):
        raise HTTPException(401)
    if timer:
        timer.cancel()
    if cmd.state == "on":
        relay.on()
        timer = threading.Timer(min(cmd.seconds, 300), relay.off)  # always auto-off
        timer.start()
    else:
        relay.off()
    return {"state": cmd.state}
