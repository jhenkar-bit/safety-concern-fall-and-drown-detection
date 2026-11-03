# SafeGuard AI





SafeGuard AI watches a camera or a video file and looks for two things: a person falling down in a room, and a swimmer who may be in trouble in a pool. When it sees one, it shows a red alarm on screen, rings a siren, and can send a photo to your phone on Telegram.

There are two ways to run it:

* **The Streamlit page** is for demos. You upload a video or paste a camera link and watch it work.
* **The always-on version** is for leaving at home. It runs without a browser, restarts itself, and tells you on Telegram if the camera goes offline.

It is a small project, not a medical or safety product. It can miss a fall and it can raise a false alarm, so use it as an extra pair of eyes, not as the only protection for anyone.



## What you need



* A laptop or PC with Python 3.10 or newer (I used Windows 11 and Python 3.12)
* Internet for the first install, and to download the model files
* About 5 GB of free disk space (PyTorch is big)
* A video file, or a camera. An Android phone with the IP Webcam app works as a camera.







## What is in the folder



|**File/Folder**|Description|
|-|-|
|safeguard-ai/ streamlit\_app.py|the demo page (start here)|
|start\_safeguard.bat|starts the always-on version (Windows)|
|download\_models.py|downloads the YOLO model files|
|config.yaml |settings|
|requirements.txt| packages to install|
|.env |Telegram token, chat id, camera link|
|HOME\_SETUP.md|step-by-step guide for the always-on version---|
|app/ config.py|alerts.py  sound.py  db.py  pipeline.py  supervisor.py|
|main.py |the always-on service (FastAPI)|
| detectors/fall.py|fall logic|
|drowning.py|pool logic|

##### &#x20; 

The two `\_\_init\_\_.py` files (in `app` and `app/detectors`) are empty, but they have to be there.





## Install



Open Command Prompt in the folder that contains `streamlit\_app.py`.

```
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python download\_models.py
```

The install takes a while, and near the end it looks stuck while PyTorch unpacks. Wait for the prompt to come back. `download\_models.py` fetches about 70 MB of model files (`yolo11m-pose.pt`, `yolo11s-pose.pt`, `yolo11n.pt`). If you skip it, the app downloads what it needs the first time you use it. If you ever delete the `.pt` files, run it again.





## Run the demo page

```
.venv\\Scripts\\activate
streamlit run streamlit\_app.py
```

Your browser opens at http://localhost:8501. You must activate `.venv` in every new terminal. If you see `'streamlit' is not recognized`, that is the reason.

If your `.venv` lives in another folder, activate it by its full path, for example `C:\\Users\\Lenovo\\Downloads\\files\\safeguard-ai\\.venv\\Scripts\\activate`.





## Using the demo page



**Upload a video.** Choose "Fall on land" or "Drowning in pool" in the sidebar, open the *Upload a video* tab, pick a clip and press *Analyse video*. You get the annotated video back with the alarms marked, a grid of snapshots, and a download button.

**Live camera.** Open the *Live CCTV link* tab, paste the stream address and press *Start*. Type `0` for the laptop webcam.

**Phone as a camera.** Put the phone and laptop on the same Wi-Fi, start the IP Webcam server, and use its address with `/video` on the end, for example `http://192.168.1.5:8080/video`. Without `/video` you get "Could not read from this link".

**Sidebar settings.**

* *Model size*: Fast, Balanced, Accurate (medium) or Most accurate (large). Bigger is more accurate but slower. I use medium. If your PC is slow, pick Balanced.
* *Detection resolution* 800 and *confidence* 0.3 worked better for me than the lower defaults.
* *Seconds lying still before alarm*, and *Alarm if found lying still for*: how long a person must stay down.
* *Alarm if a person falls and disappears from view*: see below.
* *Alarm sound length*: 3 to 100 seconds.
* *Play alarm through this computer's speakers*: browsers often block sound that starts by itself, so the siren plays from the computer instead. *Silence alarm* stops it.







## A person who falls and leaves the picture



If someone goes down and then vanishes, for example behind a sofa or out of the bottom of the frame, it counts as a fall once they have been gone for the set time (5 seconds by default). A person who drops suddenly in the middle of the frame and does not come back is treated the same way. A person who simply walks out through the edge of the frame, or steps behind a door while standing, is not. If an uploaded video ends right after someone goes down and disappears, it is reported at the end. A fall that is completely hidden, with no visible drop beforehand, can still be missed.



## Telegram alerts



1. In Telegram, chat with `@BotFather`, send `/newbot` and follow the questions. It gives you a token.
2. Open your new bot and send it any message, for example "hi". The bot cannot message you until you do this.
3. Get your chat id by messaging `@userinfobot`, or by opening `https://api.telegram.org/botYOUR\_TOKEN/getUpdates` in a browser.
4. Create a file called `.env` next to `streamlit\_app.py` (no spaces, no quotes):

```
TELEGRAM\_BOT\_TOKEN=123456789:AAH...
TELEGRAM\_CHAT\_ID=987654321
```

5. Restart the app and tick *Also send Telegram and hardware siren* in the sidebar.

Keep the token private. If it leaks, send `/revoke` to BotFather.





## Siren



The siren plays through the speakers of the computer running the app. For an extra hardware siren, `hardware/siren\_server.py` runs on a Raspberry Pi with a relay. Put its address in `SIREN\_URL` and the shared password in `SIREN\_KEY` in `.env`. With no Pi, leave `SIREN\_URL` empty.





## The always-on version for a home



For a home that needs watching all day, run the service on a PC that stays on, in the same house as the camera. Full steps are in `HOME\_SETUP.md`. In short:

1. Fill in `.env` with the Telegram values, `API\_KEY`, and `HOME\_CAM\_URL` (the camera link). You can leave `HOME\_CAM\_URL` out at first and add it later.
2. Double-click `start\_safeguard.bat`.
3. Open http://localhost:8000, press *Send test alert* and *Test siren*.

What it does that the demo page does not:

* starts watching by itself and restarts after a crash
* rings the siren on the PC and sends the alarm photo to Telegram
* sends a message if the camera goes offline, and another when it is back
* sends "SafeGuard started" after a reboot, and a check-in message every morning at 9
* can be set to start when Windows starts

Do not run the demo page and the always-on version at the same time on the same camera.





## Docker

```
docker compose up --build
```

Opens at http://localhost:8501. Add `--profile api` to start the always-on service too. Docker on Windows cannot use the laptop webcam or play sound on the PC, so use an uploaded video or a camera link.

## Tests

```
python -m tests.test\_logic
python -m tests.test\_drowning
python -m tests.test\_supervisor
```

These feed made-up positions and fake cameras into the logic and check the result. They need no camera and no model.





## Recording a good fall test clip



* Film in landscape, with the whole body in view, about 2 to 3 metres away
* Fall sideways onto a mattress, not toward the camera
* Stay still for 5 to 8 seconds after the fall, then get up
* Also record sitting down and bending over, to look for false alarms







## Problems you might hit

|What you see|What to do|
|-|-|
|`Could not open requirements file`|You are one folder too high. `cd` into the folder that contains `requirements.txt`.|
|`'streamlit' is not recognized`|Run `.venv\\Scripts\\activate` first.|
|`can't open file 'download\_models.py'`|The new files are not in this folder yet. Copy them from the zip.|
|`ModuleNotFoundError: No module named 'app'`|A file is in the wrong folder, or an `\_\_init\_\_.py` is missing.|
|"Could not read from this link"|Open the same link in Chrome first. For IP Webcam it must end in `/video`.|
|Fall video gives no alarm|Use the medium model, resolution 800, confidence 0.3, and a clip where the person lies still for several seconds.|
|No Telegram message|Send "hi" to the bot first, and check `.env` is not saved as `.env.txt`.|
|Streamlit prints many `use\_container\_width` warnings|Harmless. The latest `streamlit\_app.py` no longer causes them.|

## 

## Limits



* The fall logic is a set of rules on top of a pose model. It can fail when furniture hides the person, when the room is dark, or when someone lies down on purpose.
* The pool logic is a first version. It has only been checked on made-up data and a couple of clips. A normal person detector struggles with swimmers who are partly under water, so a real pool needs a model trained on its own footage.
* The medium model is slow on a laptop without a GPU. The always-on version still works, but analyses fewer frames per second.
* If the power or the Wi-Fi goes off, it stops. The morning check-in message not arriving is the sign.
* A camera records people. Ask them first, and keep cameras out of bedrooms and bathrooms.

