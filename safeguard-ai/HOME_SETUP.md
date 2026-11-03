# Always-on setup at home



This is the version that keeps watching without anyone opening a browser. It runs on one PC that stays switched on in the same house as the camera. When it sees a fall, it rings the siren on that PC and sends a photo to your phone on Telegram, wherever you are.

The Streamlit page (`streamlit run streamlit\_app.py`) is for demos. This one uses `uvicorn` and is for daily use.







## What you need at the house



* A Windows PC or laptop that can stay on all day, plugged in
* A camera on the same Wi-Fi: a real IP/CCTV camera, or an old Android phone running IP Webcam, also plugged in
* Internet, so the Telegram messages can reach you
* A speaker on the PC if you want the siren to be loud







## Step 1. Stop the PC from sleeping



Settings > System > Power \& battery > Screen and sleep. Set "When plugged in, put my device to sleep after" to **Never**. On a laptop, also set "closing the lid" to **do nothing** (Control Panel > Power Options > Choose what closing the lid does).





## Step 2. Put the project on the PC and test it first



Copy the `Safeguard` folder to the PC and install it the usual way (see README.md). Before going always-on, check that the demo works: `streamlit run streamlit\_app.py`, a live link, a fall test. If it does not work there, it will not work here either.





## Step 3. Fill in the `.env` file

Open `.env` in the project folder (create it with `notepad .env` if it is not there) and set these:

```
TELEGRAM\_BOT\_TOKEN=bot token
TELEGRAM\_CHAT\_ID=chat id
HOME\_CAM\_URL=http://192.168.1.5:8080/video
API\_KEY=choose-a-password
```

`HOME\_CAM\_URL` is the camera link. For a phone running IP Webcam it ends in `/video`. For a CCTV camera it is the `rtsp://user:password@camera-ip:554/...` link. No spaces around `=` and no quotation marks.

Give the camera a fixed address, or the link will break after a router restart. On the phone this is under Wi-Fi settings > the network > IP settings > Static. For a CCTV camera, set it in the camera or reserve the address in the router.

If you do not have the camera link yet, leave `HOME\_CAM\_URL` out. SafeGuard still starts, tells you on Telegram that no camera is set, and you can try **Send test alert** and **Test siren**. When you have the link, add the line to `.env`, close the black window and double-click `start\_safeguard.bat` again.





## Step 4. Check `config.yaml`



The camera section should look like this (it already does, if you took the latest file):

```yaml
cameras:
  - id: home1
    name: Living room
    source: "${HOME\_CAM\_URL}"
    mode: land
    autostart: true
```

`autostart: true` is what makes it start watching by itself. The `alerts` section has three settings you might change: `local\_siren` (siren from the PC speakers), `heartbeat\_hour` (the daily "still running" message, set to `null` to turn it off) and `offline\_after\_s` (how long without video counts as the camera being offline).





## Step 5. Start it



Double-click `start\_safeguard.bat`. A black window opens and says it is starting on http://localhost:8000.

If it says it cannot find the virtual environment, open the file in Notepad and change the line `set VENV=` to the real folder of your `.venv`, for example `set VENV=C:\\Users\\Lenovo\\Downloads\\files\\safeguard-ai\\.venv`.

The first time, Windows Firewall may ask for permission. Allow it on private networks.

You should get a Telegram message saying "SafeGuard started". Open http://localhost:8000 and you will see the camera and its live picture.





## Step 6. Test everything once



On the page at http://localhost:8000:

1. Press **Send test alert**. The message should arrive on your phone.
2. Press **Test siren**. The PC should ring. Turn the PC volume up first.
3. Do a safe fall in front of the camera, onto a mattress, and stay still for 5 to 8 seconds. You should get the siren and a Telegram message with a photo.

The page asks for the API key (the `API\_KEY` from `.env`) the first time you press one of those buttons.





## Step 7. Make it start by itself after a restart



1. Press Win+R, type `shell:startup` and press Enter. A folder opens.
2. Right-click `start\_safeguard.bat`, choose "Show more options" > "Send to" > "Desktop (create shortcut)". Move that shortcut into the startup folder.
3. This runs when someone logs in to Windows. For a PC nobody sits at, set it to log in by itself: press Win+R, type `netplwiz`, untick "Users must enter a user name and password", and enter the password once.

Restart the PC to test. A few minutes later you should get the "SafeGuard started" message again.





## Step 8. Looking at it from far away



You do not need to, because the alerts come to your phone anyway. If you want to see the live picture, install **Tailscale** (free) on the home PC and on your phone, sign in with the same account, and open `http://TAILSCALE-ADDRESS-OF-THE-PC:8000` on the phone. Do not open the router to the internet (port forwarding). It would expose the camera to anyone.







## What messages you will get on Telegram

|Message|Meaning|
|-|-|
|SafeGuard started on ...|The program started, for example after a power cut|
|ALARM: FALL, with a photo|A fall was detected|
|Camera ... is OFFLINE|No video for 90 seconds. Check the camera power and Wi-Fi. Nothing is being watched.|
|Camera ... is back online|It recovered|
|SafeGuard is running ...|A check-in every morning at 9. If it does not arrive, the PC is off or has no internet.|

The check-in message matters most. If one morning it does not arrive, something is wrong, even though no alarm was sent.





## If something goes wrong

* **No "SafeGuard started" message:** Telegram settings are wrong. Check `.env` is named exactly `.env` and the bot was sent a message first.
* **Camera offline right after starting:** open `HOME\_CAM\_URL` in Chrome. If the picture does not show there, the link or the Wi-Fi is the problem.
* **The siren does not ring:** PC volume is muted, or `local\_siren` is `false` in `config.yaml`.
* **A window closed and it stopped:** the black window is the program. Do not close it. Minimise it.





## Be realistic

It can miss a fall, and it can raise false alarms. It stops when the power or the internet goes off, and it only tells you if the PC itself is still running. Use it as an extra layer, with your parents' agreement, and keep the camera out of bedrooms and bathrooms. A daily phone call is still a good idea.

