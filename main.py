import os, pathlib, shutil, threading, random, math, time
import requests, json

app_root = os.environ.get("ANDROID_ARGUMENT", ".")
icon_dir = pathlib.Path(app_root, ".kivy", "icon")
shutil.rmtree(icon_dir, ignore_errors=True)
icon_dir.mkdir(parents=True, exist_ok=True)
os.environ["KIVY_HOME"] = str(icon_dir.parent)

# ---------- Kivy imports ----------
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock

# ---------- Android battery helper ----------
try:
    from jnius import autoclass
except ImportError:
    autoclass = None


def get_device_id():
    Build = autoclass("android.os.Build")
    return f"{Build.MANUFACTURER}-{Build.MODEL}"


SERVER = "https://mobile-fl-server.azurewebsites.net/update"


def get_battery_level():
    if autoclass is None:
        return 100
    try:
        Context = autoclass("android.content.Context")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        Intent = autoclass("android.content.Intent")
        IntentFilter = autoclass("android.content.IntentFilter")
        BatteryManager = autoclass("android.os.BatteryManager")
        act = PythonActivity.mActivity
        stat = act.registerReceiver(None, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return int(
            stat.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            / stat.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            * 100
        )
    except Exception:
        return -1


def proc_cpu_percent():
    import os, time

    def _ticks():
        with open("/proc/self/stat") as f:
            p = f.read().split()
        return int(p[13]) + int(p[14])

    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    a = _ticks()
    time.sleep(0.5)
    b = _ticks()
    return round((b - a) / hz / 0.5 * 100, 1)


def send_result_to_server(device_id, accuracy, battery, cpu, maxCPU):
    """POSTing the training stats to the Azure receiver."""
    try:
        payload = {
            "device_id": device_id,
            "accuracy": round(accuracy, 3),
            "battery": int(battery),
            "cpu": int(cpu),
            "maxCPU": int(maxCPU),
        }
        res = requests.post(SERVER, json=payload, timeout=5)
        if res.status_code == 200:
            print("✅ Result sent successfully.")
        else:
            print(f"⚠️ Server replied {res.status_code}")
    except Exception as e:
        print("❌ POST failed:", e)


# ---------- thresholds ----------
BATTERY_MIN = 30  # %
CPU_MAX = 50  # %

# ---------- training length control ----------
TARGET_MINUTES = 3
EPOCHS_PER_LOOP = 600


# ---------- UI ----------
class Main(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", **kw)
        self.lbl = Label(text="Tap START when ready", font_size="18sp")
        self.btn = Button(
            text="START", size_hint=(1, 0.25), on_press=self.start_pressed
        )
        self.add_widget(self.lbl)
        self.add_widget(self.btn)
        self.training = False
        self.maxCPU = 0

    def start_pressed(self, *_):
        if self.training:
            return
        batt = get_battery_level()
        cpu = proc_cpu_percent()
        self.lbl.text = f"Battery {batt}% | CPU {cpu}%"
        if batt < BATTERY_MIN or cpu > CPU_MAX:
            self.lbl.text += "\nSkipped – conditions not met"
            send_result_to_server(
                device_id=get_device_id(),
                accuracy="skipped",
                battery=batt,
                cpu=cpu,
                maxCPU="skipped",
            )
            return

        self.lbl.text += "\nTraining…"
        self.btn.disabled = True
        self.training = True
        threading.Thread(target=self.train_one_round, daemon=True).start()

    # -------- training thread --------
    def train_one_round(self):
        start = time.perf_counter()
        acc = 0.0

        while time.perf_counter() - start < TARGET_MINUTES * 60:
            N = 200
            X = [[random.random(), random.random()] for _ in range(N)]
            y = [int(x1 + x2 > 1) for x1, x2 in X]
            w1 = w2 = 0.0
            lr = 0.7
            for _ in range(EPOCHS_PER_LOOP):
                for (x1, x2), t in zip(X, y):
                    z = w1 * x1 + w2 * x2
                    p = 1 / (1 + math.exp(-z))
                    g = p - t
                    w1 -= lr * g * x1
                    w2 -= lr * g * x2
            acc = sum(((w1 * x1 + w2 * x2) > 0) == t for (x1, x2), t in zip(X, y)) / N

            elapsed = int((time.perf_counter() - start) / 60)
            if elapsed and elapsed % 1 == 0:  # every full minute
                live_cpu = proc_cpu_percent()

                if live_cpu > self.maxCPU:
                    self.maxCPU = live_cpu
                msg = f"Training… {elapsed} min elapsed\n" f"CPU now ≈ {live_cpu}%"
                Clock.schedule_once(lambda dt, m=msg: setattr(self.lbl, "text", m), 0)

        Clock.schedule_once(lambda dt, a=acc: self.show_result(a), 0)

    def show_result(self, acc):
        batt = get_battery_level()
        cpu = proc_cpu_percent()

        self.acc = acc
        self.battery = batt
        self.cpu = cpu

        self.lbl.text = (
            f"Battery {batt}% | CPU {cpu}%\n" f"✓ Finished – acc ≈ {acc:.2f}"
        )
        self.btn.disabled = False
        self.training = False
        try:
            send_result_to_server(
                device_id=get_device_id(),
                accuracy=self.acc,
                battery=self.battery,
                cpu=self.cpu,
                maxCPU=self.maxCPU,
            )
        except Exception as e:
            print("POST failed:", e)


class MobileFL(App):
    def build(self):
        return Main()


if __name__ == "__main__":
    MobileFL().run()
