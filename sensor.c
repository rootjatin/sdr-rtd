/*
  ESP32 Sensor Web Interface
  - Hosts a webpage
  - Lets user choose a sensor
  - Returns live sensor data as JSON

  Required libraries:
  - WiFi.h
  - WebServer.h
  - DHT.h

  Install DHT library by Adafruit from Library Manager.
*/

#include <WiFi.h>
#include <WebServer.h>
#include <DHT.h>

// ===================== WIFI SETTINGS =====================
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// ===================== SENSOR PINS =====================
// DHT sensor
#define DHTPIN 4
#define DHTTYPE DHT11   // change to DHT22 if needed

// LDR sensor
#define LDR_PIN 34      // analog pin on ESP32

// Ultrasonic sensor
#define TRIG_PIN 5
#define ECHO_PIN 18

DHT dht(DHTPIN, DHTTYPE);
WebServer server(80);

// ===================== HTML PAGE =====================
const char MAIN_page[] PROGMEM = R"====(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ESP32 Sensor Interface</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f7fb;
      margin: 0;
      padding: 20px;
    }
    .card {
      max-width: 500px;
      margin: auto;
      background: white;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h2 {
      margin-top: 0;
      color: #222;
    }
    select, button {
      width: 100%;
      padding: 12px;
      margin-top: 12px;
      font-size: 16px;
      border-radius: 8px;
    }
    button {
      background: #007bff;
      color: white;
      border: none;
      cursor: pointer;
    }
    button:hover {
      background: #0056b3;
    }
    .output {
      margin-top: 20px;
      padding: 15px;
      background: #eef3ff;
      border-radius: 8px;
      min-height: 60px;
      white-space: pre-line;
      font-size: 18px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>ESP32 Sensor Interface</h2>

    <label for="sensorSelect">Choose Sensor:</label>
    <select id="sensorSelect">
      <option value="dht">DHT Temperature & Humidity</option>
      <option value="ldr">LDR Light Sensor</option>
      <option value="ultrasonic">Ultrasonic Distance</option>
    </select>

    <button onclick="startSensor()">Start</button>
    <button onclick="stopSensor()">Stop</button>

    <div class="output" id="output">No sensor started.</div>
  </div>

  <script>
    let intervalId = null;

    function startSensor() {
      const sensor = document.getElementById("sensorSelect").value;
      const output = document.getElementById("output");

      if (intervalId) clearInterval(intervalId);

      output.innerText = "Reading " + sensor + "...";

      intervalId = setInterval(async () => {
        try {
          const response = await fetch("/read?sensor=" + sensor);
          const data = await response.json();

          if (data.error) {
            output.innerText = "Error: " + data.error;
            return;
          }

          if (sensor === "dht") {
            output.innerText =
              "Temperature: " + data.temperature + " °C\n" +
              "Humidity: " + data.humidity + " %";
          } else if (sensor === "ldr") {
            output.innerText =
              "Light Value: " + data.light;
          } else if (sensor === "ultrasonic") {
            output.innerText =
              "Distance: " + data.distance + " cm";
          }
        } catch (err) {
          output.innerText = "Request failed.";
        }
      }, 2000);
    }

    function stopSensor() {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
      document.getElementById("output").innerText = "Stopped.";
    }
  </script>
</body>
</html>
)====";

// ===================== SENSOR FUNCTIONS =====================
float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;

  float distance = duration * 0.0343 / 2.0;
  return distance;
}

// ===================== HANDLERS =====================
void handleRoot() {
  server.send(200, "text/html", MAIN_page);
}

void handleReadSensor() {
  if (!server.hasArg("sensor")) {
    server.send(400, "application/json", "{\"error\":\"Missing sensor parameter\"}");
    return;
  }

  String sensor = server.arg("sensor");
  String json = "{";

  if (sensor == "dht") {
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();

    if (isnan(humidity) || isnan(temperature)) {
      server.send(500, "application/json", "{\"error\":\"Failed to read DHT sensor\"}");
      return;
    }

    json += "\"temperature\":" + String(temperature, 1) + ",";
    json += "\"humidity\":" + String(humidity, 1);
  }
  else if (sensor == "ldr") {
    int lightValue = analogRead(LDR_PIN);
    json += "\"light\":" + String(lightValue);
  }
  else if (sensor == "ultrasonic") {
    float distance = readDistanceCM();
    if (distance < 0) {
      server.send(500, "application/json", "{\"error\":\"Failed to read ultrasonic sensor\"}");
      return;
    }

    json += "\"distance\":" + String(distance, 1);
  }
  else {
    server.send(400, "application/json", "{\"error\":\"Invalid sensor type\"}");
    return;
  }

  json += "}";
  server.send(200, "application/json", json);
}

// ===================== SETUP =====================
void setup() {
  Serial.begin(115200);

  dht.begin();

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Connected!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/read", handleReadSensor);

  server.begin();
  Serial.println("Web server started.");
}

// ===================== LOOP =====================
void loop() {
  server.handleClient();
}
