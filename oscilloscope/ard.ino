/*
  Arduino Oscilloscope Sender
  Reads analog input from A0 and sends values over Serial.
*/

const int analogPin = A0;
const unsigned long sampleIntervalMicros = 500; 
// 500 us = about 2000 samples/second

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  unsigned long now = micros();

  if (now - lastSampleTime >= sampleIntervalMicros) {
    lastSampleTime = now;

    int value = analogRead(analogPin); // 0 to 1023
    Serial.println(value);
  }
}