#include <LiquidCrystal.h>
#include <Keypad.h>
#include <SoftwareSerial.h>

LiquidCrystal lcd(12, 6, A4, A5, 7, 13);
SoftwareSerial BT(4, 5);

const byte ROWS = 4;
const byte COLS = 4;
char keys[ROWS][COLS] = {
  {'1','2','3','A'}, {'4','5','6','B'}, {'7','8','9','C'}, {'*','0','#','D'}
};
byte rowPins[ROWS] = {A0, A1, A2, A3};
byte colPins[COLS] = {11, 10, 9, 8};
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

String inputData        = "";
String currentMod       = "16QAM";
String currentFEC       = "HAM(7,4)";
int    lastSentBits     = 0;
int    totalPacketsSent = 0;
unsigned long lastHeartbeat = 0;

// Variables for live display
String txStatus = "ON";
String currentSNR = "15";
String currentChannel = "AWGN";
unsigned long lastLCDupdate = 0;

// ── helpers ───────────────────────────────────────────────────
void updateLiveDisplay() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("TX STATUS:");
  lcd.setCursor(11, 0); lcd.print(txStatus);
  
  lcd.setCursor(0, 1); lcd.print("MSG:");
  if (inputData.length() > 0) {
    lcd.setCursor(4, 1); lcd.print(inputData);
    lcd.print(" ");
  } else {
    lcd.setCursor(4, 1); lcd.print("      ");
  }
  
  lcd.setCursor(0, 2); lcd.print("SNR:");
  lcd.setCursor(4, 2); lcd.print(currentSNR);
  lcd.setCursor(8, 2); lcd.print("MOD:");
  lcd.setCursor(12, 2); lcd.print(currentMod.substring(0, 5));
  
  lcd.setCursor(0, 3); lcd.print("CHNL:");
  lcd.setCursor(5, 3); lcd.print(currentChannel);
}

// Heartbeat: keeps connection indicator alive
void sendHeartbeat() {
  Serial.print("TX_STATUS_LIVE:");
  Serial.print(currentMod);  Serial.print("|");
  Serial.print(currentFEC);  Serial.print("|");
  Serial.println(totalPacketsSent);
}

// ── setup ─────────────────────────────────────────────────────
void setup() {
  lcd.begin(20, 4);
  BT.begin(9600);
  Serial.begin(9600);

  // Welcome screen for 3 seconds
  lcd.clear();
  lcd.setCursor(4, 0); lcd.print("*WELCOME*");
  lcd.setCursor(2, 1); lcd.print("BER ADAPTIVE SYS");
  lcd.setCursor(5, 2); lcd.print("TX READY!!!");
  delay(3000);

  // Main live display
  updateLiveDisplay();

  Serial.println("TX_STATUS:READY");
}

// ── loop ──────────────────────────────────────────────────────
void loop() {

  // 1. HEARTBEAT every 2 seconds
  if (millis() - lastHeartbeat >= 2000) {
    lastHeartbeat = millis();
    sendHeartbeat();
  }

  // 2. RECEIVE MOD COMMANDS FROM RX VIA BLUETOOTH
  while (BT.available()) {
    String cmd = BT.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("CMD_MOD:")) {
      currentMod = cmd.substring(8);
      currentMod.trim();
      updateLiveDisplay();
      Serial.print("TX_MOD:"); Serial.println(currentMod);
      Serial.println("TX_ACK:MOD_UPDATED");
    }
  }

  // 3. RECEIVE COMMANDS FROM PYTHON DASHBOARD VIA USB SERIAL
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("FEC:")) {
      currentFEC = cmd.substring(4);
      currentFEC.trim();
      Serial.print("TX_FEC:"); Serial.println(currentFEC);
    }
    else if (cmd.startsWith("SET_MOD:")) {
      currentMod = cmd.substring(8);
      currentMod.trim();
      updateLiveDisplay();
      Serial.print("TX_MOD:"); Serial.println(currentMod);
    }
    else if (cmd.startsWith("SNR:")) {
      currentSNR = cmd.substring(4);
      currentSNR.trim();
      updateLiveDisplay();
      Serial.print("TX_SNR_UPDATED:"); Serial.println(currentSNR);
    }
    else if (cmd == "CH:AWG") {
      currentChannel = "AWGN";
      updateLiveDisplay();
      Serial.println("TX_CH_UPDATED:AWGN");
    }
    else if (cmd == "CH:RAY") {
      currentChannel = "RAYLEIGH";
      updateLiveDisplay();
      Serial.println("TX_CH_UPDATED:RAYLEIGH");
    }
    else if (cmd == "STATUS") {
      Serial.println("TX_STATUS:ONLINE");
      Serial.print("TX_MOD:"); Serial.println(currentMod);
      Serial.print("TX_FEC:"); Serial.println(currentFEC);
    }
  }

  // 4. KEYPAD INPUT
  char key = keypad.getKey();
  if (key) {

    if (key == '#') {
      // ── TRANSMIT on # ─────────────────────────────────────
      if (inputData.length() > 0) {
        txStatus = "SENT";
        updateLiveDisplay();
        
        // Build BT packet with FEC prefix
        String packet = "";
        if      (currentFEC == "HAM(7,4)") packet = "H:";
        else if (currentFEC == "REP3X")    packet = "R:";
        packet += inputData;
        BT.println(packet);

        lastSentBits = inputData.length() * 8;
        totalPacketsSent++;

        // Send to dashboard
        Serial.print("TX_BITS:"); Serial.println(lastSentBits);
        Serial.print("TX_DATA:"); Serial.println(inputData);

        inputData = "";
        delay(600);
        
        txStatus = "ON";
        updateLiveDisplay();
      }

    } else if (key == '*') {
      // ── CLEAR on * ────────────────────────────────────────
      inputData = "";
      txStatus = "ON";
      updateLiveDisplay();
      Serial.println("TX_CLEARED");

    } else {
      // ── REGULAR KEY ────────────────────────────────────────
      if (inputData.length() < 12) {
        inputData += key;
        txStatus = "TYPING...";
        updateLiveDisplay();
        
        Serial.print("TX_TYPING:"); Serial.println(inputData);
        
        // Reset status after 1 second of no typing
        unsigned long startTime = millis();
        while (keypad.getKey() == 0 && (millis() - startTime) < 1000) {
          delay(10);
        }
        if (keypad.getKey() == 0) {
          txStatus = "ON";
          updateLiveDisplay();
        }
      }
    }
  }
  
  // Update LCD periodically for any changes
  if (millis() - lastLCDupdate > 500) {
    lastLCDupdate = millis();
    if (txStatus == "ON") {
      updateLiveDisplay();
    }
  }
}