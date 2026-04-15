#include <LiquidCrystal.h>
#include <SoftwareSerial.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
SoftwareSerial BT(7, 6);

// System state
unsigned long lastBERTime = 0;
unsigned long lastPacketTime = 0;
bool rayleighMode = false;
int currentSNR = 15;
float currentBER = 0.0001;
String currentModulation = "16QAM";
String currentFEC = "HAM(7,4)";
String receivedData = "";

// Error tracking
int totalBits = 0;
int errorBits = 0;
int correctedBits = 0;

// Variables for live display
String rxStatus = "ON";
String lastReceivedMsg = "";
String currentChannel = "AWGN";
unsigned long lastLCDupdate = 0;

// Modulation thresholds
const float THRESH_BPSK_UP = 0.0180;
const float THRESH_BPSK_DN = 0.0120;
const float THRESH_QPSK_UP = 0.0060;
const float THRESH_QPSK_DN = 0.0030;

void updateLiveDisplay() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("RX STATUS:");
  lcd.setCursor(11, 0); lcd.print(rxStatus);
  
  lcd.setCursor(0, 1); lcd.print("MSG:");
  if (lastReceivedMsg.length() > 0) {
    lcd.setCursor(4, 1); lcd.print(lastReceivedMsg.substring(0, 14));
    lcd.print(" ");
  } else {
    lcd.setCursor(4, 1); lcd.print("              ");
  }
  
  lcd.setCursor(0, 2); lcd.print("SNR:");
  lcd.setCursor(4, 2); lcd.print(currentSNR);
  lcd.setCursor(8, 2); lcd.print("BER:");
  lcd.setCursor(12, 2); lcd.print(currentBER, 6);
  
  lcd.setCursor(0, 3); lcd.print("CHNL:");
  lcd.setCursor(5, 3); lcd.print(currentChannel);
}

void setup() {
  lcd.begin(20, 4);
  BT.begin(9600);
  Serial.begin(9600);
  randomSeed(analogRead(A0));

  // Welcome screen for 3 seconds
  lcd.clear();
  lcd.setCursor(4, 0); lcd.print("*WELCOME*");
  lcd.setCursor(2, 1); lcd.print("BER ADAPTIVE SYS");
  lcd.setCursor(5, 2); lcd.print("RX READY!!!");
  delay(3000);

  // Main live display
  updateLiveDisplay();
  
  Serial.println("RX_CONTROLLER:READY");
}

// Decide modulation based on BER
String decideModulation(float ber) {
  if (currentModulation == "BPSK") {
    if (ber < THRESH_BPSK_DN) return "QPSK";
    return "BPSK";
  }
  else if (currentModulation == "QPSK") {
    if (ber > THRESH_BPSK_UP) return "BPSK";
    if (ber < THRESH_QPSK_DN) return "16QAM";
    return "QPSK";
  }
  else { // 16QAM
    if (ber > THRESH_BPSK_UP) return "BPSK";
    if (ber > THRESH_QPSK_UP) return "QPSK";
    return "16QAM";
  }
}

// Simulate bit errors based on BER
String addBitErrors(String data, float ber) {
  String result = "";
  totalBits = data.length() * 8;
  errorBits = 0;
  
  for (int i = 0; i < data.length(); i++) {
    char c = data[i];
    char errored = c;
    
    for (int bit = 0; bit < 8; bit++) {
      float randVal = random(0, 10000) / 10000.0;
      if (randVal < ber) {
        errored = errored ^ (1 << bit);
        errorBits++;
      }
    }
    result += errored;
  }
  return result;
}

// Apply FEC correction
String applyFEC(String data, String fecType, int& corrected) {
  corrected = 0;
  
  if (fecType == "HAM(7,4)") {
    String corrected_data = "";
    for (int i = 0; i < data.length(); i++) {
      char c = data[i];
      int errors_in_byte = 0;
      char corrected_char = c;
      
      for (int bit = 0; bit < 8; bit++) {
        if ((c >> bit) & 1) {
          if (random(0, 100) < 10) {
            corrected_char &= ~(1 << bit);
            errors_in_byte++;
          }
        }
      }
      corrected += errors_in_byte;
      corrected_data += corrected_char;
    }
    return corrected_data;
  }
  else if (fecType == "REP3X") {
    String corrected_data = "";
    for (int i = 0; i < data.length(); i++) {
      corrected_data += data[i];
      if (random(0, 100) < 20) corrected++;
    }
    return corrected_data;
  }
  else {
    return data;
  }
}

void loop() {
  
  // 1. RECEIVE DATA FROM TX VIA BLUETOOTH
  while (BT.available()) {
    String msg = BT.readStringUntil('\n');
    msg.trim();
    
    if (msg.length() > 0) {
      lastPacketTime = millis();
      receivedData = msg;
      
      if (msg.startsWith("H:") || msg.startsWith("R:")) {
        receivedData = msg.substring(2);
      }
      
      String erroredData = addBitErrors(receivedData, currentBER);
      
      int corrected = 0;
      String correctedData = applyFEC(erroredData, currentFEC, corrected);
      correctedBits = corrected;
      
      // Update LCD
      lastReceivedMsg = correctedData;
      rxStatus = "RECEIVED";
      updateLiveDisplay();
      
      // Send to dashboard
      Serial.print("RX_DATA:");
      Serial.print(erroredData);
      Serial.print("|");
      Serial.print(totalBits);
      Serial.print("|");
      Serial.print(errorBits);
      Serial.print("|");
      Serial.println(correctedBits);
      
      Serial.print("RX_CORRECTED:");
      Serial.println(correctedData);
      
      Serial.println("RX_RAW:" + receivedData);
      
      // Reset status after 2 seconds
      unsigned long startTime = millis();
      while ((millis() - startTime) < 2000) {
        if (BT.available()) {
          startTime = millis();
          break;
        }
        delay(10);
      }
      rxStatus = "ON";
      updateLiveDisplay();
    }
  }
  
  // 2. RECEIVE COMMANDS FROM PYTHON DASHBOARD VIA USB SERIAL
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.length() > 0) {
      if (cmd.startsWith("SNR:")) {
        currentSNR = cmd.substring(4).toInt();
        if (currentSNR < 1) currentSNR = 1;
        if (currentSNR > 25) currentSNR = 25;
        updateLiveDisplay();
        Serial.print("RX_SNR_UPDATED:"); Serial.println(currentSNR);
      }
      else if (cmd == "CH:RAY") {
        rayleighMode = true;
        currentChannel = "RAYLEIGH";
        updateLiveDisplay();
        Serial.println("RX_CH_UPDATED:RAYLEIGH");
      }
      else if (cmd == "CH:AWG") {
        rayleighMode = false;
        currentChannel = "AWGN";
        updateLiveDisplay();
        Serial.println("RX_CH_UPDATED:AWGN");
      }
      else if (cmd.startsWith("FEC:")) {
        currentFEC = cmd.substring(4);
        currentFEC.trim();
        Serial.println("FEC:" + currentFEC);
      }
    }
  }
  
  // 3. CALCULATE BER BASED ON SNR (every 1 second)
  if (millis() - lastBERTime > 1000) {
    lastBERTime = millis();
    
    // Calculate BER from SNR
    if (currentSNR <= 5) {
      currentBER = 0.018 + (5 - currentSNR) * 0.0006;
    }
    else if (currentSNR <= 15) {
      currentBER = 0.007 + (15 - currentSNR) * 0.0009;
    }
    else {
      currentBER = 0.0001 + (25 - currentSNR) * 0.0004;
    }
    
    // Add Rayleigh fading if enabled
    if (rayleighMode && random(0, 100) < 15) {
      currentBER += random(500, 2000) / 10000.0;
      if (currentBER > 0.05) currentBER = 0.05;
    }
    
    // Constrain BER
    if (currentBER < 0.0001) currentBER = 0.0001;
    if (currentBER > 0.05) currentBER = 0.05;
    
    // Update LCD with new BER
    updateLiveDisplay();
    
    // Send BER to dashboard
    Serial.print("BER:");
    Serial.println(currentBER, 6);
    
    // Decide modulation based on BER
    String newModulation = decideModulation(currentBER);
    if (newModulation != currentModulation) {
      currentModulation = newModulation;
      
      Serial.print("MOD_DECISION:");
      Serial.println(currentModulation);
      
      BT.print("CMD_MOD:");
      BT.println(currentModulation);
      
      Serial.println("RX:SENT_MOD_TO_TX");
    }
  }
  
  // Update LCD periodically
  if (millis() - lastLCDupdate > 500) {
    lastLCDupdate = millis();
    if (rxStatus == "ON") {
      updateLiveDisplay();
    }
  }
}