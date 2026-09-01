#include <Servo.h>

Servo pump;
Servo valve;

const int PUMP_PIN = 3;
const int VALVE_PIN = 2;

const int PUMP_ON_ANGLE = 180;
const int PUMP_OFF_ANGLE = 0;

const int VALVE_OPEN_ANGLE = 180;
const int VALVE_CLOSE_ANGLE = 0;

const unsigned long RELEASE_TIME_MS = 300;

bool gripped = false;

void setup()
{
  Serial.begin(115200);

  // 문자열 수신 대기 시간을 짧게 설정
  Serial.setTimeout(100);

  pump.attach(PUMP_PIN);
  valve.attach(VALVE_PIN);

  // 시작 시 공압 OFF
  pump.write(PUMP_OFF_ANGLE);
  valve.write(VALVE_CLOSE_ANGLE);
}

void loop()
{
  if (Serial.available() <= 0)
  {
    return;
  }

  String command = Serial.readStringUntil('\n');

  command.trim();
  command.toUpperCase();

  if (command == "ON")
  {
    if (!gripped)
    {
      // 배기 밸브를 닫고 흡착 펌프 작동
      valve.write(VALVE_CLOSE_ANGLE);
      pump.write(PUMP_ON_ANGLE);

      gripped = true;
      Serial.println("PNEUMATIC_ON");
    }
  }
  else if (command == "OFF")
  {
    if (gripped)
    {
      // 밸브를 열어 진공 해제
      valve.write(VALVE_OPEN_ANGLE);
      delay(RELEASE_TIME_MS);

      // 펌프 정지 후 밸브 닫기
      pump.write(PUMP_OFF_ANGLE);
      valve.write(VALVE_CLOSE_ANGLE);

      gripped = false;
      Serial.println("PNEUMATIC_OFF");
    }
  }
  else
  {
    Serial.print("UNKNOWN_COMMAND: ");
    Serial.println(command);
  }
}