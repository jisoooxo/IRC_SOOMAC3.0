/* ============================================================
   레일 절대 회전수 제어
   TB6560 + NEMA17 42HS48-1404 + AccelStepper

   - NEMA17: 200 step/rev
   - 마이크로스텝: 4
   - 따라서 800 step/rev

   - HOME:
       현재 내부 위치값과 상관없이
       센서 방향으로 계속 이동
       센서 감지 시 즉시 정지
       감지 위치를 0 rev로 재설정
       "HOME DONE" 전송

   - 일반 이동:
       원점 기준 절대 회전수 위치로 이동

   ------------------------------------------------------------
   시리얼 프로토콜 (115200 baud)

     H        HOME 시작
     R<rev>   절대 회전수 위치 이동
              예) R75.0
     X        정지
     Z        현재 위치를 강제로 0 설정
     ?        현재 상태 출력

   ============================================================ */

#include <AccelStepper.h>


/* ==================== 핀 ==================== */

#define DIR_PIN   8
#define STEP_PIN  9
#define KILL_PIN  2


/* ==================== 모터 설정 ==================== */

// 200 step/rev × microstep 4
// 설치 방향 때문에 부호 반전
const float STEPS_PER_REV = -800.0;


/* ==================== 일반 이동 설정 ==================== */

const float MAX_SPEED_REV = 4.0;      // rev/s
const float ACCEL_REV     = 18.75;    // rev/s^2


/* ==================== HOME 설정 ==================== */

// HOME 전용 step 속도
// 현재 설치 기준 센서 방향
const float HOME_STEP_SPEED = 1600.0;


/* ==================== 센서 설정 ==================== */

const unsigned long DEBOUNCE_MS = 20;


/* ==================== STEPPER ==================== */

AccelStepper stepper(
  AccelStepper::DRIVER,
  STEP_PIN,
  DIR_PIN
);


/* ==================== 상태 ==================== */

char buf[32];
uint8_t idx = 0;

unsigned long lastReport = 0;

bool wasMoving = false;
bool killNow = false;
bool move_home = false;


/* ==================== SETUP ==================== */

void setup() {
  Serial.begin(115200);

  stepper.setMaxSpeed(
    MAX_SPEED_REV * abs(STEPS_PER_REV)
  );

  stepper.setAcceleration(
    ACCEL_REV * abs(STEPS_PER_REV)
  );

  // 부팅 시 내부 좌표는 임시 0
  // 실제 원점은 HOME 수행 후 센서 위치로 다시 설정됨
  stepper.setCurrentPosition(0);

  pinMode(
    KILL_PIN,
    INPUT_PULLUP
  );

  Serial.println(F("READY"));
}


/* ==================== LOOP ==================== */

void loop() {
  killNow = (
    digitalRead(KILL_PIN) == LOW
  );

  handleSerial();
  checkKill();

  // HOME 중에는 현재 위치와 관계없이 고정 속도로 이동
  if (move_home) {
    stepper.runSpeed();
  } else {
    stepper.run();
  }

  reportStatus();
}


/* ==================== HOME ==================== */

void startHoming() {
  if (move_home) {
    Serial.println(
      F("HOME ALREADY RUNNING")
    );
    return;
  }

  // 이미 물리적으로 센서가 눌려있다면
  // 그 위치를 바로 원점으로 설정
  if (
    digitalRead(KILL_PIN) == LOW
  ) {
    stepper.setSpeed(0);
    stepper.setCurrentPosition(0);

    Serial.println(
      F("HOME DONE")
    );

    return;
  }

  move_home = true;
  wasMoving = false;

  // 현재 좌표값과 상관없이 센서 방향으로 계속 이동
  stepper.setSpeed(
    HOME_STEP_SPEED
  );

  Serial.println(
    F("MOVE_HOME")
  );
}


/* ==================== 정지 ==================== */

void emergencyStop() {
  move_home = false;

  stepper.setSpeed(0);
  stepper.stop();

  wasMoving = false;

  Serial.println(
    F("STOPPING")
  );
}


/* ==================== 원점 센서 ==================== */

void killSwitchHit() {
  // HOME이든 일반 이동이든 센서에 닿으면 즉시 정지
  stepper.setSpeed(0);
  stepper.stop();

  // 센서가 눌린 실제 위치를 새 원점으로 설정
  stepper.setCurrentPosition(0);

  if (move_home) {
    move_home = false;

    Serial.println(
      F("HOME DONE")
    );

  } else {
    Serial.println(
      F("SET ZERO")
    );
  }

  wasMoving = false;
}


void checkKill() {
  static int stable = HIGH;
  static int lastRead = HIGH;
  static unsigned long tChange = 0;

  int r =
    killNow ? LOW : HIGH;

  if (r != lastRead) {
    lastRead = r;
    tChange = millis();
  }

  if (
    millis() - tChange > DEBOUNCE_MS &&
    r != stable
  ) {
    stable = r;

    if (stable == LOW) {
      killSwitchHit();

    } else {
      Serial.println(
        F("KILL RELEASED")
      );
    }
  }
}


/* ==================== 일반 절대 회전 이동 ==================== */

void gotoRotation(float rev) {
  // 센서가 눌린 상태에서는 음수 방향 이동 금지
  if (
    killNow &&
    rev < 0
  ) {
    Serial.println(
      F("ERR: kill switch ON - negative move rejected")
    );

    return;
  }

  long targetStep =
    (long)(
      rev * STEPS_PER_REV
    );

  stepper.moveTo(
    targetStep
  );

  Serial.print(
    F("GOTO R")
  );

  Serial.println(
    rev,
    3
  );
}


/* ==================== SERIAL ==================== */

void handleSerial() {
  while (
    Serial.available()
  ) {
    char c =
      Serial.read();

    if (
      c == '\n' ||
      c == '\r'
    ) {
      if (idx > 0) {
        buf[idx] = 0;

        parseCmd(
          buf
        );

        idx = 0;
      }

    } else if (
      idx < sizeof(buf) - 1
    ) {
      buf[idx++] = c;
    }
  }
}


void parseCmd(char* s) {
  switch (s[0]) {

    /* HOME */
    case 'H':
    case 'h':
      startHoming();
      break;


    /* 절대 회전 위치 이동 */
    case 'R':
    case 'r':

      if (move_home) {
        Serial.println(
          F("ERR: move_home")
        );

        break;
      }

      gotoRotation(
        atof(s + 1)
      );

      break;


    /* 정지 */
    case 'X':
    case 'x':

      emergencyStop();

      break;


    /* 현재 위치 강제 0 설정 */
    case 'Z':
    case 'z':

      if (move_home) {
        Serial.println(
          F("ERR: move_home")
        );

        break;
      }

      stepper.setSpeed(0);
      stepper.setCurrentPosition(0);

      Serial.println(
        F("ZEROED")
      );

      break;


    /* 현재 위치 확인 */
    case '?':

      printPosition(
        "POS"
      );

      break;


    default:

      Serial.println(
        F("ERR cmd")
      );

      break;
  }
}


/* ==================== 상태 출력 ==================== */

void printPosition(
  const char* tag
) {
  float rev =
    stepper.currentPosition()
    / STEPS_PER_REV;

  char revStr[12];

  dtostrf(
    rev,
    5,
    3,
    revStr
  );

  char line[48];

  snprintf(
    line,
    sizeof(line),
    "%s %s rev %s",
    tag,
    revStr,
    move_home
      ? "MOVE_HOME"
      : (
          stepper.isRunning()
            ? "MOVING"
            : "IDLE"
        )
  );

  Serial.println(
    line
  );
}


/* ==================== 일반 이동 상태 ==================== */

void reportStatus() {
  // HOME 완료는 센서 감지 시 HOME DONE으로 따로 처리
  if (move_home) {
    return;
  }

  bool moving =
    stepper.isRunning();

  // 일반 R 이동 완료
  if (
    wasMoving &&
    !moving
  ) {
    printPosition(
      "DONE"
    );
  }

  wasMoving = moving;

  // 일반 이동 중 위치 출력
  if (
    moving &&
    millis() - lastReport >= 100
  ) {
    lastReport =
      millis();

    printPosition(
      "POS"
    );
  }
}