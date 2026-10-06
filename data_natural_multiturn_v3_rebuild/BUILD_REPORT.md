# Natural Multiturn V3 Dataset Build Report

참고 기준: [`agent_contract.py`](../src/soomac_irc/soomac_irc/agent_contract.py), [`decision_model.py`](../src/soomac_irc/soomac_irc/decision_model.py), [`response_model.py`](../src/soomac_irc/soomac_irc/response_model.py)

→ runtime `25ce994`를 freeze된 계약으로 사용하며 runtime source는 수정하지 않았다.

## Decision v3

| split | rows |
|---|---:|
| train | 2595 |
| validation | 285 |
| test | 285 |
| challenge | 495 |
| holdout | 377 |

- schema invalid: 0
- legacy field: 0
- duplicate: 0
- contamination: 0
- high-risk review: 200
- unresolved suspicious mutation: 0

주요 분포:

```json
{
  "route_task": 2874,
  "order": 2165,
  "negative": 254,
  "restrictions": 518,
  "multi_intent": 353,
  "preferences": 65,
  "reference": 430,
  "recommendation_request": 155,
  "recommendation_revise": 22,
  "pending_interaction": 232,
  "confirmation_accept": 77,
  "commit": 204,
  "question": 1222,
  "clarify": 198,
  "stt_noisy": 92,
  "confirmation_reject": 129,
  "route_general": 676,
  "route_mixed": 487,
  "hypothetical": 40
}
```

## Response v1

| split | rows |
|---|---:|
| train | 898 |
| validation | 101 |
| test | 86 |
| challenge | 75 |
| holdout | 65 |

- invalid input: 0
- empty reply: 0
- fact-consistency failure: 0
- next_prompt failure: 0
- duplicate: 0
- contamination: 0
- high-risk review: 200
- state contradiction: 0
- runtime contract mismatch: 0
- malformed Korean: 0

주요 분포:

```json
{
  "current_execution_pending_fixture": 1,
  "next_prompt_present": 691,
  "next_prompt_execution": 197,
  "question_or_direction_included": 548,
  "pending_execution": 197,
  "policy_state_update_only": 458,
  "current_recommendation_fixture": 1,
  "next_prompt_recommendation": 121,
  "pending_recommendation": 121,
  "policy_recommendation_proposed": 121,
  "current_missing_order_fixture": 1,
  "next_prompt_missing_order": 54,
  "policy_missing_order": 54,
  "current_partial_unsupported_fixture": 1,
  "next_prompt_validation_issues": 303,
  "policy_unsupported": 121,
  "current_restriction_conflict_fixture": 1,
  "policy_restriction_conflict": 121,
  "current_physical_conflict_fixture": 1,
  "policy_physical_state": 61,
  "current_confirmation_reject_fixture": 1,
  "policy_confirmation_rejected": 76,
  "prompt_boundary_general": 1,
  "policy_general": 16,
  "current_execution_allowed_fixture": 1,
  "policy_execution_allowed": 61,
  "current_section_transition_fixture": 1,
  "next_prompt_section_prompt": 16,
  "policy_section_transition": 16,
  "normal_order_mutation": 60,
  "execution_accept": 60,
  "recommendation_revise": 60,
  "recommendation_unavailable": 60,
  "policy_recommendation_unavailable": 60,
  "recommendation_reject": 15,
  "preference_update": 20,
  "restriction_update": 60,
  "missing_order": 53,
  "unsupported": 60,
  "restriction_conflict": 60,
  "physical_conflict": 60,
  "partial_success": 60,
  "status_question": 60,
  "general_question": 15,
  "mixed_turn": 60,
  "section_transition": 15,
  "understanding_clarify": 60,
  "policy_understanding": 60,
  "stt_noisy": 15,
  "pending_status_question": 60,
  "multiple_validation_issues": 60,
  "current_preselected_refresh_fixture": 1,
  "preselected_pending": 60,
  "legacy_robot_completed_fixture": 1,
  "completion": 60,
  "recommendation_request": 60,
  "execution_reject": 60
}
```

## Source salvage

| source | rows |
|---|---:|
| 실제 runtime review Decision | 13 |
| legacy natural Decision | 3152 |
| frozen challenge/holdout | 872 |
| current reviewed Response | 12 |
| 신규 structured Response | 1213 |
| Decision discard | 188 |
| Response discard | 227 |

legacy target은 학습 row로 복사하지 않고 current 8-field contract로 재라벨하였다. model prediction 파일은 source로 사용하지 않았다.

## 대표 Decision gold 10개

### 1. STT noise — `DV3-L01298`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "noodle",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "면양 적개로 해줘"
  },
  "target": {
    "route": "task",
    "order": {
      "noodle_portion": "low"
    }
  }
}
```

### 2. reference — `DV3-L00958`

```json
{
  "input": {
    "recent_history": [
      {
        "role": "user",
        "content": "양파 보통으로 해줘"
      },
      {
        "role": "assistant",
        "content": "양파는 보통으로 저장했습니다."
      }
    ],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "양파": "normal"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "아까 그거 많이"
  },
  "target": {
    "route": "task",
    "order": {
      "toppings": {
        "양파": "high"
      }
    }
  }
}
```

### 3. compound intent — `DV3-L00731`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "noodle",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "게살 알레르기 있으니까 게살도 빼줘"
  },
  "target": {
    "route": "task",
    "order": {
      "toppings": {
        "게살": "none"
      }
    },
    "restrictions": [
      {
        "target": "게살",
        "reason": "allergy",
        "action": "add"
      }
    ]
  }
}
```

### 4. execution pending confirmation — `DV3-L01418`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": "토마토",
      "noodle_type": "얇은면",
      "noodle_portion": "normal",
      "toppings": {
        "치즈": "high",
        "페퍼론치노": "low"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": {
      "type": "execution",
      "source": "preselected",
      "section": "extra",
      "targets": [
        "치즈",
        "페퍼론치노"
      ],
      "items": {
        "치즈": "high",
        "페퍼론치노": "low"
      },
      "candidate": {
        "sauce": null,
        "noodle_type": null,
        "noodle_portion": null,
        "toppings": {
          "치즈": "high",
          "페퍼론치노": "low"
        }
      }
    },
    "robot_state": {
      "section": "extra",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "그 조합은 아니고 치즈만 유지해서 바로 담아줘"
  },
  "target": {
    "route": "task",
    "order": {
      "toppings": {
        "치즈": "high"
      }
    },
    "commit": true,
    "confirmation": "reject"
  }
}
```

### 5. recommendation revise — `DV3-L00853`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": {
      "type": "recommendation",
      "candidate": {
        "sauce": null,
        "noodle_type": null,
        "noodle_portion": null,
        "toppings": {
          "치즈": "normal"
        }
      },
      "reason_tags": [
        "매콤한 취향"
      ]
    },
    "robot_state": {
      "section": "extra",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "아까 추천 바꿔줘"
  },
  "target": {
    "route": "task",
    "recommendation": {
      "action": "revise"
    }
  }
}
```

### 6. unsupported meaning preserved — `DV3-L01049`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "meat",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "베이컨 많이 넣어줘"
  },
  "target": {
    "route": "task",
    "order": {
      "toppings": {
        "베이컨": "high"
      }
    }
  }
}
```

### 7. restriction — `DV3-L00501`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "버섯": "normal"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "noodle",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "양파 알레르기 있어"
  },
  "target": {
    "route": "task",
    "restrictions": [
      {
        "target": "양파",
        "reason": "allergy",
        "action": "add"
      }
    ]
  }
}
```

### 8. preference — `DV3-L00793`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [
      {
        "value": "담백하게"
      }
    ],
    "pending": null,
    "robot_state": {
      "section": "noodle",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "담백하게는 취소해"
  },
  "target": {
    "route": "task",
    "preferences": [
      {
        "value": "담백하게",
        "action": "remove"
      }
    ]
  }
}
```

### 9. question boundary — `DV3-L02105`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "extra",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "양파가 뭐야?"
  },
  "target": {
    "route": "general"
  }
}
```

### 10. clarify — `DV3-L01059`

```json
{
  "input": {
    "recent_history": [],
    "order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "robot_state": {
      "section": "meat",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "message": "면 양은 한 60프로"
  },
  "target": {
    "route": "task",
    "clarify": true
  }
}
```

## 대표 Response gold 10개

### 1. partial success — `RV1-S0786`

```json
{
  "input": {
    "user_text": "페퍼론치노 조금 넣고 베이컨도 추가해줘",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "페퍼론치노": "low"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "warning",
      "reason": "unsupported",
      "execute": false,
      "conflicts": [],
      "issues": {
        "unsupported": [
          {
            "field": "toppings.베이컨",
            "item": "베이컨",
            "value": "normal"
          }
        ],
        "invalid": [],
        "protected": [],
        "restriction_conflicts": [],
        "physical_conflicts": []
      },
      "unsupported": [
        {
          "field": "toppings.베이컨",
          "item": "베이컨",
          "value": "normal"
        }
      ],
      "protected": [],
      "invalid": [],
      "restriction_blocked": []
    },
    "applied_this_turn": {
      "order_changes": {
        "toppings.페퍼론치노": "low"
      },
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": {
      "type": "validation_issues",
      "issues": {
        "unsupported": [
          {
            "field": "toppings.베이컨",
            "item": "베이컨",
            "value": "normal"
          }
        ],
        "invalid": [],
        "protected": [],
        "restriction_conflicts": [],
        "physical_conflicts": []
      }
    },
    "robot_state": {
      "section": "extra",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "페퍼론치노는 조금만 반영했어요. 베이컨은 지원하지 않아 추가하지 않았어요. 지원되는 재료를 다시 고르거나 넘어갈 수 있어요."
  }
}
```

### 2. validation failure — `RV1-S1381`

```json
{
  "input": {
    "user_text": "양파 조금, 햄이랑 치즈도 넣어줘",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "양파": "low"
      },
      "restrictions": [
        {
          "target": "치즈",
          "reason": "allergy"
        }
      ]
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "warning",
      "reason": "restriction_conflict",
      "execute": false,
      "conflicts": [
        {
          "key": "toppings.치즈",
          "item": "치즈",
          "restriction": {
            "target": "치즈",
            "reason": "allergy"
          }
        }
      ],
      "issues": {
        "unsupported": [
          {
            "field": "toppings.햄",
            "item": "햄",
            "value": "normal"
          }
        ],
        "invalid": [],
        "protected": [],
        "restriction_conflicts": [
          {
            "key": "toppings.치즈",
            "item": "치즈",
            "restriction": {
              "target": "치즈",
              "reason": "allergy"
            }
          }
        ],
        "physical_conflicts": []
      },
      "unsupported": [
        {
          "field": "toppings.햄",
          "item": "햄",
          "value": "normal"
        }
      ],
      "protected": [],
      "invalid": [],
      "restriction_blocked": [
        {
          "key": "toppings.치즈",
          "item": "치즈",
          "restriction": {
            "target": "치즈",
            "reason": "allergy"
          }
        }
      ]
    },
    "applied_this_turn": {
      "order_changes": {
        "toppings.양파": "low"
      },
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": {
      "type": "validation_issues",
      "issues": {
        "unsupported": [
          {
            "field": "toppings.햄",
            "item": "햄",
            "value": "normal"
          }
        ],
        "invalid": [],
        "protected": [],
        "restriction_conflicts": [
          {
            "key": "toppings.치즈",
            "item": "치즈",
            "restriction": {
              "target": "치즈",
              "reason": "allergy"
            }
          }
        ],
        "physical_conflicts": []
      }
    },
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "양파는 조금만 반영했어요. 햄은 지원하지 않고, 치즈는 알레르기 제한 때문에 추가하지 않았어요. 지원 재료를 다시 고르거나 넘어갈 수 있어요."
  }
}
```

### 3. section transition — `RV1-S1021`

```json
{
  "input": {
    "user_text": "",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "pass",
      "reason": "section_transition",
      "execute": false,
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": {
      "type": "section_prompt",
      "section": "veggie",
      "text": "다음은 야채를 고르실 차례입니다. 양파와 버섯 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 말씀하셔도 돼요."
    },
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [
        {
          "class": "얇은면",
          "repeat_count": 2
        }
      ],
      "robot_started": true,
      "section_transition": {
        "type": "section_transition",
        "section": "noodle",
        "outcome": "completed",
        "next_section": "veggie"
      }
    },
    "recent_action_history": [
      {
        "type": "section_transition",
        "section": "noodle",
        "outcome": "completed",
        "next_section": "veggie"
      }
    ],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "다음은 채소 단계예요. 양파나 버섯을 고를 수 있고, 원하지 않으면 넘어가도 돼요."
  }
}
```

### 4. status question — `RV1-S0841`

```json
{
  "input": {
    "user_text": "지금까지 뭐 담았어?",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "양파": "low"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "pass",
      "reason": "state_update_only",
      "execute": false,
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": null,
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [
      {
        "type": "turn_applied",
        "order_changes": {
          "toppings.양파": "low"
        },
        "restriction_added": [],
        "restriction_removed": [],
        "preference_added": [],
        "preference_removed": []
      }
    ],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "현재 양파를 조금만 선택했고 채소 단계예요."
  }
}
```

### 5. pending status question — `RV1-S1261`

```json
{
  "input": {
    "user_text": "지금까지 뭐 담았어?",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "양파": "low"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": {
      "type": "execution",
      "source": "current_update",
      "section": "veggie",
      "targets": [
        "양파"
      ]
    },
    "policy": {
      "status": "pass",
      "reason": "state_update_only",
      "execute": false,
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": {
      "type": "execution",
      "source": "current_update",
      "section": "veggie",
      "targets": [
        "양파"
      ],
      "items": {}
    },
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [
      {
        "type": "turn_applied",
        "order_changes": {
          "toppings.양파": "low"
        },
        "restriction_added": [],
        "restriction_removed": [],
        "preference_added": [],
        "preference_removed": []
      }
    ],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "현재 양파를 조금만 선택했고 채소 단계예요. 실행 확인이 남아 있어요. 이대로 진행할까요?"
  }
}
```

### 6. recommendation revise — `RV1-S0241`

```json
{
  "input": {
    "user_text": "다른 걸로 추천해줘",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": {
      "type": "recommendation",
      "candidate": {
        "sauce": null,
        "noodle_type": null,
        "noodle_portion": null,
        "toppings": {
          "양파": "low"
        }
      },
      "reason_tags": [
        "수정 요청"
      ]
    },
    "policy": {
      "status": "pass",
      "reason": "recommendation_proposed",
      "execute": false,
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": {
      "proposal": {
        "sauce": null,
        "noodle_type": null,
        "noodle_portion": null,
        "toppings": {
          "양파": "low"
        }
      },
      "reason_tags": [
        "수정 요청"
      ],
      "accepted_fields": [
        "toppings.양파"
      ],
      "rejected_fields": []
    },
    "next_prompt": {
      "type": "recommendation",
      "candidate": {
        "sauce": null,
        "noodle_type": null,
        "noodle_portion": null,
        "toppings": {
          "양파": "low"
        }
      },
      "reason_tags": [
        "수정 요청"
      ]
    },
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "새 후보로 양파를 적은 양으로 추천드려요. 이 후보로 할까요?"
  }
}
```

### 7. recommendation unavailable — `RV1-S0301`

```json
{
  "input": {
    "user_text": "다른 걸 추천해줘",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "양파": "low"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "clarify",
      "reason": "recommendation_unavailable",
      "execute": false,
      "conflicts": [],
      "rejected_fields": [
        {
          "field": "toppings.양파",
          "item": "양파",
          "value": "low",
          "reason": "already_selected"
        }
      ]
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": {
      "proposal": {
        "sauce": null,
        "noodle_type": null,
        "noodle_portion": null,
        "toppings": {}
      },
      "reason_tags": [],
      "accepted_fields": [],
      "rejected_fields": [
        {
          "field": "toppings.양파",
          "item": "양파",
          "value": "low",
          "reason": "already_selected"
        }
      ]
    },
    "next_prompt": null,
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "현재 조건에서는 새 추천을 만들지 못했어요."
  }
}
```

### 8. missing order — `RV1-S0541`

```json
{
  "input": {
    "user_text": "이대로 진행해",
    "recent_history": [],
    "confirmed_order": {
      "sauce": "오일",
      "noodle_type": null,
      "noodle_portion": "low",
      "toppings": {},
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "clarify",
      "reason": "missing_order",
      "execute": false,
      "missing": [
        "noodle_type"
      ],
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": {
      "type": "missing_order",
      "fields": [
        "noodle_type"
      ]
    },
    "robot_state": {
      "section": "noodle",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "면 종류가 아직 정해지지 않았어요. 얇은면과 넓은면 중 어떤 걸로 할까요?"
  }
}
```

### 9. understanding clarify — `RV1-S1081`

```json
{
  "input": {
    "user_text": "그걸로 해줘",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "양파": "low"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": null,
    "policy": {
      "status": "clarify",
      "reason": "understanding",
      "execute": false,
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {},
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": null,
    "robot_state": {
      "section": "veggie",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "어떤 항목을 말씀하시는지 확인이 필요해요. 재료나 양을 다시 알려주세요."
  }
}
```

### 10. STT noisy — `RV1-S1211`

```json
{
  "input": {
    "user_text": "페페론치노 보통",
    "recent_history": [],
    "confirmed_order": {
      "sauce": null,
      "noodle_type": null,
      "noodle_portion": null,
      "toppings": {
        "페퍼론치노": "normal"
      },
      "restrictions": []
    },
    "preferences": [],
    "pending": {
      "type": "execution",
      "source": "current_update",
      "section": "extra",
      "targets": [
        "페퍼론치노"
      ]
    },
    "policy": {
      "status": "pass",
      "reason": "state_update_only",
      "execute": false,
      "conflicts": []
    },
    "applied_this_turn": {
      "order_changes": {
        "toppings.페퍼론치노": "normal"
      },
      "restriction_added": [],
      "restriction_removed": [],
      "preference_added": [],
      "preference_removed": []
    },
    "future_changes": [],
    "recommendation_result": null,
    "next_prompt": {
      "type": "execution",
      "source": "current_update",
      "section": "extra",
      "targets": [
        "페퍼론치노"
      ],
      "items": {}
    },
    "robot_state": {
      "section": "extra",
      "task_queue": [],
      "active_task": null,
      "completed_tasks": [],
      "robot_started": false
    },
    "recent_action_history": [],
    "execution_authorized": false,
    "starting_now": []
  },
  "target": {
    "reply": "페퍼론치노는 보통으로 선택했어요. 이대로 담을까요?"
  }
}
```

## 최종 판정

**READY FOR TRAINING**

Cause: legacy input과 current contract가 달라 input/state/target을 같은 기준으로 다시 구성해야 했다.

Effect: Decision과 Response를 독립적으로 학습하고 frozen evaluation으로 회귀를 검사할 수 있다.
