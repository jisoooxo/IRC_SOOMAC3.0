# LangGraph 주문과 ROS 실행이 함께 사용하는 메뉴·단계 정본

SAUCES = ("오일", "토마토", "크림")
NOODLE_TYPES = ("얇은면", "넓은면")
TOPPINGS = ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노")
AMOUNTS = ("low", "normal", "high")

# main_vlm.py가 소스 완료 뒤 cover를 직접 실행하고 /llm/reset을 보냄
SECTION_ORDER = ("noodle", "veggie", "meat", "extra", "lid", "sauce")

RESTRICTION_CATEGORY_ITEMS = {
    "유제품": ("크림", "치즈"),
    "갑각류": ("게살",),
    "육류": ("소시지",),
    "비건": ("소시지",),
}

RESTRICTION_REASONS = (
    "dislike",
    "cannot_eat",
    "allergy",
    "dietary_rule",
)

RESTRICTION_PRIORITY = (
    "allergy",
    "cannot_eat",
    "dietary_rule",
    "dislike",
)

RESTRICTION_CATEGORIES = (
    "유제품",
    "갑각류",
    "육류",
    "비건",
)

# 주문 집합 표현이 가리키는 canonical 토핑 묶음
ORDER_COLLECTION_TARGETS = {
    "야채": ("양파", "버섯"),
    "채소": ("양파", "버섯"),
    "추가 재료": ("치즈", "페퍼론치노"),
}