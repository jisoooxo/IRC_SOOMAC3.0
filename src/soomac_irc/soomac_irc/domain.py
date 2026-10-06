# LangGraph 주문과 ROS 실행이 함께 사용하는 메뉴·단계 정본

SAUCES = ("오일", "토마토", "크림")
NOODLE_TYPES = ("얇은면", "넓은면")
TOPPINGS = ("양파", "버섯", "소시지", "게살", "치즈", "페퍼론치노")
AMOUNTS = ("low", "normal", "high")

# lid section이 cover(뚜껑) task를 소유한다. main_vlm.py는 소스(마지막 작업) 완료 뒤 /llm/reset을 보냄
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
