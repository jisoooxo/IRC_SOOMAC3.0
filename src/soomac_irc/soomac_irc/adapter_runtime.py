def activate_adapter(model, adapter_name: str | None) -> bool:
    """요청한 LoRA Adapter가 있을 때만 활성화한다."""
    if adapter_name is None:
        return False
    set_adapter = getattr(model, "set_adapter", None)
    if not callable(set_adapter):
        raise TypeError(f"{adapter_name} Adapter를 활성화할 수 없는 모델임")
    set_adapter(adapter_name)
    return True
