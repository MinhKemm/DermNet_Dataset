from typing import Dict, Any, List

def can_generate_location(extracted_facts: Dict[str, Any]) -> bool:
    """Task 1: Skip if location == 'Không xác định được trên ảnh' or 'Không đủ dữ liệu quan sát'"""
    loc = extracted_facts.get("location", "")
    if loc in ["Không xác định được trên ảnh", "Không đủ dữ liệu quan sát", ""]:
        return False
    return True

def can_generate_attribute(extracted_facts: Dict[str, Any], key_attribute: str) -> bool:
    """Task 2: Skip if KEY_ATTRIBUTE value is empty or 'không rõ'"""
    attr_val = extracted_facts.get(key_attribute, "")
    if not attr_val or attr_val.lower() == "không rõ":
        return False
    return True

def can_generate_lesion(extracted_facts: Dict[str, Any]) -> bool:
    """Task 3: Skip if lesion is not clearly determined"""
    lesion = extracted_facts.get("lesion", "")
    if not lesion or "không" in lesion.lower() or "rõ" in lesion.lower():
        # Adjust logic as needed based on actual 'not clearly determined' strings
        if lesion in ["", "Không rõ", "Không xác định"]:
            return False
    return True

def can_generate_lesion_reasoning(extracted_facts: Dict[str, Any]) -> bool:
    """Task 4: Must have both lesion AND lesion_reasoning"""
    lesion = extracted_facts.get("lesion", "")
    reasoning = extracted_facts.get("lesion_reasoning", "")
    if not lesion or not reasoning:
        return False
    return True

def can_generate_diagnosis(extracted_facts: Dict[str, Any]) -> bool:
    """Task 5: Always available"""
    return True

def get_available_tasks(extracted_facts: Dict[str, Any], key_attribute: str) -> List[int]:
    """Returns list of task numbers [1,2,3,4,5] that are available"""
    tasks = []
    if can_generate_location(extracted_facts):
        tasks.append(1)
    if can_generate_attribute(extracted_facts, key_attribute):
        tasks.append(2)
    if can_generate_lesion(extracted_facts):
        tasks.append(3)
    if can_generate_lesion_reasoning(extracted_facts):
        tasks.append(4)
    if can_generate_diagnosis(extracted_facts):
        tasks.append(5)
    return tasks
