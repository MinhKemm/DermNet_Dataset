ANSWER_FORMAT_MARKER = "[DermNet answer format]"


def answer_instruction(question_type, language):
    question_type = str(question_type).strip().lower()
    language = "en" if language == "en" else "vi"

    if question_type == "multi_choice":
        if language == "en":
            return (
                "Answer using only the uppercase letter sequence for the correct "
                "option(s), for example A or ACD. Do not add spaces, punctuation, "
                "or explanation."
            )
        return (
            "Chỉ trả lời chuỗi chữ cái in hoa của (các) đáp án đúng, ví dụ A hoặc "
            "ACD. Không thêm khoảng trắng, dấu câu hoặc giải thích."
        )

    if question_type == "judgement":
        if language == "en":
            return "Answer with exactly 'Yes' or 'No'. Do not add any other text."
        return "Chỉ trả lời chính xác 'Có' hoặc 'Không'. Không thêm nội dung khác."

    if question_type == "fill_in_blank":
        if language == "en":
            return "Answer in English with the missing term or phrase only. Do not write a full sentence."
        return "Chỉ điền thuật ngữ hoặc cụm từ còn thiếu bằng tiếng Việt. Không viết thành câu đầy đủ."

    if question_type == "short_answer":
        if language == "en":
            return (
                "Answer directly in English with a concise phrase or one short sentence. "
                "Do not add an introduction or unrelated explanation."
            )
        return (
            "Trả lời bằng tiếng Việt, trực tiếp bằng cụm từ hoặc một câu ngắn. Không thêm câu dẫn "
            "hay giải thích không liên quan."
        )

    if language == "en":
        return "Answer directly and concisely in English."
    return "Trả lời trực tiếp và ngắn gọn bằng tiếng Việt."


def append_answer_instruction(question, question_type, language):
    question = str(question).rstrip()
    if ANSWER_FORMAT_MARKER in question:
        return question
    instruction = answer_instruction(question_type, language)
    return f"{question}\n{ANSWER_FORMAT_MARKER} {instruction}"
