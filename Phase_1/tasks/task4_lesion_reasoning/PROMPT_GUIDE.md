# Hướng dẫn Prompt cho Task 4: Lesion Reasoning

Task này sinh các cặp VQA về lý do (reasoning) nhận diện tổn thương (lesion).

- **4.1 Multi-choice**: Cần LLM để sinh distractors hoặc dùng fallback.
- **4.2 Judgement**: Sinh ra 2 câu (Có/Không) dựa trên target_flag.
- **4.3 Short answer**: Hỏi trực tiếp lý do.
- **4.4 Fill-in-blank**: Xóa các từ khóa lâm sàng bằng Regex và yêu cầu mô hình điền.
