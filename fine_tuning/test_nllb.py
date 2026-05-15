import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def test_nllb_translation():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备: {device}")

    model_name = "facebook/nllb-200-distilled-600M"

    # 1. 加载分词器（指定源语言为英语）
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")

    # 2. 加载模型（使用半精度以降低显存）
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    # 3. 测试文本
    texts = [
        "The VGG16 model uses a deep convolutional neural network architecture.",
        "Artificial intelligence is transforming the way we interact with technology.",
        "Machine learning models require large amounts of data for training.",
        "Do you know what is the UPC?"
    ]

    for text in texts:
        print(f"\n原文: {text}")

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),
                max_length=128,
                num_beams=4,
            )

        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(f"译文: {result}")


if __name__ == "__main__":
    test_nllb_translation()
