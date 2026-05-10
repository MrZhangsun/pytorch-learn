import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_nllb_translation():
    print("🚀 正在加载 NLLB-200 模型 (600M)...")
    print(f"💻 当前设备: {torch.accelerator.current_accelerator() if torch.backends.mps.is_available() else 'cpu'}")

    # 1. 加载模型和分词器
    model_name = "facebook/nllb-200-distilled-600M"

    # 指定 src_lang 为英语
    AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")

    # 使用 float16 精度以节省 Mac 的 36GB 内存
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("mps")

    # 2. 准备测试文本（包含一# ... existing code ...
    # 2. 准备测试文本（包含一些技术术语）
    text = "The VGG16 model uses a deep convolutional neural network architecture."
    print(f"\n📝 原文 (English): {text}")

    # 3. 预处理输入
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("mps")

    # 4. 进行翻译 (修正语言代码的获取方式)
    # NLLB 的语言代码通常是 'zho_Hans' (简体中文) 或 'eng_Latn' (英语)
    target_lang_id = tokenizer.convert_tokens_to_ids["zho_Hans"]

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=128
        )
    text = "The VGG16 model uses a deep convolutional neural network architecture."
    print(f"\n📝 原文 (English): {text}")

    # 3. 预处理输入
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("mps")

    # 4. 进行翻译 (forced_bos_token_id 强制指定目标语言为简体中文)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids["zho_Hans"],
            max_length=128
        )

    # 5. 解码并输出结果
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"🌏 译文 (中文): {result}")
    print("-" * 50)

if __name__ == '__main__':
    try:
        test_nllb_translation()
    except Exception as e:
        print(f"❌ 发生错误: {e}")
