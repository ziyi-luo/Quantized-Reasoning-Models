# 该测试代码适用于原模型和flatquant未做重参数化的推理测试情况

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch_npu

# 模型名称（可以换成本地路径或 HuggingFace Hub 的模型名）
model_name = "PATH_TO_MODEL"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # 半精度节省显存
    device_map="auto"            # 自动分配到 GPU / NPU / CPU
)

# 输入提示
prompt = "hello, how are you?"

# 编码
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,      # 生成的最大长度
        temperature=0.7,         # 控制生成的多样性
        top_p=0.9,               # nucleus sampling
        do_sample=True           # 开启采样
    )

# 解码
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
