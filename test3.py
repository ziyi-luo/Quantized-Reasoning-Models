import torch
import torch_npu

def test_nonzero_issue():
    # 初始化测试数据
    # 构造一些数据，确保有满足 (xmin == 0) & (xmax == 0) 的情况
    # 例如第一个元素 (0, 0) 应该触发条件
    xmin = torch.tensor([0.0, 1.0, 0.0, -5.0, 0.0]).npu()
    xmax = torch.tensor([0.0, 2.0, 5.0, 0.0, 0.0]).npu()
    
    print("--- 修改前 ---")
    print(f"xmin: {xmin}, xmin.device: {xmin.device}")
    print(f"xmax: {xmax}")

    # 待测试的代码片段
    tmp = (xmin == 0) & (xmax == 0)
    
    print(f"\ntmp (mask): {tmp}")
    
    # 这里会触发nonzero对应的底层算子aclnnNonzeroV2报错 (如果在 NPU 环境下)
    try:
        xmin[tmp] = -1
        xmax[tmp] = +1
        print("\n--- 执行成功 ---")
    except Exception as e:
        print(f"\n--- 执行出错 ---")
        print(e)

    print("\n--- 修改后 ---")
    print(f"xmin: {xmin}")
    print(f"xmax: {xmax}")

    # 验证结果
    # 预期：索引0和4的位置，xmin变为-1，xmax变为1
    expected_xmin = torch.tensor([-1.0, 1.0, 0.0, -5.0, -1.0]).npu()
    expected_xmax = torch.tensor([1.0, 2.0, 5.0, 0.0, 1.0]).npu()
    
    if torch.allclose(xmin, expected_xmin) and torch.allclose(xmax, expected_xmax):
        print("\n结果验证: 通过 ✅")
    else:
        print("\n结果验证: 失败 ❌")

if __name__ == "__main__":
    # 如果在昇腾 NPU 环境，可能需要设置 device
    # device = "npu" if torch.npu.is_available() else "cpu"
    # 这里默认在 CPU 上运行逻辑验证，如果为了复现 NPU 报错，请确保环境支持并取消注释相关代码将 tensor 移到 npu
    test_nonzero_issue()
