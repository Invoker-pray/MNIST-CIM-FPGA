#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   bash onboard_A/scripts/setup_onboard_A.sh [项目根目录]
#
# 例子:
#   bash onboard_A/scripts/setup_onboard_A.sh .
#
# 说明:
# - 默认认为当前工作目录下有 hw/、hw_A/、onboard_A/
# - 优先使用 hw_A 中的方案A板级文件
# - 缺失时回退到 hw 中查找
# - 已存在同名文件会覆盖

ROOT_DIR="${1:-.}"
ONBOARD_DIR="${ROOT_DIR}/onboard_A"
HW_DIR="${ROOT_DIR}/hw"
HWA_DIR="${ROOT_DIR}/hw_A"

echo "[INFO] ROOT_DIR    = ${ROOT_DIR}"
echo "[INFO] ONBOARD_DIR = ${ONBOARD_DIR}"
echo "[INFO] HW_DIR      = ${HW_DIR}"
echo "[INFO] HWA_DIR     = ${HWA_DIR}"

if [[ ! -d "${ONBOARD_DIR}" ]]; then
	echo "[ERROR] onboard_A 目录不存在: ${ONBOARD_DIR}"
	exit 1
fi

mkdir -p \
	"${ONBOARD_DIR}/archive" \
	"${ONBOARD_DIR}/constr" \
	"${ONBOARD_DIR}/data/expected" \
	"${ONBOARD_DIR}/data/quant" \
	"${ONBOARD_DIR}/data/samples" \
	"${ONBOARD_DIR}/data/weights" \
	"${ONBOARD_DIR}/docs" \
	"${ONBOARD_DIR}/rtl/core" \
	"${ONBOARD_DIR}/rtl/ctrl" \
	"${ONBOARD_DIR}/rtl/mem" \
	"${ONBOARD_DIR}/rtl/pkg" \
	"${ONBOARD_DIR}/rtl/top" \
	"${ONBOARD_DIR}/rtl/uart" \
	"${ONBOARD_DIR}/scripts" \
	"${ONBOARD_DIR}/tb" \
	"${ONBOARD_DIR}/vivado"

copy_first_found() {
	local dst="$1"
	shift
	for src in "$@"; do
		if [[ -f "${src}" ]]; then
			cp -f "${src}" "${dst}"
			echo "[COPY] ${src} -> ${dst}"
			return 0
		fi
	done
	echo "[WARN] 未找到候选文件: $*"
	return 1
}

copy_if_exists() {
	local src="$1"
	local dst="$2"
	if [[ -f "${src}" ]]; then
		cp -f "${src}" "${dst}"
		echo "[COPY] ${src} -> ${dst}"
	else
		echo "[WARN] 文件不存在，跳过: ${src}"
	fi
}

copy_glob_if_exists() {
	local pattern="$1"
	local dst="$2"
	shopt -s nullglob
	local files=(${pattern})
	shopt -u nullglob
	if ((${#files[@]} > 0)); then
		cp -f "${files[@]}" "${dst}"
		for f in "${files[@]}"; do
			echo "[COPY] ${f} -> ${dst}"
		done
	else
		echo "[WARN] 没有匹配到文件: ${pattern}"
	fi
}

echo
echo "== 1) 复制 package/top =="
copy_first_found "${ONBOARD_DIR}/rtl/pkg/" \
	"${HWA_DIR}/rtl/package.sv" \
	"${HW_DIR}/rtl/package.sv" \
	"${HW_DIR}/rtl_shared_buffer_ib/package.sv"

copy_first_found "${ONBOARD_DIR}/rtl/top/" \
	"${HWA_DIR}/rtl_ip/mnist_cim_demo_a_top.sv"

echo
echo "== 2) 复制 core 核心模块 =="
CORE_FILES=(
	"argmax_int8.sv"
	"fc1_multi_block_shared_input.sv"
	"fc1_ob_engine_shared_input.sv"
	"cim_tile.sv"
	"psum_accum.sv"
	"fc1_to_fc2_top_with_file.sv"
	"fc1_relu_requantize_with_file.sv"
	"fc2_core_with_file.sv"
)

for f in "${CORE_FILES[@]}"; do
	copy_first_found "${ONBOARD_DIR}/rtl/core/" \
		"${HWA_DIR}/rtl_shared_buffer_ib/${f}" \
		"${HWA_DIR}/rtl/${f}" \
		"${HW_DIR}/rtl_shared_buffer_ib/${f}" \
		"${HW_DIR}/rtl/${f}"
done

# 板级封装/推理核心，放 core
copy_first_found "${ONBOARD_DIR}/rtl/core/" \
	"${HWA_DIR}/rtl_ip/mnist_cim_accel_ip.sv"

copy_first_found "${ONBOARD_DIR}/rtl/core/" \
	"${HWA_DIR}/rtl_ip/mnist_inference_core_board.sv"

echo
echo "== 3) 复制 mem 模块 =="
MEM_FILES=(
	"fc1_weight_bank.sv"
	"fc1_bias_bank.sv"
	"fc2_weight_bank.sv"
	"fc2_bias_bank.sv"
	"quantize_param_bank.sv"
	"mnist_sample_rom.sv"
	"fc1_multi_block_shared_sample_rom.sv"
)

for f in "${MEM_FILES[@]}"; do
	copy_first_found "${ONBOARD_DIR}/rtl/mem/" \
		"${HWA_DIR}/rtl_ip/${f}" \
		"${HWA_DIR}/rtl_shared_buffer_ib/${f}" \
		"${HWA_DIR}/rtl/${f}" \
		"${HW_DIR}/rtl_shared_buffer_ib/${f}" \
		"${HW_DIR}/rtl/${f}"
done

echo
echo "== 4) 复制 uart 模块 =="
UART_FILES=(
	"uart_tx.sv"
	"uart_pred_sender.sv"
)

for f in "${UART_FILES[@]}"; do
	copy_first_found "${ONBOARD_DIR}/rtl/uart/" \
		"${HWA_DIR}/rtl_ip/${f}" \
		"${HWA_DIR}/rtl/${f}" \
		"${HW_DIR}/rtl/${f}"
done

echo
echo "== 5) 复制板级数据 =="
# 优先使用 hw_A/data 中已经整理好的板级数据
copy_first_found "${ONBOARD_DIR}/data/weights/" \
	"${HWA_DIR}/data/fc1_weight_int8.hex" \
	"${HWA_DIR}/route_b_output_2/fc1_weight_int8.hex" \
	"${HW_DIR}/route_b_output_2/fc1_weight_int8.hex"

copy_first_found "${ONBOARD_DIR}/data/weights/" \
	"${HWA_DIR}/data/fc1_bias_int32.hex" \
	"${HWA_DIR}/route_b_output_2/fc1_bias_int32.hex" \
	"${HW_DIR}/route_b_output_2/fc1_bias_int32.hex"

copy_first_found "${ONBOARD_DIR}/data/weights/" \
	"${HWA_DIR}/data/fc2_weight_int8.hex" \
	"${HWA_DIR}/route_b_output_2/fc2_weight_int8.hex" \
	"${HW_DIR}/route_b_output_2/fc2_weight_int8.hex"

copy_first_found "${ONBOARD_DIR}/data/weights/" \
	"${HWA_DIR}/data/fc2_bias_int32.hex" \
	"${HWA_DIR}/route_b_output_2/fc2_bias_int32.hex" \
	"${HW_DIR}/route_b_output_2/fc2_bias_int32.hex"

copy_first_found "${ONBOARD_DIR}/data/quant/" \
	"${HWA_DIR}/data/quant_params.hex" \
	"${HWA_DIR}/route_b_output_2/quant_params.hex" \
	"${HW_DIR}/route_b_output_2/quant_params.hex"

copy_first_found "${ONBOARD_DIR}/data/quant/" \
	"${HWA_DIR}/route_b_output_2/quant_config.json" \
	"${HW_DIR}/route_b_output_2/quant_config.json"

# 板级多样本ROM源文件
copy_first_found "${ONBOARD_DIR}/data/samples/" \
	"${HWA_DIR}/data/mnist_samples_route_b_output_2.hex"

# 同时复制若干单样本hex，便于调试/对拍
copy_glob_if_exists "${HWA_DIR}/route_b_output_2/input_*.hex" "${ONBOARD_DIR}/data/samples/"
if ! ls "${ONBOARD_DIR}/data/samples"/input_*.hex >/dev/null 2>&1; then
	copy_glob_if_exists "${HW_DIR}/route_b_output_2/input_*.hex" "${ONBOARD_DIR}/data/samples/"
fi

# 期望输出
copy_glob_if_exists "${HWA_DIR}/route_b_output_2/pred_*.txt" "${ONBOARD_DIR}/data/expected/"
if ! ls "${ONBOARD_DIR}/data/expected"/pred_*.txt >/dev/null 2>&1; then
	copy_glob_if_exists "${HW_DIR}/route_b_output_2/pred_*.txt" "${ONBOARD_DIR}/data/expected/"
fi

copy_first_found "${ONBOARD_DIR}/data/expected/" \
	"${HWA_DIR}/route_b_output_2/preds.txt" \
	"${HW_DIR}/route_b_output_2/preds.txt"

copy_first_found "${ONBOARD_DIR}/data/expected/" \
	"${HWA_DIR}/route_b_output_2/labels.txt" \
	"${HW_DIR}/route_b_output_2/labels.txt"

echo
echo "== 6) 复制 testbench 和脚本 =="
copy_first_found "${ONBOARD_DIR}/tb/" \
	"${HWA_DIR}/tb/tb_mnist_cim_demo_a_top.sv"

copy_first_found "${ONBOARD_DIR}/scripts/" \
	"${HWA_DIR}/scripts/run_tb_mnist_cim_demo_a_top.sh"

copy_first_found "${ONBOARD_DIR}/scripts/" \
	"${HWA_DIR}/scripts/gen_board_sample_rom.py"

echo
echo "== 7) 复制约束和文档（若存在） =="
copy_glob_if_exists "${HW_DIR}/constraints/*.xdc" "${ONBOARD_DIR}/constr/"
copy_glob_if_exists "${HWA_DIR}/constraints/*.xdc" "${ONBOARD_DIR}/constr/"

copy_if_exists "${HWA_DIR}/README.md" "${ONBOARD_DIR}/archive/hw_A_README.md"
copy_if_exists "${HW_DIR}/README.md" "${ONBOARD_DIR}/archive/hw_README.md"
copy_if_exists "${ROOT_DIR}/tree.txt" "${ONBOARD_DIR}/archive/original_tree.txt"

echo
echo "== 8) 生成 onboard_A README =="
cat >"${ONBOARD_DIR}/README.md" <<'EOF'
# onboard_A

方案A上板目录。

## 目标
- 使用共享输入版 FC1 + 板级 sample ROM
- 在 FPGA 板上完成一次 MNIST 推理
- UART 输出预测类别
- 支持样本切换与按键启动

## 目录说明
- rtl/top: 板级顶层
- rtl/core: 推理主链与板级封装
- rtl/mem: 权重/偏置/样本/量化参数相关模块
- rtl/uart: 串口发送模块
- rtl/ctrl: 板级控制逻辑（后续补 debounce/onepulse/done latch）
- data: 上板所需 hex/json/txt
- constr: 约束文件
- tb: 关键板级 testbench
- scripts: 仿真/生成辅助脚本

## 建议下一步
1. 检查 mnist_cim_demo_a_top.sv 的例化依赖是否全部齐全
2. 检查所有 $readmemh 路径是否改为 onboard_A/data 下的相对路径
3. 补充 ctrl 下的 debounce / onepulse / done_latch
4. 添加板卡 .xdc
5. 重新跑一次方案A testbench
EOF

echo
echo "== 9) 生成后续待办清单 =="
cat >"${ONBOARD_DIR}/docs/bringup_checklist.md" <<'EOF'
# Bring-up Checklist

- [ ] 检查顶层端口与板卡引脚是否一致
- [ ] 检查时钟频率与 UART 波特率是否匹配
- [ ] 检查 sample ROM 初始化文件路径
- [ ] 检查权重/偏置/量化参数加载路径
- [ ] 增加按键消抖
- [ ] 增加 start 单脉冲
- [ ] 增加 done 锁存
- [ ] 板上串口打印预测结果
- [ ] 至少验证 4~10 个样本
EOF

echo
echo "[DONE] onboard_A 文件架构与方案A所需文件已整理完成。"
echo "[TIP] 下一步先检查 rtl/top/mnist_cim_demo_a_top.sv 的依赖和数据路径。"
