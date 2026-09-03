# MedAgentSim

MedAgentSim là hệ thống mô phỏng tác vụ lâm sàng đa tác tử.  
Trong phiên bản này, chúng tôi đã chỉnh sửa và tích hợp mã nguồn gốc với các dòng comment có tiền tố `MedPIGuard` để bổ sung các chức năng liên quan đến MedPIGuard, bao gồm kịch bản tấn công, chèn prompt và cơ chế phòng vệ.

Hệ thống vẫn đảm bảo chạy đúng kịch bản gốc nếu không truyền các tham số liên quan đến tấn công hoặc phòng thủ.

## Yêu cầu

- Python 3.10+
- Các thư viện phụ thuộc của dự án
- Môi trường có quyền đọc/ghi thư mục output

Cài đặt phụ thuộc theo hướng dẫn của dự án gốc nếu có.

## Chạy cơ bản

Chạy MedAgentSim với cấu hình mặc định:

```bash
python medsim/main.py --agent_dataset MIMICIV --output_dir results4test --num_scenarios 5
```

## Chạy với kịch bản tấn công

### 1. Chạy với `attack_scenario` và `injection_strategy`

```bash
python medsim/main.py --agent_dataset MIMICIV --output_dir results4test --num_scenarios 5 --attack_scenario T1 --injection_strategy I2
```

### 2. Chạy với biến thể của kịch bản tấn công

```bash
python medsim/main.py --agent_dataset MIMICIV --output_dir results4test --num_scenarios 5 --attack_scenario T1 --injection_strategy I1 --attack_scenario_variant
```

### 3. Chạy với phòng vệ `protectai`

```bash
python medsim/main.py --agent_dataset MIMICIV --output_dir results4test --num_scenarios 5 --attack_scenario T1 --injection_strategy I1 --attack_scenario_variant --defense_mode protectai
```

## Các tuỳ chọn MedPI

Các cập nhật MedPI được tích hợp trong `medsim/main.py` và có thể được dùng thông qua các tham số sau:

- `--attack_scenario`: chọn kịch bản tấn công
- `--injection_strategy`: chọn chiến lược chèn prompt
- `--attack_scenario_variant`: bật biến thể của kịch bản tấn công
- `--defense_mode`: bật chế độ phòng vệ, ví dụ `protectai`

## Ví dụ quy trình chạy

1. Chọn dataset:
   - `MIMICIV`
2. Chọn số lượng kịch bản:
   - ví dụ `--num_scenarios 5`
3. Nếu cần đánh giá tấn công:
   - thêm `--attack_scenario` và `--injection_strategy`
4. Nếu cần kiểm tra phòng vệ:
   - thêm `--defense_mode protectai`

## Output

Kết quả sẽ được lưu trong thư mục chỉ định bởi `--output_dir`, ví dụ:

```bash
results4test
```

## Ghi chú

- Đảm bảo thư mục output tồn tại hoặc có quyền tạo mới.
- Có thể thay đổi `--num_scenarios` để tăng/giảm số lượng ca chạy thử.
- Nếu cần tái lập thí nghiệm, nên cố định các tham số đầu vào và thư mục output.

## Ví dụ đầy đủ

Chạy thử 5 kịch bản có tấn công và phòng vệ:

```bash
python medsim/main.py \
  --agent_dataset MIMICIV \
  --output_dir results4test \
  --num_scenarios 5 \
  --attack_scenario T1 \
  --injection_strategy I1 \
  --attack_scenario_variant \
  --defense_mode protectai
```