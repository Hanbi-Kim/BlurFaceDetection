# config.py

# 이미지 입력 크기
IMG_SIZE = (224, 224)  # (W, H)

# 블러 커널 크기
KERNEL_SIZE = (15, 15)

# 학습 설정
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 5

# 장치 설정
DEVICE = 'cuda'

# 클래스 이름 및 라벨 매핑
CLASS_NAMES = ['Child', 'Adult']
label_map = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
idx_to_class = {idx: cls for cls, idx in label_map.items()}
