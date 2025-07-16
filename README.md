# BlurFaceDetection

## 📁 프로젝트 구조
```text
BlurFaceDetection/
├── datasets/
│   ├── train/
│   │   ├── Adult/
│   │   └── Child/
│   ├── val/
│   │   ├── Adult/
│   │   └── Child/
│   └── test/
│       ├── Adult/
│       └── Child/
│
├── logs/                  # 학습 중 생성된 로그 및 기록
│
├── models/
│   └── weights/           # 훈련된 모델 가중치 저장 위치
│
├── config.py              # 학습 설정 (하이퍼파라미터 등)
├── train.py               # 모델 학습 스크립트
├── dataset.py             # 데이터 로딩 함수 정의
├── data_preparation.py    # 데이터 전처리 및 분할 스크립트
├── notebooks/             # 실험 노트북 저장소
└── README.md              # 프로젝트 설명 파일
```


## 📦 데이터 출처 (Data Source)
본 프로젝트는 다음 공개 데이터셋을 기반으로 합니다:  
- **데이터명**: 안면 인식 에이징(aging) 이미지 데이터  
- **출처**: [AI Hub - 인공지능허브](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71415)
