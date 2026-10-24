# Traffic Sign Classification with CNN

German Traffic Sign Recognition Benchmark (GTSRB)의 **43개 교통표지판 클래스**를 분류하는 CNN 학습 프로젝트입니다. 이미지 전처리부터 CNN 구조 비교, 평가 과정에서 발견한 클래스 매핑 문제 해결까지 다룹니다.

> **프로젝트 성격:** 딥러닝 학습/실험 프로젝트. 실도로 환경의 안전성이나 배포 가능성을 검증한 시스템은 아닙니다.

## 1. 프로젝트 개요

- **목표:** 교통표지판 이미지 43개 클래스를 CNN으로 분류
- **데이터:** [German Traffic Sign Recognition Benchmark (GTSRB) — Kaggle 배포본](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **기술:** Python, TensorFlow/Keras, pandas, NumPy, Matplotlib
- **입력 이미지:** 32 × 32 RGB, 픽셀을 [0, 1]로 정규화
- **분리:** Train 디렉터리 39,209개 중 학습 31,368개 / 검증 7,841개(80:20); 제공된 테스트 이미지 12,630개

## 2. 원본 실험 결과

| 모델 | 구조(요약) | 학습 정확도 | 검증 정확도 | 테스트 정확도 |
| --- | --- | ---: | ---: | ---: |
| Baseline CNN | Conv2D(5) → MaxPool → Flatten → Dense(43) | 원본 최종 값 미기록 | **84.44%** | 클래스 순서 오류로 인한 잘못된 측정치만 기록 |
| Improved CNN | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(64) → Dropout(0.2) → Dense(43) | **96.13%** | **98.58%** | **91.58%** |

- 개선 모델 학습: 10 epochs, Adam(learning rate=0.001), categorical cross-entropy.
- 위 결과는 **원본 Colab 실습 노트북에 남아 있는 로그**를 옮긴 것이며, 이 저장소에서 재실행하여 검증한 결과는 아닙니다.
- Baseline의 테스트 정확도와 개선 모델 테스트 정확도를 **동일한 정상 평가 조건으로 비교한 실험은 없음**에 유의하세요.

### 평가 과정의 핵심 발견: 클래스 인덱스 불일치

원본 실습에서 개선 모델의 테스트 정확도가 처음에는 **5.69%**로 출력됐습니다. 학습 데이터의 클래스는 폴더명 **문자열 정렬** 기준(`0, 1, 10, 11, …, 2, …`)인데, 테스트 제너레이터는 `0, 1, 2, 3, …` 순서로 만들어 **라벨 인덱스가 일치하지 않았기 때문**입니다.

모델을 새로 학습한 것이 아니라, 테스트 데이터 생성 시 **학습 데이터와 동일한 클래스 순서**를 지정하여 재평가했고 **91.58%**를 얻었습니다. 따라서 `5.69% → 91.58%`는 모델 개선 효과가 아닌 **평가 파이프라인 오류 수정 결과**입니다.

## 3. 저장소 구성

```text
.
├── notebooks/
│   └── traffic_sign_classification.ipynb  # 정리된 재현용 분석/학습 코드
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 4. 실행 방법

1. Kaggle 데이터셋을 직접 다운로드해 압축을 해제합니다. 원본 이미지는 이 저장소에 포함하지 않습니다.
2. 프로젝트 환경에서 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

3. 압축 해제한 GTSRB 데이터 루트 경로를 환경 변수 `GTSRB_DATA_DIR`에 지정하거나 노트북 상단의 `DATA_ROOT`를 수정합니다. 데이터 루트에는 `Train/` 폴더와 `Test.csv` 파일이 있어야 합니다.
4. `notebooks/traffic_sign_classification.ipynb`를 실행합니다.

> 정리된 노트북은 Colab Drive 경로에 의존하지 않도록 작성했지만 **재실행 여부는 아직 검증하지 않았습니다.** TensorFlow / Keras 버전 및 파일 구조에 따라 조정이 필요할 수 있습니다.

## 5. 한계와 후속 작업

- 클래스별 precision/recall/F1, confusion matrix, 오분류 이미지 분석이 아직 수행되지 않았습니다.
- 검증 정확도보다 테스트 정확도가 낮아 데이터 차이 및 일반화 성능에 대한 추가 분석이 필요합니다.
- 클래스별 표본 수 불균형, 촬영 조건 변화, 실제 도로 영상에서의 성능은 확인되지 않았습니다.
- 고정 학습/검증 분할에 대한 실험 반복, 모델 저장 및 배포 검증은 수행하지 않았습니다.

## 6. 출처 및 라이선스

- **데이터:** [GTSRB Kaggle 데이터셋](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). **원본 데이터의 라이선스 및 재배포 허용 범위는 별도로 확인해야 합니다.**
- **코드:** 저장소 내 `LICENSE` (MIT). MIT는 이 저장소에서 작성한 코드에만 적용되며, 외부 데이터나 제3자 코드에 대한 권리를 부여하지 않습니다.
- **확인 필요:** 원본 실습 코드의 제공자·외부 코드 사용 여부 및 사용 조건.

---

본 프로젝트는 학습용 실험이며 상용 교통안전 판단에 사용하도록 검증되지 않았습니다.
