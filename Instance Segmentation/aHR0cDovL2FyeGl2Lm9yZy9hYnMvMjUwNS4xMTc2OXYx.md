# Technical Report for ICRA 2025 GOOSE 2D Semantic Segmentation Challenge: Boosting Off-Road Segmentation via Photometric Distortion and Exponential Moving Average

Wonjune Kim, Lae-Kyoung Lee and Su-Yong An (2025)

## 🧩 Problem to Solve

본 연구는 비정형 오프로드(off-road) 환경에서 자율 주행을 위한 2D 시맨틱 세그멘테이션(Semantic Segmentation) 문제를 해결하고자 한다. 오프로드 환경은 도시 환경과 달리 도로 표지선이나 연석과 같은 구조적 단서가 부족하며, 진흙, 눈, 밀집된 덤불 등 외관의 다양성이 매우 크기 때문에 픽셀 수준의 해석이 매우 어렵다. 특히 극심하고 빠르게 변화하는 날씨와 조명 조건 아래에서 주행 가능 지형, 식생, 장애물을 정확하게 구분해내는 것이 필수적이다.

논문에서 제시하는 주요 난제는 두 가지이다. 첫째는 심각한 클래스 불균형(Class Imbalance)으로, 전체 픽셀의 약 90%가 식생(vegetation), 지형(terrain), 하늘(sky)의 세 가지 클래스에 집중되어 있어 장애물(obstacle)이나 사람(human)과 같은 안전 필수 클래스의 데이터가 매우 부족하다는 점이다. 둘째는 모호하고 낮은 대비의 경계(Ambiguous, low-contrast boundaries) 문제이다. 자연물은 서로 서서히 섞이는 경향이 있어(예: 풀과 흙, 물과 진흙), 표준적인 Cross-Entropy 최적화만으로는 정교한 경계 추출이 어렵다. 따라서 본 논문의 목표는 이러한 제약 조건이 있는 GOOSE 데이터셋에서 높은 성능을 달성하는 강건한 세그멘테이션 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 새로운 네트워크 아키텍처를 설계하는 대신, 검증된 고성능 구성 요소들을 조합하고 오프로드 환경의 특성에 맞게 훈련 레시피(Training Recipe)를 최적화하는 것이다. 특히 광범위한 조명 변화를 모사하기 위한 강력한 Photometric Distortion 증강 기법과, 가중치의 Exponential Moving Average (EMA)를 결합하여 모델의 일반화 성능을 높이고 라벨 노이즈에 대한 안정성을 확보한 것이 중심적인 설계 전략이다.

## 📎 Related Works

본 논문은 최신 컴퓨터 비전 연구의 성과물들을 기반으로 한다. 백본 네트워크로는 InternImage-B를 확장하여 연산 효율을 높인 FlashInternImage-B를 사용하였으며, 이는 DCNv4(Deformable Convolution v4) 연산자를 통해 기존 DCNv3보다 약 1.8배 빠른 속도를 제공하면서도 정확도를 유지한다. 디코더로는 다중 스케일 정보를 융합하는 FPN(Feature Pyramid Network) 브랜치와 전역적 문맥을 포착하는 PSP(Pyramid Pooling Module) 브랜치가 통합된 UPerNet을 채택하였다.

기존의 일반적인 세그멘테이션 접근 방식이 기하학적 증강(Geometric Augmentation)에 의존했다면, 본 연구는 오프로드 환경의 극심한 색상 변화를 해결하기 위해 Photometric Distortion을 명시적으로 추가함으로써 기존 방식과의 차별점을 두었다.

## 🛠️ Methodology

### 전체 시스템 구조
본 시스템은 FlashInternImage-B 백본과 UPerNet 디코더로 구성된 고용량 세그멘테이션 파이프라인을 따른다. 백본에서 추출된 $\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}$ 해상도의 특징 맵(Feature maps)이 UPerNet 디코더로 전달되며, 최종적으로 9개의 클래스에 대한 로짓(Logits)을 생성한 후 bilinear upsampling을 통해 원본 이미지 크기로 복원한다.

### 학습 절차 및 최적화
학습은 AdamW 옵티마이저(초기 학습률 $6 \times 10^{-5}$)와 Poly learning-rate schedule을 사용하여 총 96k iteration 동안 진행되었다. 입력 이미지는 $[0.5, 2.0]$ 범위 내에서 무작위로 스케일링된 후 $2048 \times 2048$ 크기로 크롭 또는 패딩되었다. 손실 함수로는 픽셀 단위의 Softmax Cross-Entropy를 사용하였다.

### Photometric Distortion
오프로드 씬의 다양한 조명 조건을 극복하기 위해 학습 과정에서 밝기(Brightness), 대비(Contrast), 채도(Saturation), 색조(Hue)를 각각 0.5의 확률로 독립적으로 변형시키는 Photometric Distortion을 적용하였다. 이는 네트워크가 단순한 색상 정보보다는 형태(Shape)와 질감(Texture)에 더 의존하도록 유도하여, 다양한 조명 환경에서의 강건성을 높이는 역할을 한다.

### Exponential Moving Average (EMA)
최적화 과정을 안정화하고 라벨 노이즈의 영향을 줄이기 위해 네트워크 파라미터의 EMA를 유지하였다. EMA 가중치 $\theta_{EMA}^{(t)}$는 다음과 같은 방정식에 의해 매 iteration마다 업데이트된다.

$$\theta_{EMA}^{(t)} = \alpha \theta_{EMA}^{(t-1)} + (1-\alpha)\theta_{CURRENT}^{(t)}$$

여기서 $\alpha=0.999$이며, 검증 및 최종 평가에는 이 EMA 스냅샷 가중치가 사용된다.

## 📊 Results

### 실험 설정
- **데이터셋**: GOOSE 훈련 세트(약 8k 장)와 GOOSE-EX 훈련 세트(약 4k 장)를 통합하여 학습하였으며, 공식 검증 세트(약 1.4k 장)를 통해 평가하였다.
- **하드웨어**: NVIDIA RTX 3090 GPU 4대를 사용하였으며, GPU당 배치 사이즈는 2로 설정하고 Mixed precision 학습을 수행하였다.
- **평가 지표**: 평균 교차 합집합(mean Intersection-over-Union, mIoU)을 사용하였다.

### 정량적 결과
FlashInternImage-B baseline 모델에서 시작하여 각 기법을 추가했을 때의 mIoU 결과는 다음과 같다.

- **Baseline (FlashInternImage-B)**: $87.2\%$ mIoU
- **+ Photometric Distortion**: $87.76\%$ mIoU ($+0.48$ 증가)
- **+ Photometric Distortion + EMA**: $88.88\%$ mIoU ($+1.12$ 추가 증가)

최종적으로 검증 세트에서 $88.88\%$ mIoU를 달성하였으며, 공식 테스트 서버 제출 결과 $84.5\%$ mIoU로 퍼블릭 리더보드 2위를 기록하였다. 특히 EMA는 데이터 수가 적은 obstacle과 human 클래스의 성능 향상에 크게 기여하였고, Photometric Distortion은 sky와 other 클래스에서 효과가 두드러졌다.

### 정성적 결과
정성적 분석 결과, Photometric Distortion을 적용한 모델은 베이스라인이 인공 지면(artificial ground)과 자연 지면(natural ground)을 혼동하는 문제를 해결하고 이를 명확히 구분해냈다. 또한 EMA를 추가했을 때 대규모 균질 영역에서 발생하는 스펙클 아티팩트(Speckle artifacts)가 억제되고 클래스 경계가 더욱 날카롭게 형성되는 것을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 오프로드 환경이라는 특수성 속에서 모델의 용량을 키우는 것(High-capacity backbone)뿐만 아니라, 데이터의 분포를 인위적으로 확장하는 증강 기법과 최적화 안정화 기법(EMA)의 조합이 얼마나 중요한지를 입증하였다. 특히 Photometric Distortion이 단순한 색상 변화를 넘어 모델이 기하학적 특징에 집중하게 함으로써, 오프로드의 모호한 경계 문제를 완화시켰다는 점이 주목할 만하다.

다만, 본 논문은 새로운 알고리즘의 제안보다는 기존 기법들의 최적 조합을 찾는 데 집중하였다. 따라서 제안된 파이프라인이 다른 비정형 환경 데이터셋에서도 동일한 수준의 성능 향상을 보일지는 추가적인 검증이 필요하다. 또한 클래스 불균형 문제를 해결하기 위해 손실 함수 자체를 수정(예: Focal Loss 등)하는 대신 EMA와 증강 기법에 의존했다는 점은 향후 개선 여지로 남는다.

## 📌 TL;DR

본 논문은 ICRA 2025 GOOSE 2D 챌린지를 위해 FlashInternImage-B와 UPerNet을 기반으로, Photometric Distortion과 EMA를 적용한 고성능 오프로드 시맨틱 세그멘테이션 파이프라인을 제안한다. 이를 통해 검증 세트 기준 $88.88\%$ mIoU를 달성하였으며, 특히 소수 클래스에 대한 예측 정확도와 경계 명확성을 크게 개선하였다. 이 연구는 극한의 조명 변화와 클래스 불균형이 존재하는 야외 환경 인식 시스템 구축에 있어 실무적인 최적화 방향성을 제시한다.