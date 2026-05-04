# IG-TRACK: IOU GUIDED SIAMESE NETWORKS FOR VISUAL OBJECT TRACKING

Mohana Murali Dasari, Rama Krishna Sai Subrahmanyam Gorthi (2020)

## 🧩 Problem to Solve

본 논문은 시각적 객체 추적(Visual Object Tracking) 분야에서 Siamese Network 기반의 Region Proposal 방식이 가지는 한계점을 해결하고자 한다. 기존의 SiamRPN 및 SiamRPN++와 같은 추적기들은 네트워크 학습 단계와 실제 테스트 단계 사이의 불일치 문제가 존재한다. 구체적으로, 이들은 학습된 네트워크의 출력값에 대해 테스트 과정에서 추가적인 후처리 연산(Post-processing)을 수행하여 최종 Bounding Box를 예측하는데, 이러한 방식은 예측된 Bounding Box의 정밀도(Precision)를 저하시키는 원인이 된다.

또한, ATOM과 같은 일부 최신 추적기는 Ground Truth 좌표에 가우시안 노이즈를 추가하여 후보군을 생성하는 방식을 사용하는데, 저자들은 이러한 무작위적인 후보 생성 방식이 추적 방법론의 일관성과 맞지 않는다고 주장한다. 따라서 본 논문의 목표는 학습 과정에서 Intersection Over Union (IOU)을 직접적으로 가이드하여, 이미지 도메인에서 정밀한 Bounding Box를 직접 예측할 수 있는 End-to-End 학습 가능 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Siamese Region Proposal 네트워크 상단에 **IOU 모듈**을 추가하여, 학습 단계에서 예측된 Bounding Box와 Ground Truth 간의 IOU를 최대화하도록 유도하는 것이다.

주요 기여 사항은 다음과 같다.
1. 네트워크 학습 시 IOU 모듈을 통합하여 추적 성능을 가이드하는 구조를 제안하였다.
2. 예측된 IOU를 최대화하기 위한 새로운 손실 함수(Loss Function)를 도입함으로써 Bounding Box 예측의 정밀도를 향상시켰다.
3. 기존 SiamRPN++에서 사용되던 Scale 및 Ratio Penalty와 같은 불필요한 후처리 연산이 오히려 오버피팅을 유발함을 분석하고 이를 제거하여 효율성을 높였다.

## 📎 Related Works

시각적 객체 추적은 초기 프레임에서 주어진 객체의 위치를 이후 프레임에서 계속해서 찾아내는 작업이다. 최근에는 딥러닝 기반의 Siamese Network가 효율성과 정확성 덕분에 널리 사용되고 있다.

- **Siamese Networks**: 템플릿 이미지와 검색 이미지의 특징 맵(Feature Map) 간의 상관관계(Correlation)를 계산하여 객체 위치를 찾는다.
- **SiamRPN 및 SiamRPN++**: 객체 탐지(Object Detection)의 Region Proposal Network(RPN) 개념을 도입하여 Bounding Box를 회귀(Regression)한다. 하지만 이들은 학습된 네트워크의 출력에 대해 역변환 및 스케일/비율 페널티를 적용하는 등의 추가 연산이 필요하며, 이는 진정한 의미의 End-to-End 학습이라고 보기 어렵다.
- **ATOM**: IOU 변조(Modulation) 및 예측을 통해 정밀도를 높였으나, 후보 Bounding Box를 생성할 때 가우시안 노이즈를 더하는 무작위적 방식을 사용한다는 한계가 있다.

본 연구는 이러한 기존 방식들의 불일치성과 불필요한 연산을 제거하고, IOU 기반의 직접적인 가이드를 통해 정밀도를 높이는 방향으로 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인
제안된 **IG-Track**은 SiamRPN++를 베이스라인으로 하며, 다음과 같은 절차로 동작한다.
1. **입력**: 첫 프레임의 타겟 이미지 $z$ ($127 \times 127$)와 이후 프레임의 검색 이미지 $x$ ($255 \times 255$)를 입력으로 받는다.
2. **특징 추출**: ResNet50 네트워크를 통해 두 이미지의 특징 맵을 추출한다.
3. **응답 맵 생성**: 추출된 특징 맵은 Classification 블록과 Regression 블록으로 전달되어 특징 도메인에서의 응답 맵을 생성한다.
4. **IOU 모듈 (학습 시 핵심)**: 특징 도메인의 응답 맵을 이미지 도메인으로 변환하고, 앵커 박스를 이용하여 상위 $K$개의 '유망한 Bounding Box(Probable Bounding Boxes)'를 선정한다. 동시에 학습 가능한 파라미터를 가진 Motion Estimation 블록을 통해 이전 프레임으로부터 '추정된 Bounding Box(Estimated Bounding Box)'를 생성한다.
5. **예측 및 보정**: 추정된 박스와 유망한 박스들 간의 IOU를 계산하여 IOU 응답 맵을 만들고, 이 중 최대값에 해당하는 박스를 최종 '예측된 Bounding Box(Predicted Bounding Box)'로 결정한다. 이때 크기의 급격한 변화를 막기 위해 선형 보간법(Linear Interpolation)을 사용한다.

### 손실 함수 및 학습 절차
네트워크는 다음과 같은 통합 손실 함수를 최소화하는 방향으로 학습된다.

$$ \text{Loss} = L_{cls} + L_{reg} + L_{iou} $$

여기서 $L_{cls}$는 분류 손실, $L_{reg}$는 Bounding Box 회귀 손실이며, 새롭게 추가된 $L_{iou}$는 다음과 같이 정의된다.

$$ L_{iou} = 1 - \text{IOU}(\text{predbb}, \text{gtbb}) $$

이 수식은 예측된 Bounding Box($\text{predbb}$)와 실제 Ground Truth Bounding Box($\text{gtbb}$) 사이의 IOU를 최대화함으로써 정밀도를 높이는 역할을 한다.

## 📊 Results

### 실험 설정
- **특징 추출기**: Pre-trained ResNet50 사용.
- **학습 데이터셋**: ImageNet VID, ImageNet DET, COCO.
- **평가 데이터셋**: VOT2018, OTB2015, GOT-10k.
- **학습 환경**: NVIDIA GeForce GTX Ti-1080 4장, 40 에포크(epoch) 동안 학습, 학습률은 $0.005$에서 $0.00005$까지 지수적으로 감소.

### 주요 결과
공정한 비교를 위해 베이스라인인 SiamRPN++를 동일한 데이터셋(YouTube-BB 제외)으로 재학습하여 비교하였다.

**1. VOT2018 데이터셋 결과**
- **EAO (Expected Average Overlap)**: $0.290 \rightarrow 0.327$ (약 13% 향상)
- **Robustness**: $0.347 \rightarrow 0.309$ (약 11% 향상, 수치가 낮을수록 좋음)
- **Accuracy**: $0.571 \rightarrow 0.565$ (약 1% 감소하였으나 거의 유사한 수준 유지)

**2. GOT-10k 데이터셋 결과**
- **AO (Average Overlap)**: $0.453 \rightarrow 0.459$ (1% 향상)
- **$SR_{0.5}$ (Success Rate at 0.5)**: $0.546 \rightarrow 0.558$ (2% 향상)
- **$SR_{0.75}$ (Success Rate at 0.75)**: $0.195 \rightarrow 0.220$ (12% 향상)

특히 $SR_{0.75}$ 지표의 큰 상승은 IOU 모듈과 IOU 손실 함수가 Bounding Box의 예측 정밀도를 실질적으로 개선했음을 입증한다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 기존 Siamese RPN 계열의 추적기들이 가진 '학습과 테스트 간의 괴리'라는 본질적인 문제를 정확히 짚어냈다. 단순히 네트워크를 깊게 쌓는 것이 아니라, 학습 목표 자체에 IOU라는 기하학적 지표를 직접 포함시킴으로써 후처리 과정 없이도 이미지 도메인에서 높은 정밀도를 얻을 수 있음을 보여주었다. 특히 고정밀도 측정 지표인 $SR_{0.75}$에서의 성능 향상은 매우 유의미하다.

### 한계 및 미해결 질문
1. **모션 모델의 단순함**: 현재 Motion Estimation 블록이 학습 가능하다고 언급되어 있으나, 구체적인 구조에 대한 설명이 부족하며 단순히 선형 보간법에 의존하고 있다. 저자들 또한 결론에서 이를 RNN 등을 이용해 개선할 필요가 있음을 인정하고 있다.
2. **데이터셋 의존성**: 비교 실험에서 YouTube-BB 데이터셋을 제외하고 공정성을 기했으나, 실제 최신 SOTA 모델(ATOM 등)들이 사용하는 거대 데이터셋(LaSOT, TrackingNet)과의 비교가 이루어지지 않아 절대적인 성능 수준을 판단하기 어렵다.
3. **추론 속도**: IOU 모듈이 학습 단계에서 추가되었으나, 이것이 추론(Inference) 단계의 속도에 어떤 영향을 미치는지에 대한 정량적 분석이 제시되지 않았다.

## 📌 TL;DR

본 논문은 SiamRPN++를 기반으로, 학습 과정에서 예측 박스와 정답 박스 간의 **IOU를 직접 최대화하는 IOU 모듈과 손실 함수를 도입한 IG-Track**을 제안한다. 이를 통해 불필요한 후처리 연산을 제거하고 End-to-End 학습을 가능하게 하여, VOT2018 및 GOT-10k 벤치마크에서 특히 고정밀도 영역(EAO 및 $SR_{0.75}$)의 성능을 유의미하게 향상시켰다. 이 연구는 향후 Siamese 추적기의 학습 목표를 정교화하는 방향에 중요한 기초 자료가 될 수 있다.