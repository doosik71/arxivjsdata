# ICPR 2024 Competition on Domain Adaptation and GEneralization for Character Classification (DAGECC)

Sofia Marino, Jennifer Vandoni, Emanuel Aldea, Ichraq Lemghari, Sylvie Le Hégarat-Mascle, and Frédéric Jurie (2024)

## 🧩 Problem to Solve

본 연구 및 경쟁회의 핵심 해결 과제는 산업 현장에서 발생하는 시리얼 번호 인식의 자동화를 위한 강건한 문자 분류 모델을 개발하는 것이다. 산업 환경에서는 부품의 재질, 조명 조건, 촬영 각도, 각인 방식(레이저, 연필, 마이크로 퍼커션 등)의 다양성으로 인해 데이터 분포가 변화하는 Domain Shift 현상이 빈번하게 발생한다. 이러한 환경 변화는 일반적인 머신러닝 모델의 성능을 급격히 저하시키며, 기존의 수동 기록 방식은 노동 집약적이고 오류 발생 가능성이 높다는 문제가 있다.

따라서 본 논문의 목표는 Domain Adaptation(도메인 적응)과 Domain Generalization(도메인 일반화) 기술을 활용하여, 학습 시 보지 못한 새로운 도메인에서도 높은 인식 성능을 유지하는 경량화된 산업용 문자 인식 시스템의 가능성을 탐색하고 이를 위한 고품질의 실세계 데이터셋인 Safran-MNIST를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여는 산업용 시리얼 번호 인식이라는 특수한 환경을 반영한 Safran-MNIST 데이터셋 스위트를 구축하고, 이를 기반으로 Domain Generalization 및 Unsupervised Domain Adaptation 성능을 겨루는 DAGECC 경진대회를 개최하여 최신 기술 동향을 분석한 것에 있다. 

핵심적인 직관은 실제 산업 데이터의 부족함을 극복하기 위해 공개 데이터셋의 활용과 더불어, 타겟 도메인의 특성(금속 표면의 질감, 각인 효과 등)을 모사한 합성 데이터(Synthetic Data) 생성 전략이 모델의 일반화 성능을 높이는 결정적인 요소가 된다는 점을 확인한 것이다.

## 📎 Related Works

논문에서는 Domain Adaptation과 Domain Generalization의 개념적 차이를 설명하며 관련 연구의 배경을 제시한다.

- **Domain Adaptation**: 소스 도메인의 지식을 활용하여 타겟 도메인의 성능을 향상시키는 전이 학습(Transfer Learning) 기법이다. 주로 소스와 타겟 간의 데이터 분포 차이를 줄이는 데 집중하며, 자율 주행이나 산업 품질 관리 등에서 중요하게 다뤄진다.
- **Domain Generalization**: 학습 과정에서 타겟 도메인의 데이터에 접근하지 않고도, 임의의 보지 못한 타겟 도메인에서 잘 작동하는 보편적인 표현(Universal Representation)을 학습하는 기법이다. 이는 타겟 도메인이 알려지지 않았거나 지속적으로 변하는 환경에서 매우 유용하다.

기존의 문자 인식 연구들은 MNIST, SVHN, EMNIST와 같은 정제된 데이터셋에 의존해 왔으나, 본 연구는 실제 항공기 엔진 부품에서 추출한 실세계 이미지 데이터를 사용함으로써 기존 벤치마크와 실제 산업 현장 간의 간극을 메우고자 한다.

## 🛠️ Methodology

### 1. 데이터셋 구성 (Safran-MNIST)
본 경진대회는 두 가지 데이터셋을 기반으로 진행된다.
- **Safran-MNIST-D**: RGB 이미지($128 \times 192$ 픽셀), 0-9까지의 숫자 10개 클래스로 구성된다.
- **Safran-MNIST-DLS**: 그레이스케일 이미지(가변 크기, $18 \times 30$ ~ $86 \times 79$ 픽셀), 숫자 10종, 알파벳 20종, 심볼 2종을 포함한 총 32개 클래스로 구성된다.

### 2. 태스크 정의
- **Task 1 (Domain Generalization)**: 타겟 도메인인 Safran-MNIST-D의 데이터에 전혀 접근하지 않고 모델을 학습시켜야 한다. 참가자들은 MNIST, SVHN, HASYv2 등 공개 데이터나 자체 생성한 합성 데이터를 자유롭게 사용할 수 있다.
- **Task 2 (Unsupervised Domain Adaptation)**: 타겟 도메인인 Safran-MNIST-DLS의 레이블이 없는(unlabeled) 데이터에 접근할 수 있다. 적절한 소스 데이터를 찾아 타겟 도메인으로 적응시키는 것이 목표이다.

### 3. 평가 지표
데이터셋의 클래스 불균형 문제를 해결하기 위해 Macro-averaged $F_1$-score를 평가지표로 사용한다.
전체 클래스 $K$에 대한 Macro $F_1$-score는 다음과 같이 정의된다.
$$F_{Macro1} = \frac{\sum_{k=1}^{K} F_1^k}{K}$$

각 클래스 $k$에 대한 $F_1$-score는 다음과 같다.
$$F_1^k = \frac{2TP_k}{2TP_k + FP_k + FN_k}$$
여기서 $TP_k$는 True Positives, $FP_k$는 False Positives, $FN_k$는 False Negatives를 의미한다.

## 📊 Results

### 1. Task 1: Domain Generalization 결과
- **1위 (Team Deng)**: ResNet50 아키텍처를 사용하였으며, ImageNet으로 사전 학습된 가중치를 초기값으로 사용했다. 특히 배경에 랜덤 색상, 노이즈, 엠보싱 효과, 가우시안 블러를 적용하고 다양한 폰트를 조합한 맞춤형 합성 데이터셋을 구축하여 파인튜닝하였다. 클래스 불균형 해결을 위해 `WeightedRandomSampler`를 적용하여 Macro $F_1$-score 0.82를 기록했다.
- **2위 (Fraunhofer IIS DEAL)**: GoogLeNet을 사용했으며, Stable Diffusion 모델을 통해 금속 표면에 각인된 형태의 합성 이미지 500장을 클래스별로 생성하여 학습에 활용했다. (Score: 0.74)

### 2. Task 2: Domain Adaptation 결과
- **1위 (Team Deng)**: Task 1과 유사하게 ResNet50과 맞춤형 합성 데이터를 사용하였다. 특이사항은 제공된 타겟 도메인의 레이블 없는 데이터를 사용하지 않고도 높은 성능을 냈다는 점이다. (Score: 0.65)
- **2위 (Deep Unsupervised Trouble)**: EMNIST와 Consola 폰트로 생성한 데이터를 사용했으며, Otsu 임계값(Otsu thresholding)을 변형하여 데이터 증강을 수행하고 ResNet18로 학습하였다. (Score: 0.60)
- **3위 (Raul)**: Blender를 사용하여 금속판에 각인된 문자를 3D 렌더링하여 약 24만 장의 대규모 합성 데이터를 생성하고 residual connection 기반의 CNN을 사용했다. (Score: 0.52)

## 🧠 Insights & Discussion

### 강점 및 주요 발견
본 경진대회의 결과는 산업용 문자 인식에서 **합성 데이터 생성(Synthetic Data Generation)**의 강력한 효율성을 입증하였다. 상위권 팀들은 단순히 기존 공개 데이터셋을 사용하는 것에 그치지 않고, 전통적인 이미지 처리 기법(엠보싱, 블러링)이나 최신 생성 AI(Stable Diffusion), 3D 렌더링(Blender)을 통해 타겟 도메인의 물리적 특성을 정교하게 모사함으로써 성능을 극대화하였다.

### 한계 및 비판적 해석
흥미로운 점은 Task 2(Unsupervised Domain Adaptation)에서 상위 3개 팀 모두 제공된 **타겟 도메인의 레이블 없는 데이터를 전혀 활용하지 않았다**는 것이다. 이는 현재의 DA 기법보다 정교한 합성 데이터를 통한 소스 도메인 확장이 훨씬 더 효과적인 'Low hanging fruit'임을 시사한다. 

논문 저자들은 이러한 결과에 대해, 단순히 데이터를 많이 생성하는 것을 넘어 타겟 도메인의 unlabeled corpus에 포함된 정보를 어떻게 더 효율적으로 추출하고 활용할 것인가가 향후 연구의 핵심 과제가 될 것이라고 논의한다. 예를 들어, Machine Learning Group LTU 팀이 시도한 pseudo-labeling과 majority voting 기반의 반복 학습 구조가 더 발전된다면 합성 데이터의 한계를 넘어서는 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 산업용 시리얼 번호 인식을 위한 Safran-MNIST 데이터셋을 제안하고, 이를 활용한 Domain Generalization 및 Adaptation 경진대회 결과를 보고한다. 실험 결과, ImageNet 사전 학습 모델에 더해 타겟 도메인의 물리적 특성을 모사한 **정교한 합성 데이터를 생성하여 학습시키는 전략**이 가장 높은 성능을 보였다. 이 연구는 실세계 산업 데이터의 부족 문제를 해결하기 위한 합성 데이터의 중요성을 강조하며, 향후 레이블 없는 데이터를 더 효과적으로 활용하는 DA 기법 연구의 필요성을 제시한다.