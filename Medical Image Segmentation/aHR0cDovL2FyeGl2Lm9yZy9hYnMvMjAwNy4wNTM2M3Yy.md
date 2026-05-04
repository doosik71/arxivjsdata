# Semi-supervised Task-driven Data Augmentation for Medical Image Segmentation

Krishna Chaitanya, Neerav Karani, Christian F. Baumgartner, Ertunc Erdil, Anton Becker, Olivio Donati, Ender Konukoglu (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델을 학습시키기 위해 필수적인 대규모의 어노테이션 데이터(annotated data)를 확보하기 어렵다는 문제를 해결하고자 한다. 의료 영상의 픽셀 단위 분할 마스크를 생성하는 작업은 전문의의 많은 시간과 비용을 요구하며, 이는 임상 환경에서 매우 비효율적인 과정이다.

기존의 데이터 증강(Data Augmentation) 기법들은 주로 무작위(random) 변환에 의존해 왔으나, 이러한 방식은 제한된 라벨 데이터 환경에서 모델의 일반화 성능을 획기적으로 높이는 데 한계가 있다. 따라서 본 연구의 목표는 라벨링된 데이터가 매우 적은 상황에서도, 분할 작업(segmentation task) 자체에 최적화된 합성 데이터를 생성하여 학습 성능을 극대화하는 Task-driven 및 Semi-supervised 데이터 증강 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터 생성기(Generator)를 단순히 '실제처럼 보이게(realistic)' 만드는 것이 아니라, **'분할 네트워크의 성능을 높이는 방향'으로 최적화**하는 것이다.

1. **Task-driven Optimization**: 생성기의 파라미터를 업데이트할 때 분할 네트워크의 손실 함수(segmentation loss)를 직접 반영함으로써, 모델이 가장 학습하기 어려우면서도 유용한 데이터를 생성하도록 유도한다.
2. **Semi-supervised Learning**: 라벨이 없는 데이터($X_{UL}$)를 Adversarial Loss를 통해 활용하여, 소수의 라벨 데이터만으로는 포착할 수 없는 인구 통계학적 형태 및 강도 변이(shape and intensity variations)를 생성기가 학습하도록 한다.
3. **Domain-specific Generators**: 의료 영상의 특성을 반영하여, 비정형 공간 변환을 위한 Deformation Field Generator($G_V$)와 밝기 및 대비 변화를 위한 Additive Intensity Field Generator($G_I$)라는 두 가지 전문 생성기를 설계하였다.

## 📎 Related Works

기존의 데이터 증강 및 반지도 학습 방식은 다음과 같은 한계점을 가진다.

- **전통적 데이터 증강**: Affine 변환, Elastic 변환, Contrast 변환 등은 구현이 간단하고 과적합을 줄여주지만, 변환 파라미터가 고정되어 있거나 무작위로 설정되어 특정 작업(task)의 성능을 최적화하지 않는다.
- **GAN 기반 증강**: 실제와 유사한 이미지를 생성하는 데 집중하지만, 생성된 이미지가 실제 분할 성능 향상에 직접적으로 기여하는지를 고려하지 않는다.
- **MixUp**: 이미지와 라벨을 선형 보간하는 방식은 비현실적인 데이터를 생성함에도 성능 향상이 보고되었으나, 이 역시 작업 최적화 관점은 아니다.
- **반지도 학습(SSL)**: Pseudo-label을 사용하는 Self-training이나 이미지 수준의 Adversarial training 등이 존재하지만, 본 논문은 생성기 자체를 Task-driven으로 최적화하여 증강 세트를 구축한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 프레임워크는 분할 네트워크($S$), 두 종류의 생성기($G_V, G_I$), 그리고 판별기($D$)로 구성된다. 생성기는 입력 이미지와 무작위 벡터 $z$를 받아 변환 필드를 생성하고, 이를 통해 증강된 이미지-라벨 쌍($X_G, Y_G$)을 만든다.

### 2. 생성기 모델 및 변환 과정

- **Deformation Field Generator ($G_V$)**: 픽셀 단위의 밀집 변형 필드 $v$를 생성한다. 입력 이미지 $x_L$과 라벨 $y_L$에 동일한 워핑(warping) 연산 $\circ$를 적용하여 $\left(x_{G_V}, y_{G_V}\right) = (v \circ x_L, v \circ y_L)$를 생성한다.
- **Additive Intensity Field Generator ($G_I$)**: 가산적 강도 마스크 $\Delta I$를 생성한다. 이를 이미지에 더해 $x_{G_I} = x_L + \Delta I$를 만들며, 라벨 $y_{G_I}$는 원래의 $y_L$과 동일하게 유지된다.

### 3. 학습 목표 및 손실 함수

전체 최적화 목표는 다음과 같은 수식으로 정의된다.

$$\min_{w_G} \left( \min_{w_S} L_S(X_L \cup X_G, Y_L \cup Y_G) + L_{reg,G}(X_{UL}) \right)$$

여기서 $L_S$는 분할 네트워크의 손실 함수이며, $L_{reg,G}$는 생성기를 위한 정규화 항으로 다음과 같이 구성된다.

$$L_{reg,G} = \lambda_{adv} L_{adv,G} + \lambda_{LD} L_{LD,G}$$

- **Adversarial Loss ($L_{adv,G}$)**: 생성된 이미지가 라벨 없는 실제 이미지($X_{UL}$)의 분포와 유사하도록 하여, 현실적인 변이를 생성하게 한다.
- **Large Deviation Loss ($L_{LD,G}$)**: 생성기가 너무 작은 변환만 만들어내어 학습이 쉬운(trivial) 데이터만 생성하는 것을 방지한다.
  - $L_{LD,G_V} = -\|v\|_1$
  - $L_{LD,G_I} = -\|\Delta I\|_1$
  - 음수 부호를 통해 $L_1$ 노름을 최대화함으로써 더 큰 변형을 유도한다.

### 4. 학습 절차 (Optimization Sequence)

1. **S 사전 학습**: 라벨 데이터만으로 분할 네트워크 $S$를 몇 에포크 동안 학습시킨다.
2. **생성기 및 판별기 학습**: $S$의 손실 함수와 정규화 항을 함께 사용하여 $G_V, D_V$ 및 $G_I, D_I$를 독립적으로 학습시킨다.
3. **최적 파라미터 선택**: 검증 데이터셋(validation set)의 Dice score가 가장 높은 지점의 생성기 파라미터를 고정한다.
4. **S 최종 학습**: 고정된 생성기로 생성한 대량의 증강 데이터와 원래 데이터를 합쳐 분할 네트워크 $S$를 처음부터 다시 학습시킨다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Cardiac(ACDC), Prostate, Pancreas 세 가지 의료 데이터셋을 사용하였다.
- **제한적 설정**: 라벨 데이터의 수 $N_L$을 1개 또는 3개로 매우 제한하여 설정하였다.
- **비교 대상**: No aug, Affine(Aff), Random Elastic(RD), Random Intensity(RI), MixUp, Self-training, Adversarial training 등.
- **지표**: Dice Similarity Coefficient (DSC)를 사용하여 평가하였다.

### 2. 주요 결과

- **정량적 결과**: 제안 방법인 $GD+GI$(학습된 변형 및 강도 증강)가 모든 데이터셋에서 다른 모든 증강 및 SSL 방법보다 월등히 높은 DSC를 기록하였다.
- **개별 기여도**: 학습된 변형 필드($GD$)가 무작위 Elastic 변형($RD$)보다 성능이 높았으며, 학습된 강도 필드($GI$)가 무작위 강도 변동($RI$)보다 우수하였다.
- **결합 효과**: $GD$와 $GI$를 모두 사용했을 때 단독 사용 시보다 성능이 크게 향상되었다.

## 🧠 Insights & Discussion

### 1. 현실성과 유용성의 분리

실험 결과, 생성기가 만들어낸 이미지가 항상 해부학적으로 완벽하게 현실적이지는 않았으나, 분할 네트워크의 성능은 오히려 향상되었다. 이는 **'현실적인 이미지 생성'이 반드시 '최적의 분할 네트워크 학습'으로 이어지는 것은 아니라는 점**을 시사하며, Task-driven 방식의 정당성을 뒷받침한다.

### 2. Joint Optimization의 중요성

Ablation study(Table I)를 통해 생성기를 분할 손실 없이 독립적으로 학습시킨 경우보다, $S$의 손실 함수를 포함하여 공동 최적화(Joint optimization)했을 때 성능이 비약적으로 상승함을 확인하였다.

### 3. 한계 및 비판적 해석

- **계산 비용**: 최적의 하이퍼파라미터($\lambda_{adv}, \lambda_{LD}$)를 찾기 위해 많은 계산 자원이 필요하며, 이는 새로운 데이터셋에 적용할 때 부담이 될 수 있다.
- **범위의 제한**: 본 연구는 단일 스캐너에서 얻은 데이터를 기준으로 하였으므로, 서로 다른 스캐너 간의 강도 차이(intensity difference) 문제를 직접적으로 해결하는 구조는 아니다.

## 📌 TL;DR

본 논문은 의료 영상 분할을 위해 **분할 작업의 성능 향상을 직접 목표로 하는 Task-driven 및 Semi-supervised 데이터 증강 기법**을 제안한다. 형태($G_V$)와 강도($G_I$)를 조절하는 두 개의 생성기를 설계하고, 이를 분할 손실과 라벨 없는 데이터의 분포(Adversarial loss)를 활용해 학습시켰다. 실험 결과, 매우 적은 라벨 데이터 환경에서 기존의 무작위 증강이나 일반적인 GAN 기반 방식보다 훨씬 높은 성능을 보였으며, 이는 단순히 현실적인 이미지를 만드는 것보다 작업에 유용한 데이터를 생성하는 것이 더 중요함을 입증하였다.
