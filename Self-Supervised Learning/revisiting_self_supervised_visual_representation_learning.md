# Revisiting Self-Supervised Visual Representation Learning

Alexander Kolesnikov, Xiaohua Zhai, Lucas Beyer (2019)

## 🧩 Problem to Solve

컴퓨터 비전 분야에서 비지도 학습(Unsupervised Learning)을 통한 시각적 표현 학습(Visual Representation Learning)은 여전히 해결되지 않은 난제로 남아 있다. 특히 최근의 자기지도 학습(Self-Supervised Learning, SSL) 기법들은 다양한 벤치마크에서 우수한 성능을 보이고 있으나, 대부분의 연구는 새로운 Pretext Task(사전 학습 과제)를 제안하는 것에만 집중해 왔다.

반면, Convolutional Neural Networks(CNN) 아키텍처의 선택과 같은 모델 구조적 측면이 자기지도 학습의 결과물인 표현(Representation)의 질에 어떤 영향을 미치는지는 충분히 연구되지 않았다. 따라서 본 논문의 목표는 다양한 CNN 아키텍처를 사용하여 기존의 자기지도 학습 모델들을 대규모로 재검토하고, 아키텍처 설계가 표현 학습의 성능에 미치는 핵심적인 통찰을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 직관은 **"자기지도 학습에서 아키텍처의 선택은 Pretext Task의 선택만큼이나 중요하다"**는 것이다. 저자들은 완전 지도 학습(Fully-supervised Learning)에서 통용되는 표준 아키텍처 설계 방식이 자기지도 학습 설정에서는 그대로 적용되지 않을 수 있음을 발견하였다.

주요 기여 사항은 다음과 같다.

1. **아키텍처 영향력 분석**: 다양한 CNN 아키텍처(ResNet, RevNet, VGG)와 Widening Factor(채널 확장 계수)가 SSL 성능에 미치는 영향을 체계적으로 분석하였다.
2. **성능의 획기적 향상**: 적절한 아키텍처 선택과 모델 확장(Widening)을 통해 기존 SSL 기법들의 성능을 대폭 끌어올렸으며, 특히 Rotation 예측 과제에서 이전의 State-of-the-art(SOTA) 결과를 크게 상회하는 성과를 거두었다.
3. **표현의 질적 특성 규명**: Skip-connection이 있는 구조에서는 네트워크의 후반부에서도 표현의 질이 저하되지 않는다는 점과, 모델의 너비(Width)가 SSL 성능에 결정적인 역할을 한다는 점을 밝혔다.
4. **평가 방법론의 재검토**: 선형 분류기(Linear Classifier)를 이용한 표현 평가 시, SGD 최적화 스케줄에 따라 결과가 크게 달라질 수 있으며 수렴까지 매우 많은 epoch이 필요함을 지적하였다.

## 📎 Related Works

자기지도 학습은 데이터로부터 자동으로 레이블을 생성하여 Pretext Task를 수행함으로써, 실제 다운스트림 태스크(Downstream Task)에 유용한 표현을 학습하는 프레임워크이다.

1. **Patch-based methods**: 이미지 패치의 상대적 위치를 예측하는 Context Prediction [7]이나, 무작위로 섞인 패치들의 원래 위치를 맞추는 Jigsaw Puzzle [34] 등이 있다.
2. **Image-level classification**: 이미지를 무작위로 회전시킨 후 그 각도를 예측하는 Rotation [11] 과제가 대표적이다.
3. **Dense spatial outputs**: 이미지 인페인팅(Inpainting), 컬러라이제이션(Colorization) 등 픽셀 수준의 예측을 수행하는 방식이다.
4. **Structural constraints**: 표현 공간에 등변성(Equivariance) 제약을 가하거나, Contrastive Predictive Coding(CPC) [37]과 같이 미래의 패치를 예측하는 방식이 존재한다.

기존 연구들은 주로 "어떤 새로운 Pretext Task를 설계할 것인가"에 초점을 맞추었으나, 본 연구는 "동일한 Task를 어떤 아키텍처로 수행할 것인가"라는 관점에서 접근하여 기존 연구와의 차별점을 가진다.

## 🛠️ Methodology

### 1. 분석 대상 CNN 아키텍처

본 연구에서는 다음과 같은 모델들을 비교 분석하였다.

- **ResNet50 (v1, v2)**: Skip-connection을 사용하는 대표적인 모델이다. Widening Factor $k \in \{4, 8, 12, 16\}$를 도입하여 채널 수를 조절하였다.
- **RevNet50**: 분석적으로 역함수 계산이 가능한(Analytically Invertible) 구조를 가진 모델이다. Residual Unit의 입력을 $x_1, x_2$로 나누고 출력을 다음과 같이 정의한다.
$$y_2 := x_2, \quad y_1 := x_1 + F(x_2)$$
이를 통해 정보 손실 없이 네트워크 깊은 곳까지 정보를 전달할 수 있는지 검증하였다.
- **VGG19-BN**: Skip-connection이 없는 구조의 영향을 확인하기 위해 Batch Normalization이 추가된 VGG 모델을 사용하였다.

### 2. 사용된 Self-supervised Techniques

- **Rotation**: 이미지를 $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$로 회전시킨 후 이를 분류하는 4-class classification 문제이다.
- **Exemplar**: 각 이미지 자체를 하나의 클래스로 간주한다. 강한 데이터 증강(Augmentation)을 통해 여러 예시를 만들고, Triplet Loss를 사용하여 동일 이미지의 표현은 가깝게, 서로 다른 이미지의 표현은 멀게 학습한다.
- **Jigsaw**: 이미지를 $3 \times 3$ 패치로 나누어 무작위로 섞은 후, 원래의 순열(Permutation)을 예측한다.
- **Relative Patch Location**: 두 패치 사이의 8가지 상대적 위치 관계(예: "위", "오른쪽 위" 등)를 예측한다.

### 3. 학습 및 평가 절차

- **표현 추출**: 네트워크의 최종 레이어 직전인 Pre-logits 레이어에서 표현을 추출한다.
- **평가 방법**: 추출된 표현을 고정한 채, 그 위에 선형 로지스틱 회귀(Linear Logistic Regression) 모델을 학습시켜 ImageNet 및 Places205 데이터셋에서 정확도를 측정하는 Linear Evaluation 방식을 사용하였다.
- **최적화**: L-BFGS 및 SGD를 사용하여 선형 모델을 학습시켰으며, 특히 SGD의 경우 학습률 감소(Learning rate decay) 시점에 따른 성능 변화를 분석하였다.

## 📊 Results

### 1. 정량적 결과 (ImageNet 기준)

- **아키텍처의 중요성**: 동일한 Pretext Task라도 아키텍처에 따라 성능 차이가 매우 컸다. 예를 들어, Rotation 과제에서 RevNet50은 매우 높은 성능을 보였으나, VGG19-BN은 현저히 낮은 성능을 기록하였다.
- **SOTA 경신**: 적절한 아키텍처(RevNet50)와 넓은 채널($16\times$)을 적용했을 때, Rotation 과제에서 **55.4%**의 Top-1 정확도를 달성하여 기존 SSL SOTA를 크게 경신하였다.
- **Widening 효과**: 모델의 너비(Width)를 증가시키는 것이 SSL 성능 향상에 매우 결정적인 영향을 미쳤으며, 이는 지도 학습 때보다 더 두드러지는 경향을 보였다.

### 2. 주요 실험 결과 및 분석

- **선형 평가의 타당성**: MLP(Multi-Layer Perceptron)를 사용한 비선형 평가를 수행한 결과, 선형 평가와 비교해 성능 향상이 미미했다. 이는 선형 평가만으로도 표현의 질을 충분히 측정할 수 있음을 시사한다.
- **Pretext Task 정확도의 함정**: Pretext Task 자체의 정확도가 높다고 해서 반드시 다운스트림 태스크의 성능이 높은 것은 아니었다. 즉, 사전 학습 과제의 정확도는 아키텍처를 선택하는 기준(Proxy)으로 사용하기에 부적절하다.
- **정보 보존 특성**: VGG와 같은 구조는 네트워크 후반부로 갈수록 표현의 질이 떨어지지만, ResNet/RevNet과 같은 Skip-connection 구조는 Pre-logits 레이어까지 표현의 질이 계속 유지되거나 오히려 향상되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

본 논문은 단순한 알고리즘 제안이 아니라, 대규모 실험을 통해 SSL의 기본 전제들을 다시 검토함으로써 실무적인 가이드라인을 제공하였다. 특히 **"Pretext Task와 아키텍처의 결합(Combination)"**이 중요하다는 점을 밝혀, 향후 SSL 연구가 Task 설계와 모델 설계 두 방향 모두에서 균형 있게 이루어져야 함을 시사한다.

### 2. 한계 및 논의사항

- **Pretext Accuracy의 한계**: 사전 학습 과제의 성능과 실제 표현의 유용성 사이의 괴리가 발견되었다. 이는 레이블이 없는 환경에서 어떤 모델이 더 좋은 표현을 학습했는지 판단할 수 있는 새로운 메커니즘이 필요함을 의미한다.
- **계산 비용**: 모델의 너비를 확장함으로써 얻는 성능 향상이 크지만, 이는 메모리 사용량 및 계산 비용의 증가를 동반한다.

### 3. 비판적 해석

저자들은 RevNet이 ResNet보다 일부 과제에서 우수함을 보였으나, 모든 Task에서 일관된 우위를 점한 것은 아니었다. 이는 역함수 가능성(Invertibility)이 모든 종류의 SSL 표현 학습에 필수적인 요소는 아닐 수 있음을 시사하며, Task의 특성에 따라 최적의 아키텍처가 다를 수 있다는 점을 다시 한번 강조한다.

## 📌 TL;DR

본 논문은 자기지도 학습(SSL)에서 Pretext Task만큼이나 **CNN 아키텍처의 선택과 모델의 너비(Width)가 성능에 결정적인 영향**을 미친다는 것을 대규모 실험으로 입증하였다. 특히 Skip-connection이 있는 구조와 확장된 채널 수를 통해 기존 SSL 기법들의 성능을 획기적으로 높였으며, 지도 학습과 비지도 학습 간의 성능 격차를 절반 수준으로 줄였다. 이는 향후 SSL 연구 시 Task 설계와 모델 구조 최적화를 동시에 고려해야 함을 시사하는 중요한 연구이다.
