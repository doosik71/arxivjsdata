# Self-Knowledge Distillation via Dropout

Hyoje Lee, Yeachan Park, Hyun Seo, Myungjoo Kang (2022)

## 🧩 Problem to Solve

심층 신경망(Deep Neural Networks, DNN)은 성능 향상을 위해 모델의 깊이나 너비를 확장하는 경향이 있으며, 이는 막대한 계산 비용과 메모리 소모를 초래한다. 이를 해결하기 위해 모델 압축 기술인 Knowledge Distillation(KD)이 널리 사용되어 왔으나, 전통적인 Offline KD 방식은 거대한 Teacher 네트워크를 먼저 학습시켜야 하며, Student 모델에 적합한 최적의 Teacher 모델을 찾는 것이 어렵다는 한계가 있다.

최근에는 Teacher 없이 단일 모델 내부에서 지식을 증류하는 Self-Knowledge Distillation(SKD) 방식이 제안되었다. 하지만 기존의 SKD 방법론들은 추가적인 학습 가능 파라미터(Trainable parameters)를 가진 서브 네트워크를 필요로 하거나, 학습 데이터의 클래스 분포와 같은 추가적인 Ground-truth 레이블 정보에 의존한다는 단점이 있다. 따라서 본 논문은 추가 파라미터나 데이터 의존성 없이, 매우 단순하면서도 효과적으로 모델의 일반화 성능을 높일 수 있는 Self-Knowledge Distillation 방법을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Dropout을 활용하여 단일 모델 내에서 서로 다른 관점을 가진 여러 개의 가상 모델을 생성하고, 이들 사이의 Posterior distribution(사후 분포)을 일치시키는 것이다. 구체적으로는 Feature extraction 단계 이후에 Dropout sampling을 통해 두 개의 서로 다른 Feature vector를 추출하고, 이를 통해 생성된 두 확률 분포 간의 Kullback-Leibler divergence(KL-divergence)를 최소화함으로써 모델 내부의 지식을 상호 증류한다.

또한, 본 연구는 KD에서 흔히 사용되는 Forward KL-divergence뿐만 아니라 Reverse KL-divergence를 함께 사용하는 것이 이론적, 실험적으로 더 강력한 정규화 효과를 제공함을 입증하였다.

## 📎 Related Works

기존의 Knowledge Distillation 연구는 주로 Teacher-Student 프레임워크에 기반하여 큰 모델의 지식을 작은 모델로 전이하는 방식에 집중했다. 이후 Teacher-Student 구조를 탈피한 연구들이 등장하였는데, 대표적으로 다음과 같은 방식들이 있다.

- **Deep Mutual Learning (DML):** 두 네트워크가 서로의 지식을 상호 증류하는 방식이다.
- **Data-Distortion Guided Self-Distillation (DDGSD):** 왜곡된 데이터 간의 지식을 증류한다.
- **Be Your Own Teacher (BYOT):** 모델의 깊은 층과 얕은 층 사이의 지식을 증류한다.
- **Class-wise self-knowledge distillation (CS-KD):** 클래스 내 인스턴스 간의 사후 분포를 일치시킨다.

이러한 기존 SKD 방식들은 추가적인 파라미터를 가진 서브 네트워크를 구성해야 하거나, 입력 데이터를 왜곡시키는 추가 과정이 필요하며, 결과적으로 계산 비용이 높다는 한계가 있다. 반면, 제안된 SD-Dropout은 단일 모델 내에서 단순한 Sampling 연산만을 사용하므로 계산 효율성이 매우 높으며, 모델 구조에 상관없이 적용 가능한 Model-agnostic한 특성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 절차

SD-Dropout의 전체 파이프라인은 다음과 같다. 입력 데이터 $x$가 네트워크를 통과하여 Global feature vector $f(x)$가 생성되면, 이후 Dropout layer를 통해 두 번의 독립적인 sampling을 수행하여 서로 다른 두 개의 feature $u \odot f(x)$와 $v \odot f(x)$를 얻는다. 이 feature들은 동일한 Fully Connected(FC) layer를 통과하여 두 개의 서로 다른 Posterior distribution $p^u_\theta(y|x)$와 $p^v_\theta(y|x)$를 생성한다.

### 주요 방정식 및 손실 함수

먼저, Softmax classifier를 통한 클래스 $i$의 사후 확률은 다음과 같이 정의된다.
$$p(y=i|x;\theta,T) = \frac{\exp(z_i/T)}{\sum_{j=1}^N \exp(z_j/T)}$$
여기서 $z_i$는 클래스 $i$의 logit이며, $T$는 Temperature 하이퍼파라미터이다.

SD-Dropout은 두 sampling 결과 사이의 지식을 상호 증류하기 위해 대칭적인 KL-divergence 손실 함수 $L_{SDD}$를 정의한다.
$$L_{SDD}(x;u,v,\theta,T) := D_{KL}(p^u_\theta(y|x)||p^v_\theta(y|x)) + D_{KL}(p^v_\theta(y|x)||p^u_\theta(y|x))$$

최종 학습을 위한 전체 손실 함수 $L_{Total}$은 표준 Cross-Entropy(CE) 손실과 $L_{SDD}$의 가중 합으로 구성된다.
$$L_{Total}(x,y;u,v,\theta,T) = L_{CE}(x,y;\theta) + \lambda_{SDD} \cdot T^2 \cdot L_{SDD}(x;u,v,\theta,T)$$
여기서 $\lambda_{SDD}$는 증류 손실의 비중을 조절하는 하이퍼파라미터이다.

### Forward vs Reverse KL-Divergence 분석

논문은 Forward KL-divergence와 Reverse KL-divergence의 기울기(Gradient) 특성을 분석한다. Forward KL은 타겟 분포를 고정하고 근사 분포를 맞추는 방식(Zero-avoiding)인 반면, Reverse KL은 근사 분포의 기울기를 더 강하게 반영한다.

저자들은 Proposition 1을 통해, 특정 가정(Assumption 1, 2) 하에서 Reverse divergence의 도함수가 Forward divergence보다 더 크다는 것을 이론적으로 증명하였다. 이는 Reverse direction을 추가함으로써 두 확률 분포 사이의 연결성(Bond)을 더 강하게 만들어, 더 강력한 Self-knowledge distillation 효과를 얻을 수 있음을 시사한다.

## 📊 Results

### 실험 설정 및 지표

- **데이터셋:** CIFAR-100, CUB-200-2011, Stanford Dogs (분류), ImageNet (대규모 분류), MS COCO (객체 탐지), CIFAR-C (Distribution shift).
- **백본 모델:** ResNet-18, ResNet-34, ResNet-152, DenseNet-121.
- **지표:** Accuracy, mAP (Mean Average Precision), ECE (Expected Calibration Error), AUROC/AUPR (OOD 탐지).

### 주요 결과

1. **이미지 분류 성능:** CIFAR-100, CUB-200, Stanford Dogs 모든 데이터셋에서 기본 CE 학습 대비 성능이 향상되었다. 특히 CUB-200-2011에서는 53.8%에서 66.6%로 비약적인 상승을 보였다. 또한, 기존 SKD 방법(BYOT 등)과 결합했을 때 시너지 효과가 나타나 성능이 추가로 향상됨을 확인하였다.
2. **대규모 데이터셋 및 타 작업 확장성:** ImageNet(ResNet-152)에서 Top-1 정확도가 74.8% $\to$ 75.5%로 상승하였으며, MS COCO 객체 탐지 작업에서도 mAP가 향상되어 Vision task 전반에 걸쳐 유효함을 입증하였다.
3. **Calibration 및 Robustness:** ECE 지표가 감소하여 모델의 Overconfidence 문제가 완화되었으며, FGSM 공격에 대한 adversarial robustness와 CIFAR-C 데이터셋에서의 noise robustness가 모두 향상되었다.
4. **OOD(Out-of-Distribution) 탐지:** ODIN detector를 사용한 실험 결과, SD-Dropout을 적용한 모델이 In-distribution 데이터의 불확실성을 줄여 OOD 탐지 성능(AUROC, AUPR)을 높였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 해석

본 논문의 가장 큰 강점은 추가 파라미터나 복잡한 아키텍처 변경 없이 Dropout sampling이라는 매우 단순한 기법만으로 정규화 효과를 극대화했다는 점이다. 이론적으로 분석한 Forward/Reverse KL의 상호 보완적 관계는 단순한 실험적 발견을 넘어, 왜 대칭적 KL-divergence가 효과적인지를 수학적으로 뒷받침한다. 실제로 두 방향의 Gradient 간 코사인 유사도가 학습이 진행됨에 따라 감소한다는 결과는 두 방향의 기울기가 서로 다른 정보를 제공하며, 둘 다 필수적임을 보여준다.

### 한계 및 비판적 해석

실험 결과에서 Dropout을 적용하는 위치에 따라 성능 차이가 발생하였다. 깊은 층(Deep layer)에서 샘플링한 High-level feature가 얕은 층(Shallow layer)보다 훨씬 효과적이었다. 이는 SD-Dropout이 단순한 정규화 도구를 넘어, 모델의 고수준 표현(Representation)을 정제하는 역할을 수행함을 의미한다. 다만, 최적의 $\beta$(Dropout rate)와 $\lambda_{SDD}$ 값을 찾기 위한 하이퍼파라미터 탐색 과정이 필요하며, 이는 모델이나 데이터셋마다 다를 수 있다.

## 📌 TL;DR

본 논문은 Dropout sampling을 통해 단일 네트워크 내부에서 두 개의 가상 모델을 생성하고, 이들 간의 Posterior distribution을 대칭적 KL-divergence(Forward + Reverse)로 일치시키는 **SD-Dropout** 방법을 제안한다. 이 방법은 추가 파라미터 없이도 이미지 분류, 객체 탐지 성능을 높이며, 모델의 Calibration, Adversarial Robustness, OOD 탐지 능력을 동시에 개선한다. 단순하면서도 강력한 정규화 기법으로서, 향후 다양한 딥러닝 모델의 일반화 성능을 높이는 범용적인 방법론으로 활용될 가능성이 매우 높다.
