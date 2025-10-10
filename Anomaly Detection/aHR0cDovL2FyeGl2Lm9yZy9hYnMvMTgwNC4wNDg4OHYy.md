# Scalable and Interpretable One-class SVMs with Deep Learning and Random Fourier Features

Minh-Nghia Nguyen, Ngo Anh Vien

## 🧩 Problem to Solve

기존 One-class Support Vector Machine (OC-SVM)은 이상 탐지(Anomaly Detection, AD)에서 효과적인 방법이지만, 대규모 및 고차원 데이터셋에 적용할 때 최적화 복잡도로 인해 확장성(scalability) 문제가 발생합니다. 이전 연구들은 특징 표현 학습(representation learning)과 이상 탐지를 분리하여 처리하여 비효율적이었습니다. 또한, OC-SVM의 결정 과정을 입력 공간과 커널 공간 간의 암묵적인 매핑 때문에 설명하기 어렵다는(interpretability) 중요한 문제가 있었습니다.

## ✨ Key Contributions

- **AE-1SVM 모델 제안**: 딥 오토인코더(Deep Autoencoder)와 무작위 푸리에 특징(Random Fourier Features, RFF)을 활용한 OC-SVM을 결합하여 종단간(end-to-end) 학습이 가능한 이상 탐지 모델을 제안합니다. 이는 표현 학습과 이상 예측을 동시에 최적화합니다.
- **확장성 및 효율성 향상**: RFF를 통해 Radial Basis Function (RBF) 커널을 근사하고 확률적 경사 하강법(Stochastic Gradient Descent, SGD)을 사용하여 OC-SVM의 최적화 복잡도를 줄여 대규모 고차원 데이터셋에 대한 훈련 및 테스트 시간을 크게 단축합니다.
- **설명 가능한 이상 탐지**: 그래디언트 기반 기여도(attribution) 방법을 OC-SVM 및 전체 종단간 아키텍처에 적용하여 모델의 이상 탐지 결정 과정을 입력 특징 관점에서 설명할 수 있는 프레임워크를 개발했습니다. 이는 딥러닝 기반 이상 탐지 모델의 설명 가능성을 연구한 최초의 사례 중 하나입니다.
- **성능 향상**: 분리 학습 방식의 기존 방법론보다 우수한 이상 탐지 성능을 달성합니다.

## 📎 Related Works

- **기존 이상 탐지 기법**: Principal Component Analysis (PCA), OC-SVM, Isolation Forest, K-means, Gaussian Mixture Model (GMM) 등이 있으며, 고차원 데이터에서 비효율적이고 통합된 차원 축소 접근 방식이 부족합니다.
- **딥러닝 기반 이상 탐지**:
  - **2단계/분리 학습**: 오토인코더로 저차원 공간을 학습한 후 기존 OC-SVM을 적용하는 방식([13] 등). 이는 학습 단계가 분리되어 있어 효율적인 이상 탐지 특징을 학습하기 어렵습니다.
  - **Robust Deep Autoencoder (RDA)**: 강건한 PCA와 오토인코더 기반 차원 축소를 결합합니다.
  - **Deep Clustering Embedding (DEC)**: 비지도 오토인코더와 클러스터링을 통합하지만, 이상 탐지보다는 클러스터링에 최적화된 잠재 공간을 학습합니다.
  - **종단간 학습(밀도 추정 기반)**: Deep Energy-based Model ([37]), Autoencoder + GMM ([40]), Generative Adversarial Networks (GANs) ([24,36]) 등. 이러한 방법들은 밀도 추정을 통해 이상을 탐지하지만, 가까운 이상 패턴이 많을 경우 높은 밀도를 할당하여 오탐(false negative)을 유발할 수 있습니다.
- **커널 근사 기법**: Nyström ([33]) 및 Random Fourier Features (RFF) ([23])는 커널 머신의 확장성 문제를 해결합니다.
- **SGD를 이용한 SVM 최적화**: [26,5]에서 SVM 최적화에 SGD를 적용하는 가능성을 제시했습니다.
- **그래디언트 기반 설명 기법**: Gradient\*Input ([28]), Integrated gradients ([30]), DeepLIFT ([27]) 등 딥러닝 모델의 분류 결정 및 입력 특징 민감도를 설명하는 데 활용됩니다.

## 🛠️ Methodology

AE-1SVM 모델은 다음과 같은 두 가지 주요 구성 요소로 이루어집니다:

1. **딥 오토인코더**: 입력 공간의 차원 축소 및 특징 표현 학습을 담당합니다.
2. **OC-SVM**: 학습된 특징 공간에서 이상 예측을 수행하며, RBF 커널은 무작위 푸리에 특징(RFF)으로 근사됩니다.

- **아키텍처 통합**: 딥 오토인코더의 병목(bottleneck) 계층 출력이 OC-SVM의 입력으로 직접 연결됩니다. RFF 매퍼는 오토인코더의 잠재 공간(latent space) 표현에 적용됩니다.
- **공동 목적 함수**: 전체 모델은 다음과 같은 목적 함수를 사용하여 SGD와 역전파를 통해 종단간으로 훈련됩니다.
  $$Q(\theta, w, \rho) = \alpha L(x,x') + \frac{1}{2}\|w\|^2 - \rho+ \frac{1}{\nu n}\sum_{i=1}^{n}\max(0, \rho - w^T z(x_i))$$
  여기서 $L(x,x')$는 오토인코더의 재구성 손실(L2-norm loss), $z(x_i)$는 RFF 매핑된 특징, $\alpha$는 특징 압축과 SVM 마진 최적화 간의 균형을 조절하는 하이퍼파라미터입니다. $n$은 배치 크기입니다.
- **RFF 적용**: RBF 커널은 결합된 사인 및 코사인 매핑 $z(x) = \sqrt{\frac{1}{D}}[\cos(\omega_1^T x) \dots \cos(\omega_D^T x) \sin(\omega_1^T x) \dots \sin(\omega_D^T x)]^T$으로 근사됩니다.
- **그래디언트 기반 설명**: OC-SVM의 결정 함수 마진 $g(x) = w \cdot \phi(x) - \rho$의 입력 특징에 대한 그래디언트($\frac{\partial g}{\partial x_k}$)를 계산하고, 이를 오토인코더의 입력 계층에 대한 잠재 공간 노드의 그래디언트($\frac{\partial \text{latent}}{\partial \text{input}}$)와 연쇄 법칙(chain rule)을 사용하여 결합함으로써 종단간 그래디언트를 얻습니다. 이를 통해 각 입력 특징이 모델의 이상 결정에 기여하는 정도를 해석할 수 있습니다.

## 📊 Results

- **이상 탐지 성능**:
  - 합성 데이터셋(Gaussian) 및 5개 실제 데이터셋(ForestCover, Shuttle, KDDCup99, USPS, MNIST)에서 AUROC 및 AUPRC 지표로 평가되었습니다.
  - AE-1SVM은 모든 시나리오에서 기존 OC-SVM 및 2단계 분리 학습 방식보다 높은 정확도를 보였습니다.
  - Isolation Forest, RDA, DEC와 같은 최신 방법론과 비교했을 때도 경쟁적이거나 우수한 성능을 달성했습니다. 특히 AUPRC에서 높은 점수를 기록하여 불균형 데이터셋에서의 강점을 입증했습니다.
- **효율성**:
  - ForestCover와 같이 가장 큰 데이터셋에서 다른 방법들보다 뛰어난 훈련 시간을 보였고, KDDCup99 및 Shuttle과 같은 대규모 샘플 크기 데이터셋에서도 가장 빠른 후보 중 하나였습니다.
  - 전체 KDDCup99 데이터셋에 대한 훈련에서도 약 200초 만에 유망한 결과를 얻어 빅데이터 환경에서의 확장성을 입증했습니다.
  - 테스트 시간은 기존 OC-SVM 및 Isolation Forest에 비해 크게 향상되어 실시간 환경에서의 적용 가능성을 시사합니다.
- **설명 가능성 검증**:
  - 합성 데이터셋 실험에서 그래디언트 기반 설명 규칙이 유효함을 시각적으로 입증했습니다.
  - USPS 및 MNIST 이미지 데이터셋에 대한 그래디언트 맵을 통해 특정 픽셀이 이상 탐지 결정에 어떻게 기여하는지 시각적으로 보여주었습니다. 예를 들어, '1'을 정상으로 보고 '7'을 이상으로 탐지할 때, '1'에는 없는 '7'의 특정 획에 해당하는 픽셀이 강한 양의 그래디언트를 가지는 것을 확인했습니다.

## 🧠 Insights & Discussion

- **종단간 학습의 이점**: 오토인코더와 OC-SVM을 공동으로 최적화함으로써, 차원 축소 네트워크가 이상 탐지 작업에 더 유용한 특징 표현을 학습하도록 유도할 수 있었습니다. 이는 분리 학습 방식의 단점을 극복합니다.
- **확장성 및 실용성**: RFF를 통한 커널 근사 및 SGD의 적용은 OC-SVM의 고질적인 확장성 문제를 해결하여 대규모 고차원 데이터셋에서도 효율적인 이상 탐지를 가능하게 합니다. 이는 산업 응용 분야에서 큰 장점입니다.
- **딥러닝 이상 탐지의 설명 가능성**: 모델의 결정에 대한 그래디언트 기반 설명은 '블랙박스'로 여겨지던 딥러닝 모델의 투명성을 높여줍니다. 이는 특히 의료 진단이나 사이버 보안과 같이 의사 결정의 근거가 중요한 분야에서 유용하게 활용될 수 있습니다.
- **잠재적 한계**: 모델의 하이퍼파라미터(예: $\nu$, $\sigma$, $\alpha$) 튜닝이 성능에 중요하게 작용할 수 있으며, 최적의 설정을 찾는 데 추가적인 연구가 필요할 수 있습니다.

## 📌 TL;DR

본 연구는 대규모 고차원 데이터에서 OC-SVM의 확장성과 설명 가능성 문제를 해결하기 위해, 딥 오토인코더와 무작위 푸리에 특징을 사용한 OC-SVM을 결합한 종단간 학습 모델인 AE-1SVM을 제안합니다. 이 모델은 공동 목적 함수와 SGD를 통해 효율적으로 훈련되며, 그래디언트 기반 방법을 통해 이상 탐지 결정 과정을 설명할 수 있습니다. 실험 결과, AE-1SVM은 기존 및 최신 이상 탐지 모델 대비 우수한 성능과 효율성을 보였으며, 딥러닝 기반 이상 탐지의 설명 가능성을 입증했습니다.
