# Going Deeper with Contextual CNN for Hyperspectral Image Classification

Hyungtae Lee and Heesung Kwon (2017)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 하이퍼스펙트럴 이미지(Hyperspectral Image, HSI) 분류를 위해 더 깊고 넓은 신경망을 구축하면서도, 제한된 학습 데이터로 인해 발생하는 성능 저하 및 과적합(Overfitting) 문제를 해결하는 것이다.

일반적으로 딥러닝 모델은 파라미터 수에 비례하는 대규모 데이터셋이 필요하지만, HSI 분야에서는 대규모 데이터셋을 확보하기 어렵다. 이로 인해 기존의 CNN 기반 HSI 분류 방식들은 모델의 깊이와 너비를 제한한 소규모 네트워크를 사용해 왔으며, 이는 HSI가 가진 풍부한 분광(Spectral) 및 공간(Spatial) 정보를 충분히 활용하지 못하게 하여 분류 성능의 한계를 초래한다. 따라서 본 연구의 목표는 제한된 데이터 환경에서도 효율적으로 학습 가능한 깊고 넓은 구조의 Fully Convolutional Network(FCN)를 설계하여 분류 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 **Residual Learning**과 **Multi-scale Filter Bank**를 결합한 깊은 FCN 구조를 통해 분광 및 공간 정보를 동시에 최적화하여 추출하는 것이다.

1. **Residual Learning의 도입**: 모델의 깊이와 너비를 확장함에 따라 발생하는 학습 효율 저하 문제를 해결하기 위해 Residual Learning을 적용하여, 적은 양의 데이터로도 매우 깊은 네트워크를 안정적으로 최적화할 수 있게 하였다.
2. **Multi-scale Filter Bank 설계**: 네트워크의 초기 단계에서 서로 다른 크기의 필터($1\times1, 3\times3, 5\times5$)를 동시에 사용하는 필터 뱅크를 구축하여, 국부적인 공간 구조와 분광 상관관계를 함께 추출하는 Joint Spatio-Spectral Feature Map을 생성한다.
3. **Fully Convolutional Network (FCN) 기반의 End-to-End 구조**: 입력 이미지의 크기에 상관없이 처리 가능하며, 차원 축소와 같은 전처리 과정 없이 원본 데이터를 직접 입력으로 사용하는 깊은 FCN 구조를 제안하였다.

## 📎 Related Works

논문에서는 기존의 HSI 분류 방식과 일반적인 이미지 분류용 CNN을 다음과 같이 설명한다.

- **일반 이미지 분류 CNN**: LeNet-5, AlexNet, VGG-16, GoogLeNet, ResNet 등으로 이어지며 네트워크의 깊이가 깊어지는 추세이다. 특히 ResNet의 Residual Learning은 매우 깊은 망의 학습 효율을 획기적으로 높였다.
- **기존 HSI 분류 방식**: Kernel 방법론이나 Stacked Autoencoders(SAE), Deep Belief Networks(DBN) 등이 사용되었다. 최근의 CNN 기반 접근법들은 주로 얕은 네트워크를 사용하거나, PCA(Principal Component Analysis)와 같은 차원 축소 기법을 통해 입력 데이터를 줄여 모델의 복잡도를 낮추는 방식을 취했다.
- **기존 방식의 한계**: 기존의 딥러닝 기반 HSI 분류기들은 분광 정보와 공간 정보를 분리해서 처리하거나, 단순한 CNN 구조를 사용하여 두 정보의 상호작용을 충분히 활용하지 못했다. 또한, 데이터 부족으로 인해 모델의 깊이를 확장하지 못한 점이 주요 한계로 지적된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

제안된 네트워크는 총 9개의 레이어로 구성된 Fully Convolutional Network(FCN)이다. 전체 파이프라인은 **Multi-scale Filter Bank $\rightarrow$ Residual Learning Modules $\rightarrow$ Classification Layers** 순으로 이어진다. FCN 구조이므로 풀링(Pooling) 레이어를 사용하지 않아 입력과 출력의 공간적 크기가 유지되며, 임의의 크기를 가진 HSI 입력 데이터를 처리할 수 있다.

### 2. 주요 구성 요소 및 역할

#### (1) Multi-scale Filter Bank

입력 HSI 데이터에서 공간적/분광적 특징을 동시에 추출하기 위한 초기 모듈이다.

- **필터 구성**: $1\times1\times B, 3\times3\times B, 5\times5\times B$ (여기서 $B$는 spectral bands의 수) 크기의 세 가지 필터를 사용한다.
- **역할**: $3\times3$과 $5\times5$ 필터는 지역적 공간 상관관계를, $1\times1$ 필터는 분광 상관관계를 추출한다.
- **결합 과정**: 각 필터의 출력 크기가 다르므로, 입력 이미지 주변에 zero-padding을 수행하고 각각 $5\times5, 3\times3$ max pooling을 적용하여 모든 특징 맵의 크기를 $(H, W)$로 맞춘 뒤 하나로 결합(Concatenation)하여 Joint Spatio-Spectral Feature Map을 형성한다.

#### (2) Residual Learning

깊은 네트워크의 최적화 문제를 해결하기 위해 도입되었다.

- **학습 방식**: 입력 $x$에 대해 직접적인 매핑 대신 잔차(Residual)를 학습하는 방식을 취한다.
- **방정식**:
$$y = F(x, \{W_i\}) + x$$
여기서 $F$는 컨볼루션 필터 $W_i$를 통한 잔차 매핑 함수이며, 이는 네트워크가 $y-x$라는 차이만을 학습하게 함으로써 최적화를 용이하게 한다. 본 논문에서는 두 개의 Residual Learning 모듈을 사용한다.

#### (3) Fully Convolutional Network (FCN) 기반 분류

본 모델은 Fully Connected(FC) 레이어 대신 $1\times1$ 컨볼루션 레이어를 사용하여 픽셀 단위 분류를 수행한다. 이는 FC 레이어와 수학적으로 동일한 효과를 내면서도 이미지 전체에 동일한 가중치를 적용할 수 있게 한다.

### 3. 학습 절차 및 목표

- **데이터 증강(Data Augmentation)**: 과적합을 방지하기 위해 학습 샘플을 수평, 수직, 대각선 방향으로 미러링하여 데이터 양을 4배로 늘렸다.
- **최적화 알고리즘**: Stochastic Gradient Descent(SGD)를 사용하며, 배치 크기는 10, 총 100k 반복 학습을 수행한다.
- **학습률 제어**: 초기 학습률 0.001에서 시작하여 33,333회 및 66,666회 반복 시점에 각각 $10^{-4}, 10^{-5}$로 단계적으로 감소시킨다.
- **초기화**: 가우시안 분포를 이용하여 가중치를 초기화하며, 마지막 레이어를 제외한 모든 바이어스(Bias)는 1로 설정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Indian Pines, Salinas, University of Pavia 세 가지 벤치마크 데이터셋을 사용하였다.
- **비교 대상(Baselines)**: RBF-SVM, Two-layer NN, Three-layer NN, LeNet-5, Shallower CNN(Hu et al.), D-DBN 등이 포함된다.
- **평가 지표**: 전체 정확도(Overall Accuracy, OA)를 측정하였으며, 20번의 무작위 학습/테스트 분할을 통한 평균 및 표준편차를 보고한다.
- **학습 데이터**: 클래스당 200개의 샘플을 고정적으로 사용하여 학습시켰다.

### 2. 주요 결과

- **정량적 성능**: 제안된 네트워크는 모든 데이터셋에서 기존의 모든 baseline보다 높은 OA를 기록하였다.
  - Indian Pines: $93.61 \pm 0.56\%$
  - Salinas: $95.07 \pm 0.23\%$
  - University of Pavia: $95.97 \pm 0.46\%$
- **최적 구조 탐색**:
  - **너비(Width)**: Indian Pines와 Pavia는 128개 필터, Salinas는 192개 필터일 때 최적의 성능을 보였다.
  - **깊이(Depth)**: 모든 데이터셋에서 2개의 Residual Learning 모듈을 사용할 때 가장 성능이 좋았으며, 3개 이상은 과적합으로 인해 성능이 하락하였다.
- **필터 뱅크의 효과**: $1\times1$ 필터만 사용했을 때보다 Multi-scale filter bank를 사용했을 때 정확도가 비약적으로 상승하였다(Indian Pines 기준 약 39.94%p 상승).

## 🧠 Insights & Discussion

### 1. 강점 및 분석

본 연구는 HSI 분야에서 고질적인 문제였던 '데이터 부족으로 인한 얕은 모델 사용'의 굴레를 Residual Learning을 통해 극복하였다. 특히 분광 정보만 고려하던 기존 CNN과 달리, 초기 단계에서 다중 스케일 필터를 통해 공간 정보를 통합함으로써 HSI의 특성을 더 잘 반영한 특징 추출이 가능함을 입증하였다.

### 2. 한계 및 비판적 해석

- **경계 지역의 오분류(Spillover)**: Confusion Matrix 및 경계 분석 결과, 클래스 간 경계 지역에서 오분류가 집중적으로 발생하는 현상이 관찰되었다. 이는 $3\times3, 5\times5$와 같은 공간 필터가 경계 너머의 다른 클래스 픽셀 정보를 함께 읽어 들여 발생하는 'spillover' 효과로 해석된다.
- **데이터 의존성**: 학습 데이터 수를 늘렸을 때 성능이 단조 증가하는 경향을 보였으며, 특히 샘플 수가 적은 클래스(Corn-notill 등)에서 성능이 상대적으로 낮게 나타났다. 이는 여전히 모델의 성능이 가용 데이터의 양에 민감하게 반응함을 의미한다.

### 3. 결론적 논의

본 논문은 단순히 모델을 깊게 만드는 것보다, **어떻게 효율적으로 학습시킬 것인가(Residual Learning)**와 **어떻게 도메인 특성(Spatio-Spectral)을 추출할 것인가(Multi-scale Bank)**가 더 중요하다는 것을 보여주었다.

## 📌 TL;DR

본 논문은 제한된 학습 데이터 환경에서도 하이퍼스펙트럴 이미지 분류 성능을 극대화하기 위해 **Residual Learning**과 **Multi-scale Filter Bank**를 결합한 깊은 **Fully Convolutional Network(FCN)**를 제안한다. 제안된 모델은 분광 정보와 공간 정보를 초기 단계부터 통합적으로 추출하여 기존의 얕은 CNN 및 DBN 기반 모델들보다 월등한 분류 정확도를 달성하였다. 이 연구는 HSI 분석에서 데이터 부족 문제를 해결하며 모델의 깊이를 확장할 수 있는 구체적인 아키텍처 가이드를 제공한다는 점에서 향후 연구에 중요한 기여를 한다.
