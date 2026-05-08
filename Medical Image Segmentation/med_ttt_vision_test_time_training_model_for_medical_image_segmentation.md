# MED-TTT: VISION-TTT TRAINING MODEL FOR MEDICAL IMAGE SEGMENTATION

Jiashu Xu (2024)

## 🧩 Problem to Solve

본 연구는 의료 영상 분할(Medical Image Segmentation) 작업에서 발생하는 두 가지 핵심적인 기술적 한계를 해결하고자 한다.

첫째는 **장거리 의존성(Long-range dependencies) 모델링과 계산 복잡도 사이의 트레이드오프**이다. Convolutional Neural Networks(CNN)는 국소적 특징 추출에는 뛰어나나 전역적인 문맥 파악 능력이 부족하며, 이를 해결하기 위한 Transformer 기반 모델들은 Self-attention 메커니즘의 이차 복잡도($O(N^2)$)로 인해 고해상도 의료 영상 처리 시 계산 비용이 지나치게 높다는 문제가 있다.

둘째는 **의료 영상 특유의 다양성과 복잡한 배경**이다. 병변의 크기와 모양이 매우 다양하여 단일 해상도 추출 방식으로는 세밀한 특징과 전역적 구조를 동시에 잡기 어려우며, 공간 도메인(Spatial domain) 정보만으로는 배경과 전경을 명확히 구분하는 데 한계가 있다.

결과적으로 본 논문의 목표는 계산 효율성을 유지하면서도 강력한 전역 문맥 파악 능력을 갖추고, 다양한 스케일의 특징과 주파수 도메인 정보를 통합하여 정밀한 분할 성능을 제공하는 Med-TTT 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Test-Time Training(TTT) 레이어를 기반으로 한 Vision-TTT 백본 네트워크의 도입**과 이를 보완하기 위한 **다중 해상도 융합 및 주파수 도메인 강화 전략**에 있다.

중심적인 설계 아이디어는 고정된 가중치를 사용하는 기존 레이어와 달리, TTT 레이어를 통해 추론 시점(Inference time)에 입력 데이터에 맞게 모델의 파라미터를 동적으로 업데이트하는 것이다. 이를 통해 선형 복잡도($O(N)$)만으로도 Transformer 수준의 표현력을 확보하고, 데이터 분포의 변화에 유연하게 대응하는 자기지도 적응(Self-supervised adaptation) 능력을 부여하였다. 또한, 공간적 해상도와 주파수 특성을 동시에 고려하는 하이브리드 구조를 통해 의료 영상의 미세한 병변 특징을 극대화하여 포착하도록 설계하였다.

## 📎 Related Works

의료 영상 분할 분야에서는 전통적으로 **CNN 기반의 U-Net**과 그 변형 모델들이 널리 사용되었다. U-Net은 대칭적 인코더-디코더 구조와 Skip connection을 통해 문맥 정보를 효과적으로 전달하지만, 커널의 국소성(Locality)으로 인해 전역적인 의존성을 모델링하는 데 한계가 있다.

이를 극복하기 위해 **Transformer 기반 모델**(TransUNet, Swin-UNet 등)이 등장하였다. 이들은 전역 문맥을 자연스럽게 캡처할 수 있지만, 앞서 언급한 계산 복잡도 문제로 인해 고밀도 예측 작업인 분할 태스크에서 오버헤드가 크다는 단점이 있다.

최근 제안된 **Test-Time Training(TTT)** 기반의 시퀀스 모델링은 고정된 hidden state를 학습 가능한 모델로 대체함으로써, 선형 복잡도를 유지하면서도 매우 표현력이 높은 hidden state를 생성할 수 있음을 보여주었다. 본 논문은 이러한 TTT의 특성을 시각적 백본에 적용하여 기존 CNN의 국소성과 Transformer의 고비용 문제를 동시에 해결하고자 한다.

## 🛠️ Methodology

### 1. Vision-TTT Layer

Vision-TTT 레이어는 시퀀스 모델링의 hidden state를 학습 가능한 가중치 $W$를 가진 모델 $f$로 간주한다. 추론 과정에서 입력 $x_t$가 들어올 때마다 다음과 같이 가중치를 동적으로 업데이트한다.

$$W_t = W_{t-1} - \eta \nabla L(W_{t-1}, x_t)$$

여기서 $\eta$는 학습률이며, $L$은 자기지도 손실 함수이다. 본 모델에서는 단순 재구성을 넘어, 입력 데이터를 두 가지 뷰(View)로 투영하는 방식을 사용한다. $\theta_k$를 통해 학습에 필요한 핵심 정보를 추출한 View K와, $\theta_v$를 통해 최적화 목표가 되는 Label View V를 생성하여 다음과 같은 손실 함수를 정의한다.

$$L(W; x_t) = \|\theta_k f(x_t; W) - \theta_v x_t\|^2$$

업데이트된 가중치 $W_t$를 사용하여 최종 출력 토큰 $z_t$는 다음과 같이 생성된다.

$$z_t = f(\theta_q x_t; W_t)$$

이때 $f$는 MLP(Multi-Layer Perceptron)로 구현되며, $\theta_q$는 추론 시 가장 정보량이 많은 특징을 강조하는 역할을 수행한다.

### 2. Linear Complexity Implementation

TTT 레이어의 병렬 처리를 위해 입력을 $K \times K$ 크기의 미니 배치로 나누어 그래디언트를 계산한다. 단일 미니 배치의 복잡도는 $O(K^2)$이며, 전체 픽셀 수 $N = H \times W$에 대해 총 복잡도는 다음과 같이 선형적으로 감소한다.

$$O\left(\frac{H}{K} \times \frac{W}{K} \times K^2\right) = O(H \times W) = O(N)$$

### 3. Multi-Resolution Fusion & Frequency Domain Integration

- **Multi-Resolution Fusion**: 입력 영상을 고해상도(세부 특징), 중해상도(심층 특징), 저해상도(전역 문맥)의 세 가지 브랜치로 나누어 처리한다. 저해상도에서 고해상도 방향으로 정보를 융합하여 작은 병변부터 거대 구조까지 모두 포착한다.
- **Frequency Domain Enhancement**: 2D 푸리에 변환(Fourier Transform)을 통해 영상을 주파수 도메인으로 변환한 후, **고주파 통과 필터(High-pass filter)**를 적용한다. 이를 통해 저주파 배경 소음을 제거하고 에지(Edge)와 텍스처 등 세밀한 디테일을 강화하여 분할 정밀도를 높인다.

### 4. Loss Function

학습의 안정성을 위해 Cross-Entropy(CE) 손실과 Dice Loss를 결합한 하이브리드 손실 함수를 사용하며, 배치 수준에서 계산하여 샘플 간의 변동성을 완화한다.

$$\text{Loss}_B = (1-\alpha) \times \text{CE}_B + \alpha \times \text{Dice Loss}_B$$

여기서 $\alpha$는 가중치 파라미터로 기본값은 $0.5$이며, 각 손실 함수는 다음과 같다.

$$\text{CE}_B = -\frac{1}{B} \sum_{b=1}^{B} \text{Target}_b \log(\text{Input}_b)$$
$$\text{Dice loss}_B = 1 - \frac{2 \sum_{b=1}^{B} \text{Input}_b \times \text{Target}_b}{\sum_{b=1}^{B} (\text{Input}_b + \text{Target}_b) + \epsilon}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 피부 병변 분할 데이터셋인 ISIC2017 및 ISIC2018을 사용하였다.
- **비교 지표**: mIoU, Dice Similarity Coefficient(DSC), Accuracy(Acc), Specificity(Spe), Sensitivity(Sen)를 측정하였다.
- **비교 모델**: UNet, TransFuse, VM-UNet, HC-Mamba 등 최신 SOTA 모델 및 Mamba 기반 모델과 비교하였다.

### 정량적 결과

ISIC17 데이터셋에서 Med-TTT는 **mIoU 78.83%, DSC 88.16%, Acc 96.07%**를 기록하며 모든 지표에서 가장 우수한 성능을 보였다. 특히 최근의 강자인 HC-Mamba 대비 mIoU에서 1.01%, DSC에서 0.78%의 우위를 보였으며, 기본 UNet보다는 mIoU 1.85%, DSC 2.17% 더 높은 성능을 나타냈다. ISIC18 데이터셋에서도 유사하게 경쟁 모델들을 상회하는 결과를 얻었다.

### Ablation Study

각 구성 요소의 기여도를 분석한 결과, Multi-resolution block(MR-block)이 제거되었을 때 성능 저하가 가장 뚜렷했으며(Setting I, III), 주파수 정보(FFF)를 제외했을 때(Setting II) 전역 정보 및 주파수 도메인의 핵심 특징 활용 능력이 떨어져 성능이 하락함을 확인하였다. 이는 제안한 세 가지 핵심 요소(TTT, MR-block, FFF)가 상호 보완적으로 작용하고 있음을 입증한다.

## 🧠 Insights & Discussion

본 논문의 강점은 **계산 효율성과 모델 표현력의 균형**을 맞춘 점이다. Transformer의 성능을 원하면서도 $O(N^2)$의 비용을 지불하기 어려운 의료 영상 분야에서, TTT 레이어를 통한 선형 복잡도 구현은 매우 실용적인 대안이 될 수 있다. 특히 추론 시점에 가중치를 업데이트하는 동적 적응 방식은 데이터셋 간의 도메인 차이가 큰 의료 영상의 특성상 일반화 성능을 높이는 데 기여했을 것으로 판단된다.

다만, 몇 가지 한계와 논의 사항이 존재한다. 첫째, TTT 레이어의 동적 업데이트 과정이 실제 추론 속도(Latency)에 미치는 영향에 대한 구체적인 분석이 부족하다. 이론적인 복잡도는 $O(N)$이지만, 매 샘플마다 그래디언트를 계산하고 가중치를 업데이트하는 오버헤드가 실제 환경에서 어느 정도인지 명시되지 않았다. 둘째, 본 연구는 피부 병변 데이터셋(ISIC)에 집중되어 있어, CT나 MRI와 같은 다른 모달리티의 3D 의료 영상에서도 동일한 효율성과 성능이 유지될지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 분할을 위해 **선형 복잡도를 가진 Vision-TTT 백본**에 **다중 해상도 융합**과 **고주파 필터링 기반 주파수 정보**를 결합한 **Med-TTT** 모델을 제안한다. 이 모델은 추론 시 파라미터를 동적으로 조정하여 전역 문맥을 효율적으로 캡처하며, ISIC 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 고해상도 의료 영상 처리에서 계산 비용 문제를 해결하면서도 정밀도를 높일 수 있는 새로운 아키텍처 방향성을 제시하여, 향후 다양한 의료 영상 진단 시스템에 적용될 가능성이 높다.
