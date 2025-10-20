# Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark

Shiv Ram Dubey, Satish Kumar Singh, Bidyut Baran Chaudhuri

## 🧩 Problem to Solve

심층 신경망은 비선형적으로 분리 불가능한 입력 데이터를 계층적인 비선형 변환을 통해 보다 선형적으로 분리 가능한 추상 특징으로 변환해야 합니다. 이 과정에서 활성화 함수(AF)는 필수적인 비선형성을 부여하는 핵심 요소입니다. 그러나 기존의 AF들은 다음과 같은 주요 문제점들을 안고 있습니다:

- **기울기 소실(Vanishing Gradient):** Sigmoid 및 Tanh와 같은 AF는 입력이 포화(saturation)될 때 기울기가 거의 0이 되어 학습을 방해합니다.
- **계산 복잡성:** Sigmoid 및 Tanh 함수는 지수 함수 계산으로 인해 계산 비용이 높습니다.
- **음수 값 활용 부족:** ReLU 기반 AF는 음수 입력에 대해 출력이 0이 되어 정보 손실 및 '죽은 ReLU(Dying ReLU)' 문제를 발생시킵니다.
- **제한된 비선형성:** 일부 AF는 데이터의 복잡성을 충분히 모델링할 만큼 충분한 비선형성을 제공하지 못합니다.
- **무한 출력(Unbounded Output):** ReLU와 같은 AF의 무한 출력 범위는 학습 불안정성을 초래할 수 있습니다.
- **적응성 부족:** 대부분의 수동으로 설계된 AF는 특정 데이터셋이나 네트워크 아키텍처의 복잡성에 맞게 자동으로 조절되지 않습니다.

이 논문은 이러한 AF의 다양한 특성과 문제점을 종합적으로 분석하고, 효과적인 AF 선택을 위한 지침을 제공하는 것을 목표로 합니다.

## ✨ Key Contributions

- 광범위한 활성화 함수(AF)에 대한 상세한 분류를 제시하며, Logistic Sigmoid/Tanh 기반, Rectified Unit 기반, Exponential Unit 기반, 학습 기반 AF를 포괄적으로 다룹니다.
- 다양한 관점(출력 범위, 단조성, 평활성 등)에서 최신 AF를 분석하여 심층 학습 커뮤니티에 통찰력을 제공합니다.
- 다양한 데이터 유형(이미지, 텍스트, 음성)에 대한 AF의 적합성과 주요 특징을 요약하여 제시합니다.
- 기존 서베이 및 성능 분석과의 비교를 통해 본 연구의 포괄성과 중요성을 강조합니다.
- 4가지 벤치마크 데이터셋에서 18가지 최신 AF를 다양한 네트워크(MobileNet, VGG16, GoogLeNet, ResNet50, SENet18, DenseNet121 등)와 함께 실험적으로 비교하고 분석합니다.

## 📎 Related Works

- **Karlik and Olgac (2011) [134]:** 다층 퍼셉트론(MLP)에서 Tanh가 기존 AF보다 우수한 성능을 보임을 보고.
- **Maas et al. (2013) [34], He et al. (2015) [35]:** ReLU와 그 변형(Leaky ReLU, Parametric ReLU)이 기울기 소실 문제 해결 및 성능 향상에 기여함을 입증.
- **Clevert et al. (2016) [27]:** Exponential Linear Unit (ELU)이 ReLU의 단점(음수 값 활용 부족)을 보완하며 빠른 학습과 높은 정확도를 보임을 제시.
- **Ramachandran et al. (2018) [29]:** 자동 탐색을 통해 Swish AF를 발견했으며, 이는 Softplus보다 성능이 향상됨을 보고.
- **Hendrycks and Gimpel (2016) [101]:** Gaussian Error Linear Unit (GELU)을 제안하며 비선형성을 확률적 정규화로 간주.
- **Misra (2019) [99]:** Mish AF를 제안하며 비단조적(non-monotonic)이고 평활한 특성으로 YOLOv4 같은 모델에 적용되어 우수한 성능을 보임.
- **Molina et al. (2020) [111]:** Padé Activation Unit (PAU)을 제안, 유리 함수를 통해 AF를 학습하고 기존 AF를 근사하거나 새로운 AF를 학습할 수 있음을 보임.
- **다양한 이전 서베이들 [138, 140, 146]:** AF 목록을 제공하거나 특정 유형의 AF에 초점을 맞췄으나, 본 논문은 더 광범위한 AF를 다루고 실험적 성능 비교를 포함합니다.

## 🛠️ Methodology

본 논문은 활성화 함수(AF)에 대한 포괄적인 서베이와 벤치마크를 위해 다음과 같은 방법론을 사용합니다:

1. **AF 분류 및 특성 분석:**

   - **Logistic Sigmoid 및 Tanh 기반 AF:** Logistic Sigmoid($S(x) = \frac{1}{1+e^{-x}}$)와 Tanh($T(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$)를 기본으로 한 변형들을 분석합니다. 이들은 주로 기울기 소실 문제와 비(非)제로 평균 출력을 개선하려는 시도입니다.
   - **Rectified Linear Unit (ReLU) 기반 AF:** ReLU($\text{max}(0, x)$)와 그 단점(음수 값 미활용, 제한된 비선형성, 무한 출력)을 극복하려는 변형들(LReLU, PReLU, CReLU, BReLU, FReLU, ABReLU 등)을 다룹니다.
   - **Exponential Linear Unit (ELU) 기반 AF:** ELU($\alpha(e^x - 1)$ for $x \leq 0$, $x$ for $x > 0$)를 기반으로 음수 값을 지수 함수로 활용하여 기울기 소실을 완화하는 변형들(SELU, CELU, PELU, MPELU, EELU 등)을 분석합니다.
   - **학습/적응형 AF:** 데이터와 네트워크 복잡성에 따라 매개변수를 학습하여 비선형성을 조절하는 AF(APL, Swish, E-Swish, AAF, SLAF, MeLU 등)를 조사합니다.
   - **기타 AF:** Softplus 기반, 확률적 AF(GELU), 다항식 AF, 서브네트워크 AF, 커널 AF 등을 포함합니다.
   - 각 AF의 매개변수 유무, 단조성, 평활성, 유계성 등의 특성을 비교합니다.

2. **실험적 성능 비교:**
   - **대상 AF:** Logistic Sigmoid, Tanh, Elliott [25], ReLU [8], LReLU [34], PReLU [35], ELU [27], SELU [52], GELU [101], CELU [53], Softplus [93], Swish [29], ABReLU [44], LiSHT [24], Soft-Root-Sign (SRS) [26], Mish [99], PAU [111], PDELU [59] 총 18가지 최신 AF를 선정합니다.
   - **데이터셋:**
     - **이미지 분류:** CIFAR10, CIFAR100 (50,000개 훈련 이미지, 10,000개 테스트 이미지)
     - **언어 번역:** 독일어-영어 번역 데이터셋
     - **음성 인식:** LibriSpeech 100시간 영어 음성 데이터셋
   - **네트워크 아키텍처:**
     - **이미지 분류:** MobileNet [149], VGG16 [150], GoogLeNet [151], ResNet50 [152], SENet18 [153], DenseNet121 [154] (경량 및 대규모 모델 포함).
     - **언어 번역:** LSTM 기반 Seq2Seq 인코더-디코더 네트워크.
     - **음성 인식:** Deep Speech 2 프레임워크 (Residual CNN 2개, 양방향 GRU 2개).
   - **훈련 설정:**
     - **이미지 분류:** 100 에포크, 배치 크기 128 (CIFAR10)/64 (CIFAR100), 학습률 0.001 (처음 80 에포크)/0.0001 (마지막 20 에포크), Adam 최적화, 교차 엔트로피 손실.
     - **언어 번역:** 50 에포크, 학습률 0.001, 배치 크기 256, Adam 최적화, 교차 엔트로피 손실.
     - **음성 인식:** 10 에포크, 학습률 0.0005, 배치 크기 10.
   - **평가 지표:**
     - **이미지 분류:** 테스트 정확도 (5회 반복 평균 및 표준편차).
     - **언어 번역:** Bleu Score (4-gram, 5회 반복 평균 및 표준편차).
     - **음성 인식:** 문자 오류율(CER) 및 단어 오류율(WER) (5회 반복 평균 및 표준편차).
   - **환경:** Google Colab, 8GB GPU 데스크톱, PyTorch 프레임워크.

## 📊 Results

- **이미지 분류 (CIFAR10/CIFAR100):**

  - **MobileNet:** Softplus, ELU, CELU가 가장 좋은 성능을 보였습니다.
  - **VGG16, GoogLeNet, DenseNet:** ReLU, Mish, PDELU가 우수한 성능을 나타냈습니다.
  - **잔여 연결 기반 네트워크 (ResNet50, SENet18, DenseNet121):** ReLU, LReLU, ELU, GELU, CELU, ABReLU, PDELU가 더 좋은 성능을 보였습니다.
  - **수렴 속도:** PAU가 대부분의 경우 가장 빠른 수렴을 보였습니다. PReLU, GELU, PDELU도 일관성 있게 좋은 수렴을 나타냈습니다. 반면, Sigmoid와 Elliott는 수렴이 가장 느렸으며, SRS는 SENet18 모델에서 훈련이 발산하는 경우도 있었습니다.
  - **훈련 시간:** PDELU, SRS, Elliott는 훈련 시간이 상당히 길었습니다. ReLU, ELU, CELU, Softplus는 정확도와 훈련 시간 사이에서 좋은 균형을 제공했습니다.

- **언어 번역 (독일어-영어):**

  - Tanh와 SELU AF가 Bleu Score 측면에서 가장 적합한 것으로 나타났습니다. PReLU, LiSHT, SRS, PAU 또한 좋은 성능을 보였습니다.

- **음성 인식:**
  - PReLU, GELU, Swish, Mish, PAU AF가 문자 오류율(CER) 및 단어 오류율(WER) 측면에서 가장 적합한 것으로 나타났습니다.

## 🧠 Insights & Discussion

- **Sigmoid 및 Tanh 기반 AF:** 기울기 소실 및 비(非)제로 평균 문제를 해결하려는 시도가 많았지만, 복잡도 증가라는 단점을 가집니다. 일반적으로 CNN에서는 수렴이 좋지 않아 피해야 하지만, RNN에서는 게이트(예: LSTM, GRU)에 여전히 널리 사용됩니다.
- **ReLU 기반 AF:** 음수 값 활용 부족, 제한된 비선형성, 무한 출력 문제를 개선하려 했으나, 모든 문제 해결에는 실패했습니다. LReLU, PReLU, ABReLU는 잔여 네트워크에서 특정 이점을 보였으나, MobileNet, VGG, GoogLeNet 같은 일부 모델에서는 ReLU보다 좋지 않은 성능을 보일 수 있습니다. ReLU, LReLU, PReLU는 단순성 때문에 가장 흔히 선택되는 AF입니다.
- **지수 함수 기반 AF:** 음수 입력을 효과적으로 활용하여 비선형성을 높이고 중요한 특징에 대한 포화를 방지하려 합니다. 그러나 대부분 비평활성(non-smooth) 문제에 직면합니다.
- **학습 기반 적응형 AF:** 데이터 및 네트워크 복잡성에 따라 비선형성을 자동으로 조절하는 최근 트렌드입니다. 학습 가능한 매개변수로 인해 약간의 복잡도가 증가하지만, 전체 네트워크 매개변수에 비하면 무시할 만합니다. 적절한 기본 함수와 매개변수 초기화가 성능에 중요하며, 잘못된 초기화는 훈련 발산으로 이어질 수 있습니다.
- **일반적인 권장 사항:**
  - 훈련 속도 향상을 위해 음수 및 양수 값을 모두 활용하여 출력이 제로 평균에 가깝도록 하는 것이 중요합니다.
  - 모델과 데이터셋의 복잡성에 맞는 AF를 선택해야 합니다. AF는 이러한 간극을 자동으로 메울 수 있어야 합니다.
  - ReLU는 여전히 인기 있는 선택이지만, Swish, Mish, PAU와 같은 최신 AF도 다양한 문제에 대해 시도해 볼 가치가 있습니다.
  - 잔여 연결이 있는 네트워크(예: ResNet, SENet, DenseNet)에는 ReLU, LReLU, ELU, GELU, CELU, PDELU가 이미지 분류에 더 적합합니다.
  - 일반적으로 PAU, PReLU, PDELU와 같은 매개변수형 AF는 데이터를 학습하여 더 나은 수렴을 보입니다.
  - PDELU와 SRS는 훈련 시간을 증가시키지만, ReLU, SELU, GELU, Softplus는 정확도와 훈련 시간 사이의 좋은 균형을 제공합니다.
  - 언어 번역에는 Tanh, SELU, PReLU, LiSHT, SRS, PAU가, 음성 인식에는 PReLU, GELU, Swish, Mish, PAU가 권장됩니다.

## 📌 TL;DR

심층 신경망에서 활성화 함수(AF)는 비선형성을 도입하여 복잡한 데이터 학습을 가능하게 하는 핵심 요소이지만, 기울기 소실, 계산 효율성, 비선형성 부족, 적응성 부재 등의 문제에 직면합니다. 본 논문은 Logistic Sigmoid/Tanh, ReLU, ELU, 학습 기반 AF를 포함한 광범위한 AF들을 종합적으로 조사하고 분류합니다. 특히, 18가지 최신 AF를 이미지 분류(CIFAR10/100), 언어 번역, 음성 인식 등 다양한 벤치마크 작업과 신경망 아키텍처에 대해 실험적으로 비교 분석합니다. 결과는 특정 AF가 네트워크 유형(예: 잔여 연결 네트워크)이나 데이터 유형(예: 텍스트, 음성)에 따라 우수한 성능을 보임을 보여줍니다. 예를 들어, Tanh와 SELU는 언어 번역에, PReLU, GELU, Swish, Mish, PAU는 음성 인식에 적합하며, PAU는 빠른 수렴을 보이지만 PDELU나 SRS는 훈련 시간이 길어질 수 있음을 지적합니다. 이 연구는 AF 선택에 대한 실질적인 지침을 제공하고, 효율적인 AF 개발을 위한 미래 연구 방향을 제시합니다.
