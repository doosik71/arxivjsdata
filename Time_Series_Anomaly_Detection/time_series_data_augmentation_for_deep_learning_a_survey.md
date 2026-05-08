# Time Series Data Augmentation for Deep Learning: A Survey

Qingsong Wen, Liang Sun, Fan Yang, Xiaomin Song, Jingkun Gao, Xue Wang, Huan Xu

## 🧩 Problem to Solve

최근 딥러닝은 시계열 분석 작업에서 뛰어난 성능을 보이고 있으나, 과적합을 피하기 위해 대규모 훈련 데이터가 필수적입니다. 하지만 의료 시계열 분류나 AIOps의 이상 감지와 같이 많은 실제 시계열 응용 분야에서는 레이블이 지정된 데이터가 제한적입니다. 딥러닝 모델이 시계열 데이터에 성공적으로 적용되기 위해서는 훈련 데이터의 크기와 품질을 향상시킬 효과적인 데이터 증강(Data Augmentation) 방법이 중요합니다. 기존의 데이터 증강 방법들은 시계열 데이터의 고유한 특성(예: 시간 종속성, 주파수 도메인)을 충분히 활용하지 못하며, 이미지나 음성 처리에서 차용된 방법들은 시계열 데이터에 유효하지 않을 수 있습니다. 또한, 데이터 증강 방법은 특정 작업(예: 분류, 이상 감지, 예측)에 따라 달라질 수 있으며, 클래스 불균형 문제 해결에도 더욱 중요해집니다.

## ✨ Key Contributions

- 시계열 데이터 증강 방법을 체계적으로 검토하고 포괄적인 분류 체계(taxonomy)를 제안했습니다.
- 제안된 분류 체계를 기반으로 각 방법의 장단점을 강조하며 구조화된 리뷰를 제공했습니다.
- 시계열 분류, 이상 감지, 예측 등 다양한 시계열 작업에 대해 여러 데이터 증강 방법의 효과를 경험적으로 비교 및 평가했습니다.
- 시계열 데이터 증강 분야의 다섯 가지 주요 미래 연구 방향을 제시하여 유용한 연구 가이드를 제공했습니다.

## 📎 Related Works

본 연구는 시계열 데이터 증강에 대한 포괄적인 설문조사로, 기존 연구 중 Iwana and Uchida (2020)의 시계열 분류를 위한 데이터 증강 방법 설문과 유사하지만, 해당 연구가 분류 외의 다른 일반적인 시계열 작업(예: 예측, 이상 감지)을 다루지 않고 미래 연구 방향도 제시하지 않는다는 한계를 지적하며 본 연구의 필요성을 강조합니다. 또한, 이미지 (Shorten and Khoshgoftaar, 2019) 및 음성 (Cui et al., 2015) 분야의 데이터 증강 연구들과 차별점을 제시합니다.

## 🛠️ Methodology

본 논문은 시계열 데이터 증강 방법에 대한 설문조사 연구로, 방법론은 다음의 분류 체계를 중심으로 다양한 기술을 설명하고 비교하는 방식으로 진행됩니다.

### 시계열 데이터 증강 분류 체계

#### 1. Basic Approaches (기본 접근 방식)

- **시간 도메인(Time Domain)**: 원본 시계열을 직접 조작합니다.
  - **Window Cropping/Slicing**: 원본 시계열에서 연속적인 슬라이스를 무작위로 추출합니다.
  - **Window Warping**: 무작위 시간 범위를 선택하여 압축 또는 확장합니다.
  - **Flipping**: 시계열 값의 부호를 뒤집습니다 ($-x_t$).
  - **DTW Barycentric Averaging (DBA)**: DTW를 사용하여 새로운 시계열을 생성하고 가중 평균으로 앙상블합니다.
  - **Noise Injection**: 가우시안 노이즈, 스파이크, 스텝형 트렌드, 슬로프형 트렌드 등을 주입합니다.
  - **Label Expansion**: 이상 감지에서 레이블이 지정된 이상 지점 근처의 데이터를 이상으로 확장합니다.
- **주파수 도메인(Frequency Domain)**: 시계열을 주파수 도메인으로 변환한 후 조작합니다.
  - **Amplitude and Phase Perturbations (APP)**: 푸리에 변환 후 진폭 스펙트럼과 위상 스펙트럼에 노이즈를 주입합니다.
  - **Surrogate Data (AAFT/IAAFT)**: 푸리에 변환 후 위상 스펙트럼에 무작위 위상 셔플을 수행하고 역 푸리에 변환합니다.
- **시간-주파수 도메인(Time-Frequency Domain)**: 시계열을 시간-주파수 표현으로 변환한 후 조작합니다.
  - **STFT-based Augmentation**: STFT를 사용하여 시간-주파수 특징을 생성하고, 지역 평균 또는 특징 벡터 셔플링을 적용합니다.
  - **SpecAugment**: 음성 시계열의 Mel-Frequency 표현에 특징 워핑, 주파수 채널 마스킹, 시간 스텝 마스킹을 적용합니다.

#### 2. Advanced Approaches (고급 접근 방식)

- **분해 기반 방법(Decomposition-based Methods)**: 시계열을 트렌드($\tau_t$), 계절성($s_t$), 잔차($r_t$)로 분해한 후 증강합니다.
  - 잔차에 대한 부트스트래핑(bootstrapping) 또는 시간/주파수 도메인 증강을 적용한 후 다시 결합합니다.
- **통계적 생성 모델(Statistical Generative Models)**: 시계열의 동역학을 통계 모델로 모델링하여 새로운 시계열을 생성합니다.
  - 가우시안 트리 혼합 모델(Mixture of Gaussian Trees), LGT(Local and Global Trend), MAR(Mixture Autoregressive) 모델 등이 있습니다.
- **학습 기반 방법(Learning-based Methods)**: 딥러닝 모델을 활용하여 데이터를 생성하거나 증강 정책을 학습합니다.
  - **임베딩 공간(Embedding Space)**: 학습된 잠재 공간에서 보간(interpolation) 또는 외삽(extrapolation)을 통해 새로운 샘플을 생성합니다. (예: MODALS)
  - **딥 생성 모델(Deep Generative Models, DGMs)**: GAN(Generative Adversarial Network)을 사용하여 현실적인 시계열 데이터를 생성합니다. (예: RGAN, RCGAN, TimeGAN)
  - **자동화된 데이터 증강(Automated Data Augmentation)**: 강화 학습, 메타 학습, 진화 검색 등을 통해 최적의 데이터 증강 정책을 자동으로 탐색합니다. (예: TANDA, AutoAugment, MODALS의 진화 검색 전략)

### 경험적 평가

세 가지 일반적인 시계열 작업(분류, 이상 감지, 예측)에 대해 일부 기본 증강 방법(자르기, 워핑, 뒤집기, APP 기반 증강)을 사용하여 성능 개선 효과를 검증했습니다.

## 📊 Results

- **시계열 분류**: Alibaba Cloud 모니터링 시스템의 시계열 데이터를 사용하여, 아웃라이어(spike, step, slope) 주입 시 데이터 증강(cropping, warping, flipping)이 분류 정확도를 $0.1\%$ ~ $1.9\%$ 향상시켰습니다.
- **시계열 이상 감지**: Yahoo! 데이터셋에 대한 U-Net 기반 네트워크 평가에서, 원본 데이터($\text{U-Net-Raw}$) 대비 분해된 잔차($\text{U-Net-DeW}$) 사용 시 F1 점수가 크게 향상되었고, 여기에 데이터 증강(flipping, cropping, label expansion, APP)을 적용($\text{U-Net-DeWA}$)하면 F1 점수가 $0.403 \rightarrow 0.662 \rightarrow 0.693$으로 더욱 개선되었습니다.
- **시계열 예측**: DeepAR 및 Transformer 모델에 대해 electricity, traffic, M4 대회 데이터셋을 사용한 평가에서, 데이터 증강(cropping, warping, flipping, APP)이 MASE(Mean Absolute Scaled Error)를 평균적으로 개선하는 유망한 결과를 보였습니다. 다만, 특정 데이터/모델 조합에서는 부정적인 결과도 관찰되었습니다.

## 🧠 Insights & Discussion

시계열 데이터 증강은 딥러닝 모델의 성능을 향상시키는 데 전반적으로 효과적임을 보여주지만, 특정 데이터나 작업에 따라 그 효과가 달라질 수 있습니다. 특히, 이미지/음성 분야에서 차용된 증강 기법을 시계열에 단순 적용하는 것은 한계가 있으며, 시계열의 시간 종속성과 같은 고유한 특성을 고려한 방법론이 필요합니다.

**미래 연구 기회:**

1. **시간-주파수 도메인에서의 증강**: STFT 외에 웨이블릿 변환(CWT, DWT, MODWT 등)을 활용한 시계열 증강 연구가 필요합니다. 특히, $\text{MODWT}$ 기반의 $\text{WIAAFT}$ 또는 진폭/위상 스펙트럼 교란 기법이 유망합니다.
2. **불균형 클래스를 위한 증강**: 시계열 분류에서 빈번한 클래스 불균형 문제를 해결하기 위해 데이터 증강과 비용-민감(cost-sensitive) 학습 또는 가중치 부여 기법을 결합하는 연구가 필요합니다.
3. **증강 선택 및 조합**: 다양한 증강 방법 중 최적의 조합을 선택하고 적용하는 전략(예: $\text{RandAugment}$와 유사한 자동화된 탐색, 강화 학습 기반의 맞춤형 정책)이 필요합니다.
4. **가우시안 프로세스(GPs)를 활용한 증강**: $\text{GPs}$ 및 $\text{DGPs}$는 시계열의 보간/외삽 능력과 커널 설계를 통한 특성 제어가 가능하여 새로운 데이터 인스턴스 생성에 활용될 수 있습니다.
5. **딥 생성 모델(DGMs)을 활용한 증강**: $\text{GAN}$ 외에 $\text{DARN}$ (Deep Autoregressive Networks), $\text{NF}$ (Normalizing Flows), $\text{VAE}$ (Variational Autoencoders)와 같은 다른 $\text{DGM}$ 아키텍처를 시계열 데이터 증강에 활용하는 연구가 필요합니다.

## 📌 TL;DR

본 논문은 딥러닝 기반 시계열 분석에서 제한된 레이블 데이터로 인한 과적합 문제를 해결하기 위한 **시계열 데이터 증강 기법**을 체계적으로 조사합니다. **시간, 주파수, 시간-주파수 도메인의 기본 증강**과 **분해 기반, 통계적 생성, 학습 기반의 고급 증강**으로 구성된 분류 체계를 제시하고, 각 방법의 장단점을 분석합니다. 시계열 분류, 이상 감지, 예측 세 가지 주요 작업에 대한 경험적 평가를 통해 데이터 증강의 효과를 입증했으며, 특히 **이상 감지에서 높은 성능 향상**을 보였습니다. 웨이블릿 변환 기반 시간-주파수 증강, 불균형 클래스 처리, 자동화된 증강 선택/조합, 가우시안 프로세스 및 다양한 딥 생성 모델 활용 등 **다섯 가지 미래 연구 방향**을 제시하여 시계열 데이터 증강 분야의 발전을 위한 통찰력을 제공합니다.
