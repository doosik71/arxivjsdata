# Sleep Staging Based on Multi Scale Dual Attention Network

Huafeng Wang, Chonggang Lu, Qi Zhang, Zhimin Hu, Xiaodong Yuan, Pingshu Zhang, Wanquan Liu (2021)

## 🧩 Problem to Solve

수면 단계 분석(Sleep Staging)은 수면 장애 진단에 필수적이지만, 전통적으로 전문가가 수면다원검사(Polysomnography, PSG) 데이터를 수동으로 분류하는 방식은 시간이 매우 많이 소요된다. 또한, PSG를 위해 여러 채널의 신호를 동시에 수집하는 과정은 복잡하며, 이는 피험자의 수면 질에 부정적인 영향을 줄 수 있다. 이에 따라 단일 채널 뇌파(single-channel EEG)만을 이용한 자동 수면 단계 분석이 주요 연구 주제로 부상하였다.

기존의 단일 채널 EEG 기반 자동 분석 방법들은 전반적인 정확도는 향상되었으나, 특히 N1 단계(수면의 전이 단계)에서의 성능이 진단 요구 수준에 미치지 못하는 문제가 있다. 이는 N1 단계의 데이터 양이 상대적으로 적고, 깨어 있는 상태와 수면 상태의 특성이 혼재되어 있어 구분이 어렵기 때문이다. 따라서 본 논문의 목표는 raw EEG 신호를 직접 입력으로 사용하여 N1 단계를 포함한 전반적인 수면 단계 분류 성능을 높이는 Multi Scale Dual Attention Network(MSDAN)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 EEG 신호에 포함된 다양한 스케일의 파형 특성을 포착하고, 불필요한 노이즈를 제거하여 핵심적인 특징만을 강조하는 것이다. 이를 위해 다음과 같은 설계를 도입하였다.

1. **Multi-scale Convolution**: 서로 다른 크기의 커널을 사용하여 EEG 신호 내의 다양한 주파수 및 파형 성분을 동시에 추출한다.
2. **Dual Attention Mechanism**: Channel Attention과 Spatial Attention을 직렬로 연결하여 중요한 정보는 강조하고 불필요한 정보는 억제한다.
3. **Soft Thresholding**: Channel Attention 단계에 soft thresholding 기법을 도입하여 EEG 신호의 고유한 노이즈와 중복 정보를 효과적으로 제거한다.
4. **Residual Connections**: 네트워크가 깊어짐에 따라 발생할 수 있는 성능 저하(Degradation) 문제를 방지하기 위해 잔차 연결을 도입하였다.

## 📎 Related Works

기존의 수면 단계 분석 방법은 크게 전통적인 머신러닝 방식과 딥러닝 방식으로 나뉜다.

- **전통적 머신러닝**: STFT, 모달 분해(Modal Decomposition) 등을 통해 수작업으로 특징(handcrafted features)을 추출하고 SVM과 같은 분류기를 사용한다. 하지만 이러한 방식은 특징 추출기의 품질에 크게 의존하며, EEG 데이터의 비선형성과 개인차로 인해 범용적인 특징 추출기를 설계하는 데 많은 시간이 소요된다는 한계가 있다.
- **딥러닝 기반 방법**: CNN, ResNet, LSTM 등을 활용하여 Raw 데이터로부터 직접 특징을 학습한다. 최근 연구들은 1D-CNN이나 Deep Residual Network를 통해 성능을 높였으나, 대부분의 모델이 N1 단계에서 낮은 F1-score를 기록하며 성능 정체 현상을 보이고 있다.

본 논문은 단순한 깊은 네트워크 구성에서 벗어나, Multi-scale 추출과 Dual Attention을 통해 N1 단계의 변별력을 높임으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인

MSDAN은 Raw EEG 신호를 입력받아 수면 단계를 분류하는 end-to-end 네트워크이다. 전체 구조는 크게 세 단계로 구성된다: **Multi-scale Convolutional Layer $\rightarrow$ Residual Attention Layers $\rightarrow$ Fully Connected Layer & Softmax**.

### 주요 구성 요소

**1. Multi-scale Feature Extraction**
EEG 신호의 다양한 파형(예: alpha rhythm, spindle wave 등)을 포착하기 위해 세 개의 서로 다른 커널 크기($1 \times 3, 1 \times 5, 1 \times 7$)를 가진 병렬 브랜치를 사용한다. 각 브랜치는 두 개의 컨볼루션 층, Batch Normalization, ReLU 및 잔차 연결로 구성되며, 최종 출력은 Concatenate 연산을 통해 하나로 합쳐진다.

**2. Residual Attention Layer**
이 레이어는 세 개의 `AttentionBlock`이 직렬로 연결된 구조이다. 각 블록은 Channel Attention과 Spatial Attention이 순차적으로 적용된다.

- **Channel Attention & Soft Thresholding**:
    각 채널의 중요도를 학습하며, 특히 노이즈 제거를 위해 soft thresholding 함수를 적용한다.
    $$a' = w_{l+1} a_l + b_{l+1}$$
    $$a_{l+1} = \text{soft}(a', \theta\lambda) = \text{sgn}(a')(|a'| - \theta\lambda)_+$$
    여기서 $\theta$는 학습된 가중치이며, $\lambda$는 $0$과 $1$ 사이의 임계값이다. 이 과정은 불필요한 채널 정보를 사전에 제거하는 역할을 한다.

- **Spatial Attention**:
    1차원 신호 내의 공간적 관계를 학습한다. Global Average Pooling과 Max Pooling을 통해 특징을 생성한 후, $1 \times 3$ 컨볼루션과 Sigmoid 함수를 거쳐 가중치 $\beta$를 생성한다.
    $$\beta = \delta(\text{Conv}_{1 \times 3}(\text{AvgPool}(a_{l+1}); \text{MaxPool}(a_{l+1})))$$
    $$a_{l+2} = a'' * \beta$$
    최종적으로 이 결과는 동일 매핑(identical mapping)과 합쳐져 잔차 블록을 형성한다.

### 학습 절차 및 손실 함수

데이터셋의 클래스 불균형(Label Imbalance) 문제를 해결하기 위해 가중치가 적용된 Multi-class Cross Entropy 손실 함수를 사용한다.

가중치 $\text{weight}[\text{class}]$는 다음과 같이 결정된다:
$$\text{weight}[\text{class}] = \min(5, \max(1, \ln(\frac{1}{p(\text{class})})))$$
여기서 $p(\text{class})$는 전체 데이터에서 해당 클래스가 차지하는 비율이다. 최종 손실은 배치 내 모든 샘플의 가중 손실 평균으로 계산된다. 최적화 알고리즘으로는 Adam을 사용하였으며, 학습률은 $0.0005$, 배치 크기는 $8$로 설정하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Sleep-EDF 및 Sleep-EDFx 공개 데이터셋을 사용하였다.
- **검증 방법**: 5-fold cross-validation 및 Hold-out validation(8:2 분할)을 적용하였다.
- **평가 지표**: Accuracy, Recall, Precision, F1-score, Cohen's Kappa coefficient를 사용하였다.

### 주요 결과

실험 결과, MSDAN은 두 데이터셋 모두에서 매우 높은 성능을 기록하였다.

- **Sleep-EDF (5-fold)**: Overall Accuracy $91.74\%$, Macro F1 $0.8231$, Kappa $0.8723$.
- **Sleep-EDFx (5-fold)**: Overall Accuracy $90.35\%$, Macro F1 $0.7945$, Kappa $0.8284$.

특히 주목할 점은 가장 분류가 어려운 **N1 단계의 F1-score**가 Sleep-EDF에서 $54.41\%$, Sleep-EDFx에서 $52.79\%$를 기록하며 기존 SOTA 모델들보다 유의미하게 향상되었다는 점이다.

### 비교 실험

ResNet50, ResNet34, CNN, Utime 등의 기존 모델과 동일한 조건(Sleep-EDF, Fpz-Cz 채널, 5-fold CV)에서 비교 실험을 수행한 결과, MSDAN의 Overall Accuracy는 $91.05\%$로 가장 높았으며, N1 단계의 F1-score 역시 $53.28\%$로 타 모델 대비 약 $2\% \sim 10\%$ 이상 향상된 결과를 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 성과는 Multi-scale 컨볼루션을 통해 EEG의 다양한 주파수 특성을 잡고, Dual Attention과 Soft Thresholding을 통해 노이즈를 억제한 데서 기인한다. t-SNE 시각화 결과, Raw EEG 데이터 상태에서는 단계별 구분이 모호했으나, 모델의 Fully Connected Layer를 통과한 후에는 수면 단계별로 특징들이 명확하게 군집화됨을 확인하였다. 이는 제안한 네트워크가 수면 단계 분류에 유효한 고차원 특징을 성공적으로 추출하고 있음을 시사한다.

### 한계 및 향후 과제

전반적인 성능 향상에도 불구하고, N1 단계의 정확도는 여전히 다른 단계에 비해 낮다. 이는 N1 단계가 가지는 본질적인 전이 특성 때문이며, 저자는 이를 해결하기 위해 데이터셋의 클래스 비율을 더 정교하게 조정하거나 N1 단계의 손실 함수 가중치를 추가로 높이는 방안을 제시하였다. 또한, 계산 자원이 허용한다면 네트워크를 더 깊게 설계하여 강건성을 높일 수 있을 것으로 논의하였다.

## 📌 TL;DR

본 논문은 단일 채널 EEG 기반 수면 단계 분석에서 특히 취약했던 **N1 단계의 분류 성능을 높이기 위해 Multi Scale Dual Attention Network(MSDAN)**를 제안하였다. Multi-scale CNN으로 다양한 파형을 포착하고, Soft-thresholding이 결합된 Dual Attention으로 노이즈를 제거함으로써, 기존 SOTA 모델들을 상회하는 성능(Overall Acc $\approx 91\%$)을 달성하였다. 이 연구는 복잡한 PSG 장비 없이 단일 채널 EEG만으로도 실용적인 수준의 자동 수면 단계 분석이 가능함을 보여주어, 향후 웨어러블 수면 진단 기기 적용에 중요한 기여를 할 가능성이 크다.
