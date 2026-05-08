# AUTOKWS: KEYWORD SPOTTING WITH DIFFERENTIABLE ARCHITECTURE SEARCH

Bo Zhang, Wenfeng Li, Qingyuan Li, Weiji Zhuang, Xiangxiang Chu, Yujun Wang (2021)

## 🧩 Problem to Solve

본 논문은 스마트 오디오 기기에서 전력 소비를 줄이기 위해 항상 켜져 있는(always-on) 경량 키워드 검출(Keyword Spotting, KWS) 프로그램의 최적화 문제를 해결하고자 한다. KWS 시스템은 실시간 응답성을 위해 매우 낮은 지연 시간(latency)과 적은 연산량을 요구하는 동시에, 높은 정확도를 유지해야 하는 상충 관계(trade-off)에 직면해 있다.

기존에는 전문가가 직접 설계한 end-to-end 신경망(예: depthwise separable convolutions, temporal convolutions, LSTMs 기반 모델)이 사용되었으나, 이러한 수동 설계 방식으로는 방대한 탐색 공간 내에서 정확도, 파라미터 수, 계산 비용의 최적 조합을 찾는 데 한계가 있다. 따라서 본 연구의 목표는 미분 가능한 신경망 구조 탐색(Differentiable Neural Architecture Search, NAS)을 통해 KWS 작업에 최적화된 효율적인 네트워크 구조를 자동으로 발견하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 KWS의 특성과 하드웨어 제약 사항을 고려하여, 실용적인 TC-ResNet 기반의 탐색 공간(search space)을 설계하고 미분 가능한 NAS 알고리즘을 적용해 최적의 경량 모델을 찾아낸 것이다. 특히, DARTS의 변형 모델인 FairDARTS와 NoisyDARTS를 적용하여 기존 NAS 방식들이 가졌던 구조적 붕괴 문제를 해결하고, 매우 적은 파라미터 수로도 최첨단(state-of-the-art) 수준의 성능을 달성하는 모델을 제안하였다.

## 📎 Related Works

KWS를 위한 기존 접근 방식은 크게 네 가지로 분류된다. 첫째, DTW 알고리즘을 사용하는 특징 템플릿 매칭 방식은 학습 시간이 짧으나 강건성(robustness)이 떨어진다. 둘째, Viterbi 알고리즘 기반의 그래프 탐색 디코딩 방식은 경쟁력이 높으나 계산 비용이 비싸다. 셋째, 슬라이딩 윈도우를 이용한 후처리 방식은 자원 제한 플랫폼에 적합하지만 프레임 수준의 정렬을 위한 사전 학습된 신경망이 필요하다. 넷째, CNN이나 LSTM을 이용한 end-to-end 방식이다.

최근에는 NAS가 딥러닝 설계의 새로운 패러다임으로 등장하였으며, 특히 DARTS는 가중치 공유 메커니즘을 통해 탐색 비용을 획기적으로 줄였다. 하지만 DARTS는 재현성이 낮고 구조적 불안정성 문제가 있다. 이를 보완하기 위해 FairDARTS와 NoisyDARTS가 제안되었다. KWS 분야에서도 NAS를 적용한 시도(예: NAS2)가 있었으나, 셀 기반(cell-based)의 복잡한 위상 구조로 인해 스마트 기기에 직접 적용하기에는 제약이 많다는 한계가 있었다.

## 🛠️ Methodology

### Search Space Design

본 연구는 우수한 성능과 적은 메모리 점유율을 가진 TC-ResNet을 기반으로 탐색 공간을 설계하였다. 여기에 채널 간의 상호작용을 모델링하는 Squeeze-and-Excitation (SE) 모듈을 도입하여 성능을 높였다. 탐색 가능한 TC-ResNet-SE 블록의 구성 요소는 다음과 같다.

- **커널 크기**: $\{3, 5, 7, 9\}$ 중 선택
- **SE 모듈**: 활성화 여부 선택
- **Skip connection**: 추가 여부 선택
이러한 조합을 통해 전체 탐색 공간에는 약 $10^7$개의 모델이 존재하게 된다.

### Searching Algorithm

효율적인 탐색을 위해 DARTS, FairDARTS, NoisyDARTS 세 가지 알고리즘을 사용한다.

**1. DARTS (Differentiable Architecture Search)**
각 레이어의 후보 연산 $o \in O$에 대해 구조적 가중치 $\alpha_o$를 할당하고, 이산적인 선택을 연속적인 형태로 완화한다. $i$번째 레이어의 출력 $x_j$는 다음과 같이 가중 합산으로 계산된다.
$$x_j = \sum_{o \in O} \frac{e^{\alpha_o}}{\sum_{o' \in O} e^{\alpha_{o'}}} o(x_i)$$
이 과정은 다음과 같은 이단계 최적화(bi-level optimization) 문제로 정의된다.
$$\min_{\alpha} L_{val}(w^*(\alpha), \alpha) \quad \text{s.t.} \quad w^*(\alpha) = \arg \min_{w} L_{train}(w, \alpha)$$
여기서 $L_{val}$과 $L_{train}$은 각각 검증 및 학습 손실 함수이며, 네트워크 가중치 $w$와 구조 가중치 $\alpha$는 SGD를 통해 교대로 업데이트된다.

**2. FairDARTS**
DARTS가 사용하는 softmax 함수 대신 sigmoid 함수를 사용하여 각 연산의 독립성을 보장한다.
$$\text{sigmoid}(\alpha_o) = \frac{1}{1 + e^{-\alpha_o}}$$
FairDARTS는 특정 임계값(본 논문에서는 $\sigma = 0.8$) 이상의 연산들을 선택함으로써 DARTS의 독점적 경쟁 문제를 완화한다.

**3. NoisyDARTS**
Skip connection이 너무 강력한 그래디언트 흐름을 형성하여 모델이 단순한 잔차 구조로 붕괴되는 현상을 방지하기 위해, skip connection의 출력에 가우시안 노이즈를 추가한다.
$$f_{skip}(x) = x + N(\mu, \beta)$$
여기서 $\mu$는 0으로 설정하며, $\beta$는 노이즈의 표준편차로 그래디언트 흐름을 적절히 방해하여 더 나은 구조를 찾도록 유도한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands v1 및 v2를 사용하였으며, 10개의 키워드와 silence, unknown 클래스를 포함한 총 12개 클래스로 구성된다.
- **입력 데이터**: 인간의 청각 특성과 유사한 40차원 MFCC를 사용하였다.
- **지표**: Top-1 정확도, 파라미터 수, 연산량(multiply-adds)을 측정하였다.

### 정량적 결과

실험 결과, NoisyDARTS-TC14 모델이 가장 우수한 성능을 보였다.

- **V1 데이터셋**: NoisyDARTS-TC14는 약 $109\text{K}$개의 파라미터로 $97.2\%$의 최고 정확도를 달성하였다. 이는 기존의 NAS2 모델($886\text{K}$ 파라미터)보다 파라미터 수가 약 8배 적으면서도 대등하거나 더 높은 정확도를 보인 것이다.
- **V2 데이터셋**: 데이터 양이 더 많은 V2에서는 NoisyDARTS-TC14가 $97.44\%$의 최고 정확도를 기록하며 인간이 설계한 baseline 모델들보다 안정적으로 우수한 성능을 보였다.

### 정성적 분석 및 ROC 곡선

ROC 곡선 분석 결과, NoisyDARTS-TC14는 MHAtt-RNN과 유사한 수준의 성능을 보였으며, 파라미터 수를 늘린 TC-ResNet-365K보다 더 뛰어난 탐지 성능을 나타냈다. 이는 단순히 모델의 크기를 키우는 것보다 NAS를 통한 최적 구조 설계가 더 효율적임을 입증한다.

## 🧠 Insights & Discussion

본 논문은 DARTS 계열의 알고리즘이 KWS 모델 탐색 시 skip connection에 과도하게 의존하는 경향이 있음을 발견하였다. 이는 모델의 파라미터 수는 줄여주지만, 결과적으로 정확도를 떨어뜨리는 원인이 된다. FairDARTS와 NoisyDARTS는 이러한 '구조적 붕괴' 현상을 효과적으로 해결함으로써, 경량성과 정확도 사이의 최적의 균형점을 찾을 수 있었다.

다만, 본 연구는 탐색 과정에서 추론 시간(latency)이나 실제 연산량(MACs)을 직접적인 제약 조건으로 넣지 않고 구조적 가중치에만 의존했다는 한계가 있다. 또한, 채널 수의 가변성이나 블록 간의 더 복잡한 연결 구조를 탐색 공간에 포함하지 않았다. 향후 이러한 요소들을 탐색 목표에 통합한다면 더욱 하드웨어 친화적인 모델을 찾을 수 있을 것이다.

## 📌 TL;DR

본 논문은 TC-ResNet과 SE 모듈을 결합한 탐색 공간 내에서 NoisyDARTS 알고리즘을 적용하여 KWS를 위한 최적의 초경량 신경망 구조를 자동으로 찾아냈다. 제안된 NoisyDARTS-TC14 모델은 기존 NAS 기반 KWS 모델 대비 파라미터 수를 약 8배 줄이면서도 state-of-the-art 수준의 정확도를 달성하였으며, 이는 저전력 IoT 기기으로의 배포 가능성을 크게 높인 결과이다.
