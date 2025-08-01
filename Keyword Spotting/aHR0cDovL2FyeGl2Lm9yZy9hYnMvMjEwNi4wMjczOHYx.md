# Encoder-Decoder Neural Architecture Optimization for Keyword Spotting
Tong Mo, Bang Liu

## 🧩 Problem to Solve
키워드 스포팅(KWS)은 특정 키워드 음성 발화를 식별하는 것을 목표로 합니다. 최근 딥 컨볼루션 신경망(CNN)이 KWS 시스템에 널리 활용되고 있지만, 이들의 모델 아키텍처는 주로 VGG-Net 또는 ResNet과 같은 범용 백본에 기반하고 있어 KWS 작업에 특화되어 설계되지 않았습니다. 이로 인해 특히 리소스 제약이 있는 스마트 장치에 배포될 때 최적의 성능을 달성하지 못할 수 있습니다. 또한, 기존의 신경망 아키텍처 탐색(NAS) 방법, 예를 들어 DARTS 기반의 KWS 연구들은 최적의 아키텍처를 평가할 때 편향을 유발할 수 있는 가정을 사용합니다.

## ✨ Key Contributions
*   KWS를 위한 CNN 아키텍처를 자동으로 설계하기 위해 인코더-디코더 신경망 아키텍처 최적화(Neural Architecture Optimization, NAO) 방법을 제안합니다.
*   NAO는 특정 탐색 공간 내에서 모델 연산자와 그 연결을 효과적으로 탐색합니다.
*   NAO의 디코더는 이산 모델 아키텍처를 정확하게 복구하여, 기존 NAS 방법보다 아키텍처 선택에서 더 정확한 결정을 내릴 수 있도록 합니다.
*   Google Speech Commands Dataset (버전 1 및 버전 2)에 대한 광범위한 평가를 통해 97% 이상의 최첨단 정확도를 달성하면서도 허용 가능한 메모리 사용량을 유지합니다.
*   제안된 방법으로 탐색된 아키텍처는 기존의 수동으로 설계된 모델 및 다른 NAS 기반 KWS 모델들(NAS2, NoisyDARTS-TC14)을 능가하는 성능을 보입니다.

## 📎 Related Works
*   **CNN 기반 KWS:** Sainath et al. [1]은 KWS에 CNN을 도입했습니다. 이후 ResNet [2], 분리형 CNN [3, 4, 5, 6], 시간적 CNN [7], SincNet [8]과 같은 다양한 범용 CNN 백본이 KWS에 적용되었습니다. RNN [9], BiLSTM [10], 스트리밍 레이어 [11]와 같은 다른 딥러닝 모델과의 결합을 통해 성능 향상 노력도 있었습니다.
*   **신경망 아키텍처 탐색(NAS):** Zoph et al. [12]은 강화 학습을 사용하여 CIFAR-10용 신경망 아키텍처를 탐색하는 데 처음 적용했습니다. 이후 검색 공간을 줄이는 향상된 NAS 모델들 [13, 14, 15]과 미분 가능한 아키텍처 탐색(DARTS) [16, 17, 18] 방법들이 제안되었습니다.
*   **KWS를 위한 NAS:** 최근 연구 [19, 20]에서는 DARTS [16]를 사용하여 KWS 모델 아키텍처를 설계했습니다. 그러나 이러한 방법들은 최적의 아키텍처 평가에 편향이 있을 수 있습니다.

## 🛠️ Methodology
본 논문에서는 KWS를 위한 CNN 모델 설계를 위해 인코더-디코더 신경망 아키텍처 최적화(NAO)를 활용합니다.
*   **CNN 구조:** KWS용으로 탐색되는 CNN은 $L$개의 셀 스택으로 구성되며, 그 뒤에 분류를 수행하는 스템이 이어집니다. 음성 발화는 MFCC 특징으로 전처리됩니다.
*   **셀 유형:** 일반 셀(normal cells)과 축소 셀(reduction cells) 두 가지 유형의 셀을 탐색합니다. 각 셀은 두 개의 입력 노드, 하나의 출력 노드, 그리고 여러 개의 중간 노드를 가집니다.
*   **NAO 프레임워크:** NAO는 세 가지 주요 구성 요소로 이루어져 있습니다:
    *   **인코더 ($E$):** 셀의 신경망 아키텍처 $a$를 문자열 시퀀스 $x$로 변환한 다음, 이를 연속 표현 $e_{x} = E(x)$로 변환합니다. 인코더는 단일 계층 Long Short-Term Memory (LSTM)로 구현됩니다.
    *   **성능 예측기 ($f$):** 연속 표현 $e_{x}$를 검증 데이터셋에 대한 예측 성능 $s_{x}$로 매핑합니다. 평균 풀링 계층과 피드 포워드 네트워크로 구성되며, 다음 손실을 최소화하여 훈련됩니다:
        $$ \min L_{pred} = \sum_{x \in X} (s_{x} - f(E(x)))^{2} $$
    *   **디코더 ($D$):** 연속 표현 $e_{x}$를 아키텍처 $a$에 해당하는 문자열 시퀀스 $x$로 다시 매핑합니다. LSTM 모델에 어텐션 메커니즘 [24]이 적용되며, 다음 손실을 최소화하여 훈련됩니다:
        $$ \min L_{rec} = - \sum_{x \in X} \log P_{D}(x|E(x)) $$
*   **공동 훈련:** 인코더, 성능 예측기, 디코더는 다음 가중치 합 손실을 최소화하여 함께 훈련됩니다:
    $$ \min L = \lambda L_{pred} + (1-\lambda)L_{rec} $$
    여기서 $\lambda$는 0과 1 사이의 하이퍼파라미터입니다.
*   **아키텍처 형성:** 탐색이 끝난 후, 가장 높은 검증 정확도를 가진 일반 셀 및 축소 셀 아키텍처가 반환됩니다. 최종 CNN 아키텍처는 네트워크 총 깊이의 1/3 및 2/3 지점에 축소 셀을 배치하여 구성됩니다.
*   **탐색 설정:** 인코더와 디코더의 LSTM은 특정 크기의 임베딩 및 은닉 상태를 가지며, Adam 최적화 알고리즘을 사용하여 1000 에포크 동안 훈련됩니다. 탐색은 N=3 반복으로 진행되며, 각 셀 내 중간 노드 수는 5개로 설정됩니다. 탐색 공간은 identity, $3 \times 3$ 분리형 합성곱, $5 \times 5$ 분리형 합성곱, $3 \times 3$ 평균 풀링, $3 \times 3$ 최대 풀링과 같은 다섯 가지 간단한 연산으로만 구성됩니다.

## 📊 Results
*   **Google Speech Commands Dataset (Version 1 및 Version 2)에서 평가:**
    *   **탐색 비용:** NVIDIA Tesla V100 GPU에서 Version 1은 8 GPU일, Version 2는 10 GPU일 소요되었습니다.
    *   **Version 1 결과:** NAO1 모델은 12개 초기 채널에서 97.01% 정확도 (278K 파라미터), 16개 초기 채널에서 **97.28% 정확도 (최첨단)** (469K 파라미터)를 달성했습니다. 기존 Res15, Attention RNN, TC-ResNet, SincConv+DSConv, DenseNet-BiLSTM 모델들을 능가했습니다. 특히 NAS2 [19]보다 더 높은 정확도와 더 적은 파라미터 수를 보였습니다.
    *   **Version 2 결과:** NAO2 모델은 12개 초기 채널에서 97.50% 정확도 (274K 파라미터), 16개 초기 채널에서 **97.92% 정확도 (최첨단)** (458K 파라미터)를 달성했습니다. Attention RNN, DenseNet-BiLSTM, NoisyDARTS-TC14 [20]를 포함한 모든 기준 모델보다 뛰어난 성능을 보였습니다.
*   **기존 NAS 기반 모델과의 비교:** NAO1과 NAO2는 NAS2 및 NoisyDARTS-TC14와 비교하여 향상된 탐색 능력을 보여주었습니다. NoisyDARTS-TC14보다 모델 크기가 약간 크지만, 이는 NAO가 특정 고성능 블록에 의존하지 않고 일반적인 연산들로만 탐색 공간을 구성했기 때문입니다.

## 🧠 Insights & Discussion
*   NAO의 인코더-디코더 최적화 프레임워크는 KWS에 최적화된 CNN 아키텍처를 발견하는 데 있어 탁월한 탐색 능력을 제공합니다.
*   디코더가 이산 모델 아키텍처를 정확하게 복구하는 기능은 탐색 과정에서 더욱 정확한 결정을 내릴 수 있도록 하여, DARTS와 같은 기존 방법의 편향을 극복합니다.
*   본 연구에서 발견된 아키텍처는 수동으로 설계된 모델뿐만 아니라 기존의 NAS 기반 KWS 모델들보다 우수한 최첨단 정확도를 달성하며, 리소스 제약이 있는 KWS 작업에 대한 효과를 입증했습니다.
*   비록 일부 베이스라인(예: NoisyDARTS-TC14)보다 모델 크기가 약간 더 크지만, 이는 NAO가 특정 최적화된 소형 블록(예: [7]에서 설계된 블록)에 의존하지 않고 보편적인 연산자만으로 검색 공간을 구성했기 때문입니다. 이는 NAO의 일반화 가능성과 순수한 아키텍처 탐색 능력의 우수성을 보여줍니다.

## 📌 TL;DR
*   **문제:** 기존 KWS 모델은 KWS에 최적화되지 않은 범용 CNN을 사용하며, 기존 NAS 방법은 탐색에 편향이 있어 최적의 성능을 달성하기 어렵습니다.
*   **제안 방법:** 인코더-디코더 기반의 신경망 아키텍처 최적화(NAO)를 사용하여 KWS를 위한 CNN 아키텍처를 자동으로 탐색합니다. NAO는 인코더로 아키텍처를 연속 공간에 매핑하고, 예측기로 성능을 예측하며, 디코더로 이산 아키텍처를 복구하는 방식으로 공동 훈련됩니다.
*   **주요 결과:** Google Speech Commands Dataset (버전 1 및 2)에서 각각 97.28% 및 97.92%의 새로운 최첨단 정확도를 달성했습니다. 이는 수동으로 설계된 모델 및 기존 NAS 기반 KWS 모델들을 능가하며, NAO의 우수한 아키텍처 탐색 능력을 입증합니다.