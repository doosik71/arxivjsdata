# SMALL-FOOTPRINT SLIMMABLE NETWORKS FOR KEYWORD SPOTTING

Zuhaib Akhtar, Mohammad Omar Khursheed, Dongsu Du, Yuzong Liu (2023)

## 🧩 Problem to Solve

본 논문은 소형 풋프린트 기반의 Keyword Spotting (KWS, 핵심어 검출) 모델을 설계할 때 발생하는 자원 제약과 모델 개발 비용 문제를 해결하고자 한다. KWS는 음성 스트림에서 특정 핵심어(wake-word)의 존재 여부를 식별하는 이진 분류 문제로, 아마존의 Alexa나 애플의 Siri와 같은 음성 비서 시스템의 첫 번째 상호작용 단계로서 매우 중요하다.

최근 음성 비서 서비스가 이어버드, 모바일 기기, 스마트 스피커 등 다양한 하드웨어 플랫폼으로 확장됨에 따라, 각 기기가 가진 서로 다른 메모리 및 CPU 연산 능력(resource constraints)에 맞춘 최적의 모델을 배포해야 할 필요성이 커졌다. 그러나 각 기기의 자원 예산에 맞춰 정확도와 효율성 사이의 트레이드오프를 고려한 개별 모델들을 일일이 설계하고 학습시키는 것은 막대한 계산 비용과 연구 시간을 소모하게 만든다. 따라서 본 연구의 목표는 하나의 초거대 네트워크(super-network)를 학습시켜, 다양한 자원 제약 조건에 맞는 여러 개의 하위 네트워크(sub-networks)를 효율적으로 추출할 수 있는 Slimmable Neural Networks를 소형 KWS 모델에 적용하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 weight-sharing(가중치 공유)과 switchable normalization을 통해 단일 모델 내에서 가변적인 너비(width)를 가질 수 있는 Slimmable 구조를 설계하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **소형 풋프린트 KWS를 위한 Slimmable CNN 설계**: 파라미터 수를 250k 개 미만으로 제한한 초경량 CNN 구조에 slimmable 아키텍처를 적용하였다.
2.  **Slimmable Transformer 확장**: 기존의 CNN 기반 slimmable 네트워크를 넘어, Transformer 아키텍처의 self-attention 모듈을 너비에 따라 조절할 수 있도록 확장 설계하였다.
3.  **실용성 검증**: 대규모 내부 음성 비서 데이터셋과 공개 데이터셋인 Google Speech Commands를 통해, 제안 방법이 매우 제한적인 자원 환경(10k-250k 파라미터)에서도 효과적임을 입증하였다.

## 📎 Related Works

KWS 모델들은 일반적으로 지연 시간을 최소화하기 위해 소형 딥러닝 모델을 사용하며, 이를 위해 다양한 모델 압축 기술이 연구되어 왔다. 특히 하나의 super-network에서 여러 sub-network를 유도하는 접근 방식인 Slimmable Networks, Dynamic Neural Networks, Once-for-All (OFA) 네트워크 등이 제안되었다.

음성 처리 분야에서는 자동 음성 인식(ASR)을 위해 Dynamic Sparsity Neural Network(DSNN)나 Omni-sparsity DNN와 같이 프루닝 마스크(pruning masks)를 활용해 하위 네트워크를 샘플링하는 방식이 연구되었다. 또한 HuBERT 모델과 같은 고비용 모델에 OFA를 적용해 최적의 하위 네트워크를 찾는 시도도 있었다. 그러나 기존 연구들은 대개 모델의 규모가 어느 정도 큰 경우에 집중되어 있었으며, 본 논문은 매우 작은 풋프린트(small-footprint) 설정에서 CNN과 Transformer 모두에 slimmable 구조를 적용했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 학습 절차 (Slimmable Training)

본 논문에서 제안하는 학습 과정은 특정 너비 목록(`widthlist`, 예: $[1.0, 0.75, 0.5, 0.25]$)을 정의하고, 학습 단계에서 이 목록에 포함된 각 너비의 모델을 순차적으로 실행하는 방식을 따른다.

1.  데이터 배치를 가져와 정의된 `widthlist`의 각 너비별로 모델을 실행하여 예측값을 얻는다.
2.  각 너비별로 손실 함수(loss function)를 계산하고 그에 따른 그래디언트(gradient)를 구한다.
3.  이때 가중치는 즉시 업데이트하지 않고 모든 너비에 대한 그래디언트를 먼저 저장한다.
4.  모든 너비에 대해 계산된 그래디언트의 합($\sum \text{gradient}$)을 구하여 전체 네트워크의 가중치를 한 번에 업데이트한다.

### Slimming CNNs

CNN의 slimming은 서로 다른 너비 간에 가중치를 공유하는 방식으로 이루어진다. 
- **커널 조절**: 네트워크 너비가 변경될 때, 설정된 너비에 따라 적절한 수의 커널을 드롭(drop)하거나 커널의 너비를 줄여 모델의 크기를 조절한다.
- **Switchable BatchNorm**: 너비마다 서로 다른 통계적 특성을 가질 수 있으므로, 각 너비별로 별도의 BatchNorm 레이어를 할당하여 런타임에 선택적으로 사용한다.

### Slimming Transformers

Transformer의 경우 모든 Dense 레이어를 slimmable하게 설계하여 각 Transformer 블록 내에서 차원을 조절한다.
- **Dense 레이어 조절**: 입력 차원을 너비에 따라 결정된 하위 차원으로 투영(project)한다. 하위 네트워크의 너비에 따라 가중치의 일부를 끄는(switch off) 방식을 사용한다.
- **Slimmable Attention**: Query($Q$), Key($K$), Value($V$) 텐서의 크기를 너비에 따라 조절한다. 이를 통해 하위 네트워크는 더 낮은 차원에서 dot-product 연산을 수행하게 되어 계산 및 메모리 효율성을 높인다.
- **수식 표현**: Slimmable Attention의 동작은 다음과 같이 정의된다.
$$ \text{SlimAttn}(Q, K, V) = \text{softmax}\left(\frac{Q[:i]K[:i]^T}{\sqrt{d_k[:i]}}\right)V[:i] $$
여기서 인덱스 $i$는 너비 목록에 따라 다음과 같이 결정된다.
$$ i \in [d_k \times \text{width}[0], d_k \times \text{width}[1], \dots, d_k \times \text{width}[n]] $$
$$ \text{width} \in [1, (n-1)/n, (n-2)/n, \dots, (1/n)], n > 1 (\text{integer}) $$
- **Switchable LayerNorm**: CNN의 BatchNorm과 마찬가지로, 각 너비에 대응하는 별도의 LayerNorm 레이어를 사용하여 런타임에 전환한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 내부 음성 비서 데이터셋(단일 키워드 검출) 및 Google Speech Commands(35개 클래스 분류).
- **입력 데이터**: Log Mel Filter Bank Energies (LFBEs)를 사용하였다.
- **모델 규모**: CNN은 최대 199k 파라미터, Transformer는 67k~120k 파라미터 수준의 소형 모델을 사용하였다.
- **평가 지표**: 내부 데이터셋은 고정된 Miss Rate에서의 Relative False Accepts를, Google 데이터셋은 Accuracy를 측정하였다.

### 주요 결과
1.  **성능 비교**: 
    - 내부 데이터셋에서는 Slimmable 모델이 동일 크기의 Scratch 모델(처음부터 따로 학습한 모델)보다 약간 낮은 성능을 보였으나, 가장 작은 너비(0.25)에서는 오히려 Scratch 모델을 능가하는 모습을 보였다.
    - Google Speech Commands 데이터셋에서는 대부분의 경우 Slimmable 모델이 Scratch 모델보다 높은 정확도를 기록하였다. 이는 가중치 공유를 통한 일종의 내재적 증류(inherent distillation) 효과 때문으로 분석된다.
2.  **학습 효율성**:
    - 단일 너비 모델을 여러 개 학습시키는 것보다 super-network 하나를 학습시키는 것이 훨씬 효율적이다.
    - 너비의 수를 1개에서 40개로 늘렸을 때, 학습 시간은 단지 3.72배 증가하는 데 그쳤다. 이는 개별 모델을 모두 학습시키는 것보다 10배 이상 빠른 속도이다.
3.  **자원 효율성**: 너비가 줄어듦에 따라 파라미터 수와 연산량(Multiplies)이 선형적으로 감소함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 Slimmable Neural Networks가 매우 작은 풋프린트의 KWS 작업에서도 실용적임을 보여주었다. 특히, 가중치 공유 구조가 하위 네트워크에 일종의 정규화 또는 증류 효과를 제공하여, 일부 케이스에서는 개별 학습 모델보다 더 나은 성능을 낼 수 있다는 점이 인상적이다.

또한, 학습 시간의 비선형적 증가 특성은 실제 산업 현장에서 매우 중요한 이점을 제공한다. 다양한 하드웨어 타겟(스마트폰, 이어버드, TV 등)에 맞춰 여러 버전의 모델을 배포해야 하는 상황에서, 하나의 super-network만 학습시키면 다양한 예산의 sub-networks를 즉시 추출할 수 있기 때문이다.

**한계 및 향후 과제**:
- 본 논문은 주로 네트워크의 '너비(width)' 조절에 집중하였으며, '깊이(depth)'를 조절하는 slimming에 대해서는 다루지 않았다.
- 다양한 엣지 컴퓨팅 칩셋에서의 실제 메모리 점유율과 연산 속도에 대한 프로파일링 데이터가 부족하여, 이론적인 파라미터 수 감소가 실제 하드웨어 성능 향상으로 얼마나 직결되는지에 대한 추가 분석이 필요하다.
- RNN과 같은 다른 아키텍처로의 확장 가능성이 언급되었으나 실제 구현 결과는 제시되지 않았다.

## 📌 TL;DR

본 논문은 매우 작은 메모리 제약(< 250k parameters)을 가진 Keyword Spotting 환경에서 CNN과 Transformer를 모두 지원하는 **Slimmable Neural Networks**를 제안하였다. 단일 super-network를 학습시켜 다양한 크기의 sub-networks를 추출함으로써, **학습 비용을 획기적으로 줄이면서도(최대 10배 이상) 개별 학습 모델에 근접하거나 오히려 능가하는 성능**을 달성하였다. 이 연구는 다양한 하드웨어 사양을 가진 엣지 기기들에 효율적으로 모델을 배포해야 하는 실무 환경에 매우 유용한 방법론을 제시한다.