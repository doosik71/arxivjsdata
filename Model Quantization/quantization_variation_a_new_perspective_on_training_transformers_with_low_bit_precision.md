# Quantization Variation: A New Perspective on Training Transformers with Low-Bit Precision

Xijie Huang, Zhiqiang Shen, Pingcheng Dong, Tim Kwang-Ting CHENG (2024)

## 🧩 Problem to Solve

본 논문은 Transformer 기반 모델을 극단적인 저비트(extreme low-bit, 예: 2-bit 또는 Binary) 정밀도로 양자화할 때 발생하는 성능 저하와 학습 효율성 문제를 해결하고자 한다. Transformer는 뛰어난 성능을 보여주지만, 방대한 파라미터 수로 인해 계산 비용과 메모리 사용량이 매우 크며, 이는 하드웨어 배포의 큰 장애물이 된다.

기존의 모델 압축 기술 중 Quantization은 ConvNet에서는 성공적으로 적용되었으나, Transformer에서는 특유의 구조적 특성으로 인해 저비트 양자화 시 정확도가 급격히 떨어진다. 또한, 기존의 Quantization-Aware Training (QAT) 방법들은 학습 시간이 지나치게 오래 걸려 효율성이 낮다는 문제가 있다. 따라서 본 연구의 목표는 Transformer의 저비트 양자화를 방해하는 근본적인 원인을 분석하고, 이를 해결하여 성능과 학습 효율을 동시에 개선하는 양자화 스킴을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 저비트 양자화가 어려운 이유를 **'Variation(변동성)'**이라는 관점에서 정의하고 이를 완화하는 것이다. 저자들은 Variation을 다음 세 가지 계층의 분석으로 정의한다.

1. **모듈별 양자화 민감도 차이**: 모델 내의 서로 다른 모듈(예: MHSA vs FFN)과 심지어 동일 모듈 내의 서로 다른 Head들이 양자화에 반응하는 민감도가 서로 다르다.
2. **분포의 이상치(Outliers)**: Transformer의 가중치와 활성화 값(Activation) 분포는 ConvNet에 비해 변동성이 훨씬 크고 이상치가 빈번하게 나타난다.
3. **가중치 진동(Weight Oscillation)**: QAT 과정에서 latent weight가 양자화 경계(boundary) 근처에서 계속해서 왔다 갔다 하며 진동하는 현상이 발생하여 수렴을 방해한다.

이러한 분석을 바탕으로, 모듈별 맞춤형 양자화, 변동성을 고려한 지식 증류(Knowledge Distillation), 그리고 진동 억제 정규화라는 세 가지 해결책을 제시한다.

## 📎 Related Works

기존의 Transformer 양자화 연구는 주로 Post-Training Quantization (PTQ)에 집중되어 왔으며, 이는 재학습 없이 가중치를 직접 양자화하는 방식이다. 그러나 PTQ는 6-bit나 8-bit 수준에서는 유효하지만, 그보다 낮은 저비트에서는 성능 하락이 심각하여 한계가 있다.

QAT 방식의 경우, ConvNet에서 사용되던 LSQ(Learned Step size Quantization) 등의 방법이 제안되었으나, Transformer에 직접 적용했을 때 성능 저하가 더 크게 나타났다. 또한, 일부 mixed-precision quantization (MPQ) 연구들은 모듈마다 다른 비트 수를 할당하여 성능을 높이려 했으나, 이는 하드웨어 구현 복잡도를 크게 증가시키고 최적의 비트 할당 조합을 찾는 데 많은 시간이 소요되는 단점이 있다.

## 🛠️ Methodology

본 논문은 앞서 정의한 세 가지 Variation을 해결하기 위해 **Variation-aware Quantization Scheme**을 제안한다.

### 1. Module-dependent Quantization (MDQ)

모듈마다 양자화 민감도가 다르다는 점에 착안하여, 단순한 레이어 단위(layer-wise) 양자화가 아닌 모듈 단위(예: MHSA의 각 head별)로 세밀한 스케일 팩터($s$)를 학습한다.
특히, 분포의 이상치로 인해 발생하는 그래디언트 불균형을 해결하기 위해 가중치의 크기에 민감한 **Gradient Scaling** 기법을 도입한다. 스케일 팩터의 그래디언트를 다음과 같이 수정한다.

$$\frac{\partial L}{\partial s} \leftarrow \frac{\partial L}{\partial s} \cdot \frac{1}{\sqrt{Q^P}||w||_1}$$

여기서 $||w||_1$은 해당 모듈 가중치의 $L1$-norm이다. 이를 통해 가중치 변동성이 큰 모듈에서 스케일 팩터의 업데이트가 과도하게 일어나는 것을 방지하여 안정성을 높인다.

### 2. Multi-crop Knowledge Distillation (MCKD)

학습 안정성을 높이고 수렴 속도를 앞당기기 위해 Full-precision 모델을 교사(Teacher)로 사용하는 지식 증류(KD)를 적용한다. 특히 Vision Transformer (ViT)의 학습 효율을 높이기 위해 Multi-crop KD 방식을 제안한다.

- 이미지 한 장에서 $M$개의 무작위 영역을 크롭(crop)하여 교사 모델의 소프트 라벨(soft label)을 미리 생성하고 저장한다.
- 학습 단계에서는 저장된 라벨을 직접 사용함으로써, 매 반복마다 교사 모델을 추론해야 하는 계산 비용을 제거하여 학습 시간을 획기적으로 단축한다.

### 3. Oscillation-aware Bin Regularization (OBR)

가중치가 양자화 경계에서 진동하는 현상을 막기 위해, 가중치를 양자화 빈(bin)의 중심점으로 유도하는 정규화 항을 추가한다.

$$\mathcal{L}_{OBR} = \sum_{m=1}^{M} \left( ||w^r_m - w^q_m||_2^2 + \sum_{n=1}^{2^b} V(w^r_{n,m}) \right)$$

여기서 $w^r$은 실제 값, $w^q$는 양자화된 값이며, $V(\cdot)$는 각 양자화 빈 내부의 가중치 분산(variance)을 의미한다. 즉, 가중치가 빈 중심에 가깝게 모이도록 하여 진동을 억제한다. 최종 손실 함수는 다음과 같다.

$$\mathcal{L} = \mathcal{L}_{KD} + \lambda \mathcal{L}_{OBR}$$

$\lambda$ 계수는 학습 초기 단계의 최적화를 방해하지 않도록 Cosine annealing 스케줄에 따라 점진적으로 증가시킨다.

## 📊 Results

### 실험 설정

- **데이터셋 및 모델**: ImageNet-1K (DeiT-T, SReT-T, Swin-T/S), GLUE benchmark (BERT-base).
- **비교 대상**: LSQ+, Q-ViT, BiT 등 기존 SOTA 양자화 방법론.
- **지표**: Top-1 Accuracy (Vision), Average Accuracy (NLP), 학습 시간, 하드웨어 면적 및 전력 소모.

### 주요 결과

1. **정확도 향상**:
    - **Vision**: 2-bit Swin-T 모델에서 기존 SOTA 대비 3.35%의 성능 향상을 달성하였다. 특히 2-bit 및 3-bit와 같은 초저비트 설정에서 baseline(LSQ+) 대비 압도적인 성능 우위를 보였다.
    - **NLP**: Binary(1-bit) BERT-base 모델이 GLUE 벤치마크에서 기존 SOTA(BiT) 대비 1.4% 높은 평균 정확도(74.9%)를 기록하였다.
2. **학습 효율성**: Multi-crop KD와 최적화된 양자화 스킴 덕분에 학습 에포크(Epoch)를 300에서 150으로 줄였음에도 더 높은 성능을 냈으며, 전체 학습 시간을 약 50% 단축하였다.
3. **하드웨어 비용**: Mixed-precision quantization (MPQ) 방식과 비교했을 때, 제안 방법은 단일 정밀도를 유지하면서 모듈별 스케일링만 수행하므로 하드웨어 구현 면적(Area)과 전력 소모(Power)가 훨씬 낮음을 확인하였다 (Table 8 참조).

## 🧠 Insights & Discussion

본 논문은 Transformer 양자화의 난제가 단순히 '비트 수의 부족'이 아니라, 모델 내부의 **계층적 변동성(Variation)**에 있음을 정량적으로 분석해냈다는 점에서 학술적 가치가 크다. 특히, ConvNet에서는 관찰되지 않는 '가중치 진동(Weight Oscillation)' 현상을 포착하고 이를 해결하기 위한 Bin Regularization을 제안한 점이 독창적이다.

또한, 단순히 정확도만 높인 것이 아니라, 하드웨어 구현 관점에서의 비용(Area, Power)과 학습 시간(Training Time)을 함께 분석함으로써 실용적인 배포 가능성을 입증하였다. Mixed-precision 방식이 정확도 면에서는 유리할 수 있으나, 실제 하드웨어 칩 설계 시 발생하는 오버헤드를 고려하면 본 논문이 제안한 Module-dependent scaling 방식이 훨씬 효율적인 대안이 될 수 있음을 시사한다.

다만, 제안된 방법론이 다양한 비트 설정(2, 3, 4-bit)에서 우수함을 보였으나, 8-bit 이상의 고정밀도 양자화에서는 OBR 정규화가 오히려 최적화를 방해할 수 있다는 점이 언급되었다. 이는 정규화의 강도 $\lambda$에 대한 정밀한 튜닝이 필요함을 의미한다.

## 📌 TL;DR

이 논문은 Transformer의 저비트 양자화가 어려운 이유를 **모듈별 민감도, 분포 이상치, 가중치 진동**이라는 세 가지 **'Variation'** 관점에서 분석하고, 이를 해결하기 위해 **모듈별 스케일 학습(MDQ), Multi-crop 지식 증류(MCKD), 빈 중심 정규화(OBR)**를 제안한다. 이를 통해 2-bit Swin-T 및 Binary BERT 등에서 SOTA 성능을 달성함과 동시에 학습 시간을 50% 단축하고 하드웨어 효율성을 극대화하였다. 저비트 Transformer 모델의 실용적인 배포를 위한 핵심적인 가이드라인을 제시한 연구이다.
