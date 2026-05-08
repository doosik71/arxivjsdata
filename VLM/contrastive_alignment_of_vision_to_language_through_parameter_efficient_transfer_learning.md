# Contrastive Alignment of Vision to Language through Parameter-Efficient Transfer Learning

Zaid Khan, Yun Fu (2023)

## 🧩 Problem to Solve

본 논문은 CLIP과 같은 contrastive vision-language 모델을 구축할 때 발생하는 막대한 계산 비용과 데이터 의존성 문제를 해결하고자 한다. 기존의 방식은 시각 모델(vision model)과 언어 모델(language model)의 모든 파라미터를 대규모 데이터셋을 통해 함께 업데이트하는 전면 학습(full-model training) 방식을 취한다. 이는 에너지 소모가 매우 크며, 특히 고품질의 이미지-텍스트 쌍(image-text pairs) 데이터가 부족한 저자원 언어(low-resource languages)나 전문 도메인에서는 적용하기 어렵다는 한계가 있다.

따라서 본 연구의 목표는 이미 학습된 강력한 단일 모달(unimodal) 모델들을 기반으로, 매우 적은 수의 파라미터만 업데이트하여 두 모달리티의 표현 공간을 정렬(alignment)할 수 있는 파라미터 효율적 전이 학습(parameter-efficient transfer learning)의 가능성과 효용성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **LilT (Locked image-language tuning)**라고 명명된 방법론으로, 시각 및 언어 인코더의 대부분을 고정(lock)한 채 일부 핵심 파라미터만 선택적으로 업데이트하거나 작은 학습 가능 모듈을 삽입하여 정렬을 달성하는 것이다.

주요 기여 사항은 다음과 같다.

- 전체 파라미터의 7% 미만만 업데이트해도 전면 학습과 대등한 성능을 낼 수 있음을 보였다.
- 특히 1% 미만의 파라미터 업데이트만으로도 전면 학습 성능의 75% 수준에 도달할 수 있음을 입증하였다.
- 파라미터 효율적 학습이 기존 단일 모달 모델이 가진 지식을 더 잘 보존하며, 이는 다국어 제로샷 검색(multilingual zero-shot retrieval)과 같은 실제 시나리오에서 더 유리하게 작용함을 보였다.
- 고정된 컴퓨팅 예산 내에서 더 큰 모델을 학습시킬 수 있게 함으로써 에너지 효율적인 학습 전략을 제시하였다.

## 📎 Related Works

기존의 vision-language 정렬 연구는 크게 두 가지 방향으로 나뉜다. 첫째는 CLIP이나 ALIGN과 같이 수억 개의 이미지-텍스트 쌍을 사용하는 대규모 contrastive 학습 방식이다. 둘째는 frozen language model을 활용하여 이미지 이해를 수행하는 방식으로, 이는 주로 비대조적(non-contrastive) 방식이거나 이미 정렬된 시각 표현이 존재한다는 가정을 전제로 한다. 따라서 이러한 방식들은 지연 시간에 민감한 신경망 검색(neural search) 애플리케이션에 적용하기 어렵다.

본 연구는 PEFT(Parameter-Efficient Fine-Tuning) 기법인 adapters, BitFit 등을 contrastive alignment라는 특수한 목적에 적용했다는 점에서 기존 접근 방식과 차별화된다. 특히, 기존의 LiT(Locked-image Text tuning)가 시각 모델을 고정하고 텍스트 모델만 학습시켰다면, LilT는 두 모델 모두를 효율적으로 튜닝하여 정렬한다는 점에서 진전된 형태를 띤다.

## 🛠️ Methodology

### 전체 파이프라인

LilT의 기본 구조는 다음과 같은 절차를 따른다.

1. 강력하게 사전 학습된 시각 모델(예: DeiT)과 언어 모델(예: SimCSE)에서 초기화를 수행한다.
2. 두 모델의 모든 파라미터를 고정(lock)한다.
3. 선택적으로 일부 핵심 파라미터를 잠금 해제(unlock)하거나, 학습 가능한 작은 모듈(adapters)을 삽입한다.
4. Contrastive loss를 통해 두 모델의 표현 공간을 정렬한다.

### 학습 목표 및 손실 함수

본 연구는 CLIP 스타일의 two-tower 아키텍처를 사용하며, $\text{InfoNCE}$ 손실 함수를 통해 정렬을 수행한다. 이미지-텍스트 쌍의 배치를 $\{x^I_k, x^T_k\}_{k=1}^b$라고 할 때, 이미지-텍스트 간의 유사도 $s_{I_{k,j}}$는 코사인 유사도로 계산된다.

이미지-투-텍스트 손실 $L_{I_k}$는 다음과 같이 정의된다.
$$L_{I_k}(x^I_k, \{x^T_j\}_{j=1}^b) = -\frac{1}{b} \log \frac{\exp(s_{I_{k,k}})}{\sum_{j} \exp(s_{I_{k,j}})}$$

텍스트-투-이미지 손실 $L_{T_k}$ 역시 대칭적으로 계산되며, 최종 전체 손실 $L$은 두 손실의 평균으로 정의된다.
$$L = \frac{1}{2b} \sum_{k=1}^b (L_{I_k} + L_{T_k})$$

### 파라미터 효율적 모듈

논문에서는 세 가지 주요 전략을 탐색한다.

1. **Layerwise Adapters**: 각 Transformer 레이어의 Layer Normalization 이전에 삽입된다. [Down-sample $\rightarrow$ GELU $\rightarrow$ Up-sample] 구조와 residual connection으로 구성된다.
2. **Deep Adapters**: 기존 인코더 스택의 끝에 새로운 Transformer 레이어를 추가하여 스택을 확장하는 방식이다.
3. **Unlocking Parameters**:
    - **LayerNorm Unlocking**: Layer Normalization의 scale $\gamma$와 bias $\beta$ 파라미터만 학습 가능하게 설정한다.
    - **BitFit**: Transformer 내 모든 모듈의 가산적 편향(additive bias) 항만 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋**: COCO2014 (학습 및 테스트), ImageNetV2 (제로샷 분류), Flickr30k (제로샷 검색).
- **기준선**: 전면 학습된 CLIP, LiT 등.
- **평가 지표**: Top-1/5 Accuracy (분류), Rank-1/5/10 (검색).

### 주요 결과

- **성능 달성**: Layerwise adapters를 사용한 $\text{LilT}_{\text{LwA}}$는 전체 파라미터의 약 7.01%만 업데이트하고도 전면 학습된 CLIP(100% 학습)과 대등하거나 오히려 상회하는 검색 성능을 보였다.
- **지식 보존 및 다국어 성능**: 영어-이미지 쌍으로만 정렬 학습을 진행한 후, 학습되지 않은 다른 언어(러시아어, 한국어 등)에 대해 평가했을 때 LilT가 CLIP보다 훨씬 높은 성능을 보였다. 이는 전면 학습보다 파라미터 효율적 학습이 사전 학습된 모델의 다국어 지식을 더 잘 보존함을 의미한다.
- **확장성(Scaling)**: 데이터셋 크기를 1.5M 쌍으로 늘리거나 모델 크기를 base에서 large로 확장했을 때, LilT 역시 성능이 지속적으로 향상됨을 확인하여 확장성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 단순히 효율성을 높인 것이 아니라, 전면 학습이 오히려 사전 학습된 모델의 유용한 지식을 파괴하는 '망각(forgetting)' 현상을 일으킬 수 있음을 시사한다. 특히 다국어 검색 실험 결과는 PEFT가 단순한 계산량 절감을 넘어 일반화 성능 향상에 기여함을 보여준다.

### 정렬 과정의 비대칭성

논문은 Layer Normalization 파라미터의 변화량을 분석하여 흥미로운 발견을 하였다. 텍스트 인코더는 얕은 층(shallow layers)의 파라미터가 정렬 과정에서 많이 변하는 반면, 시각 인코더는 깊은 층(deep layers)의 파라미터가 더 많이 변하는 경향을 보인다. 이는 텍스트의 단순한 개념이 시각적으로는 복잡한 전역적 맥락(global context)을 필요로 하는 모달리티 간의 비대칭성에서 기인한 것으로 해석된다.

### 한계 및 비판적 해석

대부분의 실험이 COCO 및 ImageNetV2와 같은 특정 데이터셋에 집중되어 있어, 다른 도메인이나 더 거대한 데이터셋에서도 동일한 경향이 나타날지는 추가 검증이 필요하다. 또한, 제로샷 분류와 검색 작업에 초점을 맞추었기에, VQA(Visual Question Answering)와 같은 복잡한 추론 작업에서도 동일한 효율성이 유지될지는 명시되지 않았다.

## 📌 TL;DR

본 논문은 사전 학습된 시각-언어 모델의 극소수 파라미터($<7\%$)만 업데이트하여 정렬하는 **LilT** 방법론을 제안한다. 실험을 통해 이 방법이 전면 학습(full-model training)과 대등한 성능을 내면서도 계산 비용을 획기적으로 줄이며, 특히 사전 학습된 모델의 지식을 더 잘 보존하여 다국어 제로샷 환경에서 더 우수한 성능을 보임을 입증하였다. 이는 향후 저자원 언어의 시각-언어 모델 구축이나 에너지 효율적인 대형 모델 학습에 중요한 기반이 될 것으로 보인다.
