# Learning Equivariant Segmentation with Instance-Unique Querying

Wenguan Wang, James Liang, Dongfang Liu

## 🧩 Problem to Solve

기존의 최신 쿼리 기반(query-based) 인스턴스 분할(instance segmentation) 방법들은 개별 장면(individual scenes) 내에서만 인스턴스를 구별하도록 인스턴스 인식 임베딩(instance-aware embeddings)을 학습합니다. 이러한 "장면 내(within-scene)" 학습 전략은 각 장면에 있는 객체 인스턴스의 다양성과 복잡성이 제한적이므로, 학습된 쿼리의 식별 능력을 저해합니다. 결과적으로, 모델은 데이터셋 전체 수준에서 인스턴스를 고유하게 식별하는 데 어려움을 겪으며, 기하학적 변환(geometric transformations)에 대한 쿼리-인스턴스 매칭의 견고성(robustness)도 부족합니다.

## ✨ Key Contributions

* **새로운 학습 프레임워크 제안:** 식별력 있는 쿼리 임베딩(discriminative query embeddings) 학습을 통해 쿼리 기반 모델의 성능을 향상시키는 새로운 학습 프레임워크를 개발했습니다.
* **데이터셋 수준의 고유성 학습 (Instance-Unique Querying):** 쿼리가 개별 장면 내에서만 검색하는 대신 전체 학습 데이터셋에서 해당 인스턴스를 검색하도록 강제합니다. 이는 모델이 더 식별력 있는 쿼리를 학습하도록 하여 효과적인 인스턴스 분리를 가능하게 합니다.
* **변환 등변성(Transformation Equivariance) 부여:** 이미지(인스턴스) 표현과 쿼리가 기하학적 변환에 대해 등변성을 갖도록 장려하여, 쿼리-인스턴스 매칭의 견고성을 높였습니다. 예를 들어, 이미지를 자르거나 뒤집을 때 표현과 쿼리도 이에 따라 변하도록 합니다.
* **광범위한 적용 가능성 및 성능 향상:** CondInst, SOLOv2, SOTR, Mask2Former 등 4가지 유명 쿼리 기반 모델에 본 학습 알고리즘을 적용하여 COCO 데이터셋에서 +1.6 ~ +3.2 $AP$ (평균 정밀도)의 유의미한 성능 향상을 달성했습니다. LVISv1 데이터셋에서는 SOLOv2의 성능을 +2.7 $AP$ 향상시켰습니다.
* **경량성:** 아키텍처 변경이나 추론 속도 지연 없이 순수하게 학습 전략 개선만으로 성능 향상을 이뤄냈습니다.

## 📎 Related Works

* **인스턴스 분할 패러다임:**
  * **Top-down (detect-then-segment):** Mask R-CNN, Cascade Mask R-CNN, HTC, PointRend, QueryInst, K-Net.
  * **Bottom-up (label-then-cluster):** Associative Embedding.
  * **Single-shot (directly-predict):** YOLACT, CondInst, SOLOv2, SOTR, Mask2Former, SparseInst, SOLQ.
* **쿼리 기반 모델 (Query-based models):** 인스턴스 관련 속성(위치, 외형)을 쿼리 벡터에 인코딩하여 예측 견고성을 높이는 CondInst, SOLOv2 (동적 필터), DETR 기반 모델 (Mask2Former, SOTR, SOLQ) 등이 있습니다.
* **등변 표현 학습 (Equivariant Representation Learning):** CNN의 변환 불변성(translation equivariance)에서 영감을 받아 캡슐 네트워크(Capsule Nets), 그룹 등변 컨볼루션(Group Equivariant Convolutions), 하모닉 네트워크(Harmonic Networks) 등 다양한 변환에 대한 등변 표현 학습 연구가 이루어졌습니다. 불변성(Invariance)은 등변성의 특수한 경우로 간주됩니다.

## 🛠️ Methodology

본 연구는 쿼리-인스턴스 매칭의 두 가지 핵심 속성인 고유성(uniqueness)과 견고성(robustness)을 학습 목표로 통합합니다.

1. **장면 내(Intra-Scene) 및 장면 간(Inter-Scene) 인스턴스 고유성 학습:**
    * **장면 내 마스크 손실 ($L_{intra\_mask}$):** 기존의 $L_{mask}$와 동일하게 현재 이미지 $I$ 내에서 쿼리 $q_n$이 해당 인스턴스 $M_{\sigma(n)}$의 픽셀과 일치하고 다른 인스턴스와 불일치하도록 합니다.
    * **장면 간 마스크 손실 ($L_{inter\_mask}$):** 이미지 $I$에서 생성된 쿼리 $\{q_n\}_{n=1}^N$을 사용하여 *다른 학습 이미지* $I'$ (특징 맵은 $I' = f(I')$)도 쿼리합니다. 이때 생성된 장면 간 예측 마스크 $\{\hat{O}_{I'n}\}_{n=1}^N$에 대한 학습 목표는 크기 $H \times W$의 모든 픽셀이 0인 매칭 행렬 **0**입니다. 이는 쿼리 $q_n$이 다른 이미지 $I'$의 인스턴스와 *불일치*하도록 강제하여 데이터셋 수준에서 인스턴스 고유성을 학습시킵니다.
    * **수식:** $\sum_{n=1}^{N} (L_{intra\_mask}(\hat{M}_n, M_{\sigma(n)}) + \frac{1}{|I|-1} \sum_{I' \neq I} L_{inter\_mask}(\hat{O}_{I'n}, \mathbf{0}))$
    * $L_{inter\_mask}$는 포컬 손실(Focal Loss)로 구현되어 쉬운 부정 샘플(easy negative samples)의 영향을 줄입니다.
    * **효율성 전략:**
        * **외부 메모리:** 대규모 쿼리-인스턴스 매칭을 위해 여러 배치(batch)에서 추출한 약 100K개의 인스턴스 픽셀 임베딩을 저장하는 외부 메모리 큐를 구축합니다 (배경 픽셀은 제외).
        * **희소 샘플링(Sparse Sampling):** 각 이미지에서 소수의 인스턴스 픽셀만 무작위로 샘플링하여 메모리에 저장, 샘플 다양성을 높입니다.
        * **인스턴스 균형 샘플링(Instance-Balanced Sampling):** 작은 인스턴스의 성능 저하를 방지하기 위해 각 인스턴스 영역에서 고정된 수의 픽셀(예: 50개)을 무작위로 샘플링합니다.

2. **변환 등변성 학습:**
    * 입력 이미지 변환 $g$ (예: 수평 뒤집기, 무작위 자르기)에 대해 특징 표현과 쿼리 임베딩이 등변성 속성을 갖도록 모델을 훈련합니다. 즉, $f(g(I)) \approx g(f(I))$ 이고, 변환된 입력 $g(I)$로부터 파생된 쿼리 $q^g_n$이 변환된 인스턴스 표현 $g(I)$와 매칭될 때, 그 결과 예측 마스크 $\hat{W}^g_n$이 변환된 정답 마스크 $g(M_{\sigma(n)})$와 일치하도록 합니다.
    * **수식:** $\sum_{n=1}^{N} L_{equi}(\hat{W}^g_n, g(M_{\sigma(n)}))$, 여기서 $\hat{W}^g_n = \langle q^g_n, g(I) \rangle$입니다.
    * $L_{equi}$는 기본 세그멘터의 마스크 예측 손실과 동일한 형태로 구현됩니다.
    * 두 새로운 학습 목표 ($L_{inter\_mask}$와 $L_{equi}$)의 균형을 위해 $L_{equi}$에 계수 $\lambda$ (실험적으로 3)를 곱합니다.

## 📊 Results

* **COCO test-dev 데이터셋:**
  * CondInst (ResNet-50/-101): +3.1 / +2.8 $AP$ 향상.
  * SOLOv2 (ResNet-50/-101): +3.2 / +2.9 $AP$ 향상.
  * SOTR (ResNet-50/-101): +2.6 / +2.4 $AP$ 향상.
  * Mask2Former (Swin-S/-B/-L): +2.2 / +2.4 / +1.6 $AP$ 향상.
  * Mask2Former (Swin-L)에서 51.8 $AP$를 달성하여 COCO 인스턴스 분할의 새로운 최첨단(state-of-the-art) 기록을 세웠습니다.
  * 모든 객체 크기 ($AP_S, AP_M, AP_L$)에 걸쳐 일관된 성능 향상을 보였습니다.
* **LVISv1 val 데이터셋 (SOLOv2, ResNet-50 기반):**
  * 전반적으로 +2.7 $AP$ 향상.
  * 드문(rare) 클래스 $AP_r$에서 +4.0, 일반(common) 클래스 $AP_c$에서 +1.9, 빈번(frequent) 클래스 $AP_f$에서 +2.1의 향상을 보였습니다.
* **추가 실험 (부록):** SparseInst 및 SOLQ와 같은 다른 쿼리 기반 모델에서도 2.0~$2.3 AP$의 성능 향상을 입증하여 방법론의 일반성을 확인했습니다. 또한 Mask2Former 기반으로 panoptic segmentation task에서도 1.2 $PQ$ (Panoptic Quality)가 향상되었습니다.
* **정성적 결과:** 붐비거나 유사한 인스턴스를 더 잘 구별하고 고품질 분할 마스크를 생성하는 것으로 나타났습니다.
* **학습 속도:** 기존 학습 대비 약 5%의 미미한 지연만 발생합니다.
* **주목할 점:** 이 모든 성능 향상은 네트워크 아키텍처 변경이나 추론 시 추가 계산 비용 없이, 순전히 새로운 학습 전략을 통해 달성되었습니다.

## 🧠 Insights & Discussion

* **의미:** 제안된 프레임워크는 유연하고 강력하여, 기존 및 성장하는 쿼리 기반 인스턴스 분할 방법론 전반에 걸쳐 이점을 제공할 수 있습니다.
* **$L_{inter\_mask}$의 효과:** 데이터셋 전체에서 인스턴스 고유성을 강제함으로써, 기존의 "장면 내" 학습 방식이 겪는 "부정 인스턴스 부족" 문제를 해결하고 식별력 있는 쿼리 학습을 촉진합니다. 포컬 손실과 인스턴스 균형 샘플링은 각각 쉬운 부정 샘플의 영향 감소와 작은 인스턴스 성능 보완에 기여합니다.
* **$L_{equi}$의 효과:** 변환 등변성을 명시적으로 모델링하여 견고한 쿼리-인스턴스 매칭을 가능하게 하며, 기존의 변환 기반 데이터 증강 기법보다 훨씬 효과적입니다. 이는 쿼리가 변환된 인스턴스 패턴을 정확하게 반영하도록 합니다.
* **제한 사항:**
  * 등변 변환(equivariant transformations)이 선형 변환 연산자의 그룹 요소로 제한됩니다. 따라서 뒤집기(flipping) 및 자르기(cropping)와 같은 일반적인 선형 변환만 적용 가능하며, 임의의 광학적 변환(photometric transformations, 예: 색상 흔들림, 흐림)에는 적합하지 않습니다.
  * 향후 더 효과적인 등변성 학습 전략과 다양한 변환 연산자를 탐색할 필요가 있습니다.
* **광범위한 영향:** 자율 주행, 로봇 내비게이션, 의료 영상 등 광범위한 실제 응용 분야에 긍정적인 영향을 미칠 수 있습니다. 다만, 실제 응용에서 부정확한 예측은 인명 안전 문제로 이어질 수 있으므로, 엄격한 보안 프로토콜 마련이 필요합니다.
* **향후 연구:** 쿼리 기반 객체 탐지 및 파놉틱 분할(panoptic segmentation)과 같은 더 넓은 범위의 밀집 예측(dense prediction) 작업에 이 학습 프레임워크를 적용할 가능성을 모색할 예정입니다.

## 📌 TL;DR

쿼리 기반 인스턴스 분할 모델은 개별 장면 내에서만 학습하여 쿼리 식별력과 견고성이 제한됩니다. 본 논문은 **데이터셋 수준의 인스턴스 고유성**과 **변환 등변성**을 학습 목표로 통합하는 새로운 학습 프레임워크를 제안합니다. 이는 쿼리가 전체 데이터셋에서 인스턴스를 고유하게 식별하도록 강제하고, 기하학적 변환에 대해 쿼리와 특징이 예측 가능하게 변하도록 하여 쿼리 식별력을 향상시킵니다. 결과적으로, 기존 모델 아키텍처나 추론 속도 변경 없이 COCO에서 최대 +3.2 $AP$ 성능 향상을 달성하며, 인스턴스 분리 능력과 견고성을 크게 개선했습니다.
